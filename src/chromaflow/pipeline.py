"""Main pipeline orchestration for ChromaFlow."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from tqdm import tqdm

from chromaflow.config import PipelineConfig, ProcessingOptions
from chromaflow.models.schema import Chunk, ChunkSpeaker, VideoData, VideoMetadata
from chromaflow.stages.audio import (
    AudioProcessingError,
    TranscriptionResult,
    process_audio,
)
from chromaflow.stages.ingest import (
    IngestError,
    extract_audio,
    extract_metadata,
    is_audio_only,
)
from chromaflow.stages.scene import (
    SceneBoundary,
    SceneDetectionError,
    detect_scenes,
    detect_scenes_for_audio_only,
)
from chromaflow.stages.visual import (
    VisualProcessingError,
    process_visual,
)
from chromaflow.store.chroma import ChromaStore, ChromaStoreError
from chromaflow.utils.logging import get_logger, log_cloud_nudge

logger = get_logger(__name__)

# Supported input formats
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav"}
SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS


class PipelineError(Exception):
    """Error during pipeline processing."""

    pass


def _create_pipeline_progress(total_stages: int, desc: str = "Processing") -> tqdm:
    """Create a progress bar for pipeline stages."""
    return tqdm(
        total=total_stages,
        desc=desc,
        unit="stage",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]",
        leave=True,
    )


class Pipeline:
    """ChromaFlow processing pipeline.

    Transforms video/audio files into RAG-ready artifacts with aligned
    transcripts, speaker diarization, and visual embeddings.

    Example:
        >>> import chromaflow
        >>> pipeline = chromaflow.Pipeline(mode="local")
        >>> video_data = pipeline.process("meeting.mp4")
        >>> results = video_data.search("budget discussion")
    """

    def __init__(
        self,
        mode: str = "local",
        api_key: str | None = None,
        device: str | None = None,
        audio_device: str | None = None,
        output_dir: str = "./chromaflow_output",
        hf_token: str | None = None,
        options: dict[str, Any] | ProcessingOptions | None = None,
        enable_search: bool = True,
        persist_store: bool = False,
    ) -> None:
        """Initialize a ChromaFlow pipeline.

        Args:
            mode: Processing mode - 'local' or 'cloud'.
            api_key: API key for cloud mode (required if mode='cloud').
            device: Compute device ('cuda', 'mps', 'cpu') or None for auto-detect.
            audio_device: Override device for audio processing (Whisper/pyannote).
                Use 'cpu' if you encounter CUDA crashes with faster-whisper.
            output_dir: Directory for output files (frames, etc.).
            hf_token: HuggingFace token for pyannote. Falls back to HF_TOKEN env var.
            options: Processing options dict or ProcessingOptions instance.
            enable_search: Enable ChromaDB for search functionality.
            persist_store: Persist ChromaDB to disk (in output_dir/chromadb).

        Raises:
            NotImplementedError: If mode='cloud' (not implemented in MVP).
            ValueError: If diarization is enabled but HF_TOKEN is not set.
        """
        # Convert options dict to ProcessingOptions if needed
        if options is None:
            processing_options = ProcessingOptions()
        elif isinstance(options, dict):
            processing_options = ProcessingOptions(**options)
        else:
            processing_options = options

        self.config = PipelineConfig(
            mode=mode,  # type: ignore[arg-type]
            api_key=api_key,
            device=device,
            output_dir=output_dir,
            hf_token=hf_token,
            options=processing_options,
        )

        # Validate cloud mode
        if self.config.mode == "cloud":
            raise NotImplementedError(
                "Cloud mode is not yet available in ChromaFlow OSS.\n"
                "Use mode='local' for local processing.\n"
                "Cloud processing coming soon: https://chromaflow.ai"
            )

        # Validate diarization requirements (fail fast)
        self.config.validate_for_diarization()

        # Log device info
        self._device = self.config.get_device()
        # Audio device can be overridden (useful for CUDA crashes in faster-whisper)
        self._audio_device = audio_device if audio_device else self._device
        if audio_device and audio_device != self._device:
            logger.info(f"ChromaFlow pipeline initialized (device={self._device}, audio_device={self._audio_device})")
        else:
            logger.info(f"ChromaFlow pipeline initialized (device={self._device})")

        # Create output directory
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB store if search enabled
        self._enable_search = enable_search
        self._store: ChromaStore | None = None

        if enable_search:
            persist_dir = str(self._output_dir / "chromadb") if persist_store else None
            self._store = ChromaStore(
                collection_name="chromaflow_videos",
                persist_directory=persist_dir,
            )

    def process(
        self,
        source: str | Path,
        options: dict[str, Any] | None = None,
        extract_frames: bool = True,
    ) -> VideoData:
        """Process a video or audio file into RAG-ready artifacts.

        Args:
            source: Path to the input file (.mp4, .mov, .avi, .mkv, .mp3, .wav).
            options: Override processing options for this file.
            extract_frames: Whether to extract keyframes and generate visual embeddings.

        Returns:
            VideoData object containing chunks, embeddings, and metadata.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the file format is not supported.
            PipelineError: If processing fails.
        """
        source_path = Path(source)
        start_time = time.perf_counter()

        # Validate file exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Validate format
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        # Merge options
        effective_options = self.config.options.model_copy()
        if options:
            for key, value in options.items():
                if hasattr(effective_options, key):
                    setattr(effective_options, key, value)

        logger.info(f"Processing: {source_path.name}")

        # Generate file ID
        file_id = self._generate_file_id(source_path)

        # Determine number of stages (6 if search enabled, 5 otherwise; 4/5 if audio-only)
        audio_only = is_audio_only(source_path)
        total_stages = 5 if audio_only else 6
        if self._store is not None:
            total_stages += 1

        # Create main progress bar
        pbar = _create_pipeline_progress(total_stages, f"Processing {source_path.name}")

        try:
            # Stage 1: Ingest - Extract metadata
            pbar.set_description("Stage 1: Extracting metadata")
            metadata = extract_metadata(source_path)
            pbar.update(1)

            # Check for cloud nudge
            duration_hours = metadata.duration / 3600
            log_cloud_nudge(logger, video_count=1, duration_hours=duration_hours)

            # Stage 2: Scene Detection (or synthetic segments for audio)
            if audio_only:
                pbar.set_description("Stage 2: Creating audio segments")
                scenes = detect_scenes_for_audio_only(metadata.duration)
                audio_path = source_path  # Use source directly for audio files
                pbar.update(1)
            else:
                pbar.set_description(f"Stage 2: Detecting scenes ({metadata.duration/60:.1f} min video)")
                scenes = detect_scenes(
                    source_path,
                    threshold=effective_options.scene_threshold,
                    min_scene_len=effective_options.min_scene_duration,
                )
                pbar.update(1)

                # Extract audio for transcription
                pbar.set_description("Stage 2b: Extracting audio track")
                audio_dir = self._output_dir / file_id
                audio_path = extract_audio(source_path, audio_dir)

            # Stage 3: Audio Processing (transcription + optional diarization)
            pbar.set_description(f"Stage 3: Transcribing audio ({metadata.duration/60:.1f} min)")
            transcription = process_audio(
                audio_path,
                model_size=effective_options.whisper_model.value,
                device=self._audio_device,  # Use audio_device (may differ from visual device)
                diarize_audio=effective_options.diarize,
                hf_token=self.config.get_hf_token(),
            )
            pbar.update(1)

            # Stage 4: Visual Processing (for video files only)
            visual_results = {}
            if not audio_only and extract_frames:
                pbar.set_description(f"Stage 4: Extracting {len(scenes)} keyframes + CLIP embeddings")
                keyframe_times = [scene.keyframe_time for scene in scenes]
                frames_dir = self._output_dir / file_id / "frames"

                visual_embeddings = process_visual(
                    source=source_path,
                    keyframe_times=keyframe_times,
                    output_dir=frames_dir,
                    file_id=file_id,
                    model_name=effective_options.clip_model,
                    device=self._device,
                )

                # Map keyframe times to visual results
                for result in visual_embeddings:
                    visual_results[result.timestamp] = result
                pbar.update(1)
            elif not audio_only:
                pbar.update(1)  # Skip visual stage but count it

            # Stage 5: Build chunks by aligning scenes with transcription
            pbar.set_description("Stage 5: Building aligned chunks")
            chunks = self._build_chunks(
                file_id=file_id,
                scenes=scenes,
                transcription=transcription,
                diarize=effective_options.diarize,
                visual_results=visual_results,
            )
            pbar.update(1)

            # Build summary
            unique_speakers = set()
            for chunk in chunks:
                for speaker in chunk.speakers:
                    unique_speakers.add(speaker.label)

            summary = (
                f"A {metadata.duration / 60:.1f}-minute "
                f"{'audio' if audio_only else 'video'} with "
                f"{len(chunks)} segments"
            )
            if effective_options.diarize and unique_speakers:
                summary += f" and {len(unique_speakers)} speakers"
            summary += "."

            # Create VideoData
            video_data = VideoData(
                file_id=file_id,
                source_path=str(source_path.absolute()),
                metadata=metadata,
                chunks=chunks,
                summary=summary,
            )

            # Stage 6: Index in ChromaDB for search
            if self._store is not None:
                pbar.set_description(f"Stage 6: Indexing {len(chunks)} chunks in ChromaDB")
                self._store.add_chunks(chunks, file_id)
                video_data._store = self._store
                pbar.update(1)

            elapsed = time.perf_counter() - start_time
            rtf = elapsed / metadata.duration if metadata.duration > 0 else 0

            pbar.set_description(f"Complete: {len(chunks)} chunks in {elapsed:.1f}s (RTF: {rtf:.2f}x)")
            pbar.close()

            logger.info(
                f"Processing complete: {len(chunks)} chunks in {elapsed:.2f}s "
                f"(RTF: {rtf:.2f}x)"
            )

            return video_data

        except (IngestError, SceneDetectionError, AudioProcessingError, VisualProcessingError, ChromaStoreError) as e:
            raise PipelineError(f"Processing failed: {e}") from e

    def _generate_file_id(self, source_path: Path) -> str:
        """Generate a unique file ID based on path and content hash."""
        # Use first 8 chars of content hash + short UUID for uniqueness
        try:
            content_hash = hashlib.md5(source_path.read_bytes()[:8192]).hexdigest()[:8]
        except Exception:
            content_hash = "00000000"

        short_uuid = uuid.uuid4().hex[:4]
        return f"vid_{content_hash}_{short_uuid}"

    def _build_chunks(
        self,
        file_id: str,
        scenes: list[SceneBoundary],
        transcription: TranscriptionResult,
        diarize: bool,
        visual_results: dict | None = None,
    ) -> list[Chunk]:
        """Build chunks by aligning scenes with transcription.

        For each scene, we find the transcript segments that overlap
        with it and combine them into a single chunk.

        Args:
            file_id: Video file ID for chunk IDs.
            scenes: Detected scene boundaries.
            transcription: Transcription result with segments.
            diarize: Whether to include speaker information.
            visual_results: Optional dict mapping keyframe times to VisualEmbeddingResult.

        Returns:
            List of aligned Chunk objects.
        """
        chunks: list[Chunk] = []
        visual_results = visual_results or {}

        for i, scene in enumerate(scenes):
            chunk_id = f"{file_id}_chunk_{i:03d}"

            # Find transcript segments that overlap with this scene
            scene_transcript_parts: list[str] = []
            scene_speakers: list[ChunkSpeaker] = []

            for seg in transcription.segments:
                # Check for overlap
                overlap_start = max(scene.start, seg.start)
                overlap_end = min(scene.end, seg.end)

                if overlap_end > overlap_start:
                    # This segment overlaps with the scene
                    scene_transcript_parts.append(seg.text)

                    if diarize and seg.speaker:
                        scene_speakers.append(
                            ChunkSpeaker(
                                label=seg.speaker,
                                start=max(seg.start, scene.start),
                                end=min(seg.end, scene.end),
                                text=seg.text,
                            )
                        )

            # Combine transcript parts
            transcript_text = " ".join(scene_transcript_parts).strip()

            # Get visual embedding and screenshot path if available
            visual_embedding: list[float] = []
            screenshot_path: str | None = None

            if scene.keyframe_time in visual_results:
                visual_result = visual_results[scene.keyframe_time]
                visual_embedding = visual_result.embedding
                if visual_result.frame_path:
                    screenshot_path = str(visual_result.frame_path)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    start=scene.start,
                    end=scene.end,
                    transcript=transcript_text,
                    speakers=scene_speakers,
                    visual_embedding=visual_embedding,
                    screenshot_path=screenshot_path,
                )
            )

        return chunks

    def get_store(self) -> ChromaStore | None:
        """Get the ChromaDB store instance.

        Returns:
            ChromaStore instance if search is enabled, None otherwise.
        """
        return self._store
