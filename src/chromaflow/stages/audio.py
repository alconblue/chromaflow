"""Audio processing stage: transcription and diarization.

This stage handles:
- Speech-to-text transcription via faster-whisper
- Speaker diarization via pyannote
- Word-level timestamp alignment

Models are lazy-loaded to avoid startup overhead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline as PyannotePipeline

logger = logging.getLogger(__name__)

# Lazy-loaded model references
_whisper_model: WhisperModel | None = None
_whisper_model_size: str | None = None
_whisper_device: str | None = None

_diarization_pipeline: PyannotePipeline | None = None
_diarization_device: str | None = None


class AudioProcessingError(Exception):
    """Error during audio processing."""

    pass


@dataclass
class TranscriptSegment:
    """A transcribed segment with speaker and timing info."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Transcribed text
    speaker: str | None = None  # Speaker label (if diarization enabled)
    confidence: float = 1.0  # Transcription confidence (avg_logprob converted)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class TranscriptWord:
    """A single word with precise timing."""

    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionResult:
    """Complete transcription result with segments and words."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    words: list[TranscriptWord] = field(default_factory=list)
    language: str = "en"
    language_probability: float = 1.0

    @property
    def full_text(self) -> str:
        """Get the full transcript as a single string."""
        return " ".join(seg.text.strip() for seg in self.segments)


@dataclass
class DiarizationSegment:
    """A speaker segment from diarization."""

    start: float
    end: float
    speaker: str


def _get_compute_type(device: str) -> str:
    """Get the appropriate compute type for the device.

    Args:
        device: Target device (cuda, mps, cpu).

    Returns:
        Compute type string for faster-whisper.
    """
    if device == "cuda":
        return "float16"  # Best for NVIDIA GPUs
    elif device == "mps":
        return "float32"  # MPS doesn't support float16 well yet
    else:
        return "int8"  # CPU optimization


def _load_whisper_model(
    model_size: str = "small",
    device: str = "cpu",
) -> WhisperModel:
    """Lazy-load the Whisper model.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v3).
        device: Compute device.

    Returns:
        Loaded WhisperModel instance.
    """
    global _whisper_model, _whisper_model_size, _whisper_device

    # Return cached model if same config
    if (
        _whisper_model is not None
        and _whisper_model_size == model_size
        and _whisper_device == device
    ):
        return _whisper_model

    start_time = time.perf_counter()
    logger.info(f"Loading Whisper model '{model_size}' on {device}...")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise AudioProcessingError(
            f"Failed to import faster-whisper: {e}\n"
            "Install with: pip install faster-whisper"
        )

    compute_type = _get_compute_type(device)

    # Map device for faster-whisper (uses "cuda" or "cpu", not "mps")
    fw_device = "cuda" if device == "cuda" else "cpu"

    try:
        _whisper_model = WhisperModel(
            model_size,
            device=fw_device,
            compute_type=compute_type,
        )
        _whisper_model_size = model_size
        _whisper_device = device

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Whisper model loaded in {elapsed:.2f}s "
            f"(size={model_size}, device={fw_device}, compute_type={compute_type})"
        )

        return _whisper_model

    except Exception as e:
        raise AudioProcessingError(f"Failed to load Whisper model: {e}") from e


def _load_diarization_pipeline(
    hf_token: str,
    device: str = "cpu",
) -> PyannotePipeline:
    """Lazy-load the pyannote diarization pipeline.

    Args:
        hf_token: HuggingFace authentication token.
        device: Compute device.

    Returns:
        Loaded pyannote Pipeline instance.
    """
    global _diarization_pipeline, _diarization_device

    # Return cached pipeline if same device
    if _diarization_pipeline is not None and _diarization_device == device:
        return _diarization_pipeline

    start_time = time.perf_counter()
    logger.info(f"Loading pyannote diarization pipeline on {device}...")

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        import torch
    except ImportError as e:
        raise AudioProcessingError(
            f"Failed to import pyannote.audio: {e}\n"
            "Install with: pip install pyannote-audio"
        )

    try:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        elif device == "mps" and torch.backends.mps.is_available():
            # pyannote may have limited MPS support
            try:
                pipeline.to(torch.device("mps"))
            except Exception:
                logger.warning("MPS not fully supported by pyannote, using CPU")
        # CPU is default, no action needed

        _diarization_pipeline = pipeline
        _diarization_device = device

        elapsed = time.perf_counter() - start_time
        logger.info(f"Diarization pipeline loaded in {elapsed:.2f}s (device={device})")

        return _diarization_pipeline

    except Exception as e:
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise AudioProcessingError(
                "HuggingFace authentication failed. Please check your HF_TOKEN.\n"
                "Make sure you've accepted the model terms at:\n"
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        raise AudioProcessingError(f"Failed to load diarization pipeline: {e}") from e


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    language: str | None = None,
    word_timestamps: bool = True,
) -> TranscriptionResult:
    """Transcribe audio file to text with timestamps.

    Uses faster-whisper for efficient transcription with word-level
    timestamps when available.

    Args:
        audio_path: Path to the audio file (WAV preferred, 16kHz mono).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: Compute device (cuda, mps, cpu).
        language: Force language (None for auto-detect).
        word_timestamps: Extract word-level timestamps.

    Returns:
        TranscriptionResult with segments and words.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        AudioProcessingError: If transcription fails.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_whisper_model(model_size, device)

    logger.info(f"Transcribing {audio_path.name}...")
    start_time = time.perf_counter()

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Collect segments and words
        segments: list[TranscriptSegment] = []
        words: list[TranscriptWord] = []

        for segment in segments_iter:
            # Convert avg_logprob to confidence (0-1 range)
            # avg_logprob is typically -0.5 to 0, we map to 0.5-1.0
            confidence = min(1.0, max(0.0, 1.0 + segment.avg_logprob))

            segments.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=confidence,
                )
            )

            # Extract words if available
            if segment.words:
                for word in segment.words:
                    words.append(
                        TranscriptWord(
                            word=word.word.strip(),
                            start=word.start,
                            end=word.end,
                            probability=word.probability,
                        )
                    )

        elapsed = time.perf_counter() - start_time
        duration = info.duration if hasattr(info, "duration") else 0

        rtf = elapsed / duration if duration > 0 else 0
        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{len(words)} words in {elapsed:.2f}s "
            f"(RTF: {rtf:.2f}x, lang: {info.language})"
        )

        return TranscriptionResult(
            segments=segments,
            words=words,
            language=info.language,
            language_probability=info.language_probability,
        )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, AudioProcessingError)):
            raise
        raise AudioProcessingError(f"Transcription failed: {e}") from e


def diarize(
    audio_path: Path,
    hf_token: str,
    device: str = "cpu",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationSegment]:
    """Perform speaker diarization on audio file.

    Uses pyannote-audio to identify "who spoke when" in the audio.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token for pyannote authentication.
        device: Compute device (cuda, mps, cpu).
        min_speakers: Minimum expected speakers (None for auto).
        max_speakers: Maximum expected speakers (None for auto).

    Returns:
        List of DiarizationSegment with speaker labels.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        AudioProcessingError: If diarization fails.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_diarization_pipeline(hf_token, device)

    logger.info(f"Diarizing {audio_path.name}...")
    start_time = time.perf_counter()

    try:
        # Run diarization
        diarization = pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Convert to DiarizationSegment list
        segments: list[DiarizationSegment] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                )
            )

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        elapsed = time.perf_counter() - start_time
        unique_speakers = len(set(s.speaker for s in segments))
        logger.info(
            f"Diarization complete: {len(segments)} segments, "
            f"{unique_speakers} speakers in {elapsed:.2f}s"
        )

        return segments

    except Exception as e:
        if isinstance(e, (FileNotFoundError, AudioProcessingError)):
            raise
        raise AudioProcessingError(f"Diarization failed: {e}") from e


def merge_transcription_with_diarization(
    transcription: TranscriptionResult,
    diarization: list[DiarizationSegment],
) -> TranscriptionResult:
    """Merge transcription segments with speaker labels from diarization.

    Assigns speaker labels to transcript segments based on temporal overlap.
    Uses majority overlap to determine the speaker for each segment.

    Args:
        transcription: Transcription result with segments.
        diarization: Diarization segments with speaker labels.

    Returns:
        Updated TranscriptionResult with speaker labels assigned.
    """
    if not diarization:
        return transcription

    def find_speaker(start: float, end: float) -> str | None:
        """Find the speaker with most overlap for a time range."""
        best_speaker = None
        best_overlap = 0.0

        for d_seg in diarization:
            # Calculate overlap
            overlap_start = max(start, d_seg.start)
            overlap_end = min(end, d_seg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg.speaker

        return best_speaker

    # Assign speakers to segments
    for segment in transcription.segments:
        segment.speaker = find_speaker(segment.start, segment.end)

    # Rename speakers to friendly names (Speaker A, Speaker B, etc.)
    speaker_map: dict[str, str] = {}
    speaker_counter = 0

    for segment in transcription.segments:
        if segment.speaker and segment.speaker not in speaker_map:
            speaker_map[segment.speaker] = f"Speaker {chr(65 + speaker_counter)}"
            speaker_counter += 1
        if segment.speaker:
            segment.speaker = speaker_map[segment.speaker]

    logger.info(f"Merged {len(speaker_map)} speakers into transcript")
    return transcription


def process_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    diarize_audio: bool = True,
    hf_token: str | None = None,
) -> TranscriptionResult:
    """Complete audio processing: transcription with optional diarization.

    This is the main entry point for audio processing, combining
    transcription and diarization into a single workflow.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size.
        device: Compute device.
        diarize_audio: Whether to perform speaker diarization.
        hf_token: HuggingFace token (required if diarize_audio=True).

    Returns:
        TranscriptionResult with segments, words, and speaker labels.

    Raises:
        ValueError: If diarization requested but no HF_TOKEN provided.
        FileNotFoundError: If audio file doesn't exist.
        AudioProcessingError: If processing fails.
    """
    if diarize_audio and not hf_token:
        raise ValueError(
            "HuggingFace token required for diarization. "
            "Set HF_TOKEN environment variable or pass hf_token parameter."
        )

    # Step 1: Transcribe
    transcription = transcribe(
        audio_path,
        model_size=model_size,
        device=device,
    )

    # Step 2: Diarize (optional)
    if diarize_audio and hf_token:
        diarization = diarize(
            audio_path,
            hf_token=hf_token,
            device=device,
        )

        # Step 3: Merge
        transcription = merge_transcription_with_diarization(
            transcription,
            diarization,
        )

    return transcription
