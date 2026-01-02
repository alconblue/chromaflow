"""Tests for the ChromaFlow Pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromaflow import Pipeline, PipelineError, VideoData
from chromaflow.config import PipelineConfig, ProcessingOptions
from chromaflow.models.schema import VideoMetadata
from chromaflow.stages.audio import TranscriptionResult, TranscriptSegment
from chromaflow.stages.scene import SceneBoundary
from chromaflow.stages.visual import VisualEmbeddingResult


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_init_local_mode_with_hf_token(self, hf_token_env: str, temp_dir: Path) -> None:
        """Pipeline should initialize in local mode with HF_TOKEN set."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir))
        assert pipeline.config.mode == "local"
        assert pipeline.config.get_hf_token() == hf_token_env

    def test_init_local_mode_no_diarize_no_token(
        self, no_hf_token_env: None, temp_dir: Path
    ) -> None:
        """Pipeline should initialize without HF_TOKEN if diarization is disabled."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
        )
        assert pipeline.config.mode == "local"
        assert pipeline.config.options.diarize is False

    def test_init_fails_without_hf_token_when_diarize_enabled(
        self, no_hf_token_env: None, temp_dir: Path
    ) -> None:
        """Pipeline should fail to initialize if diarization is enabled but HF_TOKEN is missing."""
        with pytest.raises(ValueError, match="HuggingFace token"):
            Pipeline(
                mode="local",
                output_dir=str(temp_dir),
                options={"diarize": True},
            )

    def test_init_cloud_mode_not_implemented(self, hf_token_env: str, temp_dir: Path) -> None:
        """Cloud mode should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Cloud mode"):
            Pipeline(mode="cloud", api_key="test_key", output_dir=str(temp_dir))

    def test_init_creates_output_dir(self, hf_token_env: str, temp_dir: Path) -> None:
        """Pipeline should create output directory if it doesn't exist."""
        output_dir = temp_dir / "new_output_dir"
        assert not output_dir.exists()

        Pipeline(mode="local", output_dir=str(output_dir))
        assert output_dir.exists()

    def test_init_with_search_enabled(self, hf_token_env: str, temp_dir: Path) -> None:
        """Pipeline should initialize ChromaDB store when search is enabled."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=True)
        assert pipeline._store is not None
        assert pipeline.get_store() is not None

    def test_init_with_search_disabled(self, hf_token_env: str, temp_dir: Path) -> None:
        """Pipeline should not initialize ChromaDB store when search is disabled."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        assert pipeline._store is None
        assert pipeline.get_store() is None


class TestPipelineProcess:
    """Tests for Pipeline.process() method with mocked stages."""

    @pytest.fixture
    def mock_stages(self):
        """Mock all processing stages."""
        with patch("chromaflow.pipeline.extract_metadata") as mock_metadata, \
             patch("chromaflow.pipeline.is_audio_only") as mock_is_audio, \
             patch("chromaflow.pipeline.detect_scenes") as mock_scenes, \
             patch("chromaflow.pipeline.extract_audio") as mock_extract_audio, \
             patch("chromaflow.pipeline.process_audio") as mock_audio, \
             patch("chromaflow.pipeline.process_visual") as mock_visual:

            # Default mock returns
            mock_metadata.return_value = VideoMetadata(
                duration=60.0,
                resolution="1920x1080",
                fps=30.0,
                format="mp4",
                audio_channels=2,
                sample_rate=44100,
            )

            mock_is_audio.return_value = False

            mock_scenes.return_value = [
                SceneBoundary(start=0.0, end=30.0, keyframe_time=15.0),
                SceneBoundary(start=30.0, end=60.0, keyframe_time=45.0),
            ]

            mock_extract_audio.return_value = Path("/tmp/audio.wav")

            mock_audio.return_value = TranscriptionResult(
                segments=[
                    TranscriptSegment(start=0.0, end=15.0, text="First segment.", speaker="Speaker A"),
                    TranscriptSegment(start=15.0, end=30.0, text="Second segment.", speaker="Speaker B"),
                    TranscriptSegment(start=30.0, end=45.0, text="Third segment.", speaker="Speaker A"),
                    TranscriptSegment(start=45.0, end=60.0, text="Fourth segment.", speaker="Speaker B"),
                ],
                language="en",
                language_probability=0.99,
            )

            mock_visual.return_value = [
                VisualEmbeddingResult(embedding=[0.1, 0.2, 0.3], timestamp=15.0, frame_path=Path("/tmp/frame_000.jpg")),
                VisualEmbeddingResult(embedding=[0.4, 0.5, 0.6], timestamp=45.0, frame_path=Path("/tmp/frame_001.jpg")),
            ]

            yield {
                "metadata": mock_metadata,
                "is_audio": mock_is_audio,
                "scenes": mock_scenes,
                "extract_audio": mock_extract_audio,
                "audio": mock_audio,
                "visual": mock_visual,
            }

    def test_process_returns_video_data(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should return a VideoData object."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        assert isinstance(result, VideoData)
        assert result.file_id.startswith("vid_")
        assert result.source_path == str(sample_video_path.absolute())

    def test_process_file_not_found(self, hf_token_env: str, temp_dir: Path) -> None:
        """process() should raise FileNotFoundError for missing files."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        with pytest.raises(FileNotFoundError, match="not found"):
            pipeline.process("/nonexistent/video.mp4")

    def test_process_unsupported_format(
        self, hf_token_env: str, temp_dir: Path
    ) -> None:
        """process() should raise ValueError for unsupported formats."""
        bad_file = temp_dir / "test.xyz"
        bad_file.write_bytes(b"not a video")

        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline.process(bad_file)

    def test_process_creates_chunks_from_scenes(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should create chunks aligned with scenes."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        # Should have 2 chunks (one per scene)
        assert len(result.chunks) == 2

        # First chunk should cover 0-30s
        assert result.chunks[0].start == 0.0
        assert result.chunks[0].end == 30.0

        # Second chunk should cover 30-60s
        assert result.chunks[1].start == 30.0
        assert result.chunks[1].end == 60.0

    def test_process_aligns_transcript_with_scenes(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should align transcript segments with scene boundaries."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        # First chunk (0-30s) should have first two transcript segments
        assert "First segment" in result.chunks[0].transcript
        assert "Second segment" in result.chunks[0].transcript

        # Second chunk (30-60s) should have last two transcript segments
        assert "Third segment" in result.chunks[1].transcript
        assert "Fourth segment" in result.chunks[1].transcript

    def test_process_includes_speaker_info_when_diarized(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should include speaker information when diarization enabled."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        # Chunks should have speaker information
        assert len(result.chunks[0].speakers) > 0
        speaker_labels = {s.label for chunk in result.chunks for s in chunk.speakers}
        assert "Speaker A" in speaker_labels
        assert "Speaker B" in speaker_labels

    def test_process_audio_only_file(
        self, hf_token_env: str, sample_audio_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should handle audio-only files with synthetic segments."""
        mock_stages["is_audio"].return_value = True
        mock_stages["metadata"].return_value = VideoMetadata(
            duration=90.0,
            resolution="0x0",
            fps=1.0,
            format="wav",
            audio_channels=1,
            sample_rate=16000,
        )

        # Mock synthetic segments for audio
        with patch("chromaflow.pipeline.detect_scenes_for_audio_only") as mock_audio_scenes:
            mock_audio_scenes.return_value = [
                SceneBoundary(start=0.0, end=30.0, keyframe_time=15.0),
                SceneBoundary(start=30.0, end=60.0, keyframe_time=45.0),
                SceneBoundary(start=60.0, end=90.0, keyframe_time=75.0),
            ]

            pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
            result = pipeline.process(sample_audio_path)

            assert isinstance(result, VideoData)
            assert "audio" in result.summary.lower()
            mock_audio_scenes.assert_called_once_with(90.0)

    def test_process_with_options_override(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should respect options override."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(
            sample_video_path,
            options={"diarize": False},
        )

        # Should still work, but audio processing called with diarize=False
        mock_stages["audio"].assert_called_once()
        call_kwargs = mock_stages["audio"].call_args[1]
        assert call_kwargs["diarize_audio"] is False

    def test_process_wraps_stage_errors(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should wrap stage errors in PipelineError."""
        from chromaflow.stages.ingest import IngestError

        mock_stages["metadata"].side_effect = IngestError("ffprobe failed")

        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        with pytest.raises(PipelineError, match="Processing failed"):
            pipeline.process(sample_video_path)

    def test_process_generates_summary(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should generate a meaningful summary."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        assert "1.0-minute" in result.summary
        assert "2 segments" in result.summary
        assert "2 speakers" in result.summary

    def test_process_includes_visual_embeddings(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should include visual embeddings in chunks."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path, extract_frames=True)

        # Check that visual embeddings are included
        assert len(result.chunks[0].visual_embedding) > 0
        assert result.chunks[0].visual_embedding == [0.1, 0.2, 0.3]
        assert result.chunks[1].visual_embedding == [0.4, 0.5, 0.6]

    def test_process_includes_screenshot_paths(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should include screenshot paths in chunks."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path, extract_frames=True)

        assert result.chunks[0].screenshot_path == "/tmp/frame_000.jpg"
        assert result.chunks[1].screenshot_path == "/tmp/frame_001.jpg"

    def test_process_skips_visual_for_audio_only(
        self, hf_token_env: str, sample_audio_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """process() should skip visual processing for audio-only files."""
        mock_stages["is_audio"].return_value = True
        mock_stages["metadata"].return_value = VideoMetadata(
            duration=60.0, resolution="0x0", fps=1.0, format="wav",
            audio_channels=1, sample_rate=16000,
        )

        with patch("chromaflow.pipeline.detect_scenes_for_audio_only") as mock_audio_scenes:
            mock_audio_scenes.return_value = [
                SceneBoundary(start=0.0, end=60.0, keyframe_time=30.0),
            ]

            pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
            result = pipeline.process(sample_audio_path)

            # Visual processing should not be called
            mock_stages["visual"].assert_not_called()

            # Chunks should have empty visual embeddings
            assert result.chunks[0].visual_embedding == []


class TestVideoData:
    """Tests for VideoData model."""

    @pytest.fixture
    def mock_stages(self):
        """Mock all processing stages."""
        with patch("chromaflow.pipeline.extract_metadata") as mock_metadata, \
             patch("chromaflow.pipeline.is_audio_only") as mock_is_audio, \
             patch("chromaflow.pipeline.detect_scenes") as mock_scenes, \
             patch("chromaflow.pipeline.extract_audio") as mock_extract_audio, \
             patch("chromaflow.pipeline.process_audio") as mock_audio, \
             patch("chromaflow.pipeline.process_visual") as mock_visual:

            mock_metadata.return_value = VideoMetadata(
                duration=60.0, resolution="1920x1080", fps=30.0,
                format="mp4", audio_channels=2, sample_rate=44100,
            )
            mock_is_audio.return_value = False
            mock_scenes.return_value = [SceneBoundary(start=0.0, end=60.0, keyframe_time=30.0)]
            mock_extract_audio.return_value = Path("/tmp/audio.wav")
            mock_audio.return_value = TranscriptionResult(
                segments=[TranscriptSegment(start=0.0, end=60.0, text="Test transcript.")],
                language="en", language_probability=0.99,
            )
            mock_visual.return_value = [
                VisualEmbeddingResult(embedding=[0.1, 0.2], timestamp=30.0, frame_path=None),
            ]
            yield

    def test_video_data_to_dict(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """VideoData.to_dict() should return a dictionary."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        data_dict = result.to_dict()
        assert isinstance(data_dict, dict)
        assert "file_id" in data_dict
        assert "chunks" in data_dict
        assert "metadata" in data_dict

    def test_video_data_to_json(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """VideoData.to_json() should return valid JSON string."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert '"file_id"' in json_str

    def test_video_data_to_json_file(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """VideoData.to_json() should write to file when path is provided."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        output_path = temp_dir / "output.json"
        result.to_json(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert '"file_id"' in content

    def test_video_data_search_not_available_when_disabled(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """search() should raise RuntimeError when store is not configured."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)
        result = pipeline.process(sample_video_path)

        with pytest.raises(RuntimeError, match="not available"):
            result.search("test query")

    def test_video_data_search_available_when_enabled(
        self, hf_token_env: str, sample_video_path: Path, temp_dir: Path, mock_stages
    ) -> None:
        """search() should work when ChromaDB store is configured."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=True)
        result = pipeline.process(sample_video_path)

        # Should not raise, and should return results
        search_results = result.search("test", top_k=5)
        assert isinstance(search_results, list)


class TestProcessingOptions:
    """Tests for ProcessingOptions configuration."""

    def test_default_options(self) -> None:
        """Default options should have expected values."""
        options = ProcessingOptions()
        assert options.diarize is True
        assert options.ocr is False
        assert options.whisper_model.value == "small"
        assert options.clip_model == "clip-ViT-L-14"

    def test_options_from_dict(self) -> None:
        """Options should be creatable from dict."""
        options = ProcessingOptions(diarize=False, whisper_model="tiny")
        assert options.diarize is False
        assert options.whisper_model.value == "tiny"


class TestBuildChunks:
    """Tests for the _build_chunks method."""

    def test_build_chunks_aligns_correctly(self, hf_token_env: str, temp_dir: Path) -> None:
        """_build_chunks should align transcript to scenes correctly."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        scenes = [
            SceneBoundary(start=0.0, end=10.0, keyframe_time=5.0),
            SceneBoundary(start=10.0, end=20.0, keyframe_time=15.0),
        ]

        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="Hello", speaker="Speaker A"),
                TranscriptSegment(start=5.0, end=12.0, text="world", speaker="Speaker B"),
                TranscriptSegment(start=12.0, end=20.0, text="test", speaker="Speaker A"),
            ]
        )

        chunks = pipeline._build_chunks("test_id", scenes, transcription, diarize=True)

        assert len(chunks) == 2

        # First chunk (0-10s) should have "Hello" and "world" (world overlaps)
        assert "Hello" in chunks[0].transcript
        assert "world" in chunks[0].transcript

        # Second chunk (10-20s) should have "world" and "test"
        assert "world" in chunks[1].transcript
        assert "test" in chunks[1].transcript

    def test_build_chunks_empty_transcript(self, hf_token_env: str, temp_dir: Path) -> None:
        """_build_chunks should handle empty transcript."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        scenes = [SceneBoundary(start=0.0, end=10.0, keyframe_time=5.0)]
        transcription = TranscriptionResult(segments=[])

        chunks = pipeline._build_chunks("test_id", scenes, transcription, diarize=False)

        assert len(chunks) == 1
        assert chunks[0].transcript == ""
        assert chunks[0].speakers == []

    def test_build_chunks_with_visual_results(self, hf_token_env: str, temp_dir: Path) -> None:
        """_build_chunks should include visual embeddings when provided."""
        pipeline = Pipeline(mode="local", output_dir=str(temp_dir), enable_search=False)

        scenes = [
            SceneBoundary(start=0.0, end=30.0, keyframe_time=15.0),
        ]

        transcription = TranscriptionResult(
            segments=[TranscriptSegment(start=0.0, end=30.0, text="Test")]
        )

        visual_results = {
            15.0: VisualEmbeddingResult(
                embedding=[0.1, 0.2, 0.3],
                timestamp=15.0,
                frame_path=Path("/tmp/frame.jpg"),
            )
        }

        chunks = pipeline._build_chunks(
            "test_id", scenes, transcription, diarize=False, visual_results=visual_results
        )

        assert len(chunks) == 1
        assert chunks[0].visual_embedding == [0.1, 0.2, 0.3]
        assert chunks[0].screenshot_path == "/tmp/frame.jpg"
