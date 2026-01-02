"""Tests for the audio processing stage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromaflow.stages.audio import (
    AudioProcessingError,
    DiarizationSegment,
    TranscriptSegment,
    TranscriptionResult,
    TranscriptWord,
    _get_compute_type,
    diarize,
    merge_transcription_with_diarization,
    process_audio,
    transcribe,
)


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_duration_property(self) -> None:
        """Should calculate duration correctly."""
        segment = TranscriptSegment(start=10.0, end=25.0, text="Hello world")
        assert segment.duration == 15.0

    def test_default_values(self) -> None:
        """Should have correct default values."""
        segment = TranscriptSegment(start=0.0, end=1.0, text="Test")
        assert segment.speaker is None
        assert segment.confidence == 1.0


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_full_text_property(self) -> None:
        """Should combine all segment text."""
        result = TranscriptionResult(
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="Hello"),
                TranscriptSegment(start=1.0, end=2.0, text="world"),
                TranscriptSegment(start=2.0, end=3.0, text="test"),
            ]
        )
        assert result.full_text == "Hello world test"

    def test_empty_segments(self) -> None:
        """Should handle empty segments."""
        result = TranscriptionResult()
        assert result.full_text == ""
        assert result.language == "en"


class TestGetComputeType:
    """Tests for _get_compute_type function."""

    def test_cuda_returns_float16(self) -> None:
        """CUDA should use float16 for best performance."""
        assert _get_compute_type("cuda") == "float16"

    def test_mps_returns_float32(self) -> None:
        """MPS should use float32 for compatibility."""
        assert _get_compute_type("mps") == "float32"

    def test_cpu_returns_int8(self) -> None:
        """CPU should use int8 for optimization."""
        assert _get_compute_type("cpu") == "int8"

    def test_unknown_device_returns_int8(self) -> None:
        """Unknown device should fall back to int8."""
        assert _get_compute_type("unknown") == "int8"


class TestTranscribe:
    """Tests for transcribe function."""

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            transcribe(temp_dir / "missing.wav")

    def test_faster_whisper_import_error(self, temp_dir: Path) -> None:
        """Should raise AudioProcessingError when faster-whisper not installed."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Reset cached model
        import chromaflow.stages.audio as audio_module
        audio_module._whisper_model = None

        with patch.dict("sys.modules", {"faster_whisper": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(AudioProcessingError, match="Failed to import"):
                    transcribe(audio_path)

    def test_transcribe_success(self, temp_dir: Path) -> None:
        """Should return TranscriptionResult on success."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Mock Whisper model
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = " Hello world "
        mock_segment.avg_logprob = -0.3
        mock_segment.words = [
            MagicMock(word=" Hello", start=0.0, end=2.0, probability=0.95),
            MagicMock(word=" world", start=2.0, end=5.0, probability=0.90),
        ]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        import chromaflow.stages.audio as audio_module
        audio_module._whisper_model = mock_model
        audio_module._whisper_model_size = "small"
        audio_module._whisper_device = "cpu"

        try:
            result = transcribe(audio_path)

            assert isinstance(result, TranscriptionResult)
            assert len(result.segments) == 1
            assert result.segments[0].text == "Hello world"
            assert len(result.words) == 2
            assert result.language == "en"
        finally:
            audio_module._whisper_model = None


class TestDiarize:
    """Tests for diarize function."""

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            diarize(temp_dir / "missing.wav", hf_token="test_token")

    def test_pyannote_import_error(self, temp_dir: Path) -> None:
        """Should raise AudioProcessingError when pyannote not installed."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Reset cached pipeline
        import chromaflow.stages.audio as audio_module
        audio_module._diarization_pipeline = None

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(AudioProcessingError, match="Failed to import"):
                diarize(audio_path, hf_token="test_token")

    def test_diarize_success(self, temp_dir: Path) -> None:
        """Should return list of DiarizationSegment on success."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Mock diarization result
        mock_turn1 = MagicMock()
        mock_turn1.start = 0.0
        mock_turn1.end = 5.0

        mock_turn2 = MagicMock()
        mock_turn2.start = 5.0
        mock_turn2.end = 10.0

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_diarization

        import chromaflow.stages.audio as audio_module
        audio_module._diarization_pipeline = mock_pipeline
        audio_module._diarization_device = "cpu"

        try:
            result = diarize(audio_path, hf_token="test_token")

            assert len(result) == 2
            assert all(isinstance(s, DiarizationSegment) for s in result)
            assert result[0].speaker == "SPEAKER_00"
            assert result[1].speaker == "SPEAKER_01"
        finally:
            audio_module._diarization_pipeline = None


class TestMergeTranscriptionWithDiarization:
    """Tests for merge_transcription_with_diarization function."""

    def test_empty_diarization(self) -> None:
        """Should return unchanged transcription for empty diarization."""
        transcription = TranscriptionResult(
            segments=[TranscriptSegment(start=0.0, end=5.0, text="Test")]
        )
        result = merge_transcription_with_diarization(transcription, [])
        assert result.segments[0].speaker is None

    def test_assigns_speakers_by_overlap(self) -> None:
        """Should assign speaker with most overlap."""
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="First segment"),
                TranscriptSegment(start=5.0, end=10.0, text="Second segment"),
            ]
        )
        diarization = [
            DiarizationSegment(start=0.0, end=4.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=4.0, end=10.0, speaker="SPEAKER_01"),
        ]

        result = merge_transcription_with_diarization(transcription, diarization)

        # First segment: 4s overlap with SPEAKER_00, 1s with SPEAKER_01
        assert result.segments[0].speaker == "Speaker A"
        # Second segment: 0s overlap with SPEAKER_00, 5s with SPEAKER_01
        assert result.segments[1].speaker == "Speaker B"

    def test_renames_speakers_to_friendly_names(self) -> None:
        """Should rename speakers to Speaker A, B, C, etc."""
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment(start=0.0, end=3.0, text="A"),
                TranscriptSegment(start=3.0, end=6.0, text="B"),
                TranscriptSegment(start=6.0, end=9.0, text="C"),
            ]
        )
        diarization = [
            DiarizationSegment(start=0.0, end=3.0, speaker="SPK_00"),
            DiarizationSegment(start=3.0, end=6.0, speaker="SPK_01"),
            DiarizationSegment(start=6.0, end=9.0, speaker="SPK_02"),
        ]

        result = merge_transcription_with_diarization(transcription, diarization)

        assert result.segments[0].speaker == "Speaker A"
        assert result.segments[1].speaker == "Speaker B"
        assert result.segments[2].speaker == "Speaker C"


class TestProcessAudio:
    """Tests for process_audio function."""

    def test_requires_hf_token_for_diarization(self, temp_dir: Path) -> None:
        """Should raise ValueError if diarization requested without token."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        with pytest.raises(ValueError, match="HuggingFace token required"):
            process_audio(audio_path, diarize_audio=True, hf_token=None)

    def test_process_without_diarization(self, temp_dir: Path) -> None:
        """Should work without diarization when diarize_audio=False."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Mock transcription
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Test"
        mock_segment.avg_logprob = -0.2
        mock_segment.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        import chromaflow.stages.audio as audio_module
        audio_module._whisper_model = mock_model
        audio_module._whisper_model_size = "small"
        audio_module._whisper_device = "cpu"

        try:
            result = process_audio(
                audio_path,
                diarize_audio=False,
                hf_token=None,
            )

            assert isinstance(result, TranscriptionResult)
            assert result.segments[0].speaker is None
        finally:
            audio_module._whisper_model = None

    def test_process_with_diarization(self, temp_dir: Path) -> None:
        """Should include speaker labels when diarization enabled."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")

        # Mock transcription
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Hello"
        mock_segment.avg_logprob = -0.2
        mock_segment.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Mock diarization
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = 5.0

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_diarization

        import chromaflow.stages.audio as audio_module
        audio_module._whisper_model = mock_model
        audio_module._whisper_model_size = "small"
        audio_module._whisper_device = "cpu"
        audio_module._diarization_pipeline = mock_pipeline
        audio_module._diarization_device = "cpu"

        try:
            result = process_audio(
                audio_path,
                diarize_audio=True,
                hf_token="test_token",
            )

            assert result.segments[0].speaker == "Speaker A"
        finally:
            audio_module._whisper_model = None
            audio_module._diarization_pipeline = None
