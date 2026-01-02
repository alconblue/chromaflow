"""Tests for the ingest stage."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromaflow.stages.ingest import (
    IngestError,
    ProbeResult,
    _parse_probe_result,
    extract_audio,
    extract_metadata,
    is_audio_only,
    probe_file,
    validate_file,
)


class TestValidateFile:
    """Tests for validate_file function."""

    def test_validate_existing_mp4(self, temp_dir: Path) -> None:
        """Should pass for existing .mp4 file."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")
        validate_file(video_path)  # Should not raise

    def test_validate_existing_wav(self, temp_dir: Path) -> None:
        """Should pass for existing .wav file."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b"fake audio")
        validate_file(audio_path)  # Should not raise

    def test_validate_nonexistent_file(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_file(temp_dir / "nonexistent.mp4")

    def test_validate_unsupported_format(self, temp_dir: Path) -> None:
        """Should raise ValueError for unsupported format."""
        bad_file = temp_dir / "test.xyz"
        bad_file.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_file(bad_file)

    @pytest.mark.parametrize("ext", [".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav"])
    def test_validate_all_supported_formats(self, temp_dir: Path, ext: str) -> None:
        """Should pass for all supported formats."""
        test_file = temp_dir / f"test{ext}"
        test_file.write_bytes(b"data")
        validate_file(test_file)  # Should not raise


class TestParseProbeResult:
    """Tests for _parse_probe_result function."""

    def test_parse_video_with_audio(self) -> None:
        """Should parse video file with audio track."""
        data = {
            "format": {
                "duration": "120.5",
                "format_name": "mov,mp4,m4a,3gp",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "44100",
                },
            ],
        }

        result = _parse_probe_result(data)

        assert result.duration == 120.5
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0
        assert result.format_name == "mov"
        assert result.has_video is True
        assert result.has_audio is True
        assert result.audio_channels == 2
        assert result.audio_sample_rate == 44100
        assert result.codec_name == "h264"

    def test_parse_audio_only(self) -> None:
        """Should parse audio-only file."""
        data = {
            "format": {
                "duration": "180.0",
                "format_name": "wav",
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "channels": 1,
                    "sample_rate": "16000",
                },
            ],
        }

        result = _parse_probe_result(data)

        assert result.duration == 180.0
        assert result.width is None
        assert result.height is None
        assert result.fps is None
        assert result.has_video is False
        assert result.has_audio is True
        assert result.audio_channels == 1
        assert result.audio_sample_rate == 16000

    def test_parse_fractional_fps(self) -> None:
        """Should parse fractional frame rate (e.g., 29.97)."""
        data = {
            "format": {"duration": "60.0", "format_name": "mp4"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "30000/1001",  # 29.97 fps
                },
            ],
        }

        result = _parse_probe_result(data)

        assert abs(result.fps - 29.97) < 0.01

    def test_parse_empty_streams(self) -> None:
        """Should handle empty streams gracefully."""
        data = {
            "format": {"duration": "0", "format_name": "unknown"},
            "streams": [],
        }

        result = _parse_probe_result(data)

        assert result.duration == 0.0
        assert result.has_video is False
        assert result.has_audio is False


class TestProbeFile:
    """Tests for probe_file function."""

    def test_probe_file_ffprobe_not_found(self, temp_dir: Path) -> None:
        """Should raise IngestError when ffprobe is not installed."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(IngestError, match="ffprobe not found"):
                probe_file(video_path)

    def test_probe_file_timeout(self, temp_dir: Path) -> None:
        """Should raise IngestError on timeout."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            with pytest.raises(IngestError, match="timed out"):
                probe_file(video_path)

    def test_probe_file_ffprobe_error(self, temp_dir: Path) -> None:
        """Should raise IngestError when ffprobe returns error."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(IngestError, match="ffprobe failed"):
                probe_file(video_path)

    def test_probe_file_success(self, temp_dir: Path) -> None:
        """Should return ProbeResult on success."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "format": {"duration": "60.0", "format_name": "mp4"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "44100"}
            ]
        }"""

        with patch("subprocess.run", return_value=mock_result):
            result = probe_file(video_path)

        assert isinstance(result, ProbeResult)
        assert result.duration == 60.0
        assert result.width == 1920
        assert result.height == 1080


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_metadata_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_metadata(temp_dir / "missing.mp4")

    def test_extract_metadata_unsupported_format(self, temp_dir: Path) -> None:
        """Should raise ValueError for unsupported format."""
        bad_file = temp_dir / "test.xyz"
        bad_file.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported format"):
            extract_metadata(bad_file)

    def test_extract_metadata_success(self, temp_dir: Path) -> None:
        """Should return VideoMetadata on success."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "format": {"duration": "120.0", "format_name": "mp4"},
            "streams": [
                {"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "24/1"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "48000"}
            ]
        }"""

        with patch("subprocess.run", return_value=mock_result):
            metadata = extract_metadata(video_path)

        assert metadata.duration == 120.0
        assert metadata.resolution == "1280x720"
        assert metadata.fps == 24.0
        assert metadata.format == "mp4"
        assert metadata.audio_channels == 2
        assert metadata.sample_rate == 48000


class TestExtractAudio:
    """Tests for extract_audio function."""

    def test_extract_audio_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_audio(temp_dir / "missing.mp4", temp_dir)

    def test_extract_audio_ffmpeg_not_found(self, temp_dir: Path) -> None:
        """Should raise IngestError when ffmpeg is not installed."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(IngestError, match="ffmpeg not found"):
                extract_audio(video_path, temp_dir / "output")

    def test_extract_audio_creates_output_dir(self, temp_dir: Path) -> None:
        """Should create output directory if it doesn't exist."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")
        output_dir = temp_dir / "new_output"

        mock_result = MagicMock()
        mock_result.returncode = 0

        # Create the expected output file
        def create_output(*args, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_audio.wav").write_bytes(b"wav data")
            return mock_result

        with patch("subprocess.run", side_effect=create_output):
            result = extract_audio(video_path, output_dir)

        assert output_dir.exists()
        assert result == output_dir / "test_audio.wav"

    def test_extract_audio_mono_16k(self, temp_dir: Path) -> None:
        """Should extract audio as 16kHz mono by default."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")
        output_dir = temp_dir / "output"

        captured_cmd = []

        def capture_cmd(*args, **kwargs):
            captured_cmd.extend(args[0])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_audio.wav").write_bytes(b"wav")
            mock = MagicMock()
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=capture_cmd):
            extract_audio(video_path, output_dir)

        assert "-ar" in captured_cmd
        assert "16000" in captured_cmd
        assert "-ac" in captured_cmd
        assert "1" in captured_cmd


class TestIsAudioOnly:
    """Tests for is_audio_only function."""

    def test_wav_is_audio_only(self, temp_dir: Path) -> None:
        """Should return True for .wav files."""
        wav_path = temp_dir / "test.wav"
        wav_path.write_bytes(b"data")
        assert is_audio_only(wav_path) is True

    def test_mp3_is_audio_only(self, temp_dir: Path) -> None:
        """Should return True for .mp3 files."""
        mp3_path = temp_dir / "test.mp3"
        mp3_path.write_bytes(b"data")
        assert is_audio_only(mp3_path) is True

    def test_mp4_with_video_is_not_audio_only(self, temp_dir: Path) -> None:
        """Should return False for video files with video track."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "format": {"duration": "60.0", "format_name": "mp4"},
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"}]
        }"""

        with patch("subprocess.run", return_value=mock_result):
            assert is_audio_only(video_path) is False
