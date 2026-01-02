"""Tests for the scene detection stage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromaflow.stages.scene import (
    SceneBoundary,
    SceneDetectionError,
    detect_scenes,
    detect_scenes_for_audio_only,
)


class TestSceneBoundary:
    """Tests for SceneBoundary dataclass."""

    def test_duration_property(self) -> None:
        """Should calculate duration correctly."""
        scene = SceneBoundary(start=10.0, end=25.0, keyframe_time=17.5)
        assert scene.duration == 15.0

    def test_keyframe_in_middle(self) -> None:
        """Keyframe should typically be in the middle."""
        scene = SceneBoundary(start=0.0, end=60.0, keyframe_time=30.0)
        assert scene.keyframe_time == 30.0


class TestDetectScenesForAudioOnly:
    """Tests for detect_scenes_for_audio_only function."""

    def test_creates_segments(self) -> None:
        """Should create multiple segments for long audio."""
        segments = detect_scenes_for_audio_only(duration=90.0, segment_length=30.0)

        assert len(segments) == 3
        assert segments[0].start == 0.0
        assert segments[0].end == 30.0
        assert segments[1].start == 30.0
        assert segments[1].end == 60.0
        assert segments[2].start == 60.0
        assert segments[2].end == 90.0

    def test_handles_partial_final_segment(self) -> None:
        """Should handle when duration isn't evenly divisible."""
        segments = detect_scenes_for_audio_only(duration=75.0, segment_length=30.0)

        assert len(segments) == 3
        assert segments[-1].end == 75.0
        assert segments[-1].duration == 15.0

    def test_single_segment_for_short_audio(self) -> None:
        """Should create single segment for short audio."""
        segments = detect_scenes_for_audio_only(duration=15.0, segment_length=30.0)

        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 15.0

    def test_keyframe_in_middle_of_segment(self) -> None:
        """Keyframe should be in the middle of each segment."""
        segments = detect_scenes_for_audio_only(duration=60.0, segment_length=30.0)

        assert segments[0].keyframe_time == 15.0
        assert segments[1].keyframe_time == 45.0

    def test_custom_segment_length(self) -> None:
        """Should respect custom segment length."""
        segments = detect_scenes_for_audio_only(duration=120.0, segment_length=60.0)

        assert len(segments) == 2
        assert segments[0].duration == 60.0
        assert segments[1].duration == 60.0


class TestDetectScenes:
    """Tests for detect_scenes function."""

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            detect_scenes(temp_dir / "missing.mp4")

    def test_scenedetect_import_error(self, temp_dir: Path) -> None:
        """Should raise SceneDetectionError when scenedetect not installed."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        # Reset lazy-load state
        import chromaflow.stages.scene as scene_module
        scene_module._scenedetect_loaded = False

        with patch.dict("sys.modules", {"scenedetect": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # Need to reset so it tries to import again
                scene_module._scenedetect_loaded = False
                with pytest.raises(SceneDetectionError, match="Failed to import"):
                    detect_scenes(video_path)

    def test_detect_scenes_success(self, temp_dir: Path) -> None:
        """Should return list of SceneBoundary on success."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        # Mock the scenedetect components
        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        # video.duration is a FrameTimecode in scenedetect
        mock_duration = MagicMock()
        mock_duration.get_seconds.return_value = 120.0
        mock_video.duration = mock_duration

        # Create mock timecodes
        mock_tc1_start = MagicMock()
        mock_tc1_start.get_seconds.return_value = 0.0
        mock_tc1_end = MagicMock()
        mock_tc1_end.get_seconds.return_value = 45.0

        mock_tc2_start = MagicMock()
        mock_tc2_start.get_seconds.return_value = 45.0
        mock_tc2_end = MagicMock()
        mock_tc2_end.get_seconds.return_value = 90.0

        mock_tc3_start = MagicMock()
        mock_tc3_start.get_seconds.return_value = 90.0
        mock_tc3_end = MagicMock()
        mock_tc3_end.get_seconds.return_value = 120.0

        mock_scene_list = [
            (mock_tc1_start, mock_tc1_end),
            (mock_tc2_start, mock_tc2_end),
            (mock_tc3_start, mock_tc3_end),
        ]

        mock_scene_manager = MagicMock()
        mock_scene_manager.get_scene_list.return_value = mock_scene_list

        mock_open_video = MagicMock(return_value=mock_video)
        mock_SceneManager = MagicMock(return_value=mock_scene_manager)
        mock_ContentDetector = MagicMock()

        # Set up module state
        import chromaflow.stages.scene as scene_module
        scene_module._scenedetect_loaded = True
        scene_module._open_video = mock_open_video
        scene_module._SceneManager = mock_SceneManager
        scene_module._ContentDetector = mock_ContentDetector

        try:
            result = detect_scenes(video_path)

            assert len(result) == 3
            assert all(isinstance(s, SceneBoundary) for s in result)
            assert result[0].start == 0.0
            assert result[0].end == 45.0
            assert result[0].keyframe_time == 22.5
            assert result[1].start == 45.0
            assert result[2].end == 120.0
        finally:
            # Reset state
            scene_module._scenedetect_loaded = False

    def test_no_scenes_detected(self, temp_dir: Path) -> None:
        """Should create single scene when no cuts detected."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        # video.duration is a FrameTimecode in scenedetect
        mock_duration = MagicMock()
        mock_duration.get_seconds.return_value = 60.0
        mock_video.duration = mock_duration

        mock_scene_manager = MagicMock()
        mock_scene_manager.get_scene_list.return_value = []  # No scenes

        mock_open_video = MagicMock(return_value=mock_video)
        mock_SceneManager = MagicMock(return_value=mock_scene_manager)
        mock_ContentDetector = MagicMock()

        import chromaflow.stages.scene as scene_module
        scene_module._scenedetect_loaded = True
        scene_module._open_video = mock_open_video
        scene_module._SceneManager = mock_SceneManager
        scene_module._ContentDetector = mock_ContentDetector

        try:
            result = detect_scenes(video_path)

            # Should create one scene spanning the whole video
            assert len(result) == 1
            assert result[0].start == 0.0
            assert result[0].end == 60.0
            assert result[0].keyframe_time == 30.0
        finally:
            scene_module._scenedetect_loaded = False

    def test_threshold_parameter_passed(self, temp_dir: Path) -> None:
        """Should pass threshold to ContentDetector."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        # video.duration is a FrameTimecode in scenedetect
        mock_duration = MagicMock()
        mock_duration.get_seconds.return_value = 60.0
        mock_video.duration = mock_duration

        mock_scene_manager = MagicMock()
        mock_scene_manager.get_scene_list.return_value = []

        mock_open_video = MagicMock(return_value=mock_video)
        mock_SceneManager = MagicMock(return_value=mock_scene_manager)
        mock_ContentDetector = MagicMock()

        import chromaflow.stages.scene as scene_module
        scene_module._scenedetect_loaded = True
        scene_module._open_video = mock_open_video
        scene_module._SceneManager = mock_SceneManager
        scene_module._ContentDetector = mock_ContentDetector

        try:
            detect_scenes(video_path, threshold=15.0, min_scene_len=2.0)

            # Check ContentDetector was called with correct args
            mock_ContentDetector.assert_called_once()
            call_kwargs = mock_ContentDetector.call_args[1]
            assert call_kwargs["threshold"] == 15.0
            # min_scene_len should be converted to frames (2.0 * 30 fps = 60)
            assert call_kwargs["min_scene_len"] == 60
        finally:
            scene_module._scenedetect_loaded = False

    def test_detection_error_wrapped(self, temp_dir: Path) -> None:
        """Should wrap unexpected errors in SceneDetectionError."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        mock_open_video = MagicMock(side_effect=RuntimeError("Video codec error"))

        import chromaflow.stages.scene as scene_module
        scene_module._scenedetect_loaded = True
        scene_module._open_video = mock_open_video

        try:
            with pytest.raises(SceneDetectionError, match="Scene detection failed"):
                detect_scenes(video_path)
        finally:
            scene_module._scenedetect_loaded = False
