"""Integration tests for frame extraction with real video files."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from chromaflow.stages.visual import (
    VisualProcessingError,
    extract_frame,
    extract_keyframes,
)


@pytest.mark.integration
class TestFrameExtraction:
    """Real frame extraction with actual video files."""

    def test_extract_single_frame_dimensions(self, multi_scene_video: Path) -> None:
        """Frame should have correct dimensions."""
        result = extract_frame(multi_scene_video, timestamp=5.0)

        # ASSERT: Frame is valid numpy array with 3 channels
        assert result.frame.ndim == 3
        assert result.frame.shape[2] == 3  # RGB
        # ASSERT: Reasonable dimensions (not 0x0 or corrupted)
        assert result.frame.shape[0] >= 100  # height
        assert result.frame.shape[1] >= 100  # width

    def test_extract_frame_at_start(self, multi_scene_video: Path) -> None:
        """Should handle timestamp at start of video."""
        result = extract_frame(multi_scene_video, timestamp=0.0)

        assert result.frame is not None
        assert result.timestamp == 0.0

    def test_extract_frame_near_end(self, multi_scene_video: Path) -> None:
        """Should handle timestamp near end of video."""
        # Use a timestamp that should be valid for a 60s video
        result = extract_frame(multi_scene_video, timestamp=55.0)

        assert result.frame is not None
        assert result.frame.ndim == 3

    def test_keyframes_saved_to_disk(self, multi_scene_video: Path, temp_dir: Path) -> None:
        """Keyframes should be saved as valid JPEG files."""
        results = extract_keyframes(
            multi_scene_video,
            timestamps=[0.0, 15.0, 30.0],
            output_dir=temp_dir / "frames",
            file_id="test",
        )

        # ASSERT: Correct number of frames
        assert len(results) == 3

        # ASSERT: Files exist and are valid images
        for r in results:
            assert r.saved_path is not None
            assert r.saved_path.exists()
            img = Image.open(r.saved_path)
            img.verify()  # Raises if corrupted

    def test_keyframes_different_timestamps(self, multi_scene_video: Path) -> None:
        """Keyframes at different timestamps should have correct metadata."""
        timestamps = [5.0, 25.0, 45.0]
        results = extract_keyframes(multi_scene_video, timestamps=timestamps)

        assert len(results) == 3
        for result, expected_ts in zip(results, timestamps):
            assert result.timestamp == expected_ts

    def test_extract_from_short_video(self, short_talking_head_video: Path) -> None:
        """Should work with shorter videos."""
        result = extract_frame(short_talking_head_video, timestamp=10.0)

        assert result.frame is not None
        assert result.frame.ndim == 3

    def test_extract_from_silent_video(self, silent_video: Path) -> None:
        """Should work with videos that have no audio track."""
        result = extract_frame(silent_video, timestamp=5.0)

        assert result.frame is not None
        assert result.frame.ndim == 3

    def test_extract_from_low_quality_video(self, dark_low_quality_video: Path) -> None:
        """Should work with low quality/dark videos."""
        result = extract_frame(dark_low_quality_video, timestamp=10.0)

        assert result.frame is not None
        assert result.frame.ndim == 3
        # Low quality video should still have reasonable dimensions
        assert result.frame.shape[0] >= 100
        assert result.frame.shape[1] >= 100

    def test_extract_many_keyframes(self, multi_scene_video: Path, temp_dir: Path) -> None:
        """Should handle extracting many keyframes."""
        # Extract a keyframe every 2 seconds for 60s video
        timestamps = [float(i * 2) for i in range(30)]

        results = extract_keyframes(
            multi_scene_video,
            timestamps=timestamps,
            output_dir=temp_dir / "many_frames",
            file_id="many",
        )

        assert len(results) == 30

        # Verify all are valid
        for r in results:
            assert r.frame.ndim == 3
            if r.saved_path:
                assert r.saved_path.exists()
