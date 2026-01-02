"""Scene detection stage: visual segmentation.

This stage handles:
- Content-aware scene detection via PySceneDetect
- Scene boundary identification
- Keyframe timestamp extraction

The scene detector is lazy-loaded to avoid import overhead when not needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenedetect import SceneManager, VideoStream
    from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)

# Lazy-loaded module references
_scenedetect_loaded = False
_ContentDetector: type[ContentDetector] | None = None
_SceneManager: type[SceneManager] | None = None
_open_video: callable | None = None


class SceneDetectionError(Exception):
    """Error during scene detection."""

    pass


@dataclass
class SceneBoundary:
    """A detected scene boundary."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    keyframe_time: float  # Best keyframe timestamp (typically middle)

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start


def _load_scenedetect() -> None:
    """Lazy-load scenedetect module.

    This avoids the import overhead (OpenCV initialization) when
    scene detection is not needed.
    """
    global _scenedetect_loaded, _ContentDetector, _SceneManager, _open_video

    if _scenedetect_loaded:
        return

    start_time = time.perf_counter()
    logger.debug("Loading scenedetect module...")

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        _ContentDetector = ContentDetector
        _SceneManager = SceneManager
        _open_video = open_video
        _scenedetect_loaded = True

        elapsed = time.perf_counter() - start_time
        logger.info(f"scenedetect loaded in {elapsed:.2f}s")

    except ImportError as e:
        raise SceneDetectionError(
            f"Failed to import scenedetect: {e}\n"
            "Install with: pip install scenedetect[opencv]"
        )


def detect_scenes(
    source: Path,
    threshold: float = 27.0,
    min_scene_len: float = 1.0,
) -> list[SceneBoundary]:
    """Detect scene boundaries in a video file.

    Uses PySceneDetect's ContentDetector which analyzes frame-to-frame
    changes in HSV color space to detect scene cuts. This is superior
    to simple threshold detection as it adapts to the content.

    Args:
        source: Path to the input video file.
        threshold: Scene detection sensitivity (0-100).
            Lower values = more sensitive (more scenes detected).
            Default 27.0 is good for most content.
        min_scene_len: Minimum scene duration in seconds.
            Scenes shorter than this will be merged with adjacent scenes.

    Returns:
        List of SceneBoundary objects with start/end times.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        SceneDetectionError: If scene detection fails.
    """
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    # Lazy-load scenedetect
    _load_scenedetect()

    logger.info(f"Detecting scenes in {source.name} (threshold={threshold})")
    start_time = time.perf_counter()

    try:
        # Open video
        video = _open_video(str(source))  # type: ignore[misc]

        # Get video properties for min_scene_len conversion
        fps = video.frame_rate
        min_scene_len_frames = int(min_scene_len * fps)

        # Create scene manager with content detector
        scene_manager = _SceneManager()  # type: ignore[misc]
        scene_manager.add_detector(
            _ContentDetector(  # type: ignore[misc]
                threshold=threshold,
                min_scene_len=min_scene_len_frames,
            )
        )

        # Detect scenes
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Convert to SceneBoundary objects
        boundaries: list[SceneBoundary] = []

        for scene in scene_list:
            start_tc, end_tc = scene
            start_sec = start_tc.get_seconds()
            end_sec = end_tc.get_seconds()

            # Keyframe is the middle of the scene
            keyframe_time = (start_sec + end_sec) / 2

            boundaries.append(
                SceneBoundary(
                    start=start_sec,
                    end=end_sec,
                    keyframe_time=keyframe_time,
                )
            )

        elapsed = time.perf_counter() - start_time
        # video.duration is a FrameTimecode, get seconds
        video_duration_sec = video.duration.get_seconds()
        logger.info(
            f"Scene detection complete: {len(boundaries)} scenes in {elapsed:.2f}s "
            f"({elapsed / video_duration_sec * 100:.1f}% of video duration)"
        )

        # If no scenes detected, create one scene spanning the whole video
        if not boundaries:
            logger.warning("No scene changes detected, treating as single scene")
            boundaries.append(
                SceneBoundary(
                    start=0.0,
                    end=video_duration_sec,
                    keyframe_time=video_duration_sec / 2,
                )
            )

        return boundaries

    except Exception as e:
        if isinstance(e, (FileNotFoundError, SceneDetectionError)):
            raise
        raise SceneDetectionError(f"Scene detection failed: {e}") from e


def detect_scenes_for_audio_only(duration: float, segment_length: float = 30.0) -> list[SceneBoundary]:
    """Create synthetic scene boundaries for audio-only files.

    Since audio files have no visual scenes, we create fixed-length
    segments for chunking purposes.

    Args:
        duration: Total duration of the audio file in seconds.
        segment_length: Length of each segment in seconds (default: 30s).

    Returns:
        List of SceneBoundary objects representing segments.
    """
    boundaries: list[SceneBoundary] = []
    current = 0.0

    while current < duration:
        end = min(current + segment_length, duration)
        boundaries.append(
            SceneBoundary(
                start=current,
                end=end,
                keyframe_time=(current + end) / 2,
            )
        )
        current = end

    logger.info(f"Created {len(boundaries)} synthetic segments for audio-only file")
    return boundaries
