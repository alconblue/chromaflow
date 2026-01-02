"""Pytest configuration and fixtures for integration tests."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def unique_collection_name() -> str:
    """Generate a unique collection name for ChromaDB test isolation."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def short_talking_head_video():
    """Short talking head video (30s, 720p, 1 speaker)."""
    path = FIXTURES_DIR / "short_talking_head.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def multi_scene_video():
    """Multi-scene test video (60s, 1080p, 3+ scene changes)."""
    path = FIXTURES_DIR / "multi_scene.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def screencast_video():
    """Screencast test video (45s, 1080p, text-heavy)."""
    path = FIXTURES_DIR / "screencast.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def podcast_audio():
    """Podcast test audio (90s, 16kHz mono, 2 speakers)."""
    path = FIXTURES_DIR / "podcast.wav"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def silent_video():
    """Silent video with no audio track (15s, 720p)."""
    path = FIXTURES_DIR / "silent_video.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def dark_low_quality_video():
    """Dark, low quality video (30s, 480p)."""
    path = FIXTURES_DIR / "dark_low_quality.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def long_meeting_video():
    """Long video for stress testing (3-5 minutes)."""
    path = FIXTURES_DIR / "long_meeting.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture
def hf_token_env():
    """Set a mock HF_TOKEN for tests that require it."""
    original = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "test_token_for_testing"
    yield "test_token_for_testing"
    if original is None:
        os.environ.pop("HF_TOKEN", None)
    else:
        os.environ["HF_TOKEN"] = original
