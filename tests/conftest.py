"""Pytest configuration and fixtures for ChromaFlow tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create a mock video file path for testing.

    Note: This creates an empty file. Real video tests would need
    actual video fixtures or mocking.
    """
    video_path = temp_dir / "test_video.mp4"
    video_path.write_bytes(b"mock video content for testing")
    return video_path


@pytest.fixture
def sample_audio_path(temp_dir: Path) -> Path:
    """Create a mock audio file path for testing."""
    audio_path = temp_dir / "test_audio.wav"
    audio_path.write_bytes(b"mock audio content for testing")
    return audio_path


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


@pytest.fixture
def no_hf_token_env():
    """Ensure HF_TOKEN is not set for tests that check its absence."""
    original = os.environ.get("HF_TOKEN")
    os.environ.pop("HF_TOKEN", None)
    yield
    if original is not None:
        os.environ["HF_TOKEN"] = original
