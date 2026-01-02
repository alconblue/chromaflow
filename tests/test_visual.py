"""Tests for the visual processing stage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chromaflow.stages.visual import (
    FrameResult,
    VisualEmbeddingResult,
    VisualProcessingError,
    _load_clip_model,
    _load_decord,
    extract_frame,
    extract_keyframes,
    generate_embedding,
    generate_embeddings,
    process_visual,
)


class TestFrameResult:
    """Tests for FrameResult dataclass."""

    def test_create_frame_result(self) -> None:
        """Should create a FrameResult with all fields."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = FrameResult(frame=frame, timestamp=5.0, saved_path=Path("/tmp/frame.jpg"))

        assert result.frame.shape == (480, 640, 3)
        assert result.timestamp == 5.0
        assert result.saved_path == Path("/tmp/frame.jpg")

    def test_default_saved_path(self) -> None:
        """Should have None as default saved_path."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = FrameResult(frame=frame, timestamp=0.0)

        assert result.saved_path is None


class TestVisualEmbeddingResult:
    """Tests for VisualEmbeddingResult dataclass."""

    def test_create_embedding_result(self) -> None:
        """Should create a VisualEmbeddingResult with all fields."""
        result = VisualEmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            timestamp=10.0,
            frame_path=Path("/tmp/frame.jpg"),
        )

        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.timestamp == 10.0
        assert result.frame_path == Path("/tmp/frame.jpg")


class TestLoadDecord:
    """Tests for _load_decord function."""

    def test_decord_import_error(self) -> None:
        """Should raise VisualProcessingError when decord not installed."""
        import chromaflow.stages.visual as visual_module

        visual_module._decord_loaded = False

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(VisualProcessingError, match="Failed to import decord"):
                _load_decord()


class TestLoadClipModel:
    """Tests for _load_clip_model function."""

    def test_clip_import_error(self) -> None:
        """Should raise VisualProcessingError when sentence-transformers not installed."""
        import chromaflow.stages.visual as visual_module

        visual_module._clip_model = None

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(VisualProcessingError, match="Failed to import"):
                _load_clip_model("test-model", "cpu")

    def test_clip_model_caching(self) -> None:
        """Should cache the CLIP model after loading."""
        import chromaflow.stages.visual as visual_module

        mock_model = MagicMock()

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            visual_module._clip_model = None
            visual_module._clip_model_name = None
            visual_module._clip_device = None

            # First call should load
            result1 = _load_clip_model("test-model", "cpu")

            # Second call with same params should return cached
            result2 = _load_clip_model("test-model", "cpu")

            assert result1 is result2

            # Clean up
            visual_module._clip_model = None


class TestExtractFrame:
    """Tests for extract_frame function."""

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_frame(temp_dir / "missing.mp4", 0.0)

    def test_extract_frame_success(self, temp_dir: Path) -> None:
        """Should extract a frame successfully."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        # Mock decord
        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)
        mock_vr.__getitem__ = MagicMock(return_value=mock_frame)

        import chromaflow.stages.visual as visual_module

        with patch.object(visual_module, "_VideoReader", return_value=mock_vr):
            visual_module._decord_loaded = True

            result = extract_frame(video_path, 5.0)

            assert isinstance(result, FrameResult)
            assert result.frame.shape == (480, 640, 3)
            assert result.timestamp == 5.0

            # Clean up
            visual_module._decord_loaded = False

    def test_extract_frame_saves_to_file(self, temp_dir: Path) -> None:
        """Should save frame to file when output_path provided."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")
        output_path = temp_dir / "output" / "frame.jpg"

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)
        mock_vr.__getitem__ = MagicMock(return_value=mock_frame)

        import chromaflow.stages.visual as visual_module

        with patch.object(visual_module, "_VideoReader", return_value=mock_vr):
            visual_module._decord_loaded = True

            result = extract_frame(video_path, 5.0, output_path=output_path)

            assert result.saved_path == output_path
            assert output_path.exists()

            visual_module._decord_loaded = False


class TestExtractKeyframes:
    """Tests for extract_keyframes function."""

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_keyframes(temp_dir / "missing.mp4", [0.0, 5.0, 10.0])

    def test_extract_multiple_keyframes(self, temp_dir: Path) -> None:
        """Should extract multiple keyframes."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=600)
        mock_vr.__getitem__ = MagicMock(return_value=mock_frame)

        import chromaflow.stages.visual as visual_module

        with patch.object(visual_module, "_VideoReader", return_value=mock_vr):
            visual_module._decord_loaded = True

            results = extract_keyframes(video_path, [0.0, 5.0, 10.0])

            assert len(results) == 3
            assert results[0].timestamp == 0.0
            assert results[1].timestamp == 5.0
            assert results[2].timestamp == 10.0

            visual_module._decord_loaded = False


class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    def test_generate_single_embedding(self) -> None:
        """Should generate embedding for a single frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        import chromaflow.stages.visual as visual_module

        visual_module._clip_model = mock_model
        visual_module._clip_model_name = "test-model"
        visual_module._clip_device = "cpu"

        try:
            result = generate_embedding(frame, "test-model", "cpu")

            # The embedding is normalized, so check approximate values
            # [0.1, 0.2, 0.3] normalized = [0.267, 0.535, 0.802]
            assert len(result) == 3
            # Check that it's approximately normalized (L2 norm ≈ 1)
            norm = sum(x**2 for x in result) ** 0.5
            assert 0.99 < norm < 1.01
            mock_model.encode.assert_called_once()
        finally:
            visual_module._clip_model = None


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function."""

    def test_empty_frames_list(self) -> None:
        """Should return empty list for empty input."""
        result = generate_embeddings([])
        assert result == []

    def test_generate_batch_embeddings(self) -> None:
        """Should generate embeddings for multiple frames."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        import chromaflow.stages.visual as visual_module

        visual_module._clip_model = mock_model
        visual_module._clip_model_name = "test-model"
        visual_module._clip_device = "cpu"

        try:
            result = generate_embeddings(frames, "test-model", "cpu")

            assert len(result) == 2
            # Embeddings are normalized, so check approximate normalization
            for emb in result:
                norm = sum(x**2 for x in emb) ** 0.5
                assert 0.99 < norm < 1.01
        finally:
            visual_module._clip_model = None


class TestProcessVisual:
    """Tests for process_visual function."""

    def test_empty_keyframe_times(self, temp_dir: Path) -> None:
        """Should return empty list for empty keyframe times."""
        result = process_visual(
            source=temp_dir / "test.mp4",
            keyframe_times=[],
        )
        assert result == []

    def test_process_visual_full_pipeline(self, temp_dir: Path) -> None:
        """Should extract frames and generate embeddings."""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        # Mock decord
        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=600)
        mock_vr.__getitem__ = MagicMock(return_value=mock_frame)

        # Mock CLIP model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        import chromaflow.stages.visual as visual_module

        with patch.object(visual_module, "_VideoReader", return_value=mock_vr):
            visual_module._decord_loaded = True
            visual_module._clip_model = mock_model
            visual_module._clip_model_name = "test-model"
            visual_module._clip_device = "cpu"

            try:
                results = process_visual(
                    source=video_path,
                    keyframe_times=[5.0, 15.0],
                    model_name="test-model",
                    device="cpu",
                )

                assert len(results) == 2
                assert all(isinstance(r, VisualEmbeddingResult) for r in results)
                assert results[0].timestamp == 5.0
                assert results[1].timestamp == 15.0
                # Embeddings are normalized, so check normalization
                for r in results:
                    norm = sum(x**2 for x in r.embedding) ** 0.5
                    assert 0.99 < norm < 1.01
            finally:
                visual_module._decord_loaded = False
                visual_module._clip_model = None
