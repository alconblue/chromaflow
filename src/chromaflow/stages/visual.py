"""Visual processing stage: frame extraction and embedding.

This stage handles:
- Keyframe extraction via decord
- CLIP embedding generation via sentence-transformers
- Frame saving for reference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from chromaflow.utils.logging import get_logger

logger = get_logger(__name__)


class VisualProcessingError(Exception):
    """Error during visual processing."""

    pass


@dataclass
class FrameResult:
    """Result from frame extraction."""

    frame: NDArray[np.uint8]
    timestamp: float
    saved_path: Path | None = None


@dataclass
class VisualEmbeddingResult:
    """Result from visual embedding generation."""

    embedding: list[float]
    timestamp: float
    frame_path: Path | None = None


# Lazy-loaded modules and models
_decord_loaded = False
_VideoReader = None
_cpu_ctx = None

_clip_model = None
_clip_model_name: str | None = None
_clip_device: str | None = None


def _load_decord() -> None:
    """Lazy-load decord for frame extraction."""
    global _decord_loaded, _VideoReader, _cpu_ctx

    if _decord_loaded:
        return

    start_time = time.perf_counter()

    try:
        from decord import VideoReader, cpu

        _VideoReader = VideoReader
        _cpu_ctx = cpu(0)
        _decord_loaded = True

        elapsed = time.perf_counter() - start_time
        logger.info(f"decord loaded in {elapsed:.2f}s")

    except ImportError as e:
        raise VisualProcessingError(
            f"Failed to import decord: {e}\n"
            "Install with: pip install decord"
        ) from e


def _load_clip_model(model_name: str = "clip-ViT-L-14", device: str = "cpu"):
    """Lazy-load CLIP model for embedding generation.

    Args:
        model_name: HuggingFace model identifier for CLIP.
        device: Compute device (cuda, mps, cpu).

    Returns:
        Loaded SentenceTransformer model.
    """
    global _clip_model, _clip_model_name, _clip_device

    # Return cached model if same config
    if _clip_model is not None and _clip_model_name == model_name and _clip_device == device:
        return _clip_model

    start_time = time.perf_counter()
    logger.info(f"Loading CLIP model '{model_name}' on {device}...")

    try:
        from sentence_transformers import SentenceTransformer

        # Load model
        _clip_model = SentenceTransformer(model_name, device=device)
        _clip_model_name = model_name
        _clip_device = device

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"CLIP model loaded in {elapsed:.2f}s "
            f"(model={model_name}, device={device})"
        )

        return _clip_model

    except ImportError as e:
        raise VisualProcessingError(
            f"Failed to import sentence-transformers: {e}\n"
            "Install with: pip install sentence-transformers"
        ) from e
    except Exception as e:
        raise VisualProcessingError(f"Failed to load CLIP model '{model_name}': {e}") from e


def extract_frame(
    source: Path,
    timestamp: float,
    output_path: Path | None = None,
) -> FrameResult:
    """Extract a single frame from a video at a given timestamp.

    Args:
        source: Path to the video file.
        timestamp: Time in seconds to extract frame.
        output_path: Optional path to save the frame as JPEG.

    Returns:
        FrameResult with frame data and optional saved path.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        VisualProcessingError: If frame extraction fails.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    _load_decord()

    try:
        # Open video with decord
        vr = _VideoReader(str(source), ctx=_cpu_ctx)

        # Get frame rate and calculate frame index
        fps = vr.get_avg_fps()
        frame_idx = int(timestamp * fps)

        # Clamp to valid range
        frame_idx = max(0, min(frame_idx, len(vr) - 1))

        # Extract frame (returns numpy array in RGB format)
        frame = vr[frame_idx].asnumpy()

        saved_path = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JPEG using PIL
            img = Image.fromarray(frame)
            img.save(output_path, "JPEG", quality=85)
            saved_path = output_path

        return FrameResult(
            frame=frame,
            timestamp=timestamp,
            saved_path=saved_path,
        )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, VisualProcessingError)):
            raise
        raise VisualProcessingError(f"Failed to extract frame at {timestamp}s: {e}") from e


def extract_keyframes(
    source: Path,
    timestamps: list[float],
    output_dir: Path | None = None,
    file_id: str | None = None,
) -> list[FrameResult]:
    """Extract multiple keyframes from a video.

    Args:
        source: Path to the video file.
        timestamps: List of timestamps in seconds.
        output_dir: Optional directory to save frames.
        file_id: Optional file ID for naming saved frames.

    Returns:
        List of FrameResult objects.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        VisualProcessingError: If frame extraction fails.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    _load_decord()

    start_time = time.perf_counter()
    results: list[FrameResult] = []

    try:
        # Open video once for all frames
        vr = _VideoReader(str(source), ctx=_cpu_ctx)
        fps = vr.get_avg_fps()
        num_frames = len(vr)

        for i, ts in enumerate(timestamps):
            frame_idx = int(ts * fps)
            frame_idx = max(0, min(frame_idx, num_frames - 1))

            frame = vr[frame_idx].asnumpy()

            saved_path = None
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                prefix = file_id if file_id else "frame"
                frame_path = output_dir / f"{prefix}_{i:03d}.jpg"

                img = Image.fromarray(frame)
                img.save(frame_path, "JPEG", quality=85)
                saved_path = frame_path

            results.append(FrameResult(
                frame=frame,
                timestamp=ts,
                saved_path=saved_path,
            ))

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Extracted {len(results)} keyframes in {elapsed:.2f}s "
            f"(avg {elapsed / len(results) * 1000:.1f}ms/frame)"
        )

        return results

    except Exception as e:
        if isinstance(e, (FileNotFoundError, VisualProcessingError)):
            raise
        raise VisualProcessingError(f"Failed to extract keyframes: {e}") from e


def generate_embedding(
    frame: NDArray[np.uint8],
    model_name: str = "clip-ViT-L-14",
    device: str = "cpu",
    normalize: bool = True,
) -> list[float]:
    """Generate CLIP embedding for a frame.

    Args:
        frame: Frame as numpy array (H, W, C) in RGB format.
        model_name: CLIP model identifier.
        device: Compute device (cuda, mps, cpu).
        normalize: Whether to L2-normalize the embedding (default True).

    Returns:
        Embedding as list of floats (typically 768-dim for ViT-L/14).

    Raises:
        VisualProcessingError: If embedding generation fails.
    """
    model = _load_clip_model(model_name, device)

    try:
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)

        # Generate embedding
        embedding = model.encode(img, convert_to_numpy=True)

        # L2 normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()

    except Exception as e:
        raise VisualProcessingError(f"Failed to generate embedding: {e}") from e


def generate_embeddings(
    frames: list[NDArray[np.uint8]],
    model_name: str = "clip-ViT-L-14",
    device: str = "cpu",
    batch_size: int = 8,
    normalize: bool = True,
) -> list[list[float]]:
    """Generate CLIP embeddings for multiple frames.

    Args:
        frames: List of frames as numpy arrays (H, W, C) in RGB format.
        model_name: CLIP model identifier.
        device: Compute device (cuda, mps, cpu).
        batch_size: Number of frames to process at once.
        normalize: Whether to L2-normalize embeddings (default True).

    Returns:
        List of embeddings as lists of floats.

    Raises:
        VisualProcessingError: If embedding generation fails.
    """
    if not frames:
        return []

    model = _load_clip_model(model_name, device)

    start_time = time.perf_counter()

    try:
        # Convert all frames to PIL Images
        images = [Image.fromarray(f) for f in frames]

        # Generate embeddings in batches
        embeddings = model.encode(
            images,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # L2 normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
            embeddings = embeddings / norms

        result = [e.tolist() for e in embeddings]

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Generated {len(result)} embeddings in {elapsed:.2f}s "
            f"(avg {elapsed / len(result) * 1000:.1f}ms/frame)"
        )

        return result

    except Exception as e:
        raise VisualProcessingError(f"Failed to generate embeddings: {e}") from e


def process_visual(
    source: Path,
    keyframe_times: list[float],
    output_dir: Path | None = None,
    file_id: str | None = None,
    model_name: str = "clip-ViT-L-14",
    device: str = "cpu",
) -> list[VisualEmbeddingResult]:
    """Process video to extract keyframes and generate embeddings.

    This is the main entry point for visual processing, combining
    frame extraction and embedding generation.

    Args:
        source: Path to the video file.
        keyframe_times: List of timestamps for keyframes.
        output_dir: Optional directory to save frames.
        file_id: Optional file ID for naming saved frames.
        model_name: CLIP model identifier.
        device: Compute device (cuda, mps, cpu).

    Returns:
        List of VisualEmbeddingResult objects.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        VisualProcessingError: If processing fails.
    """
    if not keyframe_times:
        return []

    # Extract keyframes
    frame_results = extract_keyframes(
        source=source,
        timestamps=keyframe_times,
        output_dir=output_dir,
        file_id=file_id,
    )

    # Generate embeddings for all frames
    frames = [r.frame for r in frame_results]
    embeddings = generate_embeddings(
        frames=frames,
        model_name=model_name,
        device=device,
    )

    # Combine results
    results: list[VisualEmbeddingResult] = []
    for frame_result, embedding in zip(frame_results, embeddings):
        results.append(VisualEmbeddingResult(
            embedding=embedding,
            timestamp=frame_result.timestamp,
            frame_path=frame_result.saved_path,
        ))

    return results
