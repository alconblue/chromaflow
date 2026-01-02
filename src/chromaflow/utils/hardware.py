"""Hardware detection utilities for ChromaFlow."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the detected compute device."""

    device: str
    name: str
    memory_gb: float | None = None
    cuda_version: str | None = None
    is_available: bool = True


def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA detected, using GPU acceleration")
            return "cuda"

        if torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected, using Metal acceleration")
            return "mps"

    except ImportError:
        logger.warning("PyTorch not installed, falling back to CPU")

    logger.warning(
        "No GPU acceleration available. Processing will be slow. "
        "Consider installing PyTorch with CUDA or using Apple Silicon."
    )
    return "cpu"


def get_device_info() -> DeviceInfo:
    """Get detailed information about the compute device.

    Returns:
        DeviceInfo object with device details.
    """
    device = detect_device()

    if device == "cuda":
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            return DeviceInfo(
                device=device,
                name=gpu_name,
                memory_gb=round(memory_gb, 1),
                cuda_version=cuda_version,
            )
        except Exception as e:
            logger.warning(f"Failed to get CUDA device info: {e}")
            return DeviceInfo(device=device, name="CUDA GPU")

    if device == "mps":
        import platform

        chip = platform.processor() or "Apple Silicon"
        return DeviceInfo(device=device, name=chip)

    return DeviceInfo(device="cpu", name="CPU")
