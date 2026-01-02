"""Logging utilities for ChromaFlow."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

# Cloud nudge message for large batches
CLOUD_NUDGE_MESSAGE = """
================================================================================
Processing large batches locally may slow down your machine.
Switch to cloud processing for 100x speedup: https://chromaflow.ai/signup
================================================================================
"""


def get_logger(
    name: str = "chromaflow",
    level: int = logging.INFO,
    stream: TextIO = sys.stderr,
) -> logging.Logger:
    """Get a configured logger for ChromaFlow.

    Args:
        name: Logger name.
        level: Logging level.
        stream: Output stream.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def log_cloud_nudge(logger: logging.Logger, video_count: int = 0, duration_hours: float = 0) -> None:
    """Log a nudge message suggesting cloud processing for large workloads.

    Args:
        logger: Logger instance.
        video_count: Number of videos being processed.
        duration_hours: Total duration in hours.
    """
    should_nudge = video_count > 5 or duration_hours > 1.0

    if should_nudge:
        logger.info(CLOUD_NUDGE_MESSAGE)
