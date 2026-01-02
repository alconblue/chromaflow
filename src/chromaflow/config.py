"""Configuration and settings for ChromaFlow pipelines."""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from chromaflow.utils.hardware import detect_device


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"


class ProcessingOptions(BaseModel):
    """Options for video processing."""

    diarize: bool = Field(default=True, description="Enable speaker diarization")
    ocr: bool = Field(default=False, description="Enable OCR (not implemented in MVP)")
    whisper_model: WhisperModel = Field(
        default=WhisperModel.SMALL, description="Whisper model size to use"
    )
    clip_model: str = Field(
        default="clip-ViT-L-14",
        description="CLIP model for visual embeddings",
    )
    scene_threshold: float = Field(
        default=27.0, ge=0, le=100, description="Scene detection sensitivity threshold"
    )
    min_scene_duration: float = Field(
        default=1.0, ge=0, description="Minimum scene duration in seconds"
    )


class PipelineConfig(BaseModel):
    """Configuration for a ChromaFlow pipeline."""

    mode: Literal["local", "cloud"] = Field(
        default="local", description="Processing mode: 'local' or 'cloud'"
    )
    api_key: str | None = Field(
        default=None, description="API key for cloud mode (required if mode='cloud')"
    )
    device: str | None = Field(
        default=None,
        description="Compute device: 'cuda', 'mps', 'cpu', or None for auto-detect",
    )
    output_dir: str = Field(
        default="./chromaflow_output", description="Directory for output files (frames, etc.)"
    )
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token for pyannote models. Falls back to HF_TOKEN env var.",
    )
    options: ProcessingOptions = Field(
        default_factory=ProcessingOptions, description="Processing options"
    )

    def get_device(self) -> str:
        """Get the compute device, auto-detecting if not specified."""
        if self.device is not None:
            return self.device
        return detect_device()

    def get_hf_token(self) -> str | None:
        """Get HuggingFace token from config or environment."""
        if self.hf_token is not None:
            return self.hf_token
        return os.environ.get("HF_TOKEN")

    def validate_for_diarization(self) -> None:
        """Validate that diarization requirements are met.

        Raises:
            ValueError: If diarization is enabled but HF_TOKEN is not set.
        """
        if not self.options.diarize:
            return

        token = self.get_hf_token()
        if token is None:
            raise ValueError(
                "Speaker diarization requires a HuggingFace token.\n\n"
                "Pyannote models require authentication. To enable diarization:\n\n"
                "1. Create a HuggingFace account at https://huggingface.co\n"
                "2. Accept the pyannote model terms at:\n"
                "   https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Create an access token at:\n"
                "   https://huggingface.co/settings/tokens\n"
                "4. Set the token via environment variable:\n"
                "   export HF_TOKEN='your_token_here'\n\n"
                "Alternatively, pass hf_token to PipelineConfig or disable diarization:\n"
                "   Pipeline(options={'diarize': False})"
            )
