"""ChromaFlow: Turn video into RAG-ready artifacts in one line of code."""

from chromaflow.config import PipelineConfig, ProcessingOptions
from chromaflow.models.schema import Chunk, ChunkSpeaker, VideoData, VideoMetadata
from chromaflow.pipeline import Pipeline, PipelineError
from chromaflow.store.chroma import ChromaStore, ChromaStoreError

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineError",
    "ProcessingOptions",
    "VideoData",
    "VideoMetadata",
    "Chunk",
    "ChunkSpeaker",
    "ChromaStore",
    "ChromaStoreError",
    "__version__",
]
