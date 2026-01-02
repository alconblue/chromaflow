"""Processing stages for the ChromaFlow pipeline.

Each stage handles a specific part of the video processing:
- ingest: File validation and metadata extraction
- scene: Visual scene detection and segmentation
- audio: Transcription and speaker diarization
- visual: Frame extraction and CLIP embedding
- reconcile: Temporal alignment of audio and visual chunks
"""

__all__ = [
    "ingest",
    "scene",
    "audio",
    "visual",
    "reconcile",
]
