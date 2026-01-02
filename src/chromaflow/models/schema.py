"""Pydantic models defining the ChromaFlow data schema.

This is the "Golden Record" format - the standard way to represent
processed video data for RAG applications.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from chromaflow.store.chroma import ChromaStore


class ChunkSpeaker(BaseModel):
    """A speaker segment within a chunk."""

    label: str = Field(..., description="Speaker label (e.g., 'Speaker A', 'Speaker B')")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this speaker segment")


class Chunk(BaseModel):
    """A semantically coherent segment of the video.

    Each chunk represents a scene with aligned audio transcription
    and visual embedding.
    """

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    transcript: str = Field(default="", description="Full transcript for this chunk")
    speakers: list[ChunkSpeaker] = Field(
        default_factory=list, description="Speaker-diarized segments"
    )
    visual_embedding: list[float] = Field(
        default_factory=list, description="CLIP embedding of the keyframe (768-dim)"
    )
    screenshot_path: str | None = Field(
        default=None, description="Path to the extracted keyframe image"
    )


class VideoMetadata(BaseModel):
    """Technical metadata about the source video."""

    duration: float = Field(..., ge=0, description="Duration in seconds")
    resolution: str = Field(..., description="Resolution as 'WIDTHxHEIGHT'")
    fps: float = Field(..., gt=0, description="Frames per second")
    format: str = Field(default="unknown", description="Container format (mp4, mkv, etc.)")
    audio_channels: int = Field(default=0, ge=0, description="Number of audio channels")
    sample_rate: int | None = Field(default=None, description="Audio sample rate in Hz")


class VideoData(BaseModel):
    """The complete processed output for a video.

    This is the primary output of Pipeline.process() and contains
    all extracted information ready for RAG applications.
    """

    file_id: str = Field(..., description="Unique identifier for this video")
    source_path: str = Field(..., description="Original path to the source file")
    metadata: VideoMetadata = Field(..., description="Technical video metadata")
    chunks: list[Chunk] = Field(default_factory=list, description="Processed chunks")
    summary: str = Field(default="", description="Auto-generated summary of the video content")

    # Private attribute for search functionality (not serialized)
    _store: ChromaStore | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Search for chunks relevant to the query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of matching Chunk objects, ordered by relevance.

        Raises:
            RuntimeError: If no vector store is configured.
        """
        if self._store is None:
            raise RuntimeError(
                "Search is not available. VideoData was not created with a vector store. "
                "Ensure you processed the video with Pipeline.process()."
            )
        return self._store.search(query, top_k=top_k)

    def to_dict(self) -> dict:
        """Export to dictionary format (excludes internal store reference)."""
        return self.model_dump(exclude={"_store"})

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Export to JSON string, optionally writing to a file.

        Args:
            path: Optional file path to write JSON to.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        json_str = self.model_dump_json(exclude={"_store"}, indent=indent)
        if path is not None:
            Path(path).write_text(json_str)
        return json_str
