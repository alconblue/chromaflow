"""ChromaDB integration for vector search.

Provides embedded vector storage and semantic search capabilities
using ChromaDB as the backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from chromaflow.utils.logging import get_logger

if TYPE_CHECKING:
    from chromaflow.models.schema import Chunk

logger = get_logger(__name__)


class ChromaStoreError(Exception):
    """Error during ChromaDB operations."""

    pass


# Lazy-loaded chromadb client
_chromadb_loaded = False
_chromadb_module = None


def _load_chromadb():
    """Lazy-load ChromaDB module."""
    global _chromadb_loaded, _chromadb_module

    if _chromadb_loaded:
        return _chromadb_module

    start_time = time.perf_counter()

    try:
        import chromadb

        _chromadb_module = chromadb
        _chromadb_loaded = True

        elapsed = time.perf_counter() - start_time
        logger.info(f"chromadb loaded in {elapsed:.2f}s")

        return chromadb

    except ImportError as e:
        raise ChromaStoreError(
            f"Failed to import chromadb: {e}\n"
            "Install with: pip install chromadb"
        ) from e


class ChromaStore:
    """ChromaDB-backed vector store for video chunks.

    Provides semantic search over processed video chunks using
    their visual embeddings and transcript text.

    The store supports two search modes:
    1. Text search: Uses ChromaDB's built-in embedding function to
       embed the query and search against transcript text.
    2. Visual search (if visual embeddings available): Uses CLIP
       embeddings for image-based similarity search.
    """

    def __init__(
        self,
        collection_name: str = "chromaflow",
        persist_directory: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize ChromaDB store.

        Args:
            collection_name: Name for the ChromaDB collection.
            persist_directory: Directory for persistent storage (None = in-memory).
            embedding_model: Sentence transformer model for text embeddings.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        chromadb = _load_chromadb()

        start_time = time.perf_counter()

        # Create client (persistent or ephemeral)
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
            logger.info(f"ChromaDB persistent client initialized at {persist_path}")
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB ephemeral client initialized")

        # Get or create collection with default embedding function
        # ChromaDB will use sentence-transformers by default
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"ChromaDB collection '{collection_name}' ready in {elapsed:.2f}s "
            f"(documents: {self._collection.count()})"
        )

        # Cache for chunk data (chunk_id -> Chunk)
        self._chunk_cache: dict[str, Chunk] = {}

    def add_chunks(self, chunks: list[Chunk], file_id: str) -> None:
        """Add processed chunks to the vector store.

        Args:
            chunks: List of Chunk objects with transcripts.
            file_id: ID of the source video file.

        Raises:
            ChromaStoreError: If adding chunks fails.
        """
        if not chunks:
            return

        start_time = time.perf_counter()

        try:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []

            for chunk in chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.transcript or "")
                metadatas.append({
                    "file_id": file_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "has_visual_embedding": len(chunk.visual_embedding) > 0,
                    "speaker_count": len(set(s.label for s in chunk.speakers)),
                })

                # Cache chunk for retrieval
                self._chunk_cache[chunk.chunk_id] = chunk

            # Add to collection (ChromaDB will generate embeddings for documents)
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Added {len(chunks)} chunks to ChromaDB in {elapsed:.2f}s "
                f"(file_id={file_id})"
            )

        except Exception as e:
            raise ChromaStoreError(f"Failed to add chunks to ChromaDB: {e}") from e

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[Chunk]:
        """Search for chunks relevant to a text query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            file_id: Optional filter to search within a specific video.

        Returns:
            List of matching Chunk objects, ordered by relevance.

        Raises:
            ChromaStoreError: If search fails.
        """
        start_time = time.perf_counter()

        try:
            # Build where filter if file_id specified
            where_filter = {"file_id": file_id} if file_id else None

            # Query the collection
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )

            # Extract chunk IDs from results
            chunk_ids = results["ids"][0] if results["ids"] else []

            # Retrieve chunks from cache
            chunks: list[Chunk] = []
            for chunk_id in chunk_ids:
                if chunk_id in self._chunk_cache:
                    chunks.append(self._chunk_cache[chunk_id])

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Search completed in {elapsed:.3f}s: "
                f"query='{query[:50]}...', results={len(chunks)}"
            )

            return chunks

        except Exception as e:
            raise ChromaStoreError(f"Search failed: {e}") from e

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[Chunk]:
        """Search for chunks using a visual embedding.

        This is useful for image-based similarity search when you have
        a CLIP embedding of an image.

        Note: This searches against document embeddings, not visual embeddings.
        For true visual search, use a separate visual embeddings collection.

        Args:
            embedding: Query embedding vector.
            top_k: Number of results to return.
            file_id: Optional filter to search within a specific video.

        Returns:
            List of matching Chunk objects, ordered by relevance.

        Raises:
            ChromaStoreError: If search fails.
        """
        start_time = time.perf_counter()

        try:
            where_filter = {"file_id": file_id} if file_id else None

            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
            )

            chunk_ids = results["ids"][0] if results["ids"] else []

            chunks: list[Chunk] = []
            for chunk_id in chunk_ids:
                if chunk_id in self._chunk_cache:
                    chunks.append(self._chunk_cache[chunk_id])

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Embedding search completed in {elapsed:.3f}s: results={len(chunks)}"
            )

            return chunks

        except Exception as e:
            raise ChromaStoreError(f"Embedding search failed: {e}") from e

    def delete(self, file_id: str) -> None:
        """Delete all chunks for a video from the store.

        Args:
            file_id: ID of the video to delete.

        Raises:
            ChromaStoreError: If deletion fails.
        """
        start_time = time.perf_counter()

        try:
            # Get all chunk IDs for this file
            results = self._collection.get(
                where={"file_id": file_id},
            )

            chunk_ids = results["ids"] if results["ids"] else []

            if chunk_ids:
                # Delete from collection
                self._collection.delete(ids=chunk_ids)

                # Remove from cache
                for chunk_id in chunk_ids:
                    self._chunk_cache.pop(chunk_id, None)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Deleted {len(chunk_ids)} chunks for file_id={file_id} "
                f"in {elapsed:.2f}s"
            )

        except Exception as e:
            raise ChromaStoreError(f"Failed to delete chunks: {e}") from e

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a specific chunk by ID.

        Args:
            chunk_id: The chunk ID to retrieve.

        Returns:
            The Chunk object, or None if not found.
        """
        return self._chunk_cache.get(chunk_id)

    def count(self, file_id: str | None = None) -> int:
        """Count chunks in the store.

        Args:
            file_id: Optional filter to count only chunks for a specific video.

        Returns:
            Number of chunks in the store.
        """
        if file_id is None:
            return self._collection.count()

        results = self._collection.get(
            where={"file_id": file_id},
        )
        return len(results["ids"]) if results["ids"] else 0

    def list_file_ids(self) -> list[str]:
        """List all unique file IDs in the store.

        Returns:
            List of file IDs.
        """
        try:
            results = self._collection.get()
            if not results["metadatas"]:
                return []

            file_ids = set()
            for metadata in results["metadatas"]:
                if "file_id" in metadata:
                    file_ids.add(metadata["file_id"])

            return sorted(file_ids)

        except Exception:
            return []
