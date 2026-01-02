"""ChromaDB integration for vector search.

Provides embedded vector storage and semantic search capabilities
using ChromaDB as the backend with dual-index support for both
transcript text and visual embeddings.
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

# Lazy-loaded sentence transformer for text embeddings
_text_model = None
_text_model_name = None


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


def _load_text_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence transformer model for text embeddings."""
    global _text_model, _text_model_name

    if _text_model is not None and _text_model_name == model_name:
        return _text_model

    start_time = time.perf_counter()

    try:
        from sentence_transformers import SentenceTransformer

        _text_model = SentenceTransformer(model_name)
        _text_model_name = model_name

        elapsed = time.perf_counter() - start_time
        logger.info(f"Text embedding model '{model_name}' loaded in {elapsed:.2f}s")

        return _text_model

    except ImportError as e:
        raise ChromaStoreError(
            f"Failed to import sentence_transformers: {e}\n"
            "Install with: pip install sentence-transformers"
        ) from e


def generate_text_embedding(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """Generate text embedding for a transcript.

    Args:
        text: Text to embed.
        model_name: Sentence transformer model name.

    Returns:
        Embedding as list of floats (384-dim for MiniLM).
    """
    if not text or not text.strip():
        return []

    model = _load_text_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_text_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[list[float]]:
    """Generate text embeddings for multiple transcripts.

    Args:
        texts: List of texts to embed.
        model_name: Sentence transformer model name.
        batch_size: Batch size for encoding.

    Returns:
        List of embeddings as lists of floats.
    """
    if not texts:
        return []

    # Filter out empty texts but track positions
    non_empty_indices = []
    non_empty_texts = []
    for i, text in enumerate(texts):
        if text and text.strip():
            non_empty_indices.append(i)
            non_empty_texts.append(text)

    if not non_empty_texts:
        return [[] for _ in texts]

    model = _load_text_model(model_name)
    embeddings = model.encode(
        non_empty_texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    # Reconstruct result with empty embeddings for empty texts
    result: list[list[float]] = [[] for _ in texts]
    for i, idx in enumerate(non_empty_indices):
        result[idx] = embeddings[i].tolist()

    return result


class ChromaStore:
    """ChromaDB-backed vector store for video chunks.

    Provides semantic search over processed video chunks using
    both transcript embeddings and visual embeddings.

    The store uses two collections:
    1. Text collection: Stores transcript embeddings for text-based search
    2. Visual collection: Stores CLIP embeddings for image-based search

    Combined search merges results from both collections using
    reciprocal rank fusion (RRF) for optimal ranking.
    """

    def __init__(
        self,
        collection_name: str = "chromaflow",
        persist_directory: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize ChromaDB store.

        Args:
            collection_name: Base name for the ChromaDB collections.
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

        # Create text collection (for transcript search)
        self._text_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_text",
            metadata={"hnsw:space": "cosine"},
        )

        # Create visual collection (for visual embedding search)
        self._visual_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_visual",
            metadata={"hnsw:space": "cosine"},
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"ChromaDB collections '{collection_name}' ready in {elapsed:.2f}s "
            f"(text: {self._text_collection.count()}, visual: {self._visual_collection.count()})"
        )

        # Cache for chunk data (chunk_id -> Chunk)
        self._chunk_cache: dict[str, Chunk] = {}

    def add_chunks(self, chunks: list[Chunk], file_id: str) -> None:
        """Add processed chunks to the vector store.

        Indexes chunks in both text and visual collections for
        multimodal search capabilities.

        Args:
            chunks: List of Chunk objects with transcripts and embeddings.
            file_id: ID of the source video file.

        Raises:
            ChromaStoreError: If adding chunks fails.
        """
        if not chunks:
            return

        start_time = time.perf_counter()

        try:
            text_ids: list[str] = []
            text_embeddings: list[list[float]] = []
            text_documents: list[str] = []
            text_metadatas: list[dict] = []

            visual_ids: list[str] = []
            visual_embeddings: list[list[float]] = []
            visual_metadatas: list[dict] = []

            for chunk in chunks:
                metadata = {
                    "file_id": file_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "speaker_count": len(set(s.label for s in chunk.speakers)),
                }

                # Add to text collection if we have transcript embedding
                if chunk.transcript_embedding:
                    text_ids.append(chunk.chunk_id)
                    text_embeddings.append(chunk.transcript_embedding)
                    text_documents.append(chunk.transcript or "")
                    text_metadatas.append(metadata)

                # Add to visual collection if we have visual embedding
                if chunk.visual_embedding:
                    visual_ids.append(chunk.chunk_id)
                    visual_embeddings.append(chunk.visual_embedding)
                    visual_metadatas.append(metadata)

                # Cache chunk for retrieval
                self._chunk_cache[chunk.chunk_id] = chunk

            # Add to text collection
            if text_ids:
                self._text_collection.add(
                    ids=text_ids,
                    embeddings=text_embeddings,
                    documents=text_documents,
                    metadatas=text_metadatas,
                )

            # Add to visual collection
            if visual_ids:
                self._visual_collection.add(
                    ids=visual_ids,
                    embeddings=visual_embeddings,
                    metadatas=visual_metadatas,
                )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Added {len(chunks)} chunks to ChromaDB in {elapsed:.2f}s "
                f"(text: {len(text_ids)}, visual: {len(visual_ids)}, file_id={file_id})"
            )

        except Exception as e:
            raise ChromaStoreError(f"Failed to add chunks to ChromaDB: {e}") from e

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_id: str | None = None,
        mode: str = "hybrid",
    ) -> list[Chunk]:
        """Search for chunks relevant to a text query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            file_id: Optional filter to search within a specific video.
            mode: Search mode - 'text', 'visual', or 'hybrid' (default).
                  'hybrid' combines text and visual search using RRF.

        Returns:
            List of matching Chunk objects, ordered by relevance.

        Raises:
            ChromaStoreError: If search fails.
        """
        start_time = time.perf_counter()

        try:
            where_filter = {"file_id": file_id} if file_id else None

            if mode == "text":
                chunks = self._search_text(query, top_k, where_filter)
            elif mode == "visual":
                chunks = self._search_visual_by_text(query, top_k, where_filter)
            else:  # hybrid
                chunks = self._search_hybrid(query, top_k, where_filter)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Search completed in {elapsed:.3f}s: "
                f"query='{query[:50]}...', mode={mode}, results={len(chunks)}"
            )

            return chunks

        except Exception as e:
            raise ChromaStoreError(f"Search failed: {e}") from e

    def _search_text(
        self,
        query: str,
        top_k: int,
        where_filter: dict | None,
    ) -> list[Chunk]:
        """Search using text embeddings only."""
        # Generate query embedding
        query_embedding = generate_text_embedding(query, self.embedding_model)
        if not query_embedding:
            return []

        results = self._text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        chunk_ids = results["ids"][0] if results["ids"] else []
        return [self._chunk_cache[cid] for cid in chunk_ids if cid in self._chunk_cache]

    def _search_visual_by_text(
        self,
        query: str,
        top_k: int,
        where_filter: dict | None,
    ) -> list[Chunk]:
        """Search visual collection using CLIP text embedding."""
        try:
            # Use CLIP to encode the text query
            from chromaflow.stages.visual import _load_clip_model

            model = _load_clip_model()
            query_embedding = model.encode(query, convert_to_numpy=True).tolist()

            results = self._visual_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
            )

            chunk_ids = results["ids"][0] if results["ids"] else []
            return [self._chunk_cache[cid] for cid in chunk_ids if cid in self._chunk_cache]

        except Exception as e:
            logger.warning(f"Visual search failed, falling back to text: {e}")
            return []

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        where_filter: dict | None,
    ) -> list[Chunk]:
        """Search using both text and visual embeddings with RRF fusion."""
        # Get more results from each to ensure good fusion
        fetch_k = min(top_k * 3, 50)

        # Text search
        text_results = self._search_text(query, fetch_k, where_filter)

        # Visual search (may fail if CLIP not available)
        visual_results = self._search_visual_by_text(query, fetch_k, where_filter)

        # Reciprocal Rank Fusion
        # RRF score = sum(1 / (k + rank)) where k=60 is standard
        k = 60
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(text_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)

        for rank, chunk in enumerate(visual_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)

        # Sort by combined score and return top_k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        return [self._chunk_cache[cid] for cid in sorted_ids if cid in self._chunk_cache]

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        file_id: str | None = None,
        collection: str = "visual",
    ) -> list[Chunk]:
        """Search for chunks using an embedding vector.

        Args:
            embedding: Query embedding vector.
            top_k: Number of results to return.
            file_id: Optional filter to search within a specific video.
            collection: Which collection to search - 'text' or 'visual'.

        Returns:
            List of matching Chunk objects, ordered by relevance.

        Raises:
            ChromaStoreError: If search fails.
        """
        start_time = time.perf_counter()

        try:
            where_filter = {"file_id": file_id} if file_id else None

            coll = self._visual_collection if collection == "visual" else self._text_collection

            results = coll.query(
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
                f"Embedding search ({collection}) completed in {elapsed:.3f}s: results={len(chunks)}"
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
            # Delete from text collection
            text_results = self._text_collection.get(where={"file_id": file_id})
            text_ids = text_results["ids"] if text_results["ids"] else []
            if text_ids:
                self._text_collection.delete(ids=text_ids)

            # Delete from visual collection
            visual_results = self._visual_collection.get(where={"file_id": file_id})
            visual_ids = visual_results["ids"] if visual_results["ids"] else []
            if visual_ids:
                self._visual_collection.delete(ids=visual_ids)

            # Remove from cache (use union of both)
            all_ids = set(text_ids) | set(visual_ids)
            for chunk_id in all_ids:
                self._chunk_cache.pop(chunk_id, None)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Deleted chunks for file_id={file_id} in {elapsed:.2f}s "
                f"(text: {len(text_ids)}, visual: {len(visual_ids)})"
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

        Returns the count from the text collection (primary index).

        Args:
            file_id: Optional filter to count only chunks for a specific video.

        Returns:
            Number of chunks in the store.
        """
        if file_id is None:
            return self._text_collection.count()

        results = self._text_collection.get(where={"file_id": file_id})
        return len(results["ids"]) if results["ids"] else 0

    def list_file_ids(self) -> list[str]:
        """List all unique file IDs in the store.

        Returns:
            List of file IDs.
        """
        try:
            results = self._text_collection.get()
            if not results["metadatas"]:
                return []

            file_ids = set()
            for metadata in results["metadatas"]:
                if "file_id" in metadata:
                    file_ids.add(metadata["file_id"])

            return sorted(file_ids)

        except Exception:
            return []
