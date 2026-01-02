"""Tests for the ChromaDB store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chromaflow.models.schema import Chunk, ChunkSpeaker
from chromaflow.store.chroma import ChromaStore, ChromaStoreError, _load_chromadb


class TestLoadChromadb:
    """Tests for _load_chromadb function."""

    def test_chromadb_import_error(self) -> None:
        """Should raise ChromaStoreError when chromadb not installed."""
        import chromaflow.store.chroma as chroma_module

        chroma_module._chromadb_loaded = False

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ChromaStoreError, match="Failed to import chromadb"):
                _load_chromadb()


class TestChromaStoreInit:
    """Tests for ChromaStore initialization."""

    def test_init_ephemeral(self) -> None:
        """Should initialize ephemeral (in-memory) store."""
        store = ChromaStore(collection_name="test_collection")

        assert store.collection_name == "test_collection"
        assert store.persist_directory is None
        assert store._client is not None
        assert store._collection is not None

    def test_init_persistent(self, temp_dir: Path) -> None:
        """Should initialize persistent store."""
        persist_dir = temp_dir / "chromadb"
        store = ChromaStore(
            collection_name="test_persistent",
            persist_directory=str(persist_dir),
        )

        assert store.persist_directory == str(persist_dir)
        assert persist_dir.exists()


class TestChromaStoreAddChunks:
    """Tests for ChromaStore.add_chunks method."""

    def test_add_empty_chunks(self) -> None:
        """Should handle empty chunks list."""
        store = ChromaStore(collection_name="test_add_empty")
        store.add_chunks([], "test_file")

        assert store.count() == 0

    def test_add_chunks_success(self) -> None:
        """Should add chunks to the store."""
        store = ChromaStore(collection_name="test_add_success")

        chunks = [
            Chunk(
                chunk_id="chunk_001",
                start=0.0,
                end=30.0,
                transcript="Hello world",
                speakers=[],
                visual_embedding=[],
            ),
            Chunk(
                chunk_id="chunk_002",
                start=30.0,
                end=60.0,
                transcript="Goodbye world",
                speakers=[
                    ChunkSpeaker(label="Speaker A", start=30.0, end=45.0, text="Goodbye"),
                ],
                visual_embedding=[0.1, 0.2, 0.3],
            ),
        ]

        store.add_chunks(chunks, "file_001")

        assert store.count() == 2
        assert store.count(file_id="file_001") == 2

    def test_add_chunks_caches_chunks(self) -> None:
        """Should cache chunks for retrieval."""
        store = ChromaStore(collection_name="test_cache")

        chunk = Chunk(
            chunk_id="chunk_cached",
            start=0.0,
            end=10.0,
            transcript="Test chunk",
        )

        store.add_chunks([chunk], "file_cached")

        cached = store.get_chunk("chunk_cached")
        assert cached is not None
        assert cached.transcript == "Test chunk"


class TestChromaStoreSearch:
    """Tests for ChromaStore.search method."""

    def test_search_empty_store(self) -> None:
        """Should return empty list for empty store."""
        store = ChromaStore(collection_name="test_search_empty")

        results = store.search("hello", top_k=5)

        assert results == []

    def test_search_returns_relevant_chunks(self) -> None:
        """Should return relevant chunks for query."""
        store = ChromaStore(collection_name="test_search_relevant")

        chunks = [
            Chunk(
                chunk_id="chunk_budget",
                start=0.0,
                end=30.0,
                transcript="We need to discuss the budget for Q4",
            ),
            Chunk(
                chunk_id="chunk_lunch",
                start=30.0,
                end=60.0,
                transcript="Let's order pizza for lunch",
            ),
            Chunk(
                chunk_id="chunk_revenue",
                start=60.0,
                end=90.0,
                transcript="The revenue forecast looks promising",
            ),
        ]

        store.add_chunks(chunks, "file_search")

        # Search for budget-related content
        results = store.search("financial planning budget", top_k=2)

        assert len(results) <= 2
        # Budget chunk should be in results
        chunk_ids = [c.chunk_id for c in results]
        assert "chunk_budget" in chunk_ids or "chunk_revenue" in chunk_ids

    def test_search_with_file_filter(self) -> None:
        """Should filter by file_id when specified."""
        store = ChromaStore(collection_name="test_search_filter")

        chunks1 = [
            Chunk(chunk_id="file1_chunk", start=0.0, end=30.0, transcript="Meeting notes"),
        ]
        chunks2 = [
            Chunk(chunk_id="file2_chunk", start=0.0, end=30.0, transcript="Meeting notes"),
        ]

        store.add_chunks(chunks1, "file_001")
        store.add_chunks(chunks2, "file_002")

        results = store.search("meeting", top_k=10, file_id="file_001")

        chunk_ids = [c.chunk_id for c in results]
        assert "file1_chunk" in chunk_ids
        assert "file2_chunk" not in chunk_ids


class TestChromaStoreDelete:
    """Tests for ChromaStore.delete method."""

    def test_delete_removes_chunks(self) -> None:
        """Should delete all chunks for a file."""
        store = ChromaStore(collection_name="test_delete")

        chunks = [
            Chunk(chunk_id="del_001", start=0.0, end=30.0, transcript="First"),
            Chunk(chunk_id="del_002", start=30.0, end=60.0, transcript="Second"),
        ]

        store.add_chunks(chunks, "file_to_delete")
        assert store.count() == 2

        store.delete("file_to_delete")
        assert store.count() == 0

    def test_delete_clears_cache(self) -> None:
        """Should clear cached chunks after delete."""
        store = ChromaStore(collection_name="test_delete_cache")

        chunk = Chunk(chunk_id="cached_del", start=0.0, end=10.0, transcript="Test")
        store.add_chunks([chunk], "file_del")

        assert store.get_chunk("cached_del") is not None

        store.delete("file_del")

        assert store.get_chunk("cached_del") is None

    def test_delete_only_affects_target_file(self) -> None:
        """Should only delete chunks for specified file."""
        store = ChromaStore(collection_name="test_delete_specific")

        chunks1 = [Chunk(chunk_id="keep_001", start=0.0, end=30.0, transcript="Keep")]
        chunks2 = [Chunk(chunk_id="del_001", start=0.0, end=30.0, transcript="Delete")]

        store.add_chunks(chunks1, "file_keep")
        store.add_chunks(chunks2, "file_delete")

        store.delete("file_delete")

        assert store.count(file_id="file_keep") == 1
        assert store.count(file_id="file_delete") == 0


class TestChromaStoreGetChunk:
    """Tests for ChromaStore.get_chunk method."""

    def test_get_existing_chunk(self) -> None:
        """Should return chunk by ID."""
        store = ChromaStore(collection_name="test_get")

        chunk = Chunk(chunk_id="get_001", start=0.0, end=10.0, transcript="Test")
        store.add_chunks([chunk], "file_get")

        result = store.get_chunk("get_001")

        assert result is not None
        assert result.chunk_id == "get_001"

    def test_get_nonexistent_chunk(self) -> None:
        """Should return None for nonexistent chunk."""
        store = ChromaStore(collection_name="test_get_none")

        result = store.get_chunk("nonexistent")

        assert result is None


class TestChromaStoreCount:
    """Tests for ChromaStore.count method."""

    def test_count_all(self) -> None:
        """Should count all chunks."""
        store = ChromaStore(collection_name="test_count_all")

        chunks = [
            Chunk(chunk_id="c1", start=0.0, end=10.0, transcript="One"),
            Chunk(chunk_id="c2", start=10.0, end=20.0, transcript="Two"),
            Chunk(chunk_id="c3", start=20.0, end=30.0, transcript="Three"),
        ]
        store.add_chunks(chunks, "file_count")

        assert store.count() == 3

    def test_count_by_file(self) -> None:
        """Should count chunks for specific file."""
        store = ChromaStore(collection_name="test_count_file")

        store.add_chunks(
            [Chunk(chunk_id="f1_c1", start=0.0, end=10.0, transcript="One")],
            "file_1",
        )
        store.add_chunks(
            [
                Chunk(chunk_id="f2_c1", start=0.0, end=10.0, transcript="Two"),
                Chunk(chunk_id="f2_c2", start=10.0, end=20.0, transcript="Three"),
            ],
            "file_2",
        )

        assert store.count(file_id="file_1") == 1
        assert store.count(file_id="file_2") == 2


class TestChromaStoreListFileIds:
    """Tests for ChromaStore.list_file_ids method."""

    def test_list_empty_store(self) -> None:
        """Should return empty list for empty store."""
        store = ChromaStore(collection_name="test_list_empty")

        assert store.list_file_ids() == []

    def test_list_file_ids(self) -> None:
        """Should list all unique file IDs."""
        store = ChromaStore(collection_name="test_list_ids")

        store.add_chunks(
            [Chunk(chunk_id="a1", start=0.0, end=10.0, transcript="A")],
            "file_a",
        )
        store.add_chunks(
            [Chunk(chunk_id="b1", start=0.0, end=10.0, transcript="B")],
            "file_b",
        )
        store.add_chunks(
            [Chunk(chunk_id="c1", start=0.0, end=10.0, transcript="C")],
            "file_c",
        )

        file_ids = store.list_file_ids()

        assert sorted(file_ids) == ["file_a", "file_b", "file_c"]
