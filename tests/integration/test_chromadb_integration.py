"""Integration tests for ChromaDB with real operations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chromaflow.models.schema import Chunk, ChunkSpeaker
from chromaflow.store.chroma import ChromaStore


def make_chunk(
    chunk_id: str,
    start: float,
    end: float,
    transcript: str,
    speakers: list[ChunkSpeaker] | None = None,
    visual_embedding: list[float] | None = None,
    transcript_embedding: list[float] | None = None,
    screenshot_path: str | None = None,
) -> Chunk:
    """Create a test chunk with auto-generated transcript embedding.

    If transcript_embedding is not provided, generates a dummy 384-dim embedding.
    """
    if transcript_embedding is None and transcript:
        # Generate a simple deterministic 384-dim embedding based on transcript hash
        h = hashlib.md5(transcript.encode()).hexdigest()
        transcript_embedding = [float(int(c, 16)) / 15.0 for c in h] * 12  # 384 dims

    return Chunk(
        chunk_id=chunk_id,
        start=start,
        end=end,
        transcript=transcript,
        speakers=speakers or [],
        visual_embedding=visual_embedding or [],
        transcript_embedding=transcript_embedding or [],
        screenshot_path=screenshot_path,
    )


@pytest.mark.integration
class TestChromaDBIntegration:
    """Real ChromaDB operations."""

    def test_add_and_retrieve_chunks(self) -> None:
        """Basic add/search cycle."""
        store = ChromaStore(collection_name="test_add_retrieve")

        chunks = [
            make_chunk(
                chunk_id="c1",
                start=0,
                end=30,
                transcript="The budget meeting discussed Q4 targets",
            ),
            make_chunk(
                chunk_id="c2",
                start=30,
                end=60,
                transcript="Marketing presented the new campaign",
            ),
            make_chunk(
                chunk_id="c3",
                start=60,
                end=90,
                transcript="Engineering demo of the new feature",
            ),
        ]

        store.add_chunks(chunks, file_id="video_001")

        # ASSERT: Count is correct
        assert store.count() == 3

        # Search
        results = store.search("financial planning budget", top_k=2)

        # ASSERT: Returns results
        assert len(results) > 0
        # ASSERT: Budget chunk is in top results
        result_ids = [r.chunk_id for r in results]
        assert "c1" in result_ids

    def test_search_relevance_ordering(self) -> None:
        """More relevant results should rank higher."""
        store = ChromaStore(collection_name="test_relevance")

        chunks = [
            make_chunk(
                chunk_id="exact",
                start=0,
                end=10,
                transcript="Python programming tutorial for beginners",
            ),
            make_chunk(
                chunk_id="related",
                start=10,
                end=20,
                transcript="Software development best practices",
            ),
            make_chunk(
                chunk_id="unrelated",
                start=20,
                end=30,
                transcript="Recipe for chocolate cake",
            ),
        ]

        store.add_chunks(chunks, file_id="test")

        results = store.search("Python coding tutorial", top_k=3)

        # ASSERT: Most relevant is first
        assert results[0].chunk_id == "exact"
        # ASSERT: All 3 results returned
        assert len(results) == 3

    def test_file_id_filtering(self) -> None:
        """Search should filter by file_id."""
        store = ChromaStore(collection_name="test_filter")

        store.add_chunks(
            [make_chunk(chunk_id="f1_c1", start=0, end=10, transcript="Meeting about budgets")],
            file_id="file_1",
        )

        store.add_chunks(
            [make_chunk(chunk_id="f2_c1", start=0, end=10, transcript="Meeting about budgets")],
            file_id="file_2",
        )

        # Search only in file_1
        results = store.search("budget meeting", file_id="file_1")

        # ASSERT: Only file_1 results
        for r in results:
            assert r.chunk_id.startswith("f1")

    def test_delete_removes_from_search(self) -> None:
        """Deleted chunks should not appear in search."""
        store = ChromaStore(collection_name="test_delete")

        store.add_chunks(
            [make_chunk(chunk_id="c1", start=0, end=10, transcript="Important meeting notes")],
            file_id="to_delete",
        )

        # Verify it's searchable
        assert store.count() == 1

        # Delete
        store.delete("to_delete")

        # ASSERT: No longer searchable
        assert store.count() == 0
        results = store.search("meeting notes")
        assert len(results) == 0

    def test_persistent_storage(self, temp_dir: Path) -> None:
        """Data should persist across store instances."""
        persist_path = str(temp_dir / "chromadb")

        # Create and add
        store1 = ChromaStore(
            collection_name="persist_test", persist_directory=persist_path
        )
        store1.add_chunks(
            [make_chunk(chunk_id="c1", start=0, end=10, transcript="Persistent data")],
            file_id="test",
        )

        # Note: ChromaStore caches chunks in memory, so count will work
        # but for true persistence test, we need to check the collection directly
        del store1

        # Reopen
        store2 = ChromaStore(
            collection_name="persist_test", persist_directory=persist_path
        )

        # ASSERT: Data persisted (text collection count)
        assert store2._text_collection.count() == 1

    def test_search_determinism(self) -> None:
        """Re-running the same query returns the same top result."""
        store = ChromaStore(collection_name="test_determinism")

        chunks = [
            make_chunk(
                chunk_id="c1",
                start=0,
                end=30,
                transcript="Machine learning and artificial intelligence",
            ),
            make_chunk(
                chunk_id="c2",
                start=30,
                end=60,
                transcript="Database systems and SQL queries",
            ),
            make_chunk(
                chunk_id="c3",
                start=60,
                end=90,
                transcript="Web development with JavaScript",
            ),
            make_chunk(
                chunk_id="c4",
                start=90,
                end=120,
                transcript="Cloud computing and DevOps practices",
            ),
        ]

        store.add_chunks(chunks, file_id="test")

        query = "AI and machine learning applications"

        # Run the same query multiple times
        results_1 = store.search(query, top_k=3)
        results_2 = store.search(query, top_k=3)
        results_3 = store.search(query, top_k=3)

        # ASSERT: Top result is always the same
        assert results_1[0].chunk_id == results_2[0].chunk_id == results_3[0].chunk_id

        # ASSERT: Full result order is deterministic
        assert [r.chunk_id for r in results_1] == [r.chunk_id for r in results_2]
        assert [r.chunk_id for r in results_2] == [r.chunk_id for r in results_3]

    def test_chunks_with_speakers(self) -> None:
        """Should handle chunks with speaker information."""
        store = ChromaStore(collection_name="test_speakers")

        chunks = [
            make_chunk(
                chunk_id="c1",
                start=0,
                end=30,
                transcript="Hello, welcome to the meeting.",
                speakers=[
                    ChunkSpeaker(
                        label="Speaker A",
                        start=0,
                        end=15,
                        text="Hello, welcome to the meeting.",
                    ),
                ],
            ),
            make_chunk(
                chunk_id="c2",
                start=30,
                end=60,
                transcript="Thank you for joining us today.",
                speakers=[
                    ChunkSpeaker(
                        label="Speaker B",
                        start=30,
                        end=45,
                        text="Thank you for joining us today.",
                    ),
                ],
            ),
        ]

        store.add_chunks(chunks, file_id="meeting")

        assert store.count() == 2

        results = store.search("welcome meeting", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert len(results[0].speakers) == 1

    def test_chunks_with_visual_embeddings(self) -> None:
        """Should handle chunks with visual embeddings metadata."""
        store = ChromaStore(collection_name="test_visual")

        chunks = [
            make_chunk(
                chunk_id="c1",
                start=0,
                end=30,
                transcript="A slide about quarterly results",
                visual_embedding=[0.1] * 768,  # 768-dim CLIP embedding
                screenshot_path="/tmp/frame_000.jpg",
            ),
            make_chunk(
                chunk_id="c2",
                start=30,
                end=60,
                transcript="Discussion about revenue",
                visual_embedding=[],  # No visual embedding
            ),
        ]

        store.add_chunks(chunks, file_id="presentation")

        assert store.count() == 2

        # Retrieve and verify embeddings preserved
        chunk1 = store.get_chunk("c1")
        assert chunk1 is not None
        assert len(chunk1.visual_embedding) == 768
        assert chunk1.screenshot_path == "/tmp/frame_000.jpg"

    def test_list_file_ids(self) -> None:
        """Should list all unique file IDs."""
        store = ChromaStore(collection_name="test_list_ids")

        store.add_chunks(
            [make_chunk(chunk_id="a1", start=0, end=10, transcript="A")],
            file_id="file_a",
        )
        store.add_chunks(
            [make_chunk(chunk_id="b1", start=0, end=10, transcript="B")],
            file_id="file_b",
        )
        store.add_chunks(
            [make_chunk(chunk_id="c1", start=0, end=10, transcript="C")],
            file_id="file_c",
        )

        file_ids = store.list_file_ids()

        assert sorted(file_ids) == ["file_a", "file_b", "file_c"]

    def test_count_by_file_id(self) -> None:
        """Should count chunks for specific file."""
        store = ChromaStore(collection_name="test_count_file")

        store.add_chunks(
            [make_chunk(chunk_id="f1_c1", start=0, end=10, transcript="One")],
            file_id="file_1",
        )
        store.add_chunks(
            [
                make_chunk(chunk_id="f2_c1", start=0, end=10, transcript="Two"),
                make_chunk(chunk_id="f2_c2", start=10, end=20, transcript="Three"),
            ],
            file_id="file_2",
        )

        assert store.count(file_id="file_1") == 1
        assert store.count(file_id="file_2") == 2
        assert store.count() == 3
