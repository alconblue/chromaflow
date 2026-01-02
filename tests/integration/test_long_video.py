"""Stress tests with longer videos (3-5 minutes).

These tests are resource-intensive and may crash when run alongside other tests
due to memory pressure. Run them in isolation with:
    pytest tests/integration/test_long_video.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stress
class TestLongVideoStress:
    """Stress tests with longer videos (3-5 minutes).

    These tests require significant memory and should be run in isolation.
    """

    def test_memory_stability(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Processing long video should not leak memory."""
        import tracemalloc

        from chromaflow import Pipeline

        tracemalloc.start()

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(long_meeting_video)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # ASSERT: Peak memory < 4GB (reasonable for 3-5 min video)
        assert peak < 4 * 1024 * 1024 * 1024  # 4GB

        # ASSERT: Processing completed
        assert len(result.chunks) > 0

    def test_chroma_batching_many_chunks(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """ChromaDB should handle many chunks efficiently."""
        from chromaflow import Pipeline

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(long_meeting_video)

        # ASSERT: All chunks indexed
        store = pipeline.get_store()
        assert store.count(file_id=result.file_id) == len(result.chunks)

        # ASSERT: Search still works with many chunks
        if result.chunks and result.chunks[0].transcript:
            search_results = result.search(result.chunks[0].transcript[:50])
            assert len(search_results) > 0

    def test_lazy_loading_not_retriggered(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Models should only load once across multiple operations."""
        import chromaflow.stages.audio as audio_module
        import chromaflow.stages.visual as visual_module
        from chromaflow import Pipeline

        # Reset module state
        visual_module._clip_model = None
        visual_module._decord_loaded = False
        audio_module._whisper_model = None

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        # Process video (triggers model loading)
        result1 = pipeline.process(long_meeting_video)

        # Capture loaded state
        clip_model_after_first = visual_module._clip_model
        whisper_model_after_first = audio_module._whisper_model

        # Process same video again
        result2 = pipeline.process(long_meeting_video)

        # ASSERT: Same model instances (not reloaded)
        assert visual_module._clip_model is clip_model_after_first
        assert audio_module._whisper_model is whisper_model_after_first

    def test_many_keyframes_extraction(
        self, long_meeting_video: Path, temp_dir: Path
    ) -> None:
        """Should handle extracting many keyframes without issues."""
        from chromaflow.stages.visual import extract_keyframes
        from chromaflow.stages.ingest import extract_metadata

        # Get actual video duration first
        metadata = extract_metadata(long_meeting_video)
        video_duration = metadata.duration

        # Generate timestamps every 2 seconds, within video bounds
        timestamps = [float(i * 2) for i in range(int(video_duration / 2))]

        results = extract_keyframes(
            long_meeting_video,
            timestamps=timestamps,
            output_dir=temp_dir / "frames",
            file_id="stress_test",
        )

        # ASSERT: All frames extracted
        assert len(results) == len(timestamps)
        assert len(results) >= 60  # At least 60 frames for a 2+ min video

        # ASSERT: Spot check a few saved files are valid (not all, to save time)
        for r in results[:5]:  # Check first 5
            if r.saved_path:
                img = Image.open(r.saved_path)
                img.verify()

    def test_long_video_produces_many_chunks(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Long video should produce proportionally more chunks."""
        from chromaflow import Pipeline

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        # Skip frame extraction to reduce memory usage
        result = pipeline.process(long_meeting_video, extract_frames=False)

        # For a 3-5 minute video, expect multiple chunks
        # (scene detection typically produces 1 chunk per 10-30 seconds of content)
        assert len(result.chunks) >= 5

        # Duration should be 2-6 minutes (relaxed bounds for fixture flexibility)
        assert result.metadata.duration >= 120  # At least 2 minutes
        assert result.metadata.duration <= 360  # At most 6 minutes

    def test_long_video_all_chunks_have_embeddings(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """All chunks in long video should have visual embeddings."""
        from chromaflow import Pipeline

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(long_meeting_video, extract_frames=True)

        # Every chunk should have a 768-dim embedding
        for i, chunk in enumerate(result.chunks):
            assert len(chunk.visual_embedding) == 768, f"Chunk {i} missing embedding"
            assert chunk.screenshot_path is not None, f"Chunk {i} missing screenshot"

    def test_search_performance_with_many_chunks(
        self, long_meeting_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Search should remain fast even with many chunks."""
        import time

        from chromaflow import Pipeline

        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(long_meeting_video)

        # Measure search time
        start = time.perf_counter()
        for _ in range(10):  # Run 10 searches
            result.search("test query", top_k=5)
        elapsed = time.perf_counter() - start

        # ASSERT: 10 searches complete in under 5 seconds
        # (average 500ms per search is generous)
        assert elapsed < 5.0, f"Search too slow: {elapsed:.2f}s for 10 searches"
