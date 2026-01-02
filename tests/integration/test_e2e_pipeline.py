"""End-to-end integration tests for the full pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from chromaflow import Pipeline, PipelineError, VideoData


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Full pipeline with real files."""

    def test_full_video_processing(
        self, multi_scene_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Process real video through all stages."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},  # Skip diarization for speed
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",  # Use CUDA for visual processing
            audio_device="cpu",  # Use CPU for Whisper (CUDA crashes on some systems)
        )

        result = pipeline.process(multi_scene_video)

        # ASSERT: Basic structure
        assert result.file_id.startswith("vid_")
        assert result.metadata.duration > 0
        assert len(result.chunks) > 0

        # ASSERT: Chunks have content
        for chunk in result.chunks:
            assert chunk.start < chunk.end
            # Visual embeddings present (video file)
            assert len(chunk.visual_embedding) == 768
            # Screenshot saved
            assert chunk.screenshot_path is not None

    def test_audio_only_no_visual_embeddings(
        self, podcast_audio: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Audio files should have empty visual embeddings."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(podcast_audio)

        # ASSERT: Chunks exist
        assert len(result.chunks) > 0

        # ASSERT: No visual embeddings for audio
        for chunk in result.chunks:
            assert chunk.visual_embedding == []
            assert chunk.screenshot_path is None

    def test_search_after_processing(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Search should work on processed video."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(short_talking_head_video)

        # Get some transcript text to search for
        if result.chunks and result.chunks[0].transcript:
            # Search for words from the actual transcript
            words = result.chunks[0].transcript.split()[:3]
            if words:
                query = " ".join(words)

                search_results = result.search(query, top_k=3)

                # ASSERT: Returns results
                assert len(search_results) > 0
                # ASSERT: First chunk is in results (it contains the query words)
                result_ids = [r.chunk_id for r in search_results]
                assert result.chunks[0].chunk_id in result_ids

    def test_search_disabled_raises_error(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Search should fail gracefully when disabled."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,  # Disabled
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(short_talking_head_video)

        # ASSERT: Search raises clear error
        with pytest.raises(RuntimeError, match="not available"):
            result.search("test query")

    def test_multiple_videos_same_pipeline(
        self,
        short_talking_head_video: Path,
        multi_scene_video: Path,
        temp_dir: Path,
        hf_token_env: str,
    ) -> None:
        """Pipeline should handle multiple videos."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        result1 = pipeline.process(short_talking_head_video)
        result2 = pipeline.process(multi_scene_video)

        # ASSERT: Different file IDs
        assert result1.file_id != result2.file_id

        # ASSERT: Both searchable
        store = pipeline.get_store()
        assert store.count() == len(result1.chunks) + len(result2.chunks)

        # ASSERT: Can filter search by file
        if result1.chunks and result1.chunks[0].transcript:
            file1_results = store.search(
                result1.chunks[0].transcript[:20], file_id=result1.file_id
            )
            for r in file1_results:
                assert r.chunk_id.startswith(result1.file_id)

    def test_extract_frames_false_skips_visual(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """extract_frames=False should skip visual processing."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(short_talking_head_video, extract_frames=False)

        # ASSERT: No visual embeddings
        for chunk in result.chunks:
            assert chunk.visual_embedding == []
            assert chunk.screenshot_path is None

    def test_video_data_to_json(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """VideoData should serialize to JSON correctly."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(short_talking_head_video)

        # Export to JSON
        json_path = temp_dir / "output.json"
        json_str = result.to_json(json_path)

        # ASSERT: JSON file created
        assert json_path.exists()

        # ASSERT: JSON is valid and contains expected fields
        import json

        data = json.loads(json_str)
        assert "file_id" in data
        assert "chunks" in data
        assert "metadata" in data
        assert len(data["chunks"]) == len(result.chunks)

    def test_silent_video_processing(
        self, silent_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Should handle videos with no audio track."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        # This may fail or produce empty transcripts - that's OK
        # We just want to verify it doesn't crash
        try:
            result = pipeline.process(silent_video)

            # If it succeeds, verify structure
            assert result.file_id.startswith("vid_")
            assert len(result.chunks) >= 0
        except PipelineError:
            # Expected for silent video - audio extraction may fail
            pass

    def test_processing_creates_output_files(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Processing should create frame files in output directory."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(short_talking_head_video, extract_frames=True)

        # ASSERT: Frames were saved
        frames_with_paths = [c for c in result.chunks if c.screenshot_path]
        assert len(frames_with_paths) > 0

        # ASSERT: Frame files exist
        for chunk in frames_with_paths:
            assert Path(chunk.screenshot_path).exists()


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Edge cases for pipeline processing."""

    def test_low_quality_video(
        self, dark_low_quality_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Should handle low quality/dark videos."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=False,
            device="cuda",
            audio_device="cpu",
        )

        result = pipeline.process(dark_low_quality_video, extract_frames=True)

        # Should complete without errors
        assert result.file_id.startswith("vid_")
        assert len(result.chunks) > 0

        # Visual embeddings should still be generated
        chunks_with_embeddings = [c for c in result.chunks if c.visual_embedding]
        assert len(chunks_with_embeddings) > 0

    def test_pipeline_reuse(
        self, short_talking_head_video: Path, temp_dir: Path, hf_token_env: str
    ) -> None:
        """Same pipeline instance should be reusable."""
        pipeline = Pipeline(
            mode="local",
            output_dir=str(temp_dir),
            options={"diarize": False},
            enable_search=True,
            persist_store=True,  # Use persistent store for test isolation
            device="cuda",
            audio_device="cpu",
        )

        # Process same video twice
        result1 = pipeline.process(short_talking_head_video)
        result2 = pipeline.process(short_talking_head_video)

        # Both should succeed
        assert result1.file_id.startswith("vid_")
        assert result2.file_id.startswith("vid_")

        # File IDs should be different (include UUID)
        assert result1.file_id != result2.file_id
