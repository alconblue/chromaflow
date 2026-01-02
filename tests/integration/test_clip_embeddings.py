"""Integration tests for CLIP embedding generation with real video files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chromaflow.stages.visual import (
    extract_keyframes,
    generate_embedding,
    generate_embeddings,
    process_visual,
)


@pytest.mark.integration
@pytest.mark.slow  # Model loading takes time
class TestClipEmbeddings:
    """Real CLIP embedding generation."""

    def test_embedding_dimensions(self, multi_scene_video: Path) -> None:
        """Embeddings should be 768-dim for ViT-L/14."""
        results = process_visual(
            source=multi_scene_video,
            keyframe_times=[10.0],
            model_name="clip-ViT-L-14",
            device="cpu",
        )

        # ASSERT: Correct dimensions for ViT-L/14
        assert len(results) == 1
        assert len(results[0].embedding) == 768

    def test_embeddings_are_normalized(self, multi_scene_video: Path) -> None:
        """CLIP embeddings should be L2-normalized."""
        results = process_visual(
            source=multi_scene_video,
            keyframe_times=[10.0],
            device="cpu",
        )

        embedding = np.array(results[0].embedding)
        norm = np.linalg.norm(embedding)

        # ASSERT: Normalized (L2 norm ≈ 1.0)
        assert 0.99 < norm < 1.01

    def test_different_frames_different_embeddings(self, multi_scene_video: Path) -> None:
        """Distinct scenes should have different embeddings."""
        results = process_visual(
            source=multi_scene_video,
            keyframe_times=[0.0, 30.0],  # Different scenes
            device="cpu",
        )

        emb1 = np.array(results[0].embedding)
        emb2 = np.array(results[1].embedding)

        # ASSERT: Embeddings are not identical
        cosine_sim = np.dot(emb1, emb2)
        assert cosine_sim < 0.99  # Should be different

    def test_similar_frames_similar_embeddings(self, screencast_video: Path) -> None:
        """Adjacent frames in static content should be similar."""
        # Screencast: frames 1 second apart should be very similar
        results = process_visual(
            source=screencast_video,
            keyframe_times=[10.0, 11.0],
            device="cpu",
        )

        emb1 = np.array(results[0].embedding)
        emb2 = np.array(results[1].embedding)

        cosine_sim = np.dot(emb1, emb2)
        # ASSERT: High similarity for adjacent static frames
        assert cosine_sim > 0.8

    def test_batch_processing_matches_individual(self, multi_scene_video: Path) -> None:
        """Batch and individual processing should give same results."""
        frames = extract_keyframes(multi_scene_video, [5.0, 15.0])

        # Individual
        individual = [generate_embedding(f.frame, device="cpu") for f in frames]

        # Batch
        batch = generate_embeddings([f.frame for f in frames], device="cpu")

        # ASSERT: Results match within floating point tolerance
        # Note: batch processing may have slightly different numerical results
        # due to batched vs individual encoding, so we use a relaxed tolerance
        for ind, bat in zip(individual, batch):
            np.testing.assert_allclose(ind, bat, rtol=1e-4, atol=1e-5)

    def test_embedding_with_saved_frames(
        self, multi_scene_video: Path, temp_dir: Path
    ) -> None:
        """Should generate embeddings and save frames together."""
        results = process_visual(
            source=multi_scene_video,
            keyframe_times=[5.0, 25.0, 45.0],
            output_dir=temp_dir / "frames",
            file_id="embed_test",
            device="cpu",
        )

        assert len(results) == 3

        for result in results:
            # Each result should have embedding
            assert len(result.embedding) == 768
            # And saved frame path
            assert result.frame_path is not None
            assert result.frame_path.exists()

    def test_empty_keyframe_times_returns_empty(self, multi_scene_video: Path) -> None:
        """Empty keyframe times should return empty list."""
        results = process_visual(
            source=multi_scene_video,
            keyframe_times=[],
            device="cpu",
        )

        assert results == []

    def test_embedding_determinism(self, short_talking_head_video: Path) -> None:
        """Same frame should produce same embedding."""
        # Extract same frame twice
        results1 = process_visual(
            source=short_talking_head_video,
            keyframe_times=[5.0],
            device="cpu",
        )
        results2 = process_visual(
            source=short_talking_head_video,
            keyframe_times=[5.0],
            device="cpu",
        )

        emb1 = np.array(results1[0].embedding)
        emb2 = np.array(results2[0].embedding)

        # ASSERT: Embeddings are identical
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)
