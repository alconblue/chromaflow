#!/usr/bin/env python3
"""ChromaFlow Quickstart Example.

This script demonstrates the basic usage of ChromaFlow to process
a video file and extract RAG-ready artifacts.

Usage:
    python examples/quickstart.py path/to/video.mp4

Requirements:
    - Set HF_TOKEN environment variable for speaker diarization
    - Or run with --no-diarize flag to skip diarization
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the quickstart example."""
    import chromaflow

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python quickstart.py <video_file> [--no-diarize]")
        print("\nExample:")
        print("  python quickstart.py meeting.mp4")
        print("  python quickstart.py meeting.mp4 --no-diarize")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    diarize = "--no-diarize" not in sys.argv

    if not video_path.exists():
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    print(f"ChromaFlow v{chromaflow.__version__}")
    print(f"Processing: {video_path}")
    print(f"Diarization: {'enabled' if diarize else 'disabled'}")
    print("-" * 50)

    # Initialize the pipeline
    pipeline = chromaflow.Pipeline(
        mode="local",
        options={"diarize": diarize},
    )

    # Process the video
    video_data = pipeline.process(video_path)

    # Display results
    print(f"\nSummary: {video_data.summary}")
    print(f"File ID: {video_data.file_id}")
    print(f"Duration: {video_data.metadata.duration:.1f}s")
    print(f"Resolution: {video_data.metadata.resolution}")
    print(f"Chunks: {len(video_data.chunks)}")

    print("\n" + "=" * 50)
    print("CHUNKS")
    print("=" * 50)

    for chunk in video_data.chunks:
        print(f"\n[{chunk.start:.1f}s - {chunk.end:.1f}s] {chunk.chunk_id}")
        print(f"  Transcript: {chunk.transcript[:100]}...")

        if chunk.speakers:
            print(f"  Speakers: {', '.join(s.label for s in chunk.speakers)}")

    # Export to JSON
    output_path = video_path.with_suffix(".chromaflow.json")
    video_data.to_json(output_path)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
