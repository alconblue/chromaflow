# ChromaFlow

Turn video into RAG-ready artifacts in one line of code.

```python
import chromaflow

pipeline = chromaflow.Pipeline(mode="local")
video_data = pipeline.process("meeting.mp4")

# Immediately searchable
results = video_data.search("When did they discuss the budget?")
```

## Features

- **Semantic Scene Detection** - Split video by visual content, not arbitrary time windows
- **Speech Transcription** - High-accuracy transcription with word-level timestamps
- **Speaker Diarization** - Identify who said what (Speaker A, Speaker B, etc.)
- **Visual Embeddings** - CLIP embeddings for every scene keyframe
- **Built-in Search** - Semantic search over your video content via ChromaDB
- **Hardware Acceleration** - Auto-detects CUDA, Apple Silicon (MPS), or falls back to CPU

## Installation

### Prerequisites

- **Python 3.10, 3.11, or 3.12**
- FFmpeg (for video processing)

### CPU-Only Installation

```bash
pip install chromaflow
```

### CUDA (NVIDIA GPU) Installation

```bash
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install ChromaFlow
pip install chromaflow
```

### Apple Silicon (MPS) Installation

```bash
# PyTorch supports MPS out of the box on macOS
pip install chromaflow
```

### Development Installation

```bash
git clone https://github.com/chromaflow/chromaflow.git
cd chromaflow
poetry install
```

## Quick Start

### Python API

```python
import chromaflow

# Initialize pipeline (auto-detects GPU)
pipeline = chromaflow.Pipeline(mode="local")

# Process a video
video_data = pipeline.process(
    source="meeting.mp4",
    options={"diarize": True}
)

# Access results
print(video_data.summary)
# > "A 5.0-minute video with 3 segments and 2 speakers."

# Search semantically
results = video_data.search("revenue discussion")
for chunk in results:
    print(f"[{chunk.start:.1f}s - {chunk.end:.1f}s] {chunk.transcript}")

# Export to JSON
video_data.to_json("output.json")
```

### Command Line

```bash
# Process a video file
chromaflow process meeting.mp4 --diarize -o output.json

# Check hardware detection
chromaflow info

# Get help
chromaflow --help
```

## Supported Formats

| Type  | Formats |
|-------|---------|
| Video | `.mp4`, `.mov`, `.avi`, `.mkv` |
| Audio | `.mp3`, `.wav` |

## Configuration

```python
pipeline = chromaflow.Pipeline(
    mode="local",           # 'local' or 'cloud' (cloud coming soon)
    device="cuda",          # 'cuda', 'mps', 'cpu', or None (auto-detect)
    output_dir="./output",  # Where to store extracted frames
    options={
        "diarize": True,            # Enable speaker diarization
        "whisper_model": "small",   # tiny, base, small, medium, large-v3
    }
)
```

---

## Speaker Diarization Notice

> **Important:** ChromaFlow uses [Pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.
>
> Pyannote requires users to:
> 1. Accept its license terms on HuggingFace
> 2. May require a paid plan for commercial use
>
> **You must set a valid `HF_TOKEN` environment variable to enable diarization.**
>
> To set up:
> 1. Create a HuggingFace account at https://huggingface.co
> 2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
> 3. Create an access token at https://huggingface.co/settings/tokens
> 4. Set the environment variable:
>    ```bash
>    export HF_TOKEN='your_token_here'
>    ```
>
> To disable diarization:
> ```python
> pipeline.process("video.mp4", options={"diarize": False})
> ```

---

## System Requirements

### FFmpeg

ChromaFlow requires FFmpeg for video processing. Install it for your platform:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

### OpenCV

SceneDetect requires OpenCV. It's included automatically via the `scenedetect[opencv]` dependency.

## Output Schema

ChromaFlow outputs a structured JSON format designed for RAG applications:

```json
{
  "file_id": "vid_a1b2c3d4_e5f6",
  "source_path": "/path/to/video.mp4",
  "metadata": {
    "duration": 300.0,
    "resolution": "1920x1080",
    "fps": 30.0,
    "format": "mp4"
  },
  "chunks": [
    {
      "chunk_id": "vid_a1b2c3d4_e5f6_chunk_000",
      "start": 0.0,
      "end": 60.0,
      "transcript": "Welcome to the quarterly review...",
      "speakers": [
        {
          "label": "Speaker A",
          "start": 0.0,
          "end": 45.0,
          "text": "Welcome to the quarterly review."
        }
      ],
      "visual_embedding": [0.123, -0.456, ...],
      "screenshot_path": "./output/frames/vid_a1b2c3d4_e5f6_000.jpg"
    }
  ],
  "summary": "A 5.0-minute video with 3 segments and 2 speakers."
}
```

## Performance Targets

| Stage | Target (GPU) |
|-------|--------------|
| Transcoding | < 10% of video duration |
| Audio Pipeline | < 20% of video duration |
| Visual Pipeline | < 20% of video duration |
| **Total** | **< 50% of video duration** |

A 1-hour video should process in under 30 minutes on an M2 Pro or RTX 3060.

---

## Verification

This section helps you verify ChromaFlow is working correctly after installation.

### Running Tests

```bash
# Run all tests
./scripts/run_tests.sh

# Run with verbose output
./scripts/run_tests.sh -v

# Run specific test file
./scripts/run_tests.sh tests/test_pipeline.py

# Run with coverage
./scripts/run_tests.sh --cov
```

### Expected Test Output

```
========================================
ChromaFlow Test Suite
========================================
tests/test_audio.py::TestTranscriptSegment::test_duration_property PASSED
tests/test_audio.py::TestTranscriptionResult::test_full_text_property PASSED
tests/test_ingest.py::TestValidateFile::test_validate_existing_mp4 PASSED
tests/test_pipeline.py::TestPipelineInit::test_init_local_mode_with_hf_token PASSED
...
========================================
All tests passed!
========================================
```

### Expected Processing Logs

When processing a video, you should see logs like:

```
12:34:56 [INFO] chromaflow.pipeline: ChromaFlow pipeline initialized (device=cuda)
12:34:56 [INFO] chromaflow.store.chroma: chromadb loaded in 0.12s
12:34:56 [INFO] chromaflow.store.chroma: ChromaDB ephemeral client initialized
12:34:56 [INFO] chromaflow.store.chroma: ChromaDB collection 'chromaflow_videos' ready in 0.05s (documents: 0)
12:34:56 [INFO] chromaflow.pipeline: Processing: meeting.mp4
12:34:56 [INFO] chromaflow.pipeline: Stage 1: Extracting metadata...
12:34:57 [INFO] chromaflow.stages.scene: scenedetect loaded in 0.45s
12:34:57 [INFO] chromaflow.pipeline: Stage 2: Detecting scenes...
12:34:58 [INFO] chromaflow.stages.scene: Scene detection complete: 5 scenes in 1.23s
12:34:58 [INFO] chromaflow.stages.ingest: Extracting audio from meeting.mp4 -> meeting_audio.wav
12:34:59 [INFO] chromaflow.stages.ingest: Audio extracted in 0.87s
12:34:59 [INFO] chromaflow.pipeline: Stage 3: Processing audio...
12:34:59 [INFO] chromaflow.stages.audio: Loading Whisper model 'small' on cuda...
12:35:02 [INFO] chromaflow.stages.audio: Whisper model loaded in 2.45s (size=small, device=cuda, compute_type=float16)
12:35:02 [INFO] chromaflow.stages.audio: Transcribing meeting_audio.wav...
12:35:15 [INFO] chromaflow.stages.audio: Transcription complete: 42 segments, 856 words in 12.34s (RTF: 0.21x, lang: en)
12:35:15 [INFO] chromaflow.stages.audio: Loading pyannote diarization pipeline on cuda...
12:35:18 [INFO] chromaflow.stages.audio: Diarization pipeline loaded in 3.12s (device=cuda)
12:35:18 [INFO] chromaflow.stages.audio: Diarizing meeting_audio.wav...
12:35:25 [INFO] chromaflow.stages.audio: Diarization complete: 28 segments, 3 speakers in 6.89s
12:35:25 [INFO] chromaflow.stages.audio: Merged 3 speakers into transcript
12:35:25 [INFO] chromaflow.pipeline: Stage 4: Processing visual embeddings...
12:35:25 [INFO] chromaflow.stages.visual: decord loaded in 0.08s
12:35:26 [INFO] chromaflow.stages.visual: Extracted 5 keyframes in 0.45s (avg 90.0ms/frame)
12:35:26 [INFO] chromaflow.stages.visual: Loading CLIP model 'openai/clip-vit-large-patch14' on cuda...
12:35:30 [INFO] chromaflow.stages.visual: CLIP model loaded in 4.12s (model=openai/clip-vit-large-patch14, device=cuda)
12:35:31 [INFO] chromaflow.stages.visual: Generated 5 embeddings in 0.89s (avg 178.0ms/frame)
12:35:31 [INFO] chromaflow.pipeline: Stage 5: Building aligned chunks...
12:35:31 [INFO] chromaflow.pipeline: Stage 6: Indexing in ChromaDB...
12:35:31 [INFO] chromaflow.store.chroma: Added 5 chunks to ChromaDB in 0.12s (file_id=vid_a1b2c3d4_e5f6)
12:35:31 [INFO] chromaflow.pipeline: Processing complete: 5 chunks in 34.56s (RTF: 0.58x)
```

### Expected Runtime

| File Type | Duration | Device | Expected Time |
|-----------|----------|--------|---------------|
| Video (1080p) | 5 min | CPU | 10-20 min |
| Video (1080p) | 5 min | CUDA | 2-5 min |
| Video (1080p) | 5 min | MPS | 4-8 min |
| Audio only | 5 min | CPU | 3-8 min |
| Audio only | 5 min | CUDA | 1-2 min |

**Note:** First run is slower due to model downloads (~3GB for Whisper small + pyannote + CLIP).

### Example Output JSON

```json
{
  "file_id": "vid_a1b2c3d4_e5f6",
  "source_path": "/path/to/meeting.mp4",
  "metadata": {
    "duration": 312.5,
    "resolution": "1920x1080",
    "fps": 29.97,
    "format": "mov",
    "audio_channels": 2,
    "sample_rate": 48000
  },
  "chunks": [
    {
      "chunk_id": "vid_a1b2c3d4_e5f6_chunk_000",
      "start": 0.0,
      "end": 45.2,
      "transcript": "Good morning everyone. Welcome to our quarterly review meeting. Today we'll be covering the Q3 results and discussing our roadmap for Q4.",
      "speakers": [
        {
          "label": "Speaker A",
          "start": 0.0,
          "end": 12.5,
          "text": "Good morning everyone."
        },
        {
          "label": "Speaker A",
          "start": 12.5,
          "end": 45.2,
          "text": "Welcome to our quarterly review meeting. Today we'll be covering the Q3 results and discussing our roadmap for Q4."
        }
      ],
      "visual_embedding": [0.0234, -0.0156, 0.0412, ...],
      "screenshot_path": "./chromaflow_output/vid_a1b2c3d4_e5f6/frames/vid_a1b2c3d4_e5f6_000.jpg"
    },
    {
      "chunk_id": "vid_a1b2c3d4_e5f6_chunk_001",
      "start": 45.2,
      "end": 98.7,
      "transcript": "Let me share my screen. As you can see here, our revenue grew by 23% compared to last quarter.",
      "speakers": [
        {
          "label": "Speaker B",
          "start": 45.2,
          "end": 52.1,
          "text": "Let me share my screen."
        },
        {
          "label": "Speaker B",
          "start": 52.1,
          "end": 98.7,
          "text": "As you can see here, our revenue grew by 23% compared to last quarter."
        }
      ],
      "visual_embedding": [0.0189, 0.0267, -0.0098, ...],
      "screenshot_path": "./chromaflow_output/vid_a1b2c3d4_e5f6/frames/vid_a1b2c3d4_e5f6_001.jpg"
    }
  ],
  "summary": "A 5.2-minute video with 5 segments and 3 speakers."
}
```

### Semantic Search

After processing, you can search your video content:

```python
# Search by text query
results = video_data.search("When did they discuss revenue?")
for chunk in results:
    print(f"[{chunk.start:.1f}s - {chunk.end:.1f}s] {chunk.transcript}")
```

### Troubleshooting

**"ffprobe not found"**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**"HuggingFace token required for diarization"**
```bash
# Set your HuggingFace token
export HF_TOKEN='your_token_here'

# Or disable diarization
chromaflow process video.mp4 --no-diarize
```

**"CUDA out of memory"**
```bash
# Use a smaller Whisper model
chromaflow process video.mp4 --whisper-model tiny

# Or force CPU
chromaflow process video.mp4 --device cpu
```

**Slow processing on Mac**
- Ensure you're using Apple Silicon (M1/M2/M3)
- MPS acceleration is auto-detected
- First run downloads models (~2GB)

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Links

- Documentation: https://chromaflow.ai/docs
- Issues: https://github.com/chromaflow/chromaflow/issues
- Cloud (coming soon): https://chromaflow.ai
