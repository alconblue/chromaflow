# ChromaFlow Tests

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_audio.py        # Audio processing unit tests
├── test_chroma.py       # ChromaDB unit tests
├── test_ingest.py       # Ingest stage unit tests
├── test_pipeline.py     # Pipeline unit tests
├── test_scene.py        # Scene detection unit tests
├── test_visual.py       # Visual processing unit tests
└── integration/         # Integration tests (optional)
    ├── conftest.py      # Integration fixtures
    ├── test_frame_extraction.py
    ├── test_clip_embeddings.py
    ├── test_chromadb_integration.py
    ├── test_e2e_pipeline.py
    └── test_long_video.py
```

## Running Tests

```bash
# Run all unit tests (fast, no external dependencies)
./scripts/run_tests.sh

# Run with coverage
./scripts/run_tests.sh --cov

# Skip integration tests
./scripts/run_tests.sh -m "not integration"

# Run only integration tests
./scripts/run_tests.sh -m integration
```

## Unit Tests vs Integration Tests

### Unit Tests (Default)

- **Fast**: Complete in < 10 seconds
- **No external dependencies**: All ML models and I/O are mocked
- **Required for all PRs**: Must pass before merge
- **Location**: `tests/test_*.py`

### Integration Tests

- **Slow**: May take several minutes (model loading, real processing)
- **External dependencies required**:
  - FFmpeg installed and in PATH
  - ML models downloaded (~3GB: Whisper + pyannote + CLIP)
  - Test video/audio fixtures in `tests/fixtures/`
- **NOT required for PRs** unless touching processing stages
- **Location**: `tests/integration/`

## When Integration Tests Are Required

Run integration tests if your PR modifies:

- `src/chromaflow/stages/ingest.py`
- `src/chromaflow/stages/scene.py`
- `src/chromaflow/stages/audio.py`
- `src/chromaflow/stages/visual.py`
- `src/chromaflow/store/chroma.py`
- `src/chromaflow/pipeline.py`

## Setting Up Integration Tests

### 1. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

### 2. Download Test Fixtures

Test fixtures are not included in the repository due to size. Generate or download them:

```bash
# Create fixtures directory
mkdir -p tests/fixtures

# Generate silent video (test pattern)
ffmpeg -f lavfi -i "testsrc=duration=15:size=1280x720:rate=30" \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/silent_video.mp4

# For other fixtures, see docs/integration_test_plan_m3.md
```

### 3. Set HuggingFace Token (for diarization tests)

```bash
export HF_TOKEN='your_huggingface_token'
```

### 4. First Run (Model Downloads)

The first integration test run will download ML models (~3GB). This is expected and only happens once:

- Whisper small: ~500MB
- pyannote speaker-diarization: ~500MB
- CLIP ViT-L/14: ~2GB

## Test Markers

| Marker | Description |
|--------|-------------|
| `integration` | Requires real files and ML models |
| `slow` | Takes > 30 seconds (usually model loading) |
| `stress` | Memory-intensive, must run in isolation |

```bash
# Examples
./scripts/run_tests.sh -m "not integration"  # Skip integration
./scripts/run_tests.sh -m "not slow"         # Skip slow tests
./scripts/run_tests.sh -m "integration and not slow"  # Fast integration only
./scripts/run_tests.sh -m "not stress"       # Skip stress tests (recommended for CI)

# Run stress tests separately (required for long video tests)
poetry run pytest tests/integration/test_long_video.py -v
```

### Running Integration Tests Without Stress Tests

For faster CI runs, exclude stress tests which can cause OOM issues:

```bash
# Recommended for CI: all integration tests except stress tests
poetry run pytest tests/integration/ -m "not stress" -v
```

## CI/CD Configuration

```yaml
# Example GitHub Actions
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./scripts/run_tests.sh -m "not integration"

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || contains(github.event.pull_request.labels.*.name, 'run-integration')
    steps:
      - uses: actions/checkout@v4
      - run: ./scripts/run_tests.sh -m integration
```
