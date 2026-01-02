"""Ingestion stage: file validation and metadata extraction.

This stage handles:
- File format validation
- Metadata extraction via ffprobe
- Audio track extraction for downstream processing
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from chromaflow.models.schema import VideoMetadata

logger = logging.getLogger(__name__)

# Supported formats
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav"}
SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS


@dataclass
class ProbeResult:
    """Raw probe result from ffprobe."""

    duration: float
    width: int | None
    height: int | None
    fps: float | None
    format_name: str
    has_video: bool
    has_audio: bool
    audio_channels: int
    audio_sample_rate: int | None
    codec_name: str | None


class IngestError(Exception):
    """Error during file ingestion."""

    pass


def validate_file(source: Path) -> None:
    """Validate that a file exists and has a supported format.

    Args:
        source: Path to the input file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )


def _run_ffprobe(source: Path) -> dict:
    """Run ffprobe and return parsed JSON output.

    Args:
        source: Path to the media file.

    Returns:
        Parsed JSON from ffprobe.

    Raises:
        IngestError: If ffprobe fails or is not installed.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise IngestError(
            "ffprobe not found. Please install FFmpeg:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    except subprocess.TimeoutExpired:
        raise IngestError(f"ffprobe timed out reading: {source}")

    if result.returncode != 0:
        raise IngestError(f"ffprobe failed: {result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise IngestError(f"Failed to parse ffprobe output: {e}")


def _parse_probe_result(data: dict) -> ProbeResult:
    """Parse ffprobe JSON into a ProbeResult.

    Args:
        data: Parsed ffprobe JSON.

    Returns:
        ProbeResult with extracted metadata.
    """
    streams = data.get("streams", [])
    format_info = data.get("format", {})

    # Find video and audio streams
    video_stream = None
    audio_stream = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    # Extract duration (prefer format duration, fall back to stream)
    duration = 0.0
    if "duration" in format_info:
        duration = float(format_info["duration"])
    elif video_stream and "duration" in video_stream:
        duration = float(video_stream["duration"])
    elif audio_stream and "duration" in audio_stream:
        duration = float(audio_stream["duration"])

    # Extract video properties
    width = None
    height = None
    fps = None
    codec_name = None

    if video_stream:
        width = video_stream.get("width")
        height = video_stream.get("height")
        codec_name = video_stream.get("codec_name")

        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = video_stream.get("r_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(fps_str)

    # Extract audio properties
    audio_channels = 0
    audio_sample_rate = None

    if audio_stream:
        audio_channels = audio_stream.get("channels", 0)
        sample_rate = audio_stream.get("sample_rate")
        if sample_rate:
            audio_sample_rate = int(sample_rate)

    # Format name
    format_name = format_info.get("format_name", "unknown")
    # Take first format if multiple (e.g., "mov,mp4,m4a,3gp")
    if "," in format_name:
        format_name = format_name.split(",")[0]

    return ProbeResult(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        format_name=format_name,
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
        codec_name=codec_name,
    )


def probe_file(source: Path) -> ProbeResult:
    """Probe a media file and return metadata.

    Args:
        source: Path to the media file.

    Returns:
        ProbeResult with file metadata.

    Raises:
        IngestError: If probing fails.
    """
    start_time = time.perf_counter()

    data = _run_ffprobe(source)
    result = _parse_probe_result(data)

    elapsed = time.perf_counter() - start_time
    logger.debug(f"Probed {source.name} in {elapsed:.2f}s")

    return result


def extract_metadata(source: Path) -> VideoMetadata:
    """Extract technical metadata from a video/audio file.

    Args:
        source: Path to the input file.

    Returns:
        VideoMetadata with duration, resolution, fps, etc.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        IngestError: If metadata extraction fails.
    """
    validate_file(source)

    probe = probe_file(source)

    # Build resolution string
    if probe.width and probe.height:
        resolution = f"{probe.width}x{probe.height}"
    else:
        resolution = "0x0"  # Audio-only files

    # Default FPS for audio-only
    fps = probe.fps if probe.fps and probe.fps > 0 else 1.0

    return VideoMetadata(
        duration=probe.duration,
        resolution=resolution,
        fps=fps,
        format=probe.format_name,
        audio_channels=probe.audio_channels,
        sample_rate=probe.audio_sample_rate,
    )


def extract_audio(
    source: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio track from video file as WAV.

    Extracts audio at 16kHz mono by default, which is optimal
    for speech recognition models like Whisper.

    Args:
        source: Path to the input video/audio file.
        output_dir: Directory to write the extracted audio.
        sample_rate: Output sample rate in Hz (default: 16000).
        mono: Convert to mono if True (default: True).

    Returns:
        Path to the extracted WAV file.

    Raises:
        IngestError: If audio extraction fails or no audio track exists.
    """
    validate_file(source)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_path = output_dir / f"{source.stem}_audio.wav"

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(source),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),  # Sample rate
    ]

    if mono:
        cmd.extend(["-ac", "1"])  # Mono

    cmd.extend([
        "-y",  # Overwrite output
        str(output_path),
    ])

    logger.info(f"Extracting audio from {source.name} -> {output_path.name}")
    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
    except FileNotFoundError:
        raise IngestError(
            "ffmpeg not found. Please install FFmpeg:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    except subprocess.TimeoutExpired:
        raise IngestError(f"Audio extraction timed out for: {source}")

    if result.returncode != 0:
        # Check if it's a "no audio" error
        if "does not contain any stream" in result.stderr.lower():
            raise IngestError(f"No audio track found in: {source}")
        raise IngestError(f"Audio extraction failed: {result.stderr}")

    if not output_path.exists():
        raise IngestError(f"Audio extraction produced no output: {output_path}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Audio extracted in {elapsed:.2f}s: {output_path}")

    return output_path


def is_audio_only(source: Path) -> bool:
    """Check if a file is audio-only (no video track).

    Args:
        source: Path to the media file.

    Returns:
        True if the file has no video track.
    """
    suffix = source.suffix.lower()
    if suffix in SUPPORTED_AUDIO_FORMATS:
        return True

    # For video containers, check if they actually have video
    try:
        probe = probe_file(source)
        return not probe.has_video
    except IngestError:
        # If probe fails, assume it's not audio-only
        return False
