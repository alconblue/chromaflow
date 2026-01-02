"""Command-line interface for ChromaFlow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from chromaflow import Pipeline, __version__
from chromaflow.config import WhisperModel
from chromaflow.utils.hardware import get_device_info

app = typer.Typer(
    name="chromaflow",
    help="Turn video into RAG-ready artifacts in one line of code.",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"chromaflow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """ChromaFlow: Turn video into RAG-ready artifacts."""
    pass


@app.command()
def process(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to video or audio file (.mp4, .mov, .avi, .mkv, .mp3, .wav)",
            exists=True,
            readable=True,
        ),
    ],
    diarize: Annotated[
        bool,
        typer.Option(
            "--diarize/--no-diarize",
            help="Enable speaker diarization (requires HF_TOKEN)",
        ),
    ] = True,
    whisper_model: Annotated[
        WhisperModel,
        typer.Option(
            "--whisper-model",
            "-m",
            help="Whisper model size",
        ),
    ] = WhisperModel.SMALL,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output JSON file path (default: stdout)",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-d",
            help="Directory for extracted frames and artifacts",
        ),
    ] = Path("./chromaflow_output"),
    device: Annotated[
        Optional[str],
        typer.Option(
            "--device",
            help="Compute device: cuda, mps, or cpu (auto-detect if not specified)",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress progress output",
        ),
    ] = False,
) -> None:
    """Process a video or audio file into RAG-ready JSON.

    Example:
        chromaflow process meeting.mp4 --diarize -o output.json
    """
    import logging

    from chromaflow.utils.logging import get_logger

    # Configure logging
    log_level = logging.WARNING if quiet else logging.INFO
    logger = get_logger(level=log_level)

    try:
        # Initialize pipeline
        pipeline = Pipeline(
            mode="local",
            device=device,
            output_dir=str(output_dir),
            options={
                "diarize": diarize,
                "whisper_model": whisper_model,
            },
        )

        # Process the file
        video_data = pipeline.process(source)

        # Output results
        json_output = video_data.to_json(indent=2)

        if output:
            output.write_text(json_output)
            if not quiet:
                typer.echo(f"Output written to: {output}")
        else:
            typer.echo(json_output)

    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show system information and detected hardware."""
    typer.echo(f"ChromaFlow v{__version__}")
    typer.echo("")

    device_info = get_device_info()
    typer.echo("Hardware Detection:")
    typer.echo(f"  Device: {device_info.device}")
    typer.echo(f"  Name: {device_info.name}")

    if device_info.memory_gb:
        typer.echo(f"  Memory: {device_info.memory_gb} GB")
    if device_info.cuda_version:
        typer.echo(f"  CUDA: {device_info.cuda_version}")

    typer.echo("")
    typer.echo("Supported Formats:")
    typer.echo("  Video: .mp4, .mov, .avi, .mkv")
    typer.echo("  Audio: .mp3, .wav")


if __name__ == "__main__":
    app()
