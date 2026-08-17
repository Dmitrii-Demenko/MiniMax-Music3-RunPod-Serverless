"""Probing and transcoding the WAV that sgl-omni returns.

The engine answers with 32 kHz 16-bit stereo WAV. A 360 second track is ~46 MB of
WAV, well past what a RunPod response can carry, so transcoding is what makes a
base64 reply possible at all and what keeps bucket objects small.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from dataclasses import dataclass, replace

from config import FRAMES_PER_SECOND

OPUS_SAMPLE_RATE = 48000


class AudioError(Exception):
    """The audio could not be read or converted."""


@dataclass(frozen=True)
class AudioInfo:
    duration_s: float
    sample_rate: int
    channels: int
    frames: int


def ffmpeg_exe() -> str:
    """Path to ffmpeg.

    imageio-ffmpeg ships its own binary and is already a dependency of the engine,
    so the container never needs a system ffmpeg. A system binary is accepted as a
    fallback so the test suite runs on a developer machine without the wheel.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise AudioError(
            "no ffmpeg available: install imageio-ffmpeg or provide ffmpeg on PATH"
        )


def probe_wav(data: bytes) -> AudioInfo:
    """Read duration, rate and channel count straight from the WAV header."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            duration_s = handle.getnframes() / float(sample_rate)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError(f"not a readable WAV stream: {exc}") from exc
    return AudioInfo(
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        frames=round(duration_s * FRAMES_PER_SECOND),
    )


def _encoder_args(target_format: str, bitrate: str) -> tuple[list[str], int | None]:
    """ffmpeg arguments and the sample rate the encoder forces, if any."""
    if target_format == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3"], None
    if target_format == "flac":
        return ["-c:a", "flac", "-f", "flac"], None
    if target_format == "opus":
        # Opus only encodes at 48 kHz, so the rate we report has to change too.
        return (
            ["-c:a", "libopus", "-b:a", bitrate, "-ar", str(OPUS_SAMPLE_RATE), "-f", "ogg"],
            OPUS_SAMPLE_RATE,
        )
    raise AudioError(f"unsupported target format {target_format!r}")


def transcode(data: bytes, target_format: str, bitrate: str) -> tuple[bytes, AudioInfo]:
    """Convert WAV bytes to `target_format`, returning the bytes and their info."""
    info = probe_wav(data)
    if target_format == "wav":
        return data, info

    args, forced_rate = _encoder_args(target_format, bitrate)
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        *args,
        "pipe:1",
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", "replace").strip()[:500]
        raise AudioError(f"ffmpeg failed to produce {target_format}: {stderr}")

    if forced_rate is not None:
        info = replace(info, sample_rate=forced_rate)
    return result.stdout, info
