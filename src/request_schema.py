"""Validation and normalisation of the worker's job input.

The engine refuses unsupported parameters rather than ignoring them, and we do the
same one layer earlier so the caller sees the mistake without paying for a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import lyrics as lyrics_module
from config import FRAMES_PER_SECOND, MAX_FRAMES, SUPPORTED_FORMATS, Settings

DEFAULT_DURATION_S = 30.0
LOSSY_FORMATS = ("mp3", "opus")
_BITRATE = re.compile(r"^\d+k$")

# Sampling in this model is fixed (guidance 1.5 then top-k 50 with a seeded draw),
# there is no speaker to select, and tempo lives in the caption. Accepting these
# silently would mean quietly returning something other than what was asked for.
REJECTED_PARAMETERS: dict[str, str] = {
    "temperature": "sampling is fixed for this model (guidance 1.5, then top-k 50)",
    "top_p": "sampling is fixed for this model",
    "top_k": "sampling is fixed for this model",
    "repetition_penalty": "sampling is fixed for this model",
    "voice": "there is no speaker to select; describe the vocal in 'prompt'",
    "ref_audio": "reference-audio conditioning is not part of this contract",
    "ref_text": "reference-audio conditioning is not part of this contract",
    "language": "there is no language tag in this contract",
    "task_type": "there is no task selector in this contract",
    "speed": "tempo belongs in 'prompt', for example 'at 92 BPM'",
}

# Aliases accepted so a client written against the reference /v1/audio/speech API
# works unchanged.
_ALIASES = {"lyrics": "input", "prompt": "instructions"}


class RequestError(Exception):
    """A client mistake. Retrying the same request will not help."""

    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GenerationRequest:
    lyrics: str
    prompt: str
    max_new_tokens: int
    seed: int
    format: str
    bitrate: str
    warnings: tuple[str, ...]


def _required_text(payload: dict, canonical: str) -> str:
    alias = _ALIASES[canonical]
    if canonical in payload and alias in payload:
        if payload[canonical] != payload[alias]:
            raise RequestError(
                f"'{canonical}' and its alias '{alias}' were both given with "
                f"different values; send only one"
            )
    value = payload.get(canonical, payload.get(alias))
    if not isinstance(value, str) or value.strip() == "":
        raise RequestError(f"'{canonical}' is required and must be a non-empty string")
    return value


def _frames(payload: dict, settings: Settings) -> int:
    has_duration = payload.get("duration") is not None
    has_tokens = payload.get("max_new_tokens") is not None
    if has_duration and has_tokens:
        raise RequestError("'duration' and 'max_new_tokens' are mutually exclusive")

    if has_tokens:
        tokens = payload["max_new_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise RequestError("'max_new_tokens' must be an integer")
        if not 1 <= tokens <= MAX_FRAMES:
            raise RequestError(f"'max_new_tokens' must be in 1..{MAX_FRAMES}")
        return tokens

    duration = payload["duration"] if has_duration else DEFAULT_DURATION_S
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise RequestError("'duration' must be a number of seconds")
    if not 0 < duration <= settings.max_duration_s:
        raise RequestError(f"'duration' must be in (0, {settings.max_duration_s}] seconds")
    return max(1, round(duration * FRAMES_PER_SECOND))


def parse(job_input: object, settings: Settings) -> GenerationRequest:
    """Turn raw job input into a validated request, or raise RequestError."""
    if not isinstance(job_input, dict):
        raise RequestError("input must be a JSON object")

    for name, reason in REJECTED_PARAMETERS.items():
        if job_input.get(name) is not None:
            raise RequestError(
                f"'{name}' is not supported: {reason}", code="unsupported_parameter"
            )
    if job_input.get("stream"):
        raise RequestError(
            "'stream' must be false: this model's API is non-streaming",
            code="unsupported_parameter",
        )

    raw_lyrics = _required_text(job_input, "lyrics")
    prompt = _required_text(job_input, "prompt").strip()
    normalised_lyrics, warnings = lyrics_module.normalize(raw_lyrics)
    if normalised_lyrics.strip() == "":
        raise RequestError("'lyrics' is required and must be a non-empty string")

    seed = job_input.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RequestError("'seed' must be an integer")
    if not 0 <= seed < 2**64:
        raise RequestError("'seed' must be a non-negative 64-bit integer")

    target_format = str(job_input.get("format") or settings.default_format).lower()
    if target_format not in SUPPORTED_FORMATS:
        raise RequestError(f"'format' must be one of {list(SUPPORTED_FORMATS)}")

    bitrate = str(job_input.get("bitrate") or settings.default_bitrate)
    if not _BITRATE.match(bitrate):
        raise RequestError("'bitrate' must look like '192k'")
    if target_format not in LOSSY_FORMATS and job_input.get("bitrate") is not None:
        warnings.append(f"bitrate is ignored for the lossless format {target_format}")

    return GenerationRequest(
        lyrics=normalised_lyrics,
        prompt=prompt,
        max_new_tokens=_frames(job_input, settings),
        seed=seed,
        format=target_format,
        bitrate=bitrate,
        warnings=tuple(warnings),
    )
