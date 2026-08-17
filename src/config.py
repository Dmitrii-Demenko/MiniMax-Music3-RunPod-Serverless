"""Worker configuration, read from the environment.

Every default here is either a model limit (see the constants) or a value the
spec justifies. Nothing else in the worker reads os.environ directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL_REPO_ID = "MiniMaxAI/MiniMax-Music3"

# Model limits, not preferences: the backbone emits 25 audio frames per second
# and refuses more than 9000 frames, which is 360 seconds of audio.
FRAMES_PER_SECOND = 25
MAX_FRAMES = 9000

SUPPORTED_FORMATS = ("wav", "mp3", "opus", "flac")


class ConfigError(Exception):
    """The environment cannot produce a usable configuration."""


def _get(env: Mapping[str, str], name: str, default: str | None) -> str | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name, "1" if default else "0") or ""
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: str | None
    model_repo_id: str
    allow_hub_download: bool
    hf_home: str
    gpu_count: int
    sgl_host: str
    sgl_port: int
    sgl_extra_args: tuple[str, ...]
    server_startup_timeout_s: float
    generation_timeout_s: float
    max_concurrency: int
    max_duration_s: float
    default_format: str
    default_bitrate: str
    base64_max_encoded_bytes: int
    bucket_endpoint_url: str | None
    bucket_access_key_id: str | None
    bucket_secret_access_key: str | None
    log_level: str

    @property
    def base_url(self) -> str:
        return f"http://{self.sgl_host}:{self.sgl_port}"

    @property
    def bucket_configured(self) -> bool:
        return all(
            (
                self.bucket_endpoint_url,
                self.bucket_access_key_id,
                self.bucket_secret_access_key,
            )
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        gpu_count = _get_int(env, "GPU_COUNT", 2)
        if gpu_count < 1:
            raise ConfigError(f"GPU_COUNT must be at least 1, got {gpu_count}")

        max_duration_s = _get_float(env, "MAX_DURATION_S", 360.0)
        hard_limit = MAX_FRAMES / FRAMES_PER_SECOND
        if not 0 < max_duration_s <= hard_limit:
            raise ConfigError(
                f"MAX_DURATION_S must be in (0, {hard_limit}], got {max_duration_s}"
            )

        default_format = (_get(env, "DEFAULT_FORMAT", "mp3") or "mp3").lower()
        if default_format not in SUPPORTED_FORMATS:
            raise ConfigError(
                f"DEFAULT_FORMAT must be one of {list(SUPPORTED_FORMATS)}, "
                f"got {default_format!r}"
            )

        extra_args = _get(env, "SGL_EXTRA_ARGS", "") or ""

        return cls(
            model_path=_get(env, "MODEL_PATH", None),
            model_repo_id=_get(env, "MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID),
            allow_hub_download=_get_bool(env, "ALLOW_HUB_DOWNLOAD", False),
            hf_home=_get(env, "HF_HOME", "/runpod-volume/huggingface-cache"),
            gpu_count=gpu_count,
            sgl_host=_get(env, "SGL_HOST", "127.0.0.1"),
            sgl_port=_get_int(env, "SGL_PORT", 8000),
            sgl_extra_args=tuple(extra_args.split()),
            server_startup_timeout_s=_get_float(env, "SERVER_STARTUP_TIMEOUT_S", 1200.0),
            generation_timeout_s=_get_float(env, "GENERATION_TIMEOUT_S", 1500.0),
            max_concurrency=_get_int(env, "MAX_CONCURRENCY", 1),
            max_duration_s=max_duration_s,
            default_format=default_format,
            default_bitrate=_get(env, "DEFAULT_BITRATE", "192k"),
            base64_max_encoded_bytes=_get_int(env, "BASE64_MAX_ENCODED_BYTES", 9_500_000),
            bucket_endpoint_url=_get(env, "BUCKET_ENDPOINT_URL", None),
            bucket_access_key_id=_get(env, "BUCKET_ACCESS_KEY_ID", None),
            bucket_secret_access_key=_get(env, "BUCKET_SECRET_ACCESS_KEY", None),
            log_level=(_get(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
        )
