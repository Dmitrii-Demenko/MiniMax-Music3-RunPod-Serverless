"""Locating the MiniMax-Music3 checkpoint.

sglang-omni's resolve_checkpoint() expects this layout under the model root:

    qwen_7B/qwen_7B/                    the 8B backbone shards (~18.5 GB)
    qwen_7B/qwen3-8B-tokenizer-music/   the music tokenizer
    flowmatching_vae.pth                the flow-matching DIT (~9.8 GB)
    dav.pth                             the DAC-style decoder (~0.5 GB)

That is ~28.8 GB of the 57 GB repository; the diffusers subfolders are not used by
this runtime. Where those files live is a deployment choice — a RunPod cached model,
a network volume, or a baked image — so we look in several places and log which one
won. This is the only reason the worker does not care about storage strategy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from logging_setup import log_event

LOG = logging.getLogger("worker.model_path")

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "qwen_7B/qwen_7B",
    "qwen_7B/qwen3-8B-tokenizer-music",
    "flowmatching_vae.pth",
    "dav.pth",
)


class ModelPathError(Exception):
    """No usable checkpoint could be found."""


def missing_artifacts(root: Path) -> list[str]:
    """Return the required artifacts that are absent or of the wrong kind."""
    missing: list[str] = []
    for relative in REQUIRED_ARTIFACTS:
        target = root / relative
        expect_dir = not relative.endswith(".pth")
        if expect_dir and not target.is_dir():
            missing.append(relative)
        elif not expect_dir and not target.is_file():
            missing.append(relative)
    return missing


def _snapshot_dirs(settings: Settings) -> list[Path]:
    """Cached-model snapshots for the configured repo, newest first."""
    org, _, name = settings.model_repo_id.partition("/")
    pattern = f"hub/models--{org}--{name}/snapshots/*"
    candidates = [path for path in Path(settings.hf_home).glob(pattern) if path.is_dir()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_model_path(settings: Settings) -> str:
    """Resolve the model root, preferring an explicit path over the HF cache."""
    if settings.model_path:
        root = Path(settings.model_path)
        missing = missing_artifacts(root)
        if missing:
            raise ModelPathError(
                f"MODEL_PATH {root} is missing required artifacts: {missing}"
            )
        log_event(LOG, logging.INFO, "model resolved", source="MODEL_PATH", path=str(root))
        return str(root)

    for snapshot in _snapshot_dirs(settings):
        missing = missing_artifacts(snapshot)
        if missing:
            log_event(
                LOG,
                logging.WARNING,
                "snapshot skipped",
                path=str(snapshot),
                missing=missing,
            )
            continue
        log_event(
            LOG, logging.INFO, "model resolved", source="cached_model", path=str(snapshot)
        )
        return str(snapshot)

    if settings.allow_hub_download:
        log_event(
            LOG,
            logging.WARNING,
            "model resolved",
            source="hub_download",
            repo_id=settings.model_repo_id,
            note="the engine will download the weights inside billed cold-start time",
        )
        return settings.model_repo_id

    raise ModelPathError(
        f"no usable checkpoint under {settings.hf_home}: attach a cached model or a "
        f"network volume, set MODEL_PATH, or set ALLOW_HUB_DOWNLOAD=1 to download "
        f"the weights at startup"
    )
