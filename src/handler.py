"""RunPod Serverless entry point for MiniMax-Music3.

Nothing starts at import time: bootstrap() owns the engine, so the module can be
imported by tests without a GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

import audio
import delivery
import model_path as model_path_module
import request_schema
import server as server_module
from config import Settings
from logging_setup import log_event, setup_logging

LOG = logging.getLogger("worker.handler")
SPEECH_PATH = "/v1/audio/speech"


@dataclass
class Runtime:
    settings: Settings
    client: httpx.AsyncClient
    engine: object
    progress: object


def _error(message: str, code: str, *, refresh: bool = False) -> dict:
    payload: dict = {"error": message, "code": code}
    if refresh:
        # Ask RunPod for a fresh worker: GPU state after an engine failure or an OOM
        # is not something we should reuse.
        payload["refresh_worker"] = True
    return payload


def build_payload(request: request_schema.GenerationRequest, model_name: str) -> dict:
    """The sgl-omni /v1/audio/speech body: lyrics in `input`, caption in `instructions`."""
    return {
        "model": model_name,
        "input": request.lyrics,
        "instructions": request.prompt,
        "seed": request.seed,
        "max_new_tokens": request.max_new_tokens,
        "response_format": "wav",
        "stream": False,
    }


async def run_job(job: dict, runtime: Runtime) -> dict:
    job_id = str(job.get("id", "unknown"))
    settings = runtime.settings
    metrics: dict[str, int] = {}

    started = time.monotonic()
    try:
        request = request_schema.parse(job.get("input"), settings)
    except request_schema.RequestError as exc:
        log_event(LOG, logging.WARNING, "rejected", job_id=job_id, code=exc.code)
        return _error(str(exc), exc.code)
    metrics["validate_ms"] = int((time.monotonic() - started) * 1000)

    if not runtime.engine.is_alive():
        return _error(
            "the engine process is not running", "engine_unavailable", refresh=True
        )

    runtime.progress(job, "generating")
    started = time.monotonic()
    try:
        response = await runtime.client.post(
            SPEECH_PATH,
            json=build_payload(request, settings.model_repo_id),
            timeout=settings.generation_timeout_s,
        )
    except asyncio.CancelledError:
        # The job was cancelled; let it propagate so the GPU is released instead of
        # finishing a render nobody is waiting for.
        log_event(LOG, logging.WARNING, "cancelled", job_id=job_id)
        raise
    except httpx.TimeoutException:
        log_event(LOG, logging.ERROR, "timeout", job_id=job_id)
        return _error(f"generation exceeded {settings.generation_timeout_s}s", "timeout")
    except httpx.HTTPError as exc:
        log_event(LOG, logging.ERROR, "engine unreachable", job_id=job_id, detail=str(exc))
        return _error(
            f"could not reach the engine: {exc}", "engine_unavailable", refresh=True
        )
    metrics["generate_ms"] = int((time.monotonic() - started) * 1000)

    if response.status_code >= 500:
        return _error(
            f"the engine failed: {response.text[:500]}", "generation_failed", refresh=True
        )
    if response.status_code >= 400:
        return _error(
            f"the engine rejected the request: {response.text[:500]}", "upstream_rejected"
        )

    runtime.progress(job, "encoding")
    started = time.monotonic()
    try:
        data, info = audio.transcode(response.content, request.format, request.bitrate)
    except audio.AudioError as exc:
        log_event(LOG, logging.ERROR, "encoding failed", job_id=job_id, detail=str(exc))
        return _error(str(exc), "encoding_failed")
    metrics["encode_ms"] = int((time.monotonic() - started) * 1000)

    runtime.progress(job, "delivering")
    started = time.monotonic()
    try:
        delivered = delivery.deliver(data, request.format, job_id, settings)
    except delivery.DeliveryError as exc:
        log_event(LOG, logging.ERROR, "delivery failed", job_id=job_id, code=exc.code)
        return _error(str(exc), exc.code)
    metrics["deliver_ms"] = int((time.monotonic() - started) * 1000)

    log_event(
        LOG,
        logging.INFO,
        "completed",
        job_id=job_id,
        duration_s=round(info.duration_s, 2),
        frames=info.frames,
        requested_frames=request.max_new_tokens,
        format=request.format,
        **metrics,
    )
    return {
        **delivered,
        "format": request.format,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        # May be shorter than requested: the model ends the song on its own token.
        "duration_s": round(info.duration_s, 3),
        "frames": info.frames,
        "seed": request.seed,
        "warnings": list(request.warnings),
        "metrics": metrics,
    }


def concurrency_modifier(current: int) -> int:
    """How many jobs this worker accepts at once.

    Raising this only pays off together with SGL_EXTRA_ARGS="--max-running-requests N":
    classifier-free guidance gives every request a second decode row with its own KV
    cache for the whole song, so the engine has to be sized for rows, not requests.
    """
    return Settings.from_env().max_concurrency


def _progress(job: dict, message: str) -> None:
    import runpod

    runpod.serverless.progress_update(job, message)


def bootstrap() -> Runtime:
    """Start the engine and build the runtime. Never called at import time."""
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    resolved = model_path_module.resolve_model_path(settings)
    engine = server_module.SglOmniServer(settings, resolved)
    engine.start()
    engine.wait_ready()
    client = httpx.AsyncClient(
        base_url=settings.base_url, timeout=settings.generation_timeout_s
    )
    log_event(
        LOG,
        logging.INFO,
        "worker ready",
        gpu_count=settings.gpu_count,
        max_concurrency=settings.max_concurrency,
        delivery="bucket" if settings.bucket_configured else "base64",
    )
    return Runtime(settings=settings, client=client, engine=engine, progress=_progress)


# The RunPod SDK wiring lives in rp_handler.py at the repository root, because
# RunPod's GitHub integration looks for runpod.serverless.start() there.
