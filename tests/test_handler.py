import asyncio
import io
import json
import wave

import httpx
import pytest

import handler
from config import Settings


def make_wav(seconds=0.4):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(b"\x00\x01" * 2 * int(32000 * seconds))
    return buffer.getvalue()


class FakeEngine:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def build_runtime(responder, settings=None, engine=None, progress=None):
    settings = settings or Settings.from_env({})
    client = httpx.AsyncClient(
        base_url=settings.base_url, transport=httpx.MockTransport(responder)
    )
    return handler.Runtime(
        settings=settings,
        client=client,
        engine=engine or FakeEngine(),
        progress=progress or (lambda job, message: None),
    )


JOB = {
    "id": "job-1",
    "input": {"lyrics": "[Verse]\nline one", "prompt": "a lo-fi track at 85 BPM"},
}


def wav_responder(request):
    return httpx.Response(200, content=make_wav())


async def test_successful_job_returns_base64_audio_and_metrics():
    result = await handler.run_job(JOB, build_runtime(wav_responder))
    assert "audio_base64" in result
    assert result["format"] == "mp3"
    assert result["sample_rate"] == 32000
    assert result["channels"] == 2
    assert result["seed"] == 0
    assert result["duration_s"] == pytest.approx(0.4, abs=0.05)
    assert set(result["metrics"]) == {
        "validate_ms",
        "generate_ms",
        "encode_ms",
        "deliver_ms",
    }


async def test_request_payload_matches_the_sgl_omni_contract():
    seen = {}

    def responder(request):
        seen.update(json.loads(request.content))
        assert request.url.path == "/v1/audio/speech"
        return httpx.Response(200, content=make_wav())

    job = {"id": "job-2", "input": {**JOB["input"], "duration": 12, "seed": 42}}
    await handler.run_job(job, build_runtime(responder))
    assert seen["input"] == "[Verse]\nline one"
    assert seen["instructions"] == "a lo-fi track at 85 BPM"
    assert seen["max_new_tokens"] == 300
    assert seen["seed"] == 42
    assert seen["response_format"] == "wav"
    assert seen["stream"] is False


async def test_validation_error_is_returned_without_calling_the_engine():
    def responder(request):  # pragma: no cover - must not be reached
        raise AssertionError("the engine must not be called for invalid input")

    result = await handler.run_job({"id": "job-3", "input": {}}, build_runtime(responder))
    assert result["code"] == "invalid_request"
    assert "refresh_worker" not in result


async def test_unsupported_parameter_is_reported_with_its_code():
    def responder(request):  # pragma: no cover
        raise AssertionError("the engine must not be called")

    job = {"id": "job-4", "input": {**JOB["input"], "temperature": 0.7}}
    result = await handler.run_job(job, build_runtime(responder))
    assert result["code"] == "unsupported_parameter"


async def test_upstream_4xx_is_passed_through_without_a_worker_refresh():
    def responder(request):
        return httpx.Response(400, text="speed is not supported")

    result = await handler.run_job(JOB, build_runtime(responder))
    assert result["code"] == "upstream_rejected"
    assert "speed" in result["error"]
    assert "refresh_worker" not in result


async def test_upstream_5xx_refreshes_the_worker():
    def responder(request):
        return httpx.Response(500, text="CUDA out of memory")

    result = await handler.run_job(JOB, build_runtime(responder))
    assert result["code"] == "generation_failed"
    assert result["refresh_worker"] is True


async def test_a_dead_engine_refreshes_the_worker():
    def responder(request):  # pragma: no cover
        raise AssertionError("the engine is dead and must not be called")

    runtime = build_runtime(responder, engine=FakeEngine(alive=False))
    result = await handler.run_job(JOB, runtime)
    assert result["code"] == "engine_unavailable"
    assert result["refresh_worker"] is True


async def test_transport_failure_refreshes_the_worker():
    def responder(request):
        raise httpx.ConnectError("connection refused")

    result = await handler.run_job(JOB, build_runtime(responder))
    assert result["code"] == "engine_unavailable"
    assert result["refresh_worker"] is True


async def test_timeout_is_reported_as_a_timeout():
    def responder(request):
        raise httpx.ReadTimeout("too slow")

    result = await handler.run_job(JOB, build_runtime(responder))
    assert result["code"] == "timeout"
    assert "refresh_worker" not in result


async def test_oversized_result_is_an_error_not_a_truncated_file():
    settings = Settings.from_env(
        {"BASE64_MAX_ENCODED_BYTES": "32", "DEFAULT_FORMAT": "wav"}
    )

    def responder(request):
        return httpx.Response(200, content=make_wav(seconds=1.0))

    result = await handler.run_job(JOB, build_runtime(responder, settings=settings))
    assert result["code"] == "result_too_large"


async def test_non_wav_engine_response_is_an_encoding_error():
    def responder(request):
        return httpx.Response(200, content=b"not audio at all")

    result = await handler.run_job(JOB, build_runtime(responder))
    assert result["code"] == "encoding_failed"


async def test_lyrics_warnings_reach_the_response():
    job = {"id": "job-5", "input": {**JOB["input"], "lyrics": "[Verse] line one"}}
    result = await handler.run_job(job, build_runtime(wav_responder))
    assert len(result["warnings"]) == 1


async def test_progress_updates_are_emitted_for_each_stage():
    stages = []
    runtime = build_runtime(
        wav_responder, progress=lambda job, message: stages.append(message)
    )
    await handler.run_job(JOB, runtime)
    assert stages == ["generating", "encoding", "delivering"]


async def test_cancellation_propagates():
    def responder(request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await handler.run_job(JOB, build_runtime(responder))


async def test_bucket_delivery_returns_a_url(monkeypatch):
    import delivery

    monkeypatch.setattr(
        delivery, "upload_to_bucket", lambda path: "https://cdn.example.com/job-1.mp3"
    )
    settings = Settings.from_env(
        {
            "BUCKET_ENDPOINT_URL": "https://example.com",
            "BUCKET_ACCESS_KEY_ID": "key",
            "BUCKET_SECRET_ACCESS_KEY": "secret",
        }
    )
    result = await handler.run_job(JOB, build_runtime(wav_responder, settings=settings))
    assert result["audio_url"] == "https://cdn.example.com/job-1.mp3"
    assert "audio_base64" not in result


def test_concurrency_modifier_reports_the_configured_limit(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENCY", "4")
    assert handler.concurrency_modifier(0) == 4


def test_build_payload_uses_the_engine_field_names():
    from request_schema import GenerationRequest

    request = GenerationRequest(
        lyrics="[Verse]\nline",
        prompt="caption",
        max_new_tokens=750,
        seed=7,
        format="mp3",
        bitrate="192k",
        warnings=(),
    )
    payload = handler.build_payload(request, "MiniMaxAI/MiniMax-Music3")
    assert payload == {
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Verse]\nline",
        "instructions": "caption",
        "seed": 7,
        "max_new_tokens": 750,
        "response_format": "wav",
        "stream": False,
    }
