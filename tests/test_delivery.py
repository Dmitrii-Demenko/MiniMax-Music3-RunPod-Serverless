import base64
import os

import pytest

import delivery
from config import Settings

BUCKET_ENV = {
    "BUCKET_ENDPOINT_URL": "https://example.com",
    "BUCKET_ACCESS_KEY_ID": "key",
    "BUCKET_SECRET_ACCESS_KEY": "secret",
}


def test_without_a_bucket_the_audio_comes_back_as_base64():
    result = delivery.deliver(b"audio-bytes", "mp3", "job-1", Settings.from_env({}))
    assert base64.b64decode(result["audio_base64"]) == b"audio-bytes"
    assert "audio_url" not in result


def test_oversized_base64_is_an_error_not_a_truncated_file():
    settings = Settings.from_env({"BASE64_MAX_ENCODED_BYTES": "16"})
    with pytest.raises(delivery.DeliveryError) as excinfo:
        delivery.deliver(b"x" * 1024, "mp3", "job-1", settings)
    assert excinfo.value.code == "result_too_large"
    assert "BUCKET_ENDPOINT_URL" in str(excinfo.value)


def test_with_a_bucket_the_audio_is_uploaded_and_a_url_returned(monkeypatch):
    seen = {}

    def fake_upload(file_path):
        seen["name"] = os.path.basename(file_path)
        with open(file_path, "rb") as handle:
            seen["content"] = handle.read()
        return "https://cdn.example.com/job-1.mp3"

    monkeypatch.setattr(delivery, "upload_to_bucket", fake_upload)
    result = delivery.deliver(b"audio-bytes", "mp3", "job-1", Settings.from_env(BUCKET_ENV))
    assert result == {"audio_url": "https://cdn.example.com/job-1.mp3"}
    assert seen["name"] == "job-1.mp3"
    assert seen["content"] == b"audio-bytes"


def test_upload_failure_is_reported_as_delivery_failed(monkeypatch):
    def fake_upload(file_path):
        raise RuntimeError("bucket unreachable")

    monkeypatch.setattr(delivery, "upload_to_bucket", fake_upload)
    with pytest.raises(delivery.DeliveryError) as excinfo:
        delivery.deliver(b"audio", "mp3", "job-1", Settings.from_env(BUCKET_ENV))
    assert excinfo.value.code == "delivery_failed"


def test_temporary_file_is_removed_after_upload(monkeypatch):
    paths = []

    def fake_upload(file_path):
        paths.append(file_path)
        return "https://cdn.example.com/job-1.mp3"

    monkeypatch.setattr(delivery, "upload_to_bucket", fake_upload)
    delivery.deliver(b"audio", "mp3", "job-1", Settings.from_env(BUCKET_ENV))
    assert not os.path.exists(paths[0])
    assert not os.path.isdir(os.path.dirname(paths[0]))


def test_temporary_file_is_removed_when_the_upload_fails(monkeypatch):
    paths = []

    def fake_upload(file_path):
        paths.append(file_path)
        raise RuntimeError("nope")

    monkeypatch.setattr(delivery, "upload_to_bucket", fake_upload)
    with pytest.raises(delivery.DeliveryError):
        delivery.deliver(b"audio", "mp3", "job-1", Settings.from_env(BUCKET_ENV))
    assert not os.path.exists(paths[0])
