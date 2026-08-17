"""Getting the finished audio back to the caller.

RunPod caps /run at 10 MB and /runsync at 20 MB, and a 360 second WAV is ~46 MB, so
a bucket is the real delivery path and base64 only ever works for short clips. When
the result does not fit we fail loudly: a truncated audio file is worse than an error.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile

from config import Settings
from logging_setup import log_event

LOG = logging.getLogger("worker.delivery")


class DeliveryError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def upload_to_bucket(file_path: str) -> str:
    """Upload `file_path` to the configured bucket and return its URL.

    runpod's rp_upload reads BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID and
    BUCKET_SECRET_ACCESS_KEY from the environment itself, which is why Settings only
    needs them to decide *whether* a bucket is configured. Verify the signature
    against the pinned runpod version before changing this call.
    """
    from runpod.serverless.utils import rp_upload

    return rp_upload.upload_file_to_bucket(os.path.basename(file_path), file_path)


def deliver(
    data: bytes, target_format: str, job_id: str, settings: Settings
) -> dict[str, str]:
    """Return either {"audio_url": ...} or {"audio_base64": ...}."""
    if settings.bucket_configured:
        directory = tempfile.mkdtemp(prefix="mm3-")
        file_path = os.path.join(directory, f"{job_id}.{target_format}")
        try:
            with open(file_path, "wb") as handle:
                handle.write(data)
            url = upload_to_bucket(file_path)
        except Exception as exc:
            raise DeliveryError(
                f"could not upload the result: {exc}", code="delivery_failed"
            ) from exc
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.isdir(directory):
                os.rmdir(directory)
        log_event(LOG, logging.INFO, "delivered", job_id=job_id, via="bucket", bytes=len(data))
        return {"audio_url": url}

    encoded = base64.b64encode(data)
    if len(encoded) > settings.base64_max_encoded_bytes:
        raise DeliveryError(
            f"the result is {len(encoded)} bytes base64-encoded, above the "
            f"{settings.base64_max_encoded_bytes} byte limit. Configure "
            f"BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID and BUCKET_SECRET_ACCESS_KEY "
            f"to return a URL instead, or request a shorter duration or a lower bitrate.",
            code="result_too_large",
        )
    log_event(LOG, logging.INFO, "delivered", job_id=job_id, via="base64", bytes=len(data))
    return {"audio_base64": encoded.decode("ascii")}
