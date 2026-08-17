"""Structured JSON logging to stdout, which is where RunPod collects worker logs."""

from __future__ import annotations

import json
import logging
import sys
import time

_CONFIGURED = False


class StdoutHandler(logging.StreamHandler):
    """A stdout handler that resolves the stream at emit time.

    logging.StreamHandler captures whatever sys.stdout was when it was constructed.
    Since the handler is installed once per process, anything that replaces
    sys.stdout afterwards would stop seeing our logs.
    """

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:
        # StreamHandler.__init__ assigns to this; the stream is always sys.stdout.
        return


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str) -> None:
    """Install the JSON handler on the root logger exactly once."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return
    handler = StdoutHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def log_event(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    """Log `message` with structured `fields` attached.

    Lyrics and prompts are user content, so callers keep them out of INFO records.
    """
    logger.log(level, message, extra={"extra_fields": fields})
