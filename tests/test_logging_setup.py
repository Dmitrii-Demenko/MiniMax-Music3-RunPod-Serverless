import json
import logging

from logging_setup import log_event, setup_logging


def test_records_are_emitted_as_single_line_json(capsys):
    setup_logging("INFO")
    logger = logging.getLogger("test.worker")
    log_event(logger, logging.INFO, "generated", job_id="abc", generate_ms=41230)
    captured = capsys.readouterr().out.strip()
    assert "\n" not in captured
    payload = json.loads(captured)
    assert payload["message"] == "generated"
    assert payload["level"] == "INFO"
    assert payload["job_id"] == "abc"
    assert payload["generate_ms"] == 41230


def test_setup_logging_is_idempotent(capsys):
    setup_logging("INFO")
    setup_logging("INFO")
    logging.getLogger("test.worker").info("once")
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_exceptions_are_included(capsys):
    setup_logging("INFO")
    logger = logging.getLogger("test.worker")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    payload = json.loads(capsys.readouterr().out.strip())
    assert "ValueError: boom" in payload["exception"]
