import pytest

from config import Settings
from server import EngineUnavailable, SglOmniServer

MODEL = "/weights/minimax"


class FakeProcess:
    """Minimal stand-in for subprocess.Popen.

    `exit_after` is the number of poll() calls that report a live process before it
    reports an exit code.
    """

    def __init__(self, exit_after=None):
        self._polls = 0
        self._exit_after = exit_after
        self.stdout = None
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        self._polls += 1
        if self._exit_after is not None and self._polls > self._exit_after:
            self.returncode = 1
            return 1
        return None

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def build(settings_env=None, process=None, responses=None):
    settings = Settings.from_env(settings_env or {})
    process = process or FakeProcess()
    calls = iter(responses or [200])

    def fake_get(url):
        try:
            outcome = next(calls)
        except StopIteration:
            outcome = 200
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    server = SglOmniServer(
        settings,
        MODEL,
        popen=lambda *args, **kwargs: process,
        http_get=fake_get,
        sleep=lambda _seconds: None,
    )
    return server, process


def test_command_serves_the_resolved_model_on_localhost():
    server, _ = build()
    assert server.command() == [
        "sgl-omni",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_extra_args_are_appended():
    server, _ = build({"SGL_EXTRA_ARGS": "--max-running-requests 32"})
    assert server.command()[-2:] == ["--max-running-requests", "32"]


def test_two_gpus_produce_the_dual_device_placement():
    server, _ = build({"GPU_COUNT": "2"})
    assert server.env()["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_one_gpu_collocates_both_stages():
    server, _ = build({"GPU_COUNT": "1"})
    assert server.env()["CUDA_VISIBLE_DEVICES"] == "0"


def test_hf_home_is_passed_to_the_engine():
    server, _ = build({"HF_HOME": "/runpod-volume/huggingface-cache"})
    assert server.env()["HF_HOME"] == "/runpod-volume/huggingface-cache"


def test_wait_ready_returns_once_the_engine_answers():
    server, _ = build(responses=[ConnectionError(), ConnectionError(), 200])
    server.start()
    server.wait_ready()  # must not raise


def test_a_dead_process_fails_immediately_instead_of_waiting_out_the_timeout():
    server, _ = build(
        {"SERVER_STARTUP_TIMEOUT_S": "600"},
        process=FakeProcess(exit_after=0),
        responses=[ConnectionError()],
    )
    server.start()
    with pytest.raises(EngineUnavailable, match="exited"):
        server.wait_ready()


def test_readiness_timeout_raises_and_stops_the_process():
    server, process = build(
        {"SERVER_STARTUP_TIMEOUT_S": "0"}, responses=[ConnectionError()] * 50
    )
    server.start()
    with pytest.raises(EngineUnavailable, match="did not become ready"):
        server.wait_ready()
    assert process.terminated


def test_wait_ready_without_start_is_an_error():
    server, _ = build()
    with pytest.raises(EngineUnavailable, match="start"):
        server.wait_ready()


def test_is_alive_reflects_the_process_state():
    server, _ = build(process=FakeProcess(exit_after=1))
    assert server.is_alive() is False  # not started yet
    server.start()
    assert server.is_alive() is True
    assert server.is_alive() is False  # the fake process exits after one poll


def test_stop_escalates_to_kill_when_terminate_is_ignored():
    class Stubborn(FakeProcess):
        def wait(self, timeout=None):
            if not self.killed:
                raise TimeoutError
            return -9

    server, process = build(process=Stubborn())
    server.start()
    server.stop(timeout=0.01)
    assert process.terminated and process.killed


def test_stop_before_start_is_a_no_op():
    server, process = build()
    server.stop()
    assert process.terminated is False
