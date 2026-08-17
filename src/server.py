"""Lifecycle of the sgl-omni child process.

The worker owns exactly one engine process, on localhost. If it cannot start, the
worker must fail before accepting jobs: a worker that takes a job it cannot run is
worse than one that never started.

Devices are assigned by CUDA_VISIBLE_DEVICES. With two GPUs sgl-omni puts the
autoregressive backbone on device 0 and the flow-matching DIT plus DAV decoder on
device 1; with one GPU it collocates both stages.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

from config import Settings
from logging_setup import log_event

LOG = logging.getLogger("worker.engine")
POLL_INTERVAL_S = 2.0


class EngineUnavailable(Exception):
    """The engine process is not running or never became ready."""


def _default_http_get(url: str) -> int:
    import httpx

    try:
        return httpx.get(url, timeout=5.0).status_code
    except httpx.HTTPError as exc:
        raise ConnectionError(str(exc)) from exc


class SglOmniServer:
    def __init__(
        self,
        settings: Settings,
        model_path: str,
        *,
        popen=subprocess.Popen,
        http_get=None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self._settings = settings
        self._model_path = model_path
        self._popen = popen
        self._http_get = http_get or _default_http_get
        self._sleep = sleep
        self._monotonic = monotonic
        self._process = None

    def command(self) -> list[str]:
        return [
            "sgl-omni",
            "serve",
            "--model-path",
            self._model_path,
            "--host",
            self._settings.sgl_host,
            "--port",
            str(self._settings.sgl_port),
            *self._settings.sgl_extra_args,
        ]

    def env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(self._settings.gpu_count)
        )
        environment["HF_HOME"] = self._settings.hf_home
        return environment

    def start(self) -> None:
        command = self.command()
        log_event(LOG, logging.INFO, "engine starting", command=command)
        self._process = self._popen(
            command,
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if getattr(self._process, "stdout", None) is not None:
            threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self) -> None:
        """Forward engine output into our log so RunPod shows one stream."""
        try:
            for line in self._process.stdout:  # type: ignore[union-attr]
                stripped = line.rstrip()
                if stripped:
                    log_event(LOG, logging.INFO, stripped, source="sgl-omni")
        except Exception:  # pragma: no cover - the pipe closes when the engine exits
            return

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def wait_ready(self) -> None:
        """Block until the engine answers, it dies, or the budget runs out."""
        if self._process is None:
            raise EngineUnavailable("start() was not called")

        url = f"{self._settings.base_url}/v1/models"
        deadline = self._monotonic() + self._settings.server_startup_timeout_s
        while True:
            returncode = self._process.poll()
            if returncode is not None:
                raise EngineUnavailable(f"engine exited with code {returncode}")
            try:
                status = self._http_get(url)
            except Exception:
                status = None
            if status == 200:
                log_event(LOG, logging.INFO, "engine ready", url=url)
                return
            if self._monotonic() >= deadline:
                self.stop()
                raise EngineUnavailable(
                    f"engine did not become ready within "
                    f"{self._settings.server_startup_timeout_s}s"
                )
            self._sleep(POLL_INTERVAL_S)

    def stop(self, timeout: float = 30.0) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
        except Exception:
            self._process.kill()
            try:
                self._process.wait(timeout=timeout)
            except Exception:  # pragma: no cover - nothing else we can do
                pass
        log_event(LOG, logging.INFO, "engine stopped")
