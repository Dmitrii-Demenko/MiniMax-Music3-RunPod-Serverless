#!/usr/bin/env python3
"""RunPod Serverless entry point.

This file lives at the repository root on purpose: RunPod's GitHub integration
scans the repository for the SDK's serverless start call, and it does not reliably
find one nested under src/. Keeping that call here, on a single line, makes the
check pass without a decoy — this is the module the container actually runs
(see ENTRYPOINT in the Dockerfile).

Everything else lives in src/: this file only wires the worker to the RunPod SDK.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from handler import Runtime, bootstrap, concurrency_modifier, run_job  # noqa: E402


def main() -> None:
    import runpod

    runtime: Runtime = bootstrap()

    async def job_handler(job: dict) -> dict:
        return await run_job(job, runtime)

    # Keep this call on one line: RunPod's repository scan greps for it and reports
    # the handler as missing when it is split across lines.
    config = {"handler": job_handler, "concurrency_modifier": concurrency_modifier}
    runpod.serverless.start(config)


if __name__ == "__main__":
    main()
