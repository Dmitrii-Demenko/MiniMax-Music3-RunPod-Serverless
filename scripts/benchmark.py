#!/usr/bin/env python3
"""Measure generation throughput against a local sgl-omni engine.

Run on a Pod, inside the worker container, with the engine already serving:

    python scripts/benchmark.py --frames 250 750 1500 9000 --concurrency 1
    python scripts/benchmark.py --frames 750 --concurrency 1 4 8

For a concurrency sweep above 1, restart the engine with a matching admission
limit, because classifier-free guidance makes every request two decode rows:

    sgl-omni serve --model-path ... --max-running-requests 32

The printed table is the performance baseline that belongs in the README.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

LYRICS = "[Verse]\nCity lights are calling out my name\n[Chorus]\nAnd I keep on walking"
PROMPT = "A dreamy synthwave track with analog pads and a driving bassline at 110 BPM"

# The engine answers with 32 kHz, 16-bit stereo WAV.
BYTES_PER_SECOND = 32000 * 2 * 2
WAV_HEADER_BYTES = 44


def gpu_memory_mb() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [int(line) for line in out.split() if line.isdigit()]


def one_request(base_url: str, frames: int, seed: int, timeout: float) -> tuple[float, int]:
    payload = {
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": LYRICS,
        "instructions": PROMPT,
        "seed": seed,
        "max_new_tokens": frames,
        "response_format": "wav",
        "stream": False,
    }
    started = time.monotonic()
    response = httpx.post(f"{base_url}/v1/audio/speech", json=payload, timeout=timeout)
    response.raise_for_status()
    return time.monotonic() - started, len(response.content)


def audio_seconds(size_bytes: int) -> float:
    return max(0.0, (size_bytes - WAV_HEADER_BYTES) / BYTES_PER_SECOND)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frames", type=int, nargs="+", default=[250, 750, 1500, 9000])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--timeout", type=float, default=3000.0)
    args = parser.parse_args()

    header = f"{'frames':>7} {'conc':>5} {'wall_s':>8} {'audio_s':>8} {'ratio':>7} {'vram_mb':>16}"
    print(header)
    print("-" * len(header))

    for frames in args.frames:
        for concurrency in args.concurrency:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                started = time.monotonic()
                results = list(
                    pool.map(
                        lambda seed: one_request(
                            args.base_url, frames, seed, args.timeout
                        ),
                        range(concurrency),
                    )
                )
                wall = time.monotonic() - started

            audio = sum(audio_seconds(size) for _elapsed, size in results)
            per_request = statistics.mean(elapsed for elapsed, _size in results)
            vram = ",".join(str(value) for value in gpu_memory_mb())
            print(
                f"{frames:>7} {concurrency:>5} {wall:>8.1f} {audio:>8.1f} "
                f"{audio / wall:>7.2f} {vram:>16}   (mean/req {per_request:.1f}s, "
                f"{frames / 25:.0f}s cap)"
            )


if __name__ == "__main__":
    main()
