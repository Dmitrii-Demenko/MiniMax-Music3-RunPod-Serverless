#!/usr/bin/env python3
"""Smoke-test MiniMax-Music3 against a local engine or a deployed RunPod endpoint.

Local engine on a Pod (sgl-omni already serving on :8000):
    python scripts/smoke_test.py --mode local --duration 10

Deployed endpoint:
    python scripts/smoke_test.py --mode endpoint \
        --endpoint-id "$ENDPOINT_ID" --api-key "$RUNPOD_API_KEY" --duration 10
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import time

import httpx

LYRICS = (
    "[Verse]\nWalking down the empty street at midnight\n"
    "[Chorus]\nAnd I keep on walking"
)
PROMPT = (
    "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, "
    "soft vinyl crackle, dusty boom-bap drums, warm upright bass."
)


def run_local(args: argparse.Namespace) -> bytes:
    payload = {
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": LYRICS,
        "instructions": PROMPT,
        "seed": args.seed,
        "max_new_tokens": int(args.duration * 25),
        "response_format": "wav",
        "stream": False,
    }
    started = time.monotonic()
    response = httpx.post(
        f"{args.base_url}/v1/audio/speech", json=payload, timeout=args.timeout
    )
    response.raise_for_status()
    print(f"local generation took {time.monotonic() - started:.1f}s")
    return response.content


def run_endpoint(args: argparse.Namespace) -> bytes:
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    body = {
        "input": {
            "lyrics": LYRICS,
            "prompt": PROMPT,
            "duration": args.duration,
            "seed": args.seed,
            "format": args.format,
        }
    }

    started = time.monotonic()
    submitted = httpx.post(f"{base}/run", json=body, headers=headers, timeout=60)
    submitted.raise_for_status()
    job_id = submitted.json()["id"]
    print(f"submitted job {job_id}")

    status: dict = {}
    state = "UNKNOWN"
    while True:
        status = httpx.get(f"{base}/status/{job_id}", headers=headers, timeout=60).json()
        state = status.get("status", "UNKNOWN")
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            break
        print(f"  {state} ({time.monotonic() - started:.0f}s)")
        time.sleep(5)

    print(f"finished as {state} after {time.monotonic() - started:.1f}s")
    if state != "COMPLETED":
        print(json.dumps(status, indent=2))
        sys.exit(1)

    output = status["output"]
    if isinstance(output, dict) and output.get("error"):
        print(json.dumps(output, indent=2))
        sys.exit(1)

    printable = {key: value for key, value in output.items() if key != "audio_base64"}
    print(json.dumps(printable, indent=2))

    if "audio_base64" in output:
        return base64.b64decode(output["audio_base64"])
    return httpx.get(output["audio_url"], timeout=300).content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "endpoint"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--api-key")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--format", default="mp3")
    parser.add_argument("--timeout", type=float, default=1500.0)
    parser.add_argument("--out", default="smoke")
    args = parser.parse_args()

    if args.mode == "endpoint" and not (args.endpoint_id and args.api_key):
        parser.error("--endpoint-id and --api-key are required for --mode endpoint")

    data = run_local(args) if args.mode == "local" else run_endpoint(args)
    # The local engine always answers with WAV; the endpoint answers in the
    # format that was requested.
    suffix = "wav" if args.mode == "local" else args.format
    path = pathlib.Path(f"{args.out}.{suffix}")
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
