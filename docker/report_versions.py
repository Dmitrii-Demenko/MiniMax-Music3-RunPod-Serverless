#!/usr/bin/env python3
"""Print the versions of the packages the worker actually runs on.

Installing sglang-omni with --no-deps means the base image decides these versions,
so a mismatch against what the pinned release expects must be visible in the build
log rather than discovered on a GPU. Reference pins for sglang-omni 0.1.2:

    torch==2.11.0  transformers==5.12.1  sglang==0.5.16
    flashinfer-python==0.6.14  flash-attn-4>=4.0.0b18

A mismatch is reported, not fatal: the runtime may still work, and the fix is
`--build-arg EXTRA_PINS="..."` once you know what actually differs. Missing
packages that the model needs are caught by the import checks in the Dockerfile.
"""

from __future__ import annotations

import importlib.metadata as metadata
import sys

# Version each package should have for sglang-omni 0.1.2, or None when any version
# is acceptable.
EXPECTED: dict[str, str | None] = {
    "sglang-omni": "0.1.2",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.12.1",
    "sglang": "0.5.16",
    "flashinfer-python": "0.6.14",
    "flash-attn-4": None,
    "numpy": None,
    "runpod": "1.12.0",
    "httpx": "0.28.1",
    "imageio-ffmpeg": "0.6.0",
}


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    print("installed versions (base image + our layer):")
    mismatches: list[str] = []
    missing: list[str] = []

    for name, expected in EXPECTED.items():
        actual = installed_version(name)
        if actual is None:
            missing.append(name)
            print(f"  {name:<20} MISSING")
            continue
        if expected is not None and actual != expected:
            mismatches.append(f"{name}: have {actual}, sglang-omni 0.1.2 expects {expected}")
            print(f"  {name:<20} {actual:<16} != expected {expected}")
        else:
            print(f"  {name:<20} {actual}")

    if missing:
        print(f"\nWARNING: not installed: {', '.join(missing)}")
    if mismatches:
        print("\nWARNING: version mismatches against the pinned release:")
        for line in mismatches:
            print(f"  - {line}")
        print(
            "\nThe base image predates this release. If generation misbehaves, rebuild\n"
            'with --build-arg EXTRA_PINS="<pkg>==<version> ..." for the packages above,\n'
            "or switch SGLANG_OMNI_IMAGE to a newer base. See docker/README.md."
        )
    if not missing and not mismatches:
        print("\nall pinned versions match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
