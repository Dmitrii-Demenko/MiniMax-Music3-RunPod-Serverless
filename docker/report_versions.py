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
import re
import sys

# Leading distribution name of a requirement string: "pyzmq>=25.0.0" -> "pyzmq",
# "huggingface-hub[hf_xet]>=0.36.0" -> "huggingface-hub".
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")

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
    # Absent from the base image despite sglang_omni.pipeline.control_plane needing
    # it, so the Dockerfile installs it explicitly. Listed here to keep it visible
    # in the build log if a future base image starts shipping a different version.
    "msgpack": "1.1.0",
    "runpod": "1.12.0",
    "httpx": "0.28.1",
    "imageio-ffmpeg": "0.6.0",
}


def installed_version(name: str) -> str | None:
    for candidate in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def declared_requirements(distribution: str) -> list[str]:
    """Distribution names sglang-omni declares as install requirements.

    Requirements gated behind an extra are skipped: nothing here installs extras, so
    reporting them as missing would be noise.
    """
    try:
        raw = metadata.requires(distribution) or []
    except metadata.PackageNotFoundError:
        return []

    names: list[str] = []
    for requirement in raw:
        head, _, marker = requirement.partition(";")
        if "extra" in marker:
            continue
        match = _REQUIREMENT_NAME.match(head)
        if match:
            names.append(match.group(1))
    return names


def report_declared(distribution: str) -> list[str]:
    """Print every declared requirement and whether it is installed.

    --no-deps means nothing verifies that the base image actually satisfies the
    dependency tree. Printing the whole inventory in one pass is what turns a
    missing package into one build log to read, instead of one build per package
    discovered by crashing on a GPU.
    """
    required = declared_requirements(distribution)
    if not required:
        print(f"\ncould not read {distribution} requirements: skipping inventory")
        return []

    missing = [name for name in required if installed_version(name) is None]
    print(f"\n{distribution} declares {len(required)} install requirements")
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(sorted(missing))}")
    else:
        print("  all declared requirements are installed")
    return missing


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

    declared_missing = report_declared("sglang-omni")

    if missing:
        print(f"\nWARNING: not installed: {', '.join(missing)}")
    if declared_missing:
        print(
            "\nWARNING: the base image does not satisfy every declared requirement.\n"
            "Anything the engine imports on the serve path must be added to the\n"
            "install allowlist in the Dockerfile, or `sgl-omni serve` will exit with\n"
            "ModuleNotFoundError on the first cold start and crash-loop the worker."
        )
    if mismatches:
        print("\nWARNING: version mismatches against the pinned release:")
        for line in mismatches:
            print(f"  - {line}")
        print(
            "\nThe base image predates this release. If generation misbehaves, rebuild\n"
            'with --build-arg EXTRA_PINS="<pkg>==<version> ..." for the packages above,\n'
            "or switch SGLANG_OMNI_IMAGE to a newer base. See docker/README.md."
        )
    if not missing and not mismatches and not declared_missing:
        print("\nall pinned versions match and every declared requirement is present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
