# Build notes

## Two build paths

| Path | File | When |
|---|---|---|
| **GitHub / RunPod builder** (default) | `Dockerfile` (repository root) | Normal deploys. Thin layer on prebuilt public images, nothing compiled from source. |
| Registry | `docker/from-source.Dockerfile` | Only if the root path fails: pinned base gone, or its dependencies conflict with `sglang-omni==0.1.2`. Compiles UCX, so build it locally and push. |

## Why the root Dockerfile looks the way it does

RunPod's GitHub builder imposes limits that decide the design:

- **30 minutes** for `docker build` (160 minutes for the whole build window). Upstream's
  own Dockerfile compiles UCX from source, which does not fit. So the root Dockerfile
  starts from a prebuilt image that already has UCX and the CUDA 13 stack.
- **No GPU during the build.** `torch.compile` and FlashInfer JIT artifacts cannot be
  generated here. The FlashInfer 0.6.14 cache is copied from the image upstream itself
  uses as its cache source. That cache was validated on Ada (SM89) and Hopper (SM90a) —
  which is why the endpoint must pin L40S / RTX 6000 Ada or H100. On any other compute
  capability the kernels compile during a billed cold start.
- **Private base images are not supported.** Both pinned bases are public.
- **80 GB image limit.** The layers here land far below it; weights are never baked in.

The upstream entrypoint clones `sglang-omni` from GitHub on every container start, which
on a serverless cold start is a network dependency that can fail or stall. `Dockerfile`
sets `SGLANG_OMNI_AUTO_CLONE=0` and installs a pinned release instead.

## Pins (verified 2026-08-17)

| What | Value |
|---|---|
| `sglang-omni` | `0.1.2` — released 2026-08-16, tag `v0.1.2`, commit `0ae7669d6e92`. The first release containing `sglang_omni/models/minimax_music3`. |
| Runtime base | `lmsysorg/sglang-omni@sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e` |
| FlashInfer cache source | `hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101` |
| `lmsysorg/sglang` (source path only) | `sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7` |
| UCX (source path only) | commit `d8e50df6651b9ea5b76f23aee0aefbf053a4137a` |

Both digests come from upstream's `docker/Dockerfile` at tag `v0.1.2`. To re-verify:

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/v0.1.2/docker/Dockerfile
```

Override without editing the file:

```bash
docker build --build-arg SGLANG_OMNI_IMAGE=<image@digest> -t mm3-worker:dev .
```

## Build-time verification

The root Dockerfile fails the build rather than the first cold start if anything is
wrong. It checks that `sglang_omni.models.minimax_music3` imports, that the checkpoint
loader is importable, that ffmpeg resolves, that `sgl-omni` is on `PATH`, and that every
worker module imports with no GPU and no engine present.

If the dependency install conflicts — the runtime base predates `0.1.2` — the fix is a
pinned pre-install before the `sglang-omni` line:

```dockerfile
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        torch==2.11.0 transformers==5.12.1
```

Record here whatever turned out to be necessary.

## Local build

```bash
docker build -t mm3-worker:dev .

# fails fast, by design: no weights anywhere
docker run --rm -e HF_HOME=/nonexistent mm3-worker:dev
```

Expected: a JSON log line naming `no usable checkpoint`, then a non-zero exit.

## Cache warming (registry path only)

`scripts/warm_caches.sh` runs on a Pod with the production GPU, triggers every compiled
path, and packs `~/.cache/flashinfer` plus the Inductor cache into an archive. Add it as
a layer on the worker image. The archive is a large build artifact and is gitignored;
store it in a registry image or object storage. A cache from a different compute
capability is useless — rebuild it per GPU type.
