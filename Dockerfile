# MiniMax-Music3 RunPod Serverless worker.
#
# Built by RunPod's GitHub integration, whose limits shape this file:
#   * docker build is capped at 30 minutes, so nothing is compiled from source
#     (upstream's own Dockerfile builds UCX, which would not fit)
#   * no GPU during the build, so torch.compile and FlashInfer JIT artifacts
#     cannot be generated here - the JIT cache is copied from a prebuilt image
#   * private base images are not supported, so every base below is public
#   * the final image must stay under 80 GB
#
# Layout mirrors upstream sglang-omni: the official runtime image already carries
# UCX and the CUDA 13 stack, and the FlashInfer 0.6.14 JIT cache validated on Ada
# (SM89) and Hopper (SM90a) is copied in from the image upstream itself uses for it.
# That cache is why the endpoint must run on L40S / RTX 6000 Ada or H100: on any
# other compute capability the kernels are compiled during a billed cold start.
#
# Pins verified 2026-08-17. See docker/README.md before changing them.

ARG SGLANG_OMNI_IMAGE=lmsysorg/sglang-omni@sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101

FROM ${FLASHINFER_CACHE_IMAGE} AS flashinfer-cache

FROM ${SGLANG_OMNI_IMAGE} AS runtime

# The upstream entrypoint clones sglang-omni from GitHub on every container start.
# On a serverless cold start that is an unacceptable network dependency, so the
# version is pinned into the image instead.
ENV SGLANG_OMNI_AUTO_CLONE=0

# MiniMax-Music3 support first shipped in 0.1.2 (released 2026-08-16); earlier
# releases do not contain sglang_omni/models/minimax_music3.
ARG SGLANG_OMNI_VERSION=0.1.2

# Escape hatch for a base/release version mismatch: pass exact pins here rather
# than editing this file, e.g.
#   --build-arg EXTRA_PINS="torch==2.11.0 transformers==5.12.1"
# Never use loose specifiers here; they reintroduce the resolver problem below.
ARG EXTRA_PINS=""

# Install the package WITHOUT its dependency tree.
#
# The base image already carries that tree: it was built by upstream's Dockerfile
# from the same pyproject.toml, and upstream's own entrypoint installs the package
# with --no-deps for exactly this reason. Resolving dependencies here instead sent
# pip into 28 minutes of backtracking - it walked accelerate all the way down to
# 0.27.0 while re-downloading torch/torchvision metadata - and blew RunPod's
# 30-minute build cap.
#
# uv ships in the base image and resolves in seconds; pip is the fallback.
RUN set -eux; \
    if command -v uv >/dev/null 2>&1; then \
        INSTALL="uv pip install --system --break-system-packages"; \
    else \
        INSTALL="python3 -m pip install --no-cache-dir --break-system-packages"; \
    fi; \
    $INSTALL --no-deps "sglang-omni==${SGLANG_OMNI_VERSION}"; \
    $INSTALL "runpod==1.12.0" "httpx==0.28.1" "imageio-ffmpeg==0.6.0"; \
    if [ -n "${EXTRA_PINS}" ]; then $INSTALL ${EXTRA_PINS}; fi

# Report what the base image actually provides. --no-deps means the base decides
# these versions, so a mismatch against the pinned release must show up in the
# build log instead of on a GPU. Warns, never fails: the import checks below are
# what decide whether the runtime is usable.
COPY docker/report_versions.py /tmp/report_versions.py
RUN python3 /tmp/report_versions.py && rm /tmp/report_versions.py

COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.14 /root/.cache/flashinfer/0.6.14

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0 \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor

# Fail the build, not the first cold start, if the runtime cannot load this model.
RUN python3 -c "import sglang_omni.models.minimax_music3 as m; print('model module:', m.__file__)" \
    && python3 -c "from sglang_omni.models.minimax_music3.checkpoint import resolve_checkpoint; print('checkpoint loader: ok')" \
    && python3 -c "from imageio_ffmpeg import get_ffmpeg_exe; print('ffmpeg:', get_ffmpeg_exe())" \
    && python3 -c "import runpod, httpx; print('runpod:', runpod.__version__)" \
    && command -v sgl-omni

COPY src/ /app/src/
COPY rp_handler.py /app/rp_handler.py
# Shipped so the on-GPU workflows (smoke test, benchmark, cache warming) can be run
# from inside the container on a Pod without copying files in.
COPY scripts/ /app/scripts/

# Verify our own modules import cleanly with no engine and no GPU present.
RUN PYTHONPATH=/app/src python3 -c "import config, logging_setup, lyrics, request_schema, model_path, audio, delivery, server, handler; print('worker modules: ok')" \
    && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('rp_handler', '/app/rp_handler.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); print('entry point: ok')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/runpod-volume/huggingface-cache

ENTRYPOINT ["python3", "-u", "/app/rp_handler.py"]
