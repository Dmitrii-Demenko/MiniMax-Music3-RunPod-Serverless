# Fallback build path: assemble the runtime from upstream sources instead of
# relying on a prebuilt sglang-omni image.
#
# Use this only if the root Dockerfile fails - for example if the pinned public
# base image disappears, or if its dependency set conflicts with sglang-omni 0.1.2.
# It compiles UCX from source and takes tens of minutes, so it will exceed RunPod's
# 30-minute build cap: build it locally and push the result to a registry, then
# create the endpoint from that image rather than from GitHub.
#
#   docker build -f docker/from-source.Dockerfile -t <registry>/<user>/mm3-base:v0.1.2 \
#     --target base .
#   docker build -f docker/from-source.Dockerfile -t <registry>/<user>/mm3-worker:0.1.0 .
#
# Before building, fetch the two upstream files this expects:
#   curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/v0.1.2/pyproject.toml \
#     -o docker/upstream/pyproject.toml

ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
ARG FLASHINFER_CACHE_IMAGE=hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101

FROM ${FLASHINFER_CACHE_IMAGE} AS flashinfer-cache

FROM ${SGLANG_IMAGE} AS base

ARG UCX_COMMIT=d8e50df6651b9ea5b76f23aee0aefbf053a4137a

RUN git clone --filter=blob:none https://github.com/openucx/ucx.git /tmp/ucx \
    && git -C /tmp/ucx checkout "${UCX_COMMIT}" \
    && cd /tmp/ucx \
    && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=/usr/local/cuda \
        --with-verbs \
        --with-dm \
        --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip \
    && ldconfig \
    && rm -rf /tmp/ucx

COPY docker/upstream/pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --break-system-packages --no-build-isolation \
        -r /tmp/pyproject.toml \
    && uv pip install --system --break-system-packages --no-deps --reinstall \
        flashinfer-python==0.6.14 \
    && python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm /tmp/pyproject.toml

# Docker builds have no GPU to regenerate architecture-specific JIT artifacts, so
# reuse the cache validated on Ada (SM89) and Hopper (SM90a).
COPY --from=flashinfer-cache /root/.cache/flashinfer/0.6.14 /root/.cache/flashinfer/0.6.14

ENV FLASHINFER_WORKSPACE_BASE=/root \
    FLASHINFER_JIT_DEBUG=0 \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    SGLANG_OMNI_AUTO_CLONE=0

FROM base AS worker

ARG SGLANG_OMNI_VERSION=0.1.2

# --no-deps because the base stage installed the dependency tree from upstream's
# pyproject.toml already; resolving it again is what blew the build time budget.
RUN uv pip install --system --break-system-packages --no-deps \
        "sglang-omni==${SGLANG_OMNI_VERSION}" \
    && uv pip install --system --break-system-packages \
        "runpod==1.12.0" "httpx==0.28.1" "imageio-ffmpeg==0.6.0"

COPY docker/report_versions.py /tmp/report_versions.py
RUN python3 /tmp/report_versions.py && rm /tmp/report_versions.py

RUN python3 -c "import sglang_omni.models.minimax_music3 as m; print('model module:', m.__file__)" \
    && python3 -c "from imageio_ffmpeg import get_ffmpeg_exe; print('ffmpeg:', get_ffmpeg_exe())" \
    && command -v sgl-omni

COPY src/ /app/src/
COPY rp_handler.py /app/rp_handler.py
COPY scripts/ /app/scripts/

RUN PYTHONPATH=/app/src python3 -c "import handler; print('worker modules: ok')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/runpod-volume/huggingface-cache

ENTRYPOINT ["python3", "-u", "/app/rp_handler.py"]
