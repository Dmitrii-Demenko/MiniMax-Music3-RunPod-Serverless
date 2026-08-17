#!/usr/bin/env bash
# Warm the FlashInfer JIT and torch.compile caches, then pack them for an image layer.
#
# Only useful for the registry build path (docker/from-source.Dockerfile): RunPod's
# GitHub builder has no GPU, so it cannot produce or verify these artifacts. With the
# GitHub path, the FlashInfer cache comes from the base image instead, which is why
# the endpoint must run on SM89 (L40S / RTX 6000 Ada) or SM90a (H100).
#
# Run INSIDE the worker container, on a Pod whose GPU matches production.
#
# Usage: scripts/warm_caches.sh /weights/minimax [/workspace/caches.tar.zst]
set -euo pipefail

MODEL_PATH="${1:?usage: warm_caches.sh <model-path> [output-archive]}"
OUT="${2:-/workspace/caches.tar.zst}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/root}"

echo "== GPU =="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "== starting engine =="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  sgl-omni serve --model-path "$MODEL_PATH" --host 127.0.0.1 --port 8000 \
  >/tmp/warm-engine.log 2>&1 &
ENGINE_PID=$!
trap 'kill "$ENGINE_PID" 2>/dev/null || true' EXIT

until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "engine died; last lines of /tmp/warm-engine.log:" >&2
    tail -50 /tmp/warm-engine.log >&2
    exit 1
  fi
  sleep 2
done
echo "engine ready"

echo "== two short generations to trigger every compiled path =="
for seed in 1 2; do
  curl -sf -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"MiniMaxAI/MiniMax-Music3\",\"input\":\"[Intro]\\n(instrumental)\",\"instructions\":\"An ambient instrumental at 70 BPM with warm analog pads\",\"seed\":${seed},\"max_new_tokens\":250,\"response_format\":\"wav\",\"stream\":false}" \
    -o "/tmp/warm-${seed}.wav"
  echo "  seed ${seed}: $(wc -c <"/tmp/warm-${seed}.wav") bytes"
done

echo "== packing caches =="
tar --zstd -cf "$OUT" -C / \
  "root/.cache/flashinfer" \
  "${TORCHINDUCTOR_CACHE_DIR#/}"
ls -lh "$OUT"

cat <<EOF

Copy $OUT out of the Pod to docker/caches/caches.tar.zst, then add it as a layer on
top of the worker image. It is a build artifact and is gitignored on purpose.
A cache built on a different compute capability is useless - rebuild per GPU type.
EOF
