#!/usr/bin/env bash
# Populate a RunPod network volume with only the files sglang-omni needs (~28.8 GB
# of the 57 GB repository).
#
# This is the fallback for when RunPod Cached Models cannot serve the model. The two
# options are mutually exclusive: both mount at /runpod-volume.
#
# Prerequisites:
#   * a network volume in a datacenter that has your GPU type
#   * an S3 API key pair from the RunPod console (Settings -> S3 API keys), which is
#     separate from your normal RunPod API key
#   * awscli and the huggingface CLI available
#
# Usage:
#   DATACENTER=EU-CZ-1 VOLUME_ID=abc123 \
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
#     scripts/prepare_network_volume.sh
set -euo pipefail

: "${DATACENTER:?set DATACENTER, e.g. EU-CZ-1}"
: "${VOLUME_ID:?set VOLUME_ID}"
: "${AWS_ACCESS_KEY_ID:?set the RunPod S3 access key}"
: "${AWS_SECRET_ACCESS_KEY:?set the RunPod S3 secret}"

LOCAL_DIR="${LOCAL_DIR:-/tmp/minimax-music3}"
REMOTE_PREFIX="${REMOTE_PREFIX:-minimax-music3}"
ENDPOINT="https://s3api-$(echo "$DATACENTER" | tr '[:upper:]' '[:lower:]').runpod.io/"

echo "== downloading the runtime subset =="
hf download MiniMaxAI/MiniMax-Music3 --local-dir "$LOCAL_DIR" \
  --include "qwen_7B/**" "flowmatching_vae.pth" "dav.pth" "config.json"

echo "== verifying the layout sglang-omni requires =="
for artifact in \
  qwen_7B/qwen_7B \
  qwen_7B/qwen3-8B-tokenizer-music \
  flowmatching_vae.pth \
  dav.pth
do
  if [ ! -e "$LOCAL_DIR/$artifact" ]; then
    echo "missing $artifact - refusing to upload an unusable checkpoint" >&2
    exit 1
  fi
done
du -sh "$LOCAL_DIR"

echo "== uploading to s3://$VOLUME_ID/$REMOTE_PREFIX =="
# Files over 500 MB go multipart; the AWS CLI handles that automatically.
aws s3 sync "$LOCAL_DIR" "s3://$VOLUME_ID/$REMOTE_PREFIX" \
  --region "$DATACENTER" --endpoint-url "$ENDPOINT"

echo "== listing what landed =="
# ListObjects can be slow or fail on large trees while checksums compute; retry
# after a short wait if this errors. The upload itself is already done by then.
aws s3 ls "s3://$VOLUME_ID/$REMOTE_PREFIX/" \
  --region "$DATACENTER" --endpoint-url "$ENDPOINT" --recursive --summarize | tail -5

cat <<EOF

Done. On the endpoint:
  * remove the cached model
  * attach network volume $VOLUME_ID
  * set MODEL_PATH=/runpod-volume/$REMOTE_PREFIX

The worker will then log: "model resolved" with source=MODEL_PATH.
EOF
