# MiniMax-Music3 on RunPod Serverless

A serverless endpoint for [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3):
send lyrics and a style caption, get a finished song back. Generation runs on
[SGLang-Omni](https://github.com/sgl-project/sglang-omni) inside the worker; you pay only
for the time a track is being rendered.

- Up to **360 seconds** per track (25 audio frames per second, hard cap 9000 frames)
- 32 kHz stereo, delivered as `mp3`, `wav`, `flac` or `opus`
- Deterministic: same lyrics, caption, seed and length return byte-identical audio
- Two GPUs per worker by default (autoregressive backbone on GPU 0, flow-matching
  decoder on GPU 1); single-GPU also supported

## API

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

```json
{
  "input": {
    "lyrics": "[Verse]\nWalking down the empty street at midnight\n[Chorus]\nAnd I keep on walking",
    "prompt": "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano, soft vinyl crackle, dusty boom-bap drums, warm upright bass.",
    "duration": 30,
    "seed": 42,
    "format": "mp3",
    "bitrate": "192k"
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `lyrics` (alias `input`) | string | required | Non-empty. Structure tags on their own lines. |
| `prompt` (alias `instructions`) | string | required | Non-empty. Genre, BPM, key, instruments, vocal. |
| `duration` | number | `30` | Seconds, `1…360`. Mutually exclusive with `max_new_tokens`. |
| `max_new_tokens` | int | — | Audio frames, `1…9000`, 25 per second. |
| `seed` | int | `0` | Non-negative 64-bit. `0` is a fixed seed, not a random one. |
| `format` | string | `mp3` | `wav`, `mp3`, `opus`, `flac`. |
| `bitrate` | string | `192k` | Applies to `mp3` and `opus`. |

Response:

```json
{
  "audio_url": "https://your-bucket/job-id.mp3",
  "format": "mp3",
  "sample_rate": 32000,
  "channels": 2,
  "duration_s": 29.4,
  "frames": 735,
  "seed": 42,
  "warnings": [],
  "metrics": {"validate_ms": 1, "generate_ms": 41230, "encode_ms": 380, "deliver_ms": 540}
}
```

`audio_base64` replaces `audio_url` when no bucket is configured. **`duration_s` may be
shorter than requested** — `duration` is a cap, and the model ends the song itself when
it emits the audio-end token. That is not truncation.

### Requesting a track end to end

```bash
JOB=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"lyrics":"[Verse]\nCity lights are calling out my name","prompt":"A dreamy synthwave track with analog pads and a driving bassline at 110 BPM","duration":30,"seed":7}}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# Generation takes minutes: poll /status rather than using /runsync.
curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Or use the bundled client:

```bash
python scripts/smoke_test.py --mode endpoint \
  --endpoint-id "$ENDPOINT_ID" --api-key "$RUNPOD_API_KEY" --duration 30
```

### Errors

Errors come back as `{"error": "...", "code": "..."}` in the job output.

| Code | Meaning | Retry |
|---|---|---|
| `invalid_request` | Missing or out-of-range field | No — fix the request |
| `unsupported_parameter` | A parameter this model does not honour | No |
| `upstream_rejected` | The engine refused the request | No |
| `generation_failed` | The engine failed, e.g. out of memory | Yes, on a fresh worker |
| `engine_unavailable` | The engine is not running | Yes, on a fresh worker |
| `encoding_failed` | ffmpeg could not produce the requested format | No |
| `result_too_large` | Base64 result over the limit and no bucket configured | No — configure a bucket |
| `timeout` | Generation exceeded `GENERATION_TIMEOUT_S` | No — raise the timeout |

These parameters are **rejected rather than ignored**, so a mistake is visible instead of
silently producing something else: `temperature`, `top_p`, `top_k`, `repetition_penalty`
(sampling is fixed at guidance 1.5 then top-k 50), `voice`, `ref_audio`, `ref_text`,
`language`, `task_type` (no such conditioning), `speed` (tempo belongs in the prompt) and
`stream: true` (this model's API is non-streaming).

## Deploy from GitHub

1. Push this repository to GitHub.
2. In the RunPod console: **Serverless → New Endpoint → GitHub Repo**, pick the repo and
   branch. The Dockerfile is at the repository root, so no path override is needed.
3. Configure the endpoint:

| Setting | Value | Why |
|---|---|---|
| GPUs per worker | **2** | Backbone on device 0, DIT + DAV on device 1 |
| GPU type | **L40S / RTX 6000 Ada (48 GB)**, secondary **H100 (80 GB)** | The prebuilt FlashInfer JIT cache covers SM89 and SM90a only; on A100 (SM80) or A6000/A40 (SM86) the kernels compile during a billed cold start. Picking these by pool is not enough — see the warning below |
| CUDA version filter | **13.x** | `sglang-omni 0.1.2` ships a CUDA 13 stack |
| Execution timeout | **1800 s** | The 600 s default cannot finish a 360 s track |
| Idle timeout | **120–300 s** | Cold starts are expensive; keep workers warm between requests |
| Active workers | 0 for staging, **≥1** for production | Otherwise every first request pays a cold start |
| Max workers | 3 to start | |
| Scaling | Queue delay, 4 s | Generation is long; aggressive scaling buys nothing |
| Cached model | `MiniMaxAI/MiniMax-Music3` | Download time is not billed and cold starts drop to seconds |
| Network volume | **none** | It would collide with the cached-model mount at `/runpod-volume` |

> **The `ADA_48_PRO` pool is not Ada-only.** RunPod also files
> `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb` under it — a Blackwell
> (SM120) MIG slice, outside the SM89/SM90a the baked JIT cache covers. Selecting the
> pool therefore does not select SM89, and the REST view of the endpoint reports only
> `pools`, so an excluded SKU is the only way to keep it out. Exclude it explicitly and
> confirm `gpuTypeId` on a live worker rather than trusting the pool name:
>
> ```bash
> # gpuIds ends up as: ADA_48_PRO,-NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb
> curl -s "https://rest.runpod.io/v1/endpoints/$ENDPOINT_ID/workers" \
>   -H "Authorization: Bearer $RUNPOD_API_KEY" | jq '.[].gpuTypeId'
> ```
>
> Also check `dataCenterIds` is not left empty: with every datacenter allowed, workers
> can land where the image pull stalls indefinitely and sit `INITIALIZING` for hours,
> consuming the max-workers budget so healthy replacements never start.

4. Set environment variables (see the table below). At minimum `GPU_COUNT=2`; add the
   `BUCKET_*` trio if you want URLs instead of base64.
5. Verify:

```bash
python scripts/smoke_test.py --mode endpoint \
  --endpoint-id "$ENDPOINT_ID" --api-key "$RUNPOD_API_KEY" --duration 10
```

Then repeat with `--duration 300` and confirm it finishes inside the execution timeout.

RunPod's builder caps `docker build` at 30 minutes and has no GPU. See
[docker/README.md](docker/README.md) for what that implies and for the registry fallback
build if the GitHub build ever fails.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GPU_COUNT` | `2` | Devices given to the engine (`CUDA_VISIBLE_DEVICES`) |
| `MODEL_PATH` | — | Explicit path to the weights; overrides auto-discovery |
| `MODEL_REPO_ID` | `MiniMaxAI/MiniMax-Music3` | Used to find the cached-model snapshot |
| `HF_HOME` | `/runpod-volume/huggingface-cache` | Where RunPod puts cached models |
| `ALLOW_HUB_DOWNLOAD` | `0` | Allow downloading 57 GB at startup. Off, because that download is billed cold-start time |
| `SGL_EXTRA_ARGS` | — | Extra engine flags, e.g. `--max-running-requests 32` |
| `SGL_HOST` / `SGL_PORT` | `127.0.0.1` / `8000` | Where the engine listens |
| `SERVER_STARTUP_TIMEOUT_S` | `1200` | Budget for loading weights and capturing CUDA graphs |
| `GENERATION_TIMEOUT_S` | `1500` | Per-request timeout; keep below the endpoint execution timeout |
| `MAX_CONCURRENCY` | `1` | Jobs a single worker accepts at once |
| `MAX_DURATION_S` | `360` | Upper bound on `duration` |
| `DEFAULT_FORMAT` | `mp3` | Format when the request omits one |
| `DEFAULT_BITRATE` | `192k` | Bitrate when the request omits one |
| `BASE64_MAX_ENCODED_BYTES` | `9500000` | Headroom under RunPod's 10 MB `/run` limit |
| `BUCKET_ENDPOINT_URL` | — | Set all three and results come back as URLs |
| `BUCKET_ACCESS_KEY_ID` | — | S3 / R2 credentials |
| `BUCKET_SECRET_ACCESS_KEY` | — | |
| `LOG_LEVEL` | `INFO` | |

### Delivery: why a bucket matters

RunPod caps payloads at 10 MB for `/run` and 20 MB for `/runsync`. A 360-second WAV is
about 46 MB, and even a 192 kbps MP3 of that length exceeds the `/run` limit once
base64-encoded. Without `BUCKET_*` the worker returns base64 and refuses anything over
the limit with `result_too_large` — it never returns a truncated file. Configure a
bucket for anything beyond short clips.

### Storage: cached model vs network volume

Cached Models is the default because RunPod does not bill the download and cold starts
drop to seconds. It stores the whole 57 GB repository under
`/runpod-volume/huggingface-cache/hub/`; the worker finds the snapshot itself.

A network volume is the fallback, holding only the ~28.8 GB this runtime actually needs
(`qwen_7B/qwen_7B/`, `qwen_7B/qwen3-8B-tokenizer-music/`, `flowmatching_vae.pth`,
`dav.pth`). The two are mutually exclusive: both mount at `/runpod-volume`. A volume also
locks workers to one datacenter, which narrows GPU availability.

```bash
DATACENTER=EU-CZ-1 VOLUME_ID=<id> \
AWS_ACCESS_KEY_ID=<s3-key> AWS_SECRET_ACCESS_KEY=<s3-secret> \
  scripts/prepare_network_volume.sh
```

Then remove the cached model, attach the volume, and set
`MODEL_PATH=/runpod-volume/minimax-music3`. No code change is needed.

## Tuning concurrency

`MAX_CONCURRENCY=1` is the default: predictable latency and deterministic VRAM. Batching
is where this model is most efficient — the depth decoder costs the same per step whether
one or eight requests share the batch — so raising it improves cost per track under load.

Raise it together with the engine's admission limit:

```
MAX_CONCURRENCY=4
SGL_EXTRA_ARGS=--max-running-requests 16
```

**Size for decode rows, not requests.** Classifier-free guidance is mandatory and gives
every request a second row that holds its own KV cache for the whole song, so KV grows
twice as fast as the request count suggests. Never pass `--cuda-graph-max-bs`: this model
computes it internally and discards a supplied value.

## Prompting

- **Every structure tag on its own line.** `[Verse] Walking down the street` reaches the
  model as `[verse]` alone and the lyric is silently dropped. The worker splits such
  lines and reports it in `warnings`, but do not rely on that.
  Tags: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`,
  `[Instrumental]`, `[Solo]`, `[Outro]`.
- **The caption is the strongest control you have.** Name genre, BPM, key, instruments,
  production character, and the vocal explicitly ("warm female vocal") — otherwise the
  model may drift instrumental. Long structured captions work well: global attributes,
  emotional progression, vocal detail.
- **Iterate at 10 seconds** (`"duration": 10`), then render full length once the style is
  right. Generation time scales with length.
- For an instrumental, ask for it in the caption and keep the lyrics minimal
  (`"[Intro]\n(instrumental)"`) — lyrics must be non-empty.
- Whitespace matters. Reproducibility requires byte-identical lyrics and caption.
- The tokenized prompt is capped at 5000 tokens.

## Performance baseline

Measure before you promise anything: no throughput numbers are published upstream for
this model, so this table starts empty and must be filled from your own hardware.

On a Pod with the production GPU, inside the worker container with the engine serving:

```bash
python scripts/benchmark.py --frames 250 750 1500 9000 --concurrency 1
python scripts/benchmark.py --frames 750 --concurrency 1 4 8
```

| GPU | Image | Date | frames | conc | wall_s | audio_s | ratio | VRAM |
|---|---|---|---|---|---|---|---|---|
| _fill in_ | | | | | | | | |

Also record engine startup time (`engine ready` in the logs) — it sets the floor for
`SERVER_STARTUP_TIMEOUT_S` and dominates cold start.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

The suite runs without a GPU: pure functions are tested directly, and the handler is
tested against an `httpx.MockTransport` that returns a synthetic WAV. Anything that needs
a real engine lives in `scripts/` and is run on a Pod.

| File | Responsibility |
|---|---|
| `rp_handler.py` | RunPod SDK wiring. Lives at the root because RunPod's GitHub integration scans for the serverless start call there and does not reliably find one under `src/` |
| `src/handler.py` | Job flow and error mapping |
| `src/server.py` | sgl-omni child process: launch, readiness, shutdown |
| `src/request_schema.py` | Input validation; rejects unsupported parameters |
| `src/lyrics.py` | Puts structure tags on their own lines |
| `src/model_path.py` | Finds the checkpoint across storage strategies |
| `src/audio.py` | WAV probing and transcoding via bundled ffmpeg |
| `src/delivery.py` | Bucket upload or size-checked base64 |
| `src/config.py` | Environment into a validated `Settings` |
| `src/logging_setup.py` | Structured JSON logs |

Design documents: [spec](docs/superpowers/specs/2026-08-17-minimax-music3-runpod-serverless-design.md),
[implementation plan](docs/superpowers/plans/2026-08-17-minimax-music3-runpod-worker.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Worker exits with `no usable checkpoint` | The cached model is not attached or still downloading. Check the endpoint's cached-model setting, or set `MODEL_PATH`. |
| Jobs sit `IN_QUEUE` forever, workers cycle `IDLE`/`UNHEALTHY`, container log is empty | The container is exiting before the handler starts, so nothing ever polls for jobs. The endpoint's worker list is the diagnostic: repeated `start container ... begin` system lines with no container output is the crash-loop signature. `engine exited with code 1` in the JSON log means `sgl-omni serve` itself died — read the forwarded `source: sgl-omni` lines for its traceback. |
| `engine exited with code 1` right after `engine starting`, with `ModuleNotFoundError` from `sgl-omni` | A dependency `--no-deps` left behind. The base image does not carry the whole tree the Dockerfile comment assumes, and only the packages the build's import checks actually reach are verified. Add the missing pin next to `msgpack` in the Dockerfile and extend those checks to import the path that failed. |
| First request is extremely slow | Kernels compiling. Confirm the GPU really is SM89 or SM90a — see the GPU pool warning under [Deploy from GitHub](#deploy-from-github); raise the idle timeout and keep one active worker. |
| `engine did not become ready` | Raise `SERVER_STARTUP_TIMEOUT_S`; check worker logs for the engine's own output, which is forwarded into the JSON log with `source: sgl-omni`. |
| `result_too_large` | Configure the `BUCKET_*` trio, or request a shorter duration or lower bitrate. |
| `unsupported_parameter` | Sampling is fixed in this model; tempo and vocal belong in `prompt`. |
| Same seed gives different audio | Lyrics or caption differ by whitespace. They must be byte-identical. |

## Licence obligations

The model is under the **MiniMax-Music3 Community Licence**, and these are requirements
on whatever product calls this endpoint, not footnotes:

1. A commercial product or service using the model must **display "MiniMax-Music3"
   prominently in its UI**.
2. If aggregate yearly revenue from such products exceeds **20 million USD**, you need
   prior written authorisation from MiniMax (api@minimax.io, subject
   "MiniMax-Music3 licensing - authorization request").
3. If you let third parties generate audio with it, you must implement, maintain and
   periodically review proportionate safeguards against outputs that infringe third-party
   rights, and must not weaken or circumvent them.
4. Use must comply with applicable law and the licence's Acceptable Use Policy.

Inherited licences: Qwen3-8B (Apache 2.0), DiT from stable-audio-tools (MIT), VAE from
DAC (MIT). Full text: [LICENSE on the model page](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE).

The code in this repository is yours to license as you see fit; it does not change the
terms above.
