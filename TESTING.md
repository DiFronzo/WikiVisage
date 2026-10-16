# Testing the WikiVisage face service

The face service (`model-server/`) is the only place ML runs: SCRFD-500M face
detection and ArcFace MobileFaceNet 512D embedding on ONNX Runtime. `app.py` and
`worker.py` reach it over HTTP via `face_client.py` and ship without any ML
libraries.

This guide covers running and exercising the service locally. In production it
runs on Toolforge as its own tool, built with buildpacks — see
[how-to-run-it.md](how-to-run-it.md#face-service-deployment-second-tool). The
Dockerfile here is for local development and CI only; both install the same
fully pinned `requirements.txt` lock (generated from `requirements.in` — see
[Regenerating the Python lock](#7-regenerating-the-python-lock)).

---

## A note on the model weights

KServe conventionally mounts model weights at `/mnt/models`, so you may expect a
`-v $(pwd)/models:/mnt/models` flag. **This service does not need one.** The two
ONNX files are committed in `model-server/weights/` (about 16 MB, ordinary git
blobs, **not** Git LFS) and are copied into the image. On startup `load()`
verifies their SHA-256 against the values pinned in `backend.py` and the process
exits if either file is missing, truncated, an LFS pointer, or different.

The weights are InsightFace's `buffalo_sc` models, which are **non-commercial
research only** — read `model-server/weights/LICENSE-NOTICE`. To check or
re-create them:

```bash
scripts/fetch_weights.sh --check   # verify the committed files (no network)
scripts/fetch_weights.sh           # download from the official release, verify, install
```

---

## 1. Build the container

```bash
cd model-server
docker build -t wikivisage-face-service:local .
```

The build is two-stage: the builder stage carries a compiler toolchain (some
kserve transitive deps, notably `psutil` on arm64, have no prebuilt wheel); the
runtime image carries none.

First build takes a few minutes, mostly downloading the `onnxruntime`,
`opencv-python-headless` and `kserve` wheels. No system packages are needed
beyond `curl` (used by the image's health check).

---

## 2. Run it

```bash
docker run --rm \
  --name wikivisage-face-service \
  -p 8080:8080 \
  --memory=2g --cpus=2 \
  --read-only --tmpfs /tmp \
  --security-opt no-new-privileges:true \
  wikivisage-face-service:local
```

The resource and isolation flags are not optional extras. Before this split,
detection ran in a subprocess hardened with `RLIMIT_AS` / `RLIMIT_CPU` /
`RLIMIT_FSIZE` and a scrubbed environment, because libjpeg, libpng and the ML
runtime parse untrusted bytes. Locally those guarantees come from the container flags
above; on Toolforge from the job's `--mem` / `--cpu` limits and `--mount=none`.
Environment scrubbing is no longer needed at all — the service runs in its own
tool and never holds OAuth, database, or token-encryption secrets in the first
place, which is strictly stronger than deleting them at fork time.

Or with Compose, which applies all of the above for you:

```bash
cd model-server
docker compose up --build
```

Wait for `Model 'wikivisage' loaded (model_id=scrfd500m+arcface_mbf, ...)` in the
logs before sending requests. The line before it, `Weights verified: ...`, shows
the SHA-256 of both weight files; on a bad deploy the process instead logs
`Face backend failed to load, refusing to start` and exits 1, so it never
becomes ready.

### Readiness

```bash
# Server liveness
curl -s http://localhost:8080/v2/health/live

# Model readiness — polled by the container HEALTHCHECK, the Toolforge job
# health check, and face_client.is_healthy()
curl -s http://localhost:8080/v1/models/wikivisage
```

Expected: `{"name":"wikivisage","ready":true}`

### Authentication

Set `WIKIVISAGE_FACE_SERVICE_TOKEN` (at least 32 characters) on the container
to run it the way production does. `model-server/guard.py` then requires
`Authorization: Bearer <token>` on every request except `GET
/v1/models/wikivisage`, and caps request bodies at 40 MiB:

```bash
TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
docker run --rm -p 8080:8080 -e WIKIVISAGE_FACE_SERVICE_TOKEN="$TOKEN" wikivisage-face-service:local

curl -s -o /dev/null -w '%{http_code}\n' localhost:8080/v1/models/wikivisage           # 200, open
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8080/v1/models/wikivisage:predict  # 401
curl -s -H "Authorization: Bearer $TOKEN" localhost:8080/v1/models                     # 200
```

`face_client.is_healthy()` checks readiness *and* `GET /v1/models`, so a missing
or wrong client token reports the service as unhealthy instead of silently
failing every predict call. Without a token the guard only enforces the body
cap, and `model.py` refuses to start at all when it detects Toolforge
(`TOOL_TOOLFORGE_API_URL`) with no token.

---

## 3. Send a prediction

Images travel base64-encoded in the request body — the service performs no
outbound network I/O and will not fetch URLs. `make_input.py` builds the
request for you (standard library only, no install needed):

```bash
cd model-server

# Grab a test image with a face in it (960 px is what the worker sends)
curl -sL -o test.jpg \
  "https://commons.wikimedia.org/wiki/Special:FilePath/Douglas%20adams%20portrait%20cropped.jpg?width=960"

# Build the request body
python3 make_input.py test.jpg > input.json

# Send it
curl -s -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' \
  -d @input.json | python3 -m json.tool
```

Response shape:

```json
{
  "predictions": [
    {
      "id": "0",
      "status": "ok",
      "width": 960,
      "height": 1152,
      "model_id": "scrfd500m+arcface_mbf",
      "dim": 512,
      "faces": [
        {
          "box": {"top": 208, "right": 710, "bottom": 586, "left": 332},
          "score": 0.91,
          "encoding": "<base64 of 512 little-endian float32 values>"
        }
      ]
    }
  ]
}
```

The `encoding` decodes to exactly `dim * 4` = 2048 bytes (512 × little-endian
float32, L2-normalised, so cosine similarity is a dot product) and is written
straight into the `faces.encoding` BLOB column together with `model_id`
(`faces.embedding_model`) and `score` (`faces.det_score`). `face_client` rejects
a response whose `model_id` or `dim` is not the expected one, or whose encodings
have any other width (for example the old 1024-byte float64 format).

Detection notes: faces whose short side is under `WIKIVISAGE_MIN_FACE_PX` (20)
are dropped; a second half-scale pass finds very large faces that SCRFD-500M
misses at full size; at most `WIKIVISAGE_MAX_FACES` (300) are returned, best
score first.

### Batching

```bash
python3 make_input.py a.jpg b.jpg c.jpg > batch.json
curl -s -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' -d @batch.json | python3 -m json.tool
```

`predictions` is always the same length and order as `instances`. One bad image
yields `{"status": "error", "error": "..."}` for that entry only — its siblings
still return results.

### Encoding a known bounding box

This is the path `/api/manual-face` and `/api/update-face-bbox` use when a user
draws a box by hand. Coordinates are CSS order: `TOP,RIGHT,BOTTOM,LEFT`.

```bash
python3 make_input.py --task encode --box 208,710,586,332 test.jpg > encode.json
curl -s -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' -d @encode.json | python3 -m json.tool
```

A drawn box has no landmarks, so the service runs the detector on the box grown
by 50% and aligns the detection that overlaps the box best (IoU >= 0.3). Each
returned face carries `aligned: true` and the detector `score`; if no detection
overlaps enough, the box is just resized and embedded (`aligned: false`,
`score: null`), which gives a weaker embedding.

If the box was measured on an image of a different size than the one you send,
say so with `box_space` and the service scales it to the decoded image
(`--box-space WIDTH,HEIGHT` in `make_input.py`). The app passes the stored
`images.detection_width/height`:

```bash
python3 make_input.py --task encode --box 104,355,293,166 --box-space 480,576 test.jpg
```

### Error handling

```bash
# Malformed base64 -> per-instance error, HTTP 200
curl -s -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"id":"x","task":"detect","image":"not-base64!!"}]}'

# Missing 'instances' -> envelope-level rejection, HTTP 400
curl -s -i -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' -d '{"foo":"bar"}'
```

The distinction matters to the worker: a per-instance error is treated as
terminal and marks the image `status='error'`, while a transport or envelope
failure is treated as transient and leaves the image `pending` for the next
poll cycle.

---

## 4. Point the app and worker at it

```bash
export WIKIVISAGE_FACE_SERVICE_URL=http://localhost:8080

python app.py
python worker.py --worker-id local-1
```

The worker probes the service once at startup and logs loudly if it is
unreachable. `/health` also reports it:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# {"status":"healthy","database":"connected","limiter":"...",
#  "face_service":"reachable","degraded":false}
```

### Client configuration

| Variable | Default | Purpose |
|---|---|---|
| `WIKIVISAGE_FACE_SERVICE_URL` | `http://localhost:8080` | Base URL of the service |
| `WIKIVISAGE_FACE_SERVICE_MODEL` | `wikivisage` | KServe model name in the URL path |
| `WIKIVISAGE_FACE_SERVICE_TOKEN` | *(unset)* | Bearer token; must match the server's |
| `WIKIVISAGE_FACE_SERVICE_BATCH_SIZE` | `8` | Max images per request |
| `WIKIVISAGE_FACE_SERVICE_MAX_BATCH_BYTES` | `25165824` (24 MiB) | Max raw bytes per request |
| `WIKIVISAGE_FACE_SERVICE_TIMEOUT` | `30` | Base read timeout (seconds) |
| `WIKIVISAGE_FACE_SERVICE_TIMEOUT_PER_IMAGE` | `15` | Added per image in the batch |
| `WIKIVISAGE_FACE_MODEL_ID` | `scrfd500m+arcface_mbf` | The only `model_id` the client accepts |
| `WIKIVISAGE_FACE_EMBEDDING_DIM` | `512` | Embedding length the client expects |
| `WIKIVISAGE_DETECTION_WIDTH` | `960` | Commons thumbnail width the worker detects on and the app downloads for manual drawing (must be a standard Commons step) |

Batches split on whichever of the count/byte budgets is hit first, so a few
large images cannot produce a huge request body.

### Server configuration

| Variable | Default | Purpose |
|---|---|---|
| `WIKIVISAGE_FACE_SERVICE_TOKEN` | *(unset)* | Required bearer token (>= 32 chars); mandatory on Toolforge |
| `WIKIVISAGE_MAX_REQUEST_BYTES` | `41943040` (40 MiB) | Request body cap, enforced before parsing |
| `WIKIVISAGE_FACE_WORKERS` | `2` | Executor threads running inference on the shared (thread-safe) ONNX sessions; set to the vCPU count |
| `WIKIVISAGE_MAX_BATCH_SIZE` | `32` | Reject requests with more instances |
| `WIKIVISAGE_MAX_IMAGE_BYTES` | `20971520` (20 MiB) | Per-image decoded size ceiling |
| `WIKIVISAGE_MAX_IMAGE_PIXELS` | `100000000` | Pixel-area ceiling before decoding |
| `WIKIVISAGE_MAX_BOXES` | `64` | Max boxes per `encode` instance |
| `WIKIVISAGE_WEIGHTS_DIR` | `<model-server>/weights` | Directory holding the two ONNX files |
| `WIKIVISAGE_DET_MAX_SIDE` | `1280` | Longest image side fed to the detector |
| `WIKIVISAGE_MIN_FACE_PX` | `20` | Drop detected faces whose short side is smaller |
| `WIKIVISAGE_MAX_FACES` | `300` | Per-image cap on returned faces |
| `WIKIVISAGE_DET_COARSE_MIN_SIDE` | `640` | Long side from which the half-scale pass runs (`0` disables it) |

---

## 5. Run the Python test suite

The main suite never touches the network — `face_client` is mocked throughout.

```bash
# Unit tests only (what the CI "test" job runs). The face-service unit and
# contract tests skip here: they need onnxruntime/opencv/kserve or a service.
pytest tests/ -q

# Unit + integration (needs local MariaDB)
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# Just the client, and the request guard (stdlib only, no kserve needed)
pytest tests/test_face_client.py tests/test_face_service_guard.py -v

# The face-service unit tests: backend.py (fake ONNX sessions plus the real
# weights) and model.py (the KServe envelope with a scripted backend). They
# import model-server/ directly, so install its requirements first.
pip install -r model-server/requirements.txt -r requirements-dev.txt
pytest tests/test_face_service_backend.py tests/test_face_service_model.py -v

# Contract tests against a running container. With a token set on both sides
# the auth tests run too; this is how CI runs them.
WIKIVISAGE_FACE_SERVICE_CONTRACT=1 WIKIVISAGE_FACE_SERVICE_TOKEN="$TOKEN" \
  pytest tests/test_face_service_contract.py -v
```

Tests that need a **real face** read `tests/data/portrait_sample.jpg` (committed;
provenance in `tests/data/README.md` — a close-cropped public-domain portrait
whose ~514x756 px face at 960 px exercises the half-scale pass). Without it they
skip, unless `WIKIVISAGE_REQUIRE_SAMPLE` is an explicit affirmative
(`1`/`true`/`yes`/`on` — CI sets `1`), in which case they fail.
`WIKIVISAGE_TEST_SAMPLE_IMAGE=/path/to/photo.jpg` points them at another photo.

CI also checks that the service log has no ONNX Runtime `VerifyOutputSizes`
warnings after the contract tests (the detector's declared output shapes come
from a 640x640 trace; `backend._make_session` sets `log_severity_level = 3` so
other input sizes do not flood the log).

---

## Verified behaviour

Checked locally against a freshly started service process (real KServe 0.15.2,
real weights, Python 3.11, Apple-silicon laptop; **not** inside Docker or the
Toolforge buildpack image):

| Check | Result |
|---|---|
| Weights | Both SHA-256 values match the pinned ones; the log shows them at startup |
| Same person, 2 official portraits | Cosine similarity 0.73 |
| Different people (Obama, Biden, Harris) | Cosine similarity -0.02 – 0.15 |
| Family portrait | 4 faces |
| Large close-up (532 px face on a 960x1307 image) | Missed at full size by SCRFD-500M, found by the half-scale pass |
| Sloppy hand-drawn box around a face | Landmarks recovered (`aligned: true`), embedding 0.957 cosine to the detected face |
| Same box, unaligned fallback | 0.69 cosine to the aligned embedding — why `aligned` is reported |
| `box_space` | A box given in a 2x larger space embeds identically to the same region given directly |
| Per-instance error isolation | A corrupt image in a batch returns an error for that entry only |
| Unreachable service | Raises `FaceServiceUnavailable`, not a crash |
| Service log | No `VerifyOutputSizes` warnings after the contract suite |

Not measured: behaviour on Toolforge CPUs (expect 2–3x slower than a laptop),
recall on hard Commons images against your own labelled faces (that is the
plan's Phase 0), and encodings across machines.

### Embeddings across machines

The ONNX models run in float32, so the same image embedded on different CPU or
ONNX Runtime builds can differ in the last digits. On one machine the path is
deterministic against itself (a unit test asserts it). What *would* invalidate
stored embeddings is a different set of weights, a changed alignment template,
or changed preprocessing in `backend.py`; every face therefore records the
`embedding_model` that produced it, and the worker only compares faces of the
current model.

---

## 6. Running without Docker

```bash
cd model-server
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python model.py --model_name wikivisage --http_port 8080
```

Python 3.11 is required: `kserve==0.15.2` runs on FastAPI/uvicorn and needs a
modern interpreter. Do not downgrade to the `kserve` 0.8.x line — it pins
`numpy~=1.19`, `ray==1.9.0`, and `protobuf==3.19.1`, none of which have wheels
for or build on Python 3.11+.

---

## 7. Regenerating the Python lock

`requirements.txt` is a fully pinned transitive lock generated from
`requirements.in`. Edit `requirements.in`, then regenerate:

```bash
cd model-server
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  requirements.in -o requirements.txt
```

The target is the buildpack's Python and platform (Python 3.11, x86_64
manylinux). Never edit `requirements.txt` by hand and never reduce it to
top-level-only pins — that is how transitive versions drifted under a file
labelled "pinned". Both the Docker image and the Toolforge buildpack install
the generated lock with a plain `pip install -r requirements.txt`.

---

## 8. Building the production image locally (buildpacks)

Toolforge builds the face service with Cloud Native Buildpacks, not the
Dockerfile. To reproduce that build — worth doing after touching
`requirements.txt` or the `Procfile` — run `pack` with Toolforge's builder
against a copy of `model-server/`:

```bash
docker run --rm -u root -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/model-server":/workspace -w /workspace buildpacksio/pack \
  build wikivisage-face-bp --platform linux/amd64 \
  --builder tools-harbor.wmcloud.org/toolforge/heroku-builder:24_0.21.8 \
  --buildpack heroku/python --buildpack heroku/procfile

docker run --rm -p 8082:8000 --entrypoint web \
  -e WIKIVISAGE_FACE_SERVICE_TOKEN="$TOKEN" wikivisage-face-bp
```

The service needs no system packages (there is no `project.toml`), so the
deb-packages buildpack is not requested. The weights directory is part of the
build context, so it ends up in the image.

---

## Troubleshooting

**The service exits at startup with `Face backend failed to load, refusing to
start` / `WeightsError`** — a weights file is missing or its SHA-256 does not
match `backend.WEIGHT_SHA256`. The message names the file and shows both hashes.
Typical causes: the files were replaced by Git LFS pointer files, a download was
truncated, or someone swapped the model. Run `scripts/fetch_weights.sh --check`;
`scripts/fetch_weights.sh` restores the official files. Because the process
never reports ready, `deploy.yml`'s readiness wait fails and no worker is
restarted against it.

**`NoModelReady` on startup** — `load()` raised before setting `self.ready`.
Check the traceback above the error.

**Worker exits at startup with "Face service runs a different embedding model"**
— the service reports a `model_id` or `dim` other than `scrfd500m+arcface_mbf` /
512 (a stale service, or `WIKIVISAGE_FACE_MODEL_ID` set differently). Redeploy the
face tool first; the worker refuses to store vectors from another model.

**Every image stays `pending` and the log says "face encoding is N bytes,
expected 2048"** — an old face service (1024-byte float64 encodings) is still
answering. Same fix: deploy the face tool before the workers.

**HTTP 401 from the service** — the client and server hold different
`WIKIVISAGE_FACE_SERVICE_TOKEN` values, or the client has none. The web app
logs `rejected WIKIVISAGE_FACE_SERVICE_TOKEN` and `/health` reports
`face_service: unreachable`.

**Worker logs "Face service not ready"** — the worker probes once at startup and
continues regardless; images stay `pending` and retry every poll cycle. Confirm
`WIKIVISAGE_FACE_SERVICE_URL` and that `curl $URL/v1/models/wikivisage` returns
`ready: true`.

**Detection is slow** — inference is CPU-bound. Throughput scales with
`WIKIVISAGE_FACE_WORKERS` up to the CPU count (the Toolforge job has 2 vCPUs);
beyond that add replicas (`--replicas N` on the Toolforge job). The worker
downloads Commons thumbnails at `?width=960`, about 3x the bytes of the former
500 px, and detection on them is correspondingly heavier.

**Faces are missing on a close-up portrait** — SCRFD-500M misses very large
faces at full size; the half-scale pass covers them when the image's long side is
at least `WIKIVISAGE_DET_COARSE_MIN_SIDE` (640). If you disabled it (`0`) or set
`WIKIVISAGE_DET_MAX_SIDE` very high, that is the cause. Tiny faces below
`WIKIVISAGE_MIN_FACE_PX` (20) are dropped on purpose.

**Logs flooded with `VerifyOutputSizes` warnings** — the ONNX session's log
level was changed; `backend._make_session` must keep `log_severity_level = 3`.

**OOM / container killed** — lower `WIKIVISAGE_FACE_SERVICE_BATCH_SIZE` on the
client, `WIKIVISAGE_MAX_IMAGE_PIXELS` on the server, or `WIKIVISAGE_FACE_WORKERS`
(each concurrent decode of a large image holds a full RGB copy). A crowd photo
is embedded in chunks of 32 faces to bound activation memory.
