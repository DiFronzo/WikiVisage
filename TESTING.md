# Testing the WikiVisage face service

The face service (`model-server/`) owns the only dlib dependency in the stack.
`app.py` and `worker.py` reach it over HTTP via `face_client.py` and ship
without any ML libraries.

This guide covers running and exercising the service locally. In production it
runs on Toolforge as its own tool, built with buildpacks — see
[how-to-run-it.md](how-to-run-it.md#face-service-deployment-second-tool). The
Dockerfile here is for local development and CI only; both install the same
`requirements.txt`.

---

## A note on `/mnt/models`

KServe conventionally mounts model weights at `/mnt/models`, so you may expect
a `-v $(pwd)/models:/mnt/models` flag. **This service does not need one.** The
dlib weights ship inside the `face_recognition_models` wheel and are already in
the image after `pip install` — `load()` just imports the package and warms it.
Reintroduce a mount only if you ever swap in externally-managed weights.

---

## 1. Build the container

```bash
cd model-server
docker build -t wikivisage-face-service:local .
```

The build is two-stage: the builder stage carries a compiler toolchain (some
kserve transitive deps, notably `psutil` on arm64, have no prebuilt wheel); the
runtime image carries none.

First build takes a few minutes, mostly downloading the `dlib-bin` wheel.

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
`RLIMIT_FSIZE` and a scrubbed environment, because dlib, libjpeg, and libpng
parse untrusted bytes. Locally those guarantees come from the container flags
above; on Toolforge from the job's `--mem` / `--cpu` limits and `--mount=none`.
Environment scrubbing is no longer needed at all — the service runs in its own
tool and never holds OAuth, database, or token-encryption secrets in the first
place, which is strictly stronger than deleting them at fork time.

Or with Compose, which applies all of the above for you:

```bash
cd model-server
docker compose up --build
```

Wait for `Model 'wikivisage' loaded` in the logs before sending requests.

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

# Grab a test image with a face in it
curl -sL -o test.jpg \
  "https://commons.wikimedia.org/wiki/Special:FilePath/Douglas%20adams%20portrait%20cropped.jpg?width=500"

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
      "width": 500,
      "height": 600,
      "faces": [
        {
          "box": {"top": 104, "right": 355, "bottom": 293, "left": 166},
          "encoding": "<base64 of 1024 raw float64 bytes>"
        }
      ]
    }
  ]
}
```

The `encoding` decodes to exactly 1024 bytes (128 × float64) and is written
straight into the `faces.encoding` BLOB column.

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
python3 make_input.py --task encode --box 104,355,293,166 test.jpg > encode.json
curl -s -X POST http://localhost:8080/v1/models/wikivisage:predict \
  -H 'Content-Type: application/json' -d @encode.json | python3 -m json.tool
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

Batches split on whichever of the count/byte budgets is hit first, so a few
large images cannot produce a huge request body.

### Server configuration

| Variable | Default | Purpose |
|---|---|---|
| `WIKIVISAGE_FACE_SERVICE_TOKEN` | *(unset)* | Required bearer token (>= 32 chars); mandatory on Toolforge |
| `WIKIVISAGE_MAX_REQUEST_BYTES` | `41943040` (40 MiB) | Request body cap, enforced before parsing |
| `WIKIVISAGE_FACE_WORKERS` | `1` | Threads decoding images. dlib itself always runs one call at a time |
| `WIKIVISAGE_MAX_BATCH_SIZE` | `32` | Reject requests with more instances |
| `WIKIVISAGE_MAX_IMAGE_BYTES` | `20971520` (20 MiB) | Per-image decoded size ceiling |
| `WIKIVISAGE_MAX_IMAGE_PIXELS` | `100000000` | Pixel-area ceiling before dlib |
| `WIKIVISAGE_MAX_BOXES` | `64` | Max boxes per `encode` instance |

---

## 5. Run the Python test suite

The suite never touches the network — `face_client` is mocked throughout.

```bash
# Unit tests only (what CI runs)
pytest tests/ -q

# Unit + integration (needs local MariaDB)
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# Just the client, and the request guard (stdlib only, no kserve needed)
pytest tests/test_face_client.py tests/test_face_service_guard.py -v

# Contract tests against a running container. With a token set on both sides
# the auth tests run too; this is how CI runs them.
WIKIVISAGE_FACE_SERVICE_CONTRACT=1 WIKIVISAGE_FACE_SERVICE_TOKEN="$TOKEN" \
  pytest tests/test_face_service_contract.py -v
```

---

## Verified behaviour

The following was measured against a locally built container, comparing the
new remote path to the old in-process path on the same image.

| Check | Result |
|---|---|
| Bounding boxes | **Identical** — `[(155, 295, 229, 220)]` from both paths |
| Image dimensions | Identical (500 x 718) |
| Encoding width | Exactly 1024 bytes (128 x float64) |
| Encoding values | Agree to **7.7e-07 L2** (see below) |
| Per-instance error isolation | A corrupt image in a 3-image batch returns `FaceServiceRejected` for that entry only; siblings still return faces |
| Byte-aware batching | 20 images at `BATCH_SIZE=8` split into `[8, 8, 4]` |
| Too-small bbox | `encode_known_face` returns `None` (drives the HTTP 422 path) |
| Unreachable service | Raises `FaceServiceUnavailable`, not a crash |
| Malformed envelope | HTTP 400 |

### On encoding equivalence

The encodings are **not bit-identical across machines**, and that is expected.
dlib's ResNet runs in float32, so CPU and BLAS differences change the last
couple of digits:

```
local vs local (same process, rerun) : max delta 0.000e+00
local vs remote container            : max delta 2.086e-07
float32 epsilon                      : 1.192e-07
L2 distance local <-> remote         : 7.736e-07
classification threshold             : 0.6
ratio (drift / threshold)            : 1.29e-06
```

The path is bit-deterministic against itself, so nothing in this service
introduces nondeterminism. The residual is platform float32 drift, about one
part in a million of the decision margin — roughly a millionth of the distance
between two photos of the same person. It is not specific to this
architecture: the same drift exists between any two machines, including
between two Toolforge nodes, and existed before the split.

What *would* invalidate stored encodings is changing the `dlib-bin` version or
the `face_recognition_models` weights, or changing the dlib calls (upsampling,
the 5-point landmark model, `num_jitters`). All three are pinned and commented
in `model-server/requirements.txt` and `model.py`.

### Dropping the `face_recognition` package

The service no longer depends on `face_recognition`: its metadata requires the
source-only `dlib`, which a buildpack cannot skip the way the Dockerfile's
`--no-deps` used to. `model.py`'s `_Dlib` makes the same calls directly. This
was checked by running the old image (with the package) and the new one side
by side on the same bytes:

| Check | Result |
|---|---|
| 40 real Commons images, 67 faces — boxes | **Identical** |
| Same — `detect` encodings | **Byte-identical** |
| Same — `encode` task with known boxes | **Byte-identical** |
| Truncated JPEG (70% of the file) | Identical (both decode it) |
| Buildpack image (amd64) vs old image (arm64) | 6.5e-07 – 7.7e-07 L2, the usual cross-platform drift |

Comparing against encodings stored in March also showed three faces off by
0.06–0.2 L2 — but the old package shows exactly the same gap on those images.
Commons now serves different thumbnail bytes for them than it did then; it is
input drift, not a code change.

To re-run this comparison yourself, the checks above are in the test suite
(`tests/test_face_client.py`); the cross-path numeric comparison requires a
running container and a local dlib install, so it is not part of CI.

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

## 7. Building the production image locally (buildpacks)

Toolforge builds the face service with Cloud Native Buildpacks, not the
Dockerfile. To reproduce that build — worth doing after touching
`requirements.txt`, `project.toml` or the `Procfile` — run `pack` with
Toolforge's builder against a copy of `model-server/`:

```bash
docker run --rm -u root -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/model-server":/workspace -w /workspace buildpacksio/pack \
  build wikivisage-face-bp --platform linux/amd64 \
  --builder tools-harbor.wmcloud.org/toolforge/heroku-builder:24_0.21.8 \
  --buildpack heroku/deb-packages --buildpack heroku/python --buildpack heroku/procfile

docker run --rm -p 8082:8000 --entrypoint web \
  -e WIKIVISAGE_FACE_SERVICE_TOKEN="$TOKEN" wikivisage-face-bp
```

The deb-packages buildpack must be named explicitly here; Toolforge injects it
automatically. The image is about 1 GB.

---

## Troubleshooting

**`NoModelReady` on startup** — `load()` raised before setting `self.ready`.
Almost always a missing `face_recognition_models` install. Check the traceback
above the error.

**`could not read Username for 'https://github.com'` during build** — you are
building an older revision that still pins the `face-recognition` Git fork
(`github.com/DiFronzo/face_recognition`). That repository has been deleted and
now 404s, so pip's `git clone` falls through to an interactive credential
prompt. The current `requirements.txt` does not use `face_recognition` at all.

**Build tries to compile `dlib` (CMake errors)** — something reintroduced the
`face_recognition` package, whose metadata requires the `dlib` sdist. Keep it
out of `requirements.txt`; `model.py` calls dlib directly.

**HTTP 401 from the service** — the client and server hold different
`WIKIVISAGE_FACE_SERVICE_TOKEN` values, or the client has none. The web app
logs `rejected WIKIVISAGE_FACE_SERVICE_TOKEN` and `/health` reports
`face_service: unreachable`.

**`pkg_resources is deprecated` warning** — harmless. `face_recognition_models`
still uses it, which is why `setuptools<81` is pinned in the service's
requirements.

**Worker logs "Face service not ready"** — the worker probes once at startup and
continues regardless; images stay `pending` and retry every poll cycle. Confirm
`WIKIVISAGE_FACE_SERVICE_URL` and that `curl $URL/v1/models/wikivisage` returns
`ready: true`.

**Detection is slow** — dlib HOG is CPU-bound and, because dlib is not
thread-safe, each replica detects one image at a time. Add replicas
(`--replicas N` on the Toolforge job) rather than raising
`WIKIVISAGE_FACE_WORKERS`, which only overlaps image decoding. The worker
downloads Commons thumbnails at `?width=500`, so images are small.

**Server exits with SIGSEGV (exit 139)** — two threads were inside dlib at once.
Every dlib call must stay under `_dlib_lock` in `model.py`.

**OOM / container killed** — lower `WIKIVISAGE_FACE_SERVICE_BATCH_SIZE` on the
client, or `WIKIVISAGE_MAX_IMAGE_PIXELS` on the server. A single dlib call on a
large image can transiently allocate several hundred MB.
