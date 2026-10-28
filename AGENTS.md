# AGENTS.md — WikiVisage

## Overview

Active-learning Flask app for Wikimedia Commons. Users classify faces via yes/no UI, training a centroid-distance model that auto-classifies remaining faces. Approved matches are written as P180 (depicts) SDC claims to Commons via OAuth — triggered manually by the user from the project detail page. Hosted on Wikimedia Toolforge (Kubernetes, no GPU).

Face detection and 512D embedding (SCRFD-500M + ArcFace MobileFaceNet on ONNX Runtime) run in a **separate face service** (`model-server/`), deployed as its own Toolforge tool and reached over HTTPS. The web and worker processes carry no ML dependencies at all.


## Structure

```
WikiVisage/
├── app.py              # Flask web app: OAuth (PKCE), routes, classification API, CSP (~4900 lines; the big routes are thin orchestration over named helpers)
├── worker.py           # Background pipeline: crawl, remote detect, infer (~3900 lines). `write_sdc_claims` is a thin entry over `_SdcRun`; `main` is a sequence of small functions (see Worker)
├── face_client.py      # HTTP client for the face service. ZERO ML deps — shared by app.py + worker.py
├── liftwing_client.py  # Experimental LiftWing Qwen metadata pre-filter client (fail-open, circuit breaker, completion probe, optional bearer token) — worker-only
├── config.py           # Shared config: WAKE_FILE_PATH, HEARTBEAT_FILE_DIR, COMMONS_THUMB_STEPS, DETECTION_WIDTH, LIFTWING_PROBE_INTERVAL, SESSION_IDLE_DAYS, SESSION_MAX_AGE_DAYS, WORKER_STALE_SECONDS, WORKER_RESTART_GRACE_SECONDS, PROJECT_VERSION
├── token_crypto.py     # Fernet encrypt/decrypt helpers for OAuth tokens at rest (~125 lines)
├── redis_lock.py       # Best-effort Redis single-flight lock; used to serialize OAuth refresh (~100 lines)
├── oauth_refresh.py    # The OAuth refresh-token protocol shared by app.py and worker.py: Redis single-flight lock, DB compare-and-swap, re-read after a lost race. DB access and the token exchange are passed in as callables
├── api_errors.py       # `ApiError`: how JSON routes reject a request (status + body). Its own module on purpose: the Flask handler matches by class identity, and test fixtures reload `app`
├── database.py         # MariaDB connection pool with retry logic, plus `open_dedicated_connection()` for the worker's heartbeat thread (~570 lines)
├── model-server/       # Face detection service — the ONLY place ML runs
│   ├── model.py        # KServe V1 predictor: WikiVisageModel(kserve.Model), async predict, box_space scaling
│   ├── backend.py      # FaceBackend protocol + SCRFD-500M/ArcFace-MBF on ONNX Runtime; weight SHA-256 pins. No KServe dependency
│   ├── guard.py        # Stdlib-only ASGI middleware: bearer token + 40 MiB body cap (public-facing)
│   ├── weights/        # det_500m.onnx, w600k_mbf.onnx (plain git blobs, NOT LFS), SHA256SUMS, LICENSE-NOTICE (non-commercial terms)
│   ├── requirements.in # Top-level pins (kserve, onnxruntime, opencv-headless, Pillow, numpy); the source of requirements.txt
│   ├── requirements.txt # Fully pinned transitive lock (uv pip compile, Python 3.12/x86_64-manylinux_2_28) — plain pip, no system packages
│   ├── Procfile        # Toolforge buildpack entrypoint: `web: python model.py ... --http_port 8000`
│   ├── .python-version # 3.12 (Heroku Python buildpack)
│   ├── Dockerfile      # Local dev + CI only. Two-stage, python:3.12-slim-bookworm, uid 10001
│   ├── docker-compose.yml # Local harness with memory/CPU ceilings + read-only rootfs
│   ├── make_input.py   # Builds KServe request JSON from local image files
│   └── .dockerignore
├── scripts/
│   ├── check_coverage_floors.py # CI gate: per-module coverage floors (a ratchet; raise, never lower)
│   ├── fetch_weights.sh   # Re-create/verify model-server/weights/ from the official InsightFace release (zip SHA, then both file SHAs)
│   ├── check_no_lfs.sh    # CI guard: no .gitmodules, no LFS filter on the weights, no LFS pointer files
│   └── eval_prefilter.py  # Offline readout for the LiftWing shadow verdicts (keep recall, skip rate, detector signal, sample reasons)
├── healthcheck.sh      # Toolforge liveness health check script (per-worker heartbeat file age check; the file is touched by the worker's heartbeat thread)
├── schema.sql          # DDL for 10 tables: users, projects, images, faces, user_stats, sdc_claims, project_members, worker_heartbeats, service_status, plus the legacy single-row worker_heartbeat (still created, no longer written or read; dropped in a later release)
├── migrate.py          # Idempotent schema migration with --reset flag; one-time embedding-model wipe (apply_embedding_model_upgrade)
├── pyproject.toml      # Project config: Ruff linter/formatter rules, pytest config, markers, version
├── requirements.in     # Top-level web/worker requirements; the source of requirements.txt
├── requirements.txt    # Fully pinned universal lock (uv pip compile --universal, constrained by model-server/requirements.txt). NO ML dependencies. Never hand-edit
├── requirements-dev.txt # Dev/test deps: pytest, pytest-cov, ruff, mypy + stubs (includes requirements.txt)
├── requirements-browser.txt # Optional Playwright browser tests (includes requirements-dev.txt); never installed in production
├── babel.cfg           # pybabel extraction config (explicit file list, excludes venv)
├── messages.pot        # Extracted translatable strings template
├── CONTRIBUTING.md     # Contributor guide: setup, conventions, i18n, schema changes
├── TESTING.md          # Face service: build, run, verify, troubleshoot
├── translations/       # i18n translation files (Flask-Babel / gettext)
│   ├── en/LC_MESSAGES/ # English (identity: msgstr = msgid)
│   ├── nb/LC_MESSAGES/ # Norwegian Bokmål
│   ├── es/LC_MESSAGES/ # Spanish
│   └── fr/LC_MESSAGES/ # French
├── tests/              # Hybrid test suite: unit + integration + contract tests
│   ├── __init__.py
│   ├── conftest.py     # Integration fixture infrastructure; `_make_embedding(seed, dim)` (unit-norm float32)
│   ├── data/           # Public-domain portrait sample + provenance README used by the real-face tests
│   ├── test_app_*.py   # Web-app tests by domain: auth, classify_api, commons, face_drawing, membership, misc, project_pages, sdc, system, undo_reclassify (one 13,000-line file before). `_app_preamble.py` does the env/app import once, `_app_support.py` holds helpers shared by several files
│   ├── test_app_helpers.py # Direct tests of the helpers extracted from the big routes (box parsing, undo decision table, reclassify, image choice, project creation)
│   ├── test_config.py  # The shared User-Agent
│   ├── test_gallery_filter.py # The non-visual-media filter equals the old regex on real rows (integration)
│   ├── test_oauth_refresh.py # The refresh protocol: lock, compare-and-swap, peer wait, retry
│   ├── test_worker_sdc.py # write_sdc_claims end to end through a scripted DB + Commons API (every error path, the removal phase, the blocked-page back-off)
│   ├── test_worker_loop.py # The worker's main loop pieces: startup, supervisor, inference drain, sleep, shutdown
│   ├── test_worker_heartbeat.py # Heartbeat thread: per-worker row upsert, progress watchdog, planned-stop marker, shutdown order (one integration test against the real table)
│   ├── test_cookie_consent.py # `POST /cookie-consent`, the `locale` cookie flags, the cookie banner/footer markup
│   ├── test_gallery_thumb_browser.py # Chromium/WebKit tests of the gallery thumbnails (zoom cap, face marker, hi-res swap and its fallback, resize relayout)
│   ├── test_session_persistence_browser.py # Real HTTP + persistent Chromium/WebKit profile: the session cookie survives a browser restart and the OAuth state round-trips
│   ├── test_bbox_editor_browser.py # Chromium/WebKit tests of the rendered bbox editor (touch identity, cancel, resize, scrolling)
│   ├── test_database.py # Unit + integration tests
│   ├── test_face_client.py # Unit tests for the face service client (dim/model_id validation, box_space, startup probe)
│   ├── test_face_service_backend.py # Unit tests for model-server/backend.py (fake ONNX sessions + real weights; needs onnxruntime/cv2)
│   ├── test_face_service_model.py # Unit tests for model-server/model.py (envelope, box_space, load/exit; needs kserve + onnx stack)
│   ├── test_face_service_contract.py # Contract tests against a live service (some need a token, some need the portrait sample)
│   ├── test_face_service_guard.py # Unit tests for model-server/guard.py (loaded by path)
│   ├── test_liftwing_client.py # Unit tests for the LiftWing pre-filter client (parsing, fail-open, circuit breaker, probe, auth)
│   ├── test_eval_prefilter.py # Unit tests for scripts/eval_prefilter.py readout math
│   ├── test_migrate.py # Unit + integration tests, incl. the one-time wipe and its re-run gate
│   ├── test_token_crypto.py # Unit tests
│   ├── test_translations.py # Unit test: committed .mo catalogs match their .po sources (freshness)
│   ├── test_worker.py  # Unit + integration tests
│   └── test_security_round2.py # Round-2 security regression tests (CSP, PKCE, single-flight, /health, M3/M4)
├── static/             # Static assets
│   ├── wikivisage-logo.svg        # Full logo with text
│   ├── wikivisage-logo-notext.svg # Logo icon only
│   ├── wikivisage-common.js       # Shared JS helpers: snapThumbWidth(), commonsThumbUrl(), bbox geometry, gallery face view (computeFaceView())
│   └── view-it-tool.png           # View it! Tool icon (local copy)
├── .github/
│   ├── dependabot.yml  # Weekly update PRs: root uv, model-server uv (grouped), github-actions
│   └── workflows/
│       ├── audit.yml   # Dependency audit: pip-audit + uv lock freshness for both locks (dependency changes, weekly, manual)
│       ├── ci.yml      # CI: Ruff + mypy + no-LFS check, pytest on Python 3.12/3.15.1, MariaDB integration + coverage gates, browser, face-service unit/contract jobs
│       └── deploy.yml  # CD: Release-triggered Toolforge deploy via SSH (2 workers + health checks)
├── Procfile            # web: gunicorn (2 gthread workers x 4 threads, module-level app), worker: python -u worker.py
├── how-to-run-it.md    # Toolforge deployment guide
├── test-local.md       # Local development setup guide
├── LICENSE             # MIT license (does NOT cover model-server/weights; see its LICENSE-NOTICE)
└── .env                # Local dev env vars (gitignored)
```

> `templates/` (10 Jinja2 files, all extending `base.html`) is unchanged; see the Templates convention section.
>
> **No `project.toml` anywhere.** The root one was deleted earlier (it only installed `libopenblas0` / `liblapack3` for dlib), and `model-server/project.toml` is gone too: onnxruntime and opencv-python-headless ship self-contained wheels, so neither tool needs system deb packages.


## Architecture — Three Processes

```
+---------------------+     +---------------------+
|   Flask Web App     |     | Background Worker(s)|
|      (app.py)       |     |     (worker.py)     |
|  no ML deps         |     |  no ML deps         |
+----------+----------+     +----------+----------+
           |                           |
           |  face_client.py (HTTP, base64 image bytes)
           +-------------+-------------+
                         v
              +---------------------+
              |    Face Service     |
              | (model-server/)     |
              |  KServe + ONNX RT   |
              +---------------------+
           |                           |
           +-------------+-------------+
                         v
                   +-----------+
                   |  MariaDB  |
                   | (ToolsDB) |
                   +-----------+
```

Only the face service runs ML (ONNX Runtime + OpenCV). Both Toolforge processes
import `face_client.py`, which has zero ML dependencies — a test asserts that
importing `face_client` or `worker` loads no `onnxruntime`, `cv2`, `dlib` or
`kserve` module. dlib and `face_recognition` are gone from the repository.

### Face Service (model-server/)

Standalone KServe V1 model server. `WikiVisageModel(kserve.Model)` implements
`load()` and `async def predict(payload, headers=None, response_headers=None)`.

- **Images arrive base64-encoded, never as URLs.** The service performs no
  outbound network I/O, which removes SSRF surface and makes it deployable
  behind a default-deny egress policy.
- **ONNX Runtime sessions are thread-safe; there is no lock.** `predict` stays on
  the event loop only for parsing and dispatches each instance to a
  `ThreadPoolExecutor` (`FACE_WORKERS`, default 2 = the job's `--cpu 2`) via
  `run_in_executor`. One `InferenceSession` per model is shared by all threads
  (`intra_op_num_threads = inter_op_num_threads = 1`, sequential execution, CPU
  provider, `log_severity_level = 3` to silence ORT's per-call `VerifyOutputSizes`
  warning for non-640x640 detector inputs); `cv2.setNumThreads(1)` is called at
  import. Throughput scales with `FACE_WORKERS`. Do not add multiprocessing.
- **Weights are verified at startup, fail-closed.** `backend.verify_weights` pins
  the SHA-256 of `det_500m.onnx` and `w600k_mbf.onnx` (the values also live in
  `weights/SHA256SUMS`; a test keeps them equal). A missing, truncated, LFS-pointer
  or swapped file makes `load()` raise `WeightsError` and `main()` exit 1, so the
  service never reports ready and `deploy.yml`'s readiness wait fails before any
  worker is restarted. One log line shows both hashes on success.
- **Models and pipeline** (`backend.py`): SCRFD-500M with 5 landmarks, then
  ArcFace MobileFaceNet (512D). Detect: Pillow decode (unchanged: no EXIF
  rotation, `LOAD_TRUNCATED_IMAGES`), downscale so the long side is at most
  `WIKIVISAGE_DET_MAX_SIDE` (1280), pad bottom/right to a multiple of 32, `(x-127.5)/128`,
  decode 3 strides x 2 anchors, score 0.5, NMS 0.4, drop faces with short side below
  `WIKIVISAGE_MIN_FACE_PX` (20), return `score` per face. **SCRFD-500M misses very large
  faces** (measured: a 532 px face on a 960x1307 portrait found nothing at full size), so
  when the working image's long side is at least `WIKIVISAGE_DET_COARSE_MIN_SIDE` (640) a
  second half-scale pass runs and contributes faces of at least 240 px; results are merged
  by NMS. Align: Umeyama similarity to the ArcFace template, `cv2.warpAffine`, 112x112,
  `(x-127.5)/127.5`; embeddings are L2-normalised float32 and computed in chunks of 32.
- **`encode` task (drawn box, no landmarks):** the detector runs on the box grown by 50%
  (clipped); the detection with the highest IoU (>= 0.3) is embedded aligned, otherwise the
  box is resized to 112x112 and embedded unaligned (`aligned: false`, noticeably weaker).
  The minimum face size does not apply here. An optional `box_space: {width, height}` on the
  instance says which pixel space the boxes were measured in; the service scales them to the
  image it actually decoded (this fixed a latent mismatch: the app used to send boxes
  measured at the stored detection size against a differently sized download).
- **Response:** per prediction `id, status, width, height, model_id, dim, faces`; per face
  `box`, `score` (null for fallback) and `encoding` (base64 of `dim` little-endian float32);
  `encode` faces also carry `aligned`.
- **Pluggable backend.** `FaceBackend` (`detect`, `embed`, `embed_unaligned`, `model_id`,
  `dim`) is a `Protocol`; YuNet + SFace (permissive licences) is the intended drop-in if the
  non-commercial weights ever have to go.
- **Per-instance error isolation.** One malformed image yields a
  `status: "error"` entry (with a machine-readable `error_code`: `invalid_image`,
  `invalid_box`, `invalid_request` or `service_error`) for that instance; siblings
  still return results. `predictions` is always the same length and order as `instances`.
- Two tasks: `detect` (find all faces) and `encode` (embed known bounding
  boxes, used by the manual-draw and bbox-edit routes).

Endpoint: `POST /v1/models/wikivisage:predict`. Readiness:
`GET /v1/models/wikivisage`. See `TESTING.md` for the full wire contract.

**Production hosting: a second Toolforge tool** (`wikivisage-face`, repo
variable `FACE_TOOL`), built with buildpacks from `model-server/` and run as a
published continuous job (`face-service`, port 8000, `--mem 3Gi --cpu 2`,
`--mount=none`, HTTP health check on `/v1/models/wikivisage`). It is a separate
tool, not a job in `wikivisage`, because Toolforge envvars are tool-wide
([T405022](https://phabricator.wikimedia.org/T405022)): in the main tool the
process parsing untrusted images would receive the ToolsDB password, OAuth
secret and `WIKIVISAGE_TOKEN_KEY`. The hop is `wikivisage job → HTTPS →
wikivisage-face.toolforge.org → RequestGuard (bearer token, 40 MiB body cap) →
KServe`. Runbook: `how-to-run-it.md`.

**The guard covers every route.** KServe also serves admin routes such as
`POST /v2/repository/models/<name>/unload`; with a token set, `guard.py` lets
only `GET /v1/models/<name>` through unauthenticated (the platform health check
cannot send headers). gRPC is disabled so there is no second, unguarded port.
`model.py` refuses to start when `TOOL_TOOLFORGE_API_URL` is set without a token
or with one shorter than 32 characters.

**Deploy ordering is a safety property.** `deploy.yml` publishes a
`face-service-<tag>` git tag (commit tree = `model-server/`), rebuilds and
restarts the face tool, waits for it to report ready, then — on the main tool —
stops the workers, migrates, restarts web, and only starts the workers again
once `/health` reports `face_service: reachable`. `is_healthy()` also probes the
token-protected `GET /v1/models`, so a missing or wrong token fails the
preflight. An unreachable face service is deliberately
*not* a hard error in the worker (images stay `pending` so a restart cannot
mass-error them), so without that preflight a bad deploy would look green while
silently processing nothing. The order is unchanged by the worker heartbeat
work: the offline banner stays hidden during a deploy because each worker writes
a planned-stop marker on SIGTERM (see *Worker heartbeat & downtime banner*), not
because the workers stay up.

**Security posture.** Detection used to run in a subprocess hardened with
scrubbed env vars plus `RLIMIT_AS` / `RLIMIT_CPU` / `RLIMIT_FSIZE`, because
libjpeg/libpng (via Pillow) and the ML runtime parse untrusted bytes. On Toolforge those limits come from
the job's `--mem` / `--cpu` and `--mount=none`; locally from the container flags
(`--memory`, `--cpus`, `--read-only`, uid 10001). Env scrubbing is obsolete —
the face tool never holds WikiVisage's OAuth, database, or token-encryption
secrets at all, only its own (unused) ToolsDB and replica credentials.

### Client (face_client.py)

Shared by both Toolforge processes. Embeddings travel base64-encoded (512 little-endian
float32 values, 2048 bytes) and are handed back as raw `bytes` written straight into the
`faces.encoding` BLOB. Every prediction is validated on receipt: `model_id` must equal
`EXPECTED_MODEL_ID` and `dim` must equal `EMBEDDING_DIM` (else `FaceServiceModelMismatch`,
retryable), and each encoding must be exactly `dim * 4` bytes — a stale 1024-byte
float64 payload is rejected.

| Function | Purpose |
|---|---|
| `detect_faces(image_bytes)` | Single image → `DetectionResult` |
| `detect_faces_batch(items)` | `[(key, bytes)]` → `{key: DetectionResult \| FaceServiceError}` |
| `encode_known_face(image_bytes, bbox, box_space=None)` | Embed one known box → `KnownFaceEncoding(encoding, model_id, aligned)` or `None` (unusable region — `invalid_box` rejection or no face; image-level `invalid_image` rejections are re-raised). `box_space=(w, h)` is the stored `detection_width/height` the box was measured in |
| `check_service_model()` | One tiny real request; returns `(model_id, dim)`; used by the worker at startup |
| `is_healthy()` | Readiness probe, never raises |
| `get_session()` / `reset_session()` | Pooled `requests.Session` with retries |

Exceptions: `FaceServiceError` (base), `FaceServiceUnavailable` (transport —
retryable), `FaceServiceModelMismatch` (subclass of Unavailable: wrong model/dim; the
worker exits at startup on it), `FaceServiceRejected` (this input is unusable — terminal;
`code` carries the service's `error_code` when present). `DetectedFace` carries `score`
and `aligned`; `DetectionResult` carries `model_id` and `dim`.

Batches split on whichever of the count / total-bytes budgets is hit first, so
a few large images cannot produce a huge request body.

### Metadata pre-filter (liftwing_client.py, experimental)

An **experiment**, not a pipeline dependency: a text-only LLM (Wikimedia
LiftWing, `llm-qwen36-27b`) guesses from Commons metadata alone whether a file
could show a person. It runs in **shadow mode only** — verdicts are stored on
the image row and change nothing (`projects.prefilter_mode`: `off`/`shadow`;
`enforce` exists in the ENUM for forward compatibility but the settings route
rejects it and the worker treats any non-`off` value as shadow).

- **The worker owns it; the web app never calls LiftWing.** Only public
  Commons metadata is sent (title, description flattened to plain text and
  capped at 250 chars, up to 15 categories with the `Category:` prefix
  stripped) — never image bytes, never the MIME type. One shared client per
  worker process, requests serialised (concurrency 1) and spaced by a jittered
  minimum interval, no streaming.
- **Non-visual media never reach the model.** Files whose MIME type is
  `audio/*`, `video/*`, `application/pdf`, `application/ogg`, `image/svg+xml`
  or `image/vnd.djvu` are settled as `keep=false` (reason `non-visual media
  (<type>)`) without a call; any other type, including a missing one, is sent
  to the model (fail open). The prompt asks for a reason only on drops, so
  keeps cost one boolean each.
- **Fail open, always.** Timeout, non-200, malformed JSON, a missing verdict or
  a circuit-breaker trip means "keep everything": `filter_metadata()` returns
  `None` and the images stay unchecked for a later cycle. A per-process circuit
  breaker pauses calls for `CIRCUIT_COOLDOWN` (600 s) after
  `CIRCUIT_FAILURE_THRESHOLD` (3) consecutive failures.
- **Where it runs:** `process_project` calls `prefilter_images(project)` **after**
  the image loop — the verdicts are metadata-only, and filtering first both
  delayed downloads and left every already-processed image permanently
  unchecked. It selects `bootstrapped=0 AND prefilter_checked_at IS NULL`
  regardless of `status`, capped at `LIFTWING_MAX_IMAGES` (250) per call, in
  batches of `LIFTWING_BATCH_SIZE` (20). A future `enforce` mode must instead
  select `status='pending'` and run before detection (commented in the code).
  Metadata comes from one batched Commons API call per batch
  (`_fetch_metadata`, which maps non-canonical titles back through
  `query.normalized`); a metadata failure leaves the batch unchecked. Missing
  verdicts are stored as keep with a NULL reason. Every batch is wall-clock
  logged; measured on 2026-10-17, 20-file batches take ~20 s (p95 ~21 s)
  against the 90 s timeout. `scripts/eval_prefilter.py` reads the verdicts back
  (keep recall on human-confirmed target images, skip rate, dropped-with-faces
  and kept-without-faces detector signal, sample reasons).
- **Background shadow sweep:** a daemon thread (`_shadow_prefilter_loop`,
  started in `main()`, pausing `LIFTWING_SWEEP_INTERVAL` = 60 s between
  passes; `0` disables) covers the projects the pipelines never claim.
  `_sweep_shadow_prefilter()` atomically claims one project through
  `_claim_shadow_project()` — any non-deleted project with the mode on,
  unchecked non-bootstrapped images, and *not* claimable by
  `_claim_active_projects` or `_claim_inference_projects` (projects with a
  pending SDC write request are left to the SDC path), so a sweep never
  delays real work — and runs the same `prefilter_images()` on it (≤ 250
  files), releasing the claim when the pass ends. `prefilter_images()`
  refreshes the claim after every stored batch, so a slow pass cannot outlive
  the 15-minute claim expiry (a failed refresh stops the pass, leaving the
  remaining files for a later pass). Without this, completed,
  paused or fully processed projects never received verdicts at all (prod
  report 2026-10-17: the settings box sat at "No verdicts recorded yet" for
  every shadow-mode project). The least-recently-claimed ordering rotates
  passes across projects, and the shared
  `worker_claimed_by`/`worker_claimed_at` columns mean two worker instances
  never sweep the same project at once. The thread exists because a pass can
  take minutes, which would otherwise stall the main loop's progress marks and
  claim refreshes (the heartbeat thread would read that as a stall); `main()` keeps its handle and joins it
  (bounded by `SWEEP_JOIN_TIMEOUT`) before releasing claims and closing the
  DB pool and HTTP session.
- **Probe and status (`service_status`):** the worker probes only while at
  least one non-deleted project has `prefilter_mode != 'off'`, at most once per
  `LIFTWING_PROBE_INTERVAL` (300 s; 0 disables). The claim is one atomic
  `UPDATE service_status SET last_checked = NOW() WHERE ... last_checked < NOW() - INTERVAL %s SECOND`,
  so two worker instances never probe at once. The probe runs a **minimal real
  completion** (`max_tokens=8`) rather than `GET <base>/models`: the models
  endpoint can answer 200 while completions time out or are rate limited, which
  would show a green banner during an outage. Real filter outcomes feed the same
  hysteresis (`_record_liftwing_failure` / `_record_liftwing_success`): `down`
  after 3 consecutive failures, `up` after one success. It is called from the
  outer poll loop **and** the two 60 s keep-alive loops (`_keep_alive_during_cycle`, `_supervise`), so a long project cycle
  does not starve it.
- **Auth and rate limits:** anonymous calls are rate limited per IP (observed
  header: `x-wmf-ratelimit-class: anon-browser`). Setting
  `WIKIVISAGE_LIFTWING_API_TOKEN` (a Wikimedia personal API token) sends
  `Authorization: Bearer <token>` and moves calls to the authenticated quota.
  A 429 opens the client circuit breaker immediately and honours `Retry-After`
  (delta-seconds or HTTP-date) as the cooldown instead of hammering the gateway
  two more times. The 429 deadline is tracked separately from the breaker, so it
  also pauses the recovery probe, and a later success cannot clear it early. The
  breaker check, the request and the recorded outcome run under one lock, so
  calls queued behind a failing one see the open breaker instead of each firing
  their own request. Requests are also spaced by a jittered minimum interval
  (`WIKIVISAGE_LIFTWING_MIN_INTERVAL`, default 2 s; `0` disables) so the tool
  never bursts at the shared service. The User-Agent is `WikiVisage/<version>`
  (`config.PROJECT_VERSION`).
- **Web side:** context processor `inject_liftwing_status` reads the row (no
  network call; failures read as "no message") and `base.html` shows a
  dismissible `.system-banner` on the project detail/settings pages only, and
  only for projects using the filter. Settings shows the status and last check
  time next to the mode selector. A status older than 3 probe intervals reads
  as `unknown` (a dead worker hides the banner instead of showing stale data).
  When the mode is not `off`, settings also renders a results box built by the
  settings route from `images.prefilter_*` joined with `faces`
  (`_get_prefilter_stats`, one read-only query; DB failure degrades to an
  "unavailable" line): files checked, would-skip count and rate, skips that had
  a detected face, human-confirmed target images it would drop (the safety
  gate, also reporting whether any human-confirmed targets have been seen yet)
  and the last verdict timestamp. The web app still never calls LiftWing.
  `/health` adds `"liftwing": "reachable" | "unreachable" | "disabled"`
  (`disabled` when unused or unknown) and it **never** sets `degraded`.

## Toolforge Processes

### Web (app.py)

Flask app served by gunicorn (2 `gthread` workers x 4 threads, `app:app`; the module-level app is built at import, and `create_app()` remains as a backwards-compatible factory for tests and scripts). Handles OAuth 2.0 login, project CRUD, face classification UI, and a Commons thumbnail proxy. SDC write requests are queued via a flag; the background worker performs the actual API writes.

**Security middleware:**
- Open redirect protection on login (`_is_safe_url()`)
- CSRF protection on all POST routes (per-request CSRF token stored in Flask's signed session cookie)
- PKCE (RFC 7636, S256) on the OAuth authorization code flow as defense-in-depth against intercepted authorization codes
- Sessions: signed cookies, revocable per user through `users.session_epoch`, 30-day sliding idle lifetime, 90-day absolute cap, always persistent cookies (see *Sessions are signed cookies, revocable per user* under Known Issues)
- Rate limiting via Flask-Limiter (global 200/hour default, 10/min on bbox endpoints)
- Security headers: `Content-Security-Policy` (`script-src 'self' 'nonce-<per-request nonce>'` with **no** `'unsafe-inline'` for scripts, `script-src-attr 'none'`, plus `object-src 'none'`, `base-uri 'none'`, `frame-ancestors 'none'`; `style-src` keeps `'unsafe-inline'` for the remaining `style=""` attributes). The nonce is generated by `_generate_csp_nonce()` (`g.csp_nonce`), exposed to templates as `{{ csp_nonce }}` and emitted by `set_security_headers()`, so header and tags always carry the same value. `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`
- OAuth refresh single-flight via Redis lock (`redis_lock.single_flight()`, 15s TTL, key `wikivisage:lock:oauth-refresh:<user_id>`) — prevents concurrent token-refresh requests from invalidating each other's freshly rotated refresh tokens. Degrades to no-op if Redis is unreachable; DB-level CAS on `token_expires_at` remains as a second line of defense.

**Routes (40 total):**
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Landing page |
| `/set-language/<lang>` | GET | Set locale cookie, redirect back |
| `/cookie-consent` | POST | Apply the cookie banner choice (`choice=accepted\|declined`, CSRF, rate limit 20/min) to an existing `locale` cookie: Accept makes it a 1-year cookie, Decline session-scoped. Answers 204; an invalid choice is 400. No `locale` cookie, nothing to change |
| `/login` | GET | OAuth 2.0 redirect to Wikimedia. A logged-in user is sent to the dashboard unless `?reauth=1` is given (the one-click re-authorize: the OAuth round trip runs again and the callback replaces the stored tokens; `next` is honoured). `?reauth=1` is honoured only when `Sec-Fetch-Site` is absent, `same-origin` or `none`: a link from another site gets the dashboard instead, because the GET changes state. Answers 503 (not a redirect) while the signed-in user could not be loaded from the database |
| `/auth/callback` | GET | OAuth token exchange. Clears the session (session fixation), except after a same-account re-authorize (the old session's `user_id` equals the resolved id): then `csrf_token`, `last_classify`, `skipped_images_*` and `manual_faces_*` are carried over, so the pending undo and other open tabs survive the round trip |
| `/logout` | POST | End this device's session only: clears the cookie, leaves `session_epoch` alone, so other browsers stay logged in (CSRF protected). Not `login_required`, so it also works while the database is down; without a `user_id` in the session it just redirects |
| `/dashboard` | GET | User's project list (paginated), includes collaborator count badges |
| `/api/category-info` | GET | Return total file count for a Commons category (BFS subcategory traversal) |
| `/project/new` | GET/POST | Create project (QID + Commons category, validates P31=Q5 and category existence). Detects cross-user duplicates and offers join option |
| `/project/<id>` | GET | Project stats, classification breakdown, model results gallery with approve/reject/edit-bbox, collaborator count badge, live inference-status card |
| `/project/<id>/classify` | GET | Active learning face classification UI |
| `/project/<id>/classify/clear-skips` | POST | Reset skipped faces for this session |
| `/api/classify` | POST | Submit face classification (yes/no/none) |
| `/api/undo-classify` | POST | Undo last classification (reverts manual face draws too). Undoing a target removes its P180 claim immediately (queues the worker when removal fails or no token is available) so an undo cannot strand a claim; undoing a rejection cancels the removal the rejection queued and never deletes a claim; deletes manual faces only after their claim is gone. Flashes the claim outcome ("SDC claim was removed." / "SDC claim removal queued.") on the classify page the client redirects to |
| `/api/reclassify` | POST | Approve/reject model-classified face. Reject removes the P180 claim immediately when a written claim is on record for a non-bootstrap face, and queues a worker removal otherwise (bootstrap faces and `sdc_written=0` faces whose write may be in flight) unless a sibling target still confirms the image. Response echoes the post-action `sdc_written` so the gallery card can stay in sync without a reload |
| `/api/manual-face` | POST | Manually draw face bounding box (rate limited: 10/min) |
| `/api/update-face-bbox` | POST | Redraw face bounding box from Model Results (rate limited: 10/min) |
| `/api/write-sdc/<id>` | POST | Queue P180 depicts claims for writing by background worker (sets flag, returns immediately) |
| `/api/sdc-status/<id>` | GET | Poll SDC write progress (written/pending counts, in_progress flag, the worker's `error`, and `reauth_url` when that error is an authorization failure, i.e. in `config.SDC_REAUTH_MESSAGES`) |
| `/api/stop-sdc/<id>` | POST | Cancel an in-progress SDC write (clears sdc_write_requested flag) |
| `/api/project/<id>/gallery` | GET | Paginated JSON API for Classification Results gallery (filter by result/source/sdc) |
| `/api/progress/<id>` | GET | Poll image processing progress (processed/total/pending counts, `face_stats`, `inference_eligible`, `complete`); feeds the progress card and the live inference-status card |
| `/project/<id>/settings` | GET/POST | Edit project params (incl. `prefilter_mode`: `off`/`shadow`), view member list (owner-only). Shows the LiftWing status line next to the mode selector and, when the mode is not `off`, a shadow-results stats box |
| `/project/<id>/settings/remove-member` | POST | Remove a member from the project (owner-only, CSRF protected) |
| `/project/<id>/settings/unban-member` | POST | Unban a member so they can rejoin the project (owner-only) |
| `/project/<id>/leave` | POST | Leave a project (non-owner members only, CSRF protected) |
| `/project/<id>/invite-code` | POST | Generate or revoke invite code for the project (owner-only) |
| `/join` | POST | Join a project via invite code |
| `/project/<id>/rerun-inference` | POST | Reset model-classified faces to re-run inference with current settings (owner-only). The paused / active-SDC-write / `min_confirmed` checks and the reset run in one transaction under the project-row lock; refuses without resetting in those states, otherwise wakes the worker even if no model face needed a reset |
| `/project/<id>/delete` | POST | Delete project |
| `/account/settings` | GET/POST | User account settings (leaderboard opt-out). GET also shows a Sessions card with the "Log out everywhere" button |
| `/account/logout-everywhere` | POST | Bump `users.session_epoch` and clear the session, which ends every session of the user on every device (CSRF, rate limit 5/hour; `data-confirm` on the form). On a database error the session is kept and an error is flashed |
| `/leaderboard` | GET | Top classifiers |
| `/sw.js` | GET | Serve service worker from root scope |
| `/.well-known/appspecific/com.chrome.devtools.json` | GET | Silence Chrome DevTools auto-request |
| `/health` | GET | Health check (JSON: `status`, `database`, `limiter` (`redis`/`memory`), `face_service` (`reachable`/`unreachable`), `liftwing` (`reachable`/`unreachable`/`disabled`), `workers` (`{"fresh": n, "stopping": k, "stale": m}` from `worker_heartbeats`, or `"unknown"` when unreadable), `degraded` boolean). `liftwing` and `workers` are informational and never affect `degraded` |
| `/robots.txt` | GET | Custom robots.txt (overrides Toolforge default Disallow: /) |
| `/sitemap.xml` | GET | XML sitemap for search engines |
| `/llms.txt` | GET | Plain-text site description for LLM crawlers ([llmstxt.org](https://llmstxt.org/)); robots.txt allows it through the `Disallow: /` |
| `/commons-thumb/<path>` | GET | Redirect to Commons thumbnail URL (standard step sizes enforced) |

**Error handlers:** 400, 403, 404, 429, 503, 500 — all render `error.html` (429 returns "Rate limit exceeded"; 503 is the "temporarily unavailable" page, sent with `Retry-After: 5`).

**Database trouble does not end sessions.** A `DatabaseError` while `before_request` loads the user leaves the session alone and sets `g.user = None`, `g.user_load_failed = True`; `login_required` then answers **503** (a JSON body for `/api/*` and JSON requests, otherwise the 503 page) instead of redirecting to `/login`, and `/login` and `/` answer 503 too (the landing page would otherwise show a "Log in" button that 503s). Only a `TokenDecryptionError`, a `session_epoch` mismatch or a session past the absolute age clear the session (`_end_session()`). `_get_valid_token()` never clears it: a failed refresh leaves `g.token_refresh_transient` set, and `_token_unavailable_error()` returns **401** "Your Wikimedia authorization has expired or was revoked. Please re-authorize to continue." with `reauth_url` (`/login?reauth=1`) only for a permanent refusal (revoked or expired refresh token, `invalid_grant`) and a retryable **503** without it otherwise. Undo, reclassify and `/api/write-sdc` use it.

**One-click re-authorize.** The browser login can stay valid while Wikimedia no longer accepts the stored token (revoked grant, expired refresh token). Instead of telling the user to log out and in, every such answer carries `reauth_url`, and the page shows a **Re-authorize** button that goes to `/login?reauth=1&next=<current page>` (the server renders that link for the page: the `reauth_login_url` context value, `/login?reauth=1&next=<request.full_path>`, becomes the JS constant `REAUTH_URL` in `base.html`; `reauthHref(data)` returns it only when the API answer carries a `reauth_url`, so no response text ever reaches an `href`). Paths: the JSON errors above (`showInlineAlert(message, reauthHref(data))`, used by reject/approve, bbox edit and the classify undo), the send-edits request, and the worker's stored `sdc_write_error` through `/api/sdc-status` (`sdcShowMessage(..., reauthUrl)`). The worker stores `config.SDC_AUTH_EXPIRED_MESSAGE` for `notloggedin`, `badtoken`, `mwoauth-invalid-authorization` and a token that cannot be refreshed at the start of a run **because the grant was refused** (revoked/expired refresh token, missing user row or refresh token; `_refresh_worker_token` records `permanent` from `on_failure` in a thread-local read by `_refresh_failure_was_permanent()`); a transient refresh failure (login-service timeout, 5xx, database error) stores `config.SDC_AUTH_UNAVAILABLE_MESSAGE` ("...try sending the edits again in a moment"), which is deliberately **not** in `SDC_REAUTH_MESSAGES`; `config.SDC_REAUTH_MESSAGES` also lists the wordings older releases stored, so existing rows get the button too (the project page renders the button next to a stored `sdc_write_error` on load, through `sdc_needs_reauth`, since the status poller only runs during a write). A row an older release stored as `"mwoauth-invalid-authorization: <info>"` is not in the set and gets no button; that is accepted. A missing edit permission (`permissiondenied`) is deliberately not offered as a re-authorize: it is fixed in the OAuth consumer registration. Transient failures (Wikimedia unreachable) never show the button. `oauth_refresh.refresh_tokens()` takes an optional `on_failure(permanent: bool)` callback (the web app uses it to set the flag; the worker uses it to pick the stored message) and `is_permanent_refresh_error()` classifies the exception.

**Access control helpers (membership-aware):**
- `get_project_for_actor(project_id, user_id)` — Fetches a project if the user is the owner OR a member via `project_members`. Used by `project_detail`, `classify`, `api_sdc_status`, `api_gallery`, `api_progress`. Returns project dict or `None` (→ 404).
- `verify_image_access(image_id, project_id, user_id)` — Verifies that an image belongs to a project accessible by the user (owner or member). Used by `api_classify`, `api_undo_classify`, `api_manual_face`. Returns image dict or `None` (→ 403).

**Project join flow (`/project/new` POST):**
1. User submits QID + Commons category.
2. Own-duplicate check: if the user already has this exact project, redirect to it.
3. Cross-user duplicate check: if ANOTHER user has a project with the same QID + category, show join/create-anyway options.
4. Join: inserts into `project_members` and redirects to the existing project.
5. Create anyway: creates a new independent project for the user.

### Worker (worker.py)

Long-running background process with concurrent execution. Polls DB every 60s (`POLL_INTERVAL`). Uses `ThreadPoolExecutor` at two levels:
- **Project-level**: Up to `MAX_CONCURRENT_PROJECTS` (default 4) projects processed simultaneously
- **Image-level**: Within each project, up to `IMAGE_THREADS` (default 4) images downloaded and face-detected in parallel

**Multi-instance**: Two worker instances (`ml-worker-1`, `ml-worker-2`) run in production. Each takes a `--worker-id` CLI argument used for per-worker heartbeat files and distributed project claiming via `SELECT ... FOR UPDATE`.

**Liveness (heartbeat thread)**: a dedicated daemon thread, `_heartbeat_loop()` (started in `main()` right after `init_db`, before the face-service startup check), ticks every `HEARTBEAT_INTERVAL` (30 s, waiting in 1 s slices). Each tick first touches `$HOME/.wikivisage-worker-alive-{worker_id}` (`_touch_heartbeat_file()`), then upserts this worker's row in `worker_heartbeats` through its **own** DB connection (`database.open_dedicated_connection()`), so an exhausted pool or a slow main loop cannot make a busy worker look dead, and a DB failure never blocks the file touch (a `Heartbeat write failed` warning is logged at most once per `HEARTBEAT_WARN_INTERVAL` = 60 s). **Startup gate**: until `main()` has passed the face-service check and `_log_project_diagnostics()` it sets `_startup_done`; before that a tick only touches the file and writes **no** row, so a worker that crash-loops in startup (e.g. `FaceServiceModelMismatch` exits 1) never writes a fresh `running` row and the offline banner can appear. Each tick runs under its own `try/except` (`Heartbeat tick failed`), so one unexpected error does not end the thread. Right after `_startup_done.set()`, `_delete_retired_worker_rows()` removes `worker_heartbeats` rows silent for `RETIRED_WORKER_ROW_DAYS` (7) other than its own (best effort), so a renamed worker does not show as `stale` in `/health` for ever. The worker id is the `VARCHAR(64)` primary key, so `_parse_worker_id` refuses an id longer than 64 characters and truncates the hostname part of the default `<hostname>-<pid>` to 50. Toolforge runs `healthcheck.sh {worker_id}` every 10s; 3 consecutive failures (file missing or >5 min old) trigger automatic restart. A database outage therefore no longer fails the health check. The heartbeat directory is configured in `config.py` as `HEARTBEAT_FILE_DIR`.

**Progress watchdog**: `_mark_progress(phase)` (no I/O, callable from any thread) is called in the main loop, `_supervise`, `_sleep_until_next_cycle`, `_keepalive_sdc_write`, the `_api_request` retry sleeps, around `_purge_deleted_projects` and at the start of each pipeline stage. If nothing reports progress for `STALL_LIMIT` (900 s, `WIKIVISAGE_WORKER_STALL_LIMIT`), the heartbeat thread logs **one** error (`No worker progress for ...`) with the phase and the main thread's stack, then stops touching the file and writing the row, so `healthcheck.sh` restarts a genuinely stuck worker (about 15 min after the last progress plus the 5 min file age). Progress marks from project threads count too; the logged stack is always the main thread's. The shadow-sweep thread does **not** count as progress (`_progress_exempt` thread-local, set by `_shadow_prefilter_loop`): it reports every minute while a shadow-mode project has unchecked files and would otherwise hide a hung main loop.

**Planned stop**: on SIGTERM (`toolforge jobs delete`, deploy, restart) the signal handler only sets `shutdown_requested`; the heartbeat thread then upserts `state='stopping'` with `expected_back_until = NOW() + WORKER_RESTART_GRACE_SECONDS` (config.py, 1200 s, `WIKIVISAGE_WORKER_RESTART_GRACE`). `_release_resources(sweep_thread, heartbeat_thread=None)` joins it (`HEARTBEAT_JOIN_TIMEOUT` = 10 s) **first**, before the sweep join, claim release and `close_pool()`, because the sweep join can take 100 s+ (the thread writes the marker on its own within about a second of SIGTERM; the join only keeps the interpreter from killing the daemon thread mid-write). The marker is an upsert, so a SIGTERM before the first tick still writes a row, but **only once startup finished** (before that the worker was never "running") and **not after a reported stall** (`Not announcing a planned stop: the worker was stalled`: Toolforge killing a stuck worker is not a planned stop, so the banner shows). The first `running` tick of the restarted worker clears the deadline and resets `started_at`. SIGKILL/OOM writes nothing, so the banner shows after `WORKER_STALE_SECONDS`. Log lines to grep: `Heartbeat write failed`, `No worker progress for`, `Announced planned stop`, `Heartbeat thread still running`.

Two query paths:
1. **Active projects** → full pipeline: early `run_autonomous_inference` when candidates already exist (so a re-run does not wait for discovery) → `traverse_category` → `bootstrap_from_sparql` → `prefilter_images` (experimental, shadow) → `process_images` → final `run_autonomous_inference`
2. **Active or completed projects with unclassified faces and enough human-confirmed targets** → inference-only: `run_autonomous_inference`

**Scheduling:** the inference-only queue is drained **before** the active batch each cycle (waves of `MAX_CONCURRENT_PROJECTS`), and the wake handler plus the 60 s keep-alive checks of the running loops call `_run_new_inference_projects()` so a project whose classifications just crossed `min_confirmed` is picked up mid-cycle (measured: 31 s from gate crossing to classification while the active batch was mid-run). Inference costs ~1 s per project while an active project can hold all slots for a long time looping its image backlog, so the original order (one inference batch *after* the active batch) left ready projects showing "Model inference in progress" at a standstill. `seen_ids` bounds the cycle-start drain: a project that stays eligible (e.g. faces with unusable encodings) is only attempted once per cycle there.

**Inference writes:** `_persist_inference_batch` takes the project-row lock used by human classifications and re-reads candidate eligibility and sibling targets before each batch UPDATE. Per-image deduplication is computed from distance alone; the locked write, not the candidate snapshot, decides whether a live sibling target vetoes a match. A target approved after the snapshot forces its remaining siblings to non-matches, while a sibling undone after the snapshot lets the closest candidate become a match. Faces with a new human edit, queued P180 removal, or superseded box are left untouched. Paused/deleted projects receive no batch writes. Counts report rows actually updated.

**Note:** SDC claim writing is user-triggered via the "Send Edits to Wikimedia Commons" button on the project detail page (calls `/api/write-sdc/<id>`). The worker no longer writes SDC claims automatically.

**SDC shutdown reporting:** a shutdown signal can arrive after the final successful edit. Before recording an interruption, the worker checks both actionable writes and removals. With neither left it clears the request without an error; with work left it explains that completed edits were kept and the remaining edits need a new send request. A failed completion check reports an unverified outcome rather than claiming success. No automatic retry is added.

**Pipeline stages:**
| Function | What it does |
|----------|-------------|
| `traverse_category` | Crawls Commons category API, inserts image rows (batch INSERT IGNORE). Caps at `MAX_IMAGES_PER_PROJECT` (9000). Filters out video/audio (keeps images only). |
| `_download_image` | Downloads image with streaming 50MB size cap (`MAX_IMAGE_DOWNLOAD_BYTES`) |
| `_validate_image_dimensions` | Checks image pixel area before shipping bytes to the face service (rejects >100 megapixels) |
| `_download_for_detection` | Downloads + dimension-checks one image; marks it `error` and returns `None` on failure |
| `_persist_detection` | Batch-inserts faces (embedding, `embedding_model`, `det_score`, box), marks the image `processed`, auto-classifies single-face bootstrap images. Optional `stored_size` converts boxes into an existing detection space |
| `_handle_detection_failure` | Retry policy: `FaceServiceRejected` → mark `error` (terminal); `FaceServiceUnavailable` → leave `pending` (retry next cycle) |
| `_process_image_wave` | One memory-bounded wave: parallel download → batched remote detect → serial persist. **The only detection entry point.** |
| `process_images` | Splits the pending batch into waves of `DETECTION_WAVE_SIZE` and runs each |
| `bootstrap_from_sparql` | Seeds model from existing P180 depicts claims via SPARQL |
| `_fetch_metadata` | One batched Commons API POST (`prop=imageinfo\|categories`) per pre-filter batch; description from `ImageDescription`/`ObjectName` flattened to plain text; maps non-canonical titles back through `query.normalized`; returns `{}` on failure (batch stays unchecked) |
| `prefilter_images` | Experimental shadow-mode LiftWing stage, run after the image loop. Caps at `LIFTWING_MAX_IMAGES` (250) per call, batches of `LIFTWING_BATCH_SIZE` (20), skips bootstrapped images, selects any unchecked image regardless of `status`, stores `(keep, reason)` per image and records success/failure into `service_status`. Fail-open throughout. A failed claim refresh stops the pass |
| `_store_prefilter_verdicts` | Atomic batch UPDATE of `prefilter_keep` / `prefilter_reason` / `prefilter_checked_at` (`WHERE prefilter_checked_at IS NULL`) |
| `_liftwing_service_down` | True when `service_status` says LiftWing is down; the pre-filter skips the project entirely |
| `_record_liftwing_failure` | Increments `consecutive_failures`; flips to `down` at `LIFTWING_FAIL_THRESHOLD` (3). A real failed pre-filter batch lands here too |
| `_record_liftwing_success` | Resets `consecutive_failures`, sets `status='up'`, stamps `last_ok`; called after each stored pre-filter batch and by the probe |
| `_probe_liftwing_if_due` | Atomic once-per-`LIFTWING_PROBE_INTERVAL` claim + real probe; runs only while a project uses the filter. Called from the outer poll loop and both 60 s keep-alive loops |
| `_claim_shadow_project` | Atomic claim of one project the active/inference paths leave alone (non-deleted, mode on, unchecked non-bootstrapped backlog; skips projects with a pending SDC write); least-recently-claimed first, so sweeps rotate |
| `_sweep_shadow_prefilter` | One `prefilter_images()` pass (≤ 250 files) on a swept project, releasing the claim afterwards; skipped while LiftWing is marked down |
| `_shadow_prefilter_loop` | Daemon sweep thread started by `main()`: sweep, then sleep `LIFTWING_SWEEP_INTERVAL` in 1 s slices |
| `_load_embedding` | Validates a `faces.encoding` BLOB: exactly `dim * 4` bytes, little-endian float32, finite, L2 norm in `[0.9, 1.1]`; returns a renormalised unit vector or `None` (rejects stale 1024-byte dlib encodings) |
| `_face_distance` | Cosine distance `1 - dot(centroid, embedding)`, clamped at 0; stored in `faces.confidence` |
| `_convert_box_to_stored_space` | Re-expresses a box from one pixel space into the stored `detection_width/height` space (for re-detection on images whose boxes live in another size) |
| `_persist_inference_batch` | Takes the project-row lock, re-checks candidate eligibility and sibling targets, then applies model decisions in one transaction. Returns actual classified and match counts |
| `run_autonomous_inference` | Cosine-distance classification on unclassified faces against the renormalised mean centroid (needs >= `min_confirmed` target faces). Only reads faces whose `embedding_model` equals `face_client.EXPECTED_MODEL_ID`, and skips faces with `sdc_removal_pending = 1` so a model guess can never re-target a face whose claim a human asked to remove (the removal phase would then strip the claim of a fresh match). A candidate on an image that already has a confirmed target (`image_has_target`) is always resolved as a non-match, so a re-run cannot promote a second face after the human approved one of several model faces on the same image |
| `write_sdc_claims` | Thin entry over `_SdcRun` (`begin` → `write_phase` → `removal_phase` → `finish`; each Commons call site, the token-refresh retry, the abort and the cancel check exist once). A page held by another project's fresh claim is `BLOCKED`: the run backs off `_SDC_BLOCKED_BACKOFF_SECONDS` between fully blocked passes and ends after `_SDC_BLOCKED_MAX_PASSES` with a notice, instead of re-selecting it in a tight loop. Writes P180 claims to Commons SDC via Wikibase API (idempotent). Triggered by `sdc_write_requested` flag set from web UI. Before every write it re-reads the face's live `is_target`/`superseded_by` and skips faces that were rejected, undone or replaced while the batch was in flight (the web side queues a removal for those too; a whole unverifiable batch stops the run instead of looping). Re-checks `sdc_write_requested` every 5 faces (`_PER_FACE_CANCEL_CHECK_EVERY`) so the user can stop a running batch quickly, and calls `_keepalive_sdc_write` on the same cadence. Treats `no-such-entity` / `no-such-claim` / `no-such-statement` / `notfound` as already-gone successes (`_ALREADY_GONE_CODES`). |
| `_keepalive_sdc_write` | Calls `_mark_progress("sdc_keepalive")` (so the heartbeat thread does not read a long run as a stall) and refreshes `projects.worker_claimed_at` (only for this worker's claim) while a long SDC run blocks the main loop — otherwise a run over `CLAIM_EXPIRY_MINUTES` lets the other worker claim the same project. The claim refresh is best-effort (`try/except DatabaseError`). Called every 5 written faces and once per removal page. |
| `_heartbeat_loop` | Heartbeat thread body (see Liveness above): 30 s ticks, file touch then `worker_heartbeats` upsert on a dedicated connection, stall watchdog, planned-stop marker on shutdown |
| `_mark_progress` | Records the current phase and a monotonic timestamp; I/O free, safe from any thread |
| `_api_request` | Wrapper for Commons/Wikidata API calls with maxlag, retry, and User-Agent |
| `_get_csrf_token` | Fetches CSRF token for Wikibase API writes |

### Shared (database.py)

Both processes import from `database.py`. Thread-safe connection pool (`Queue`-based) with exponential backoff retry (3 attempts).

**Exports:**
- `init_db(pool_size)` — Initialize connection pool. The web app calls it at import with `WEB_DB_POOL_SIZE` (`WIKIVISAGE_WEB_DB_POOL_SIZE`, default 3) per gunicorn worker; the worker passes `WIKIVISAGE_DB_POOL_SIZE` when set, else `min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 5)`.
- `execute_query(sql, params, fetch)` — Universal read/write. Returns `List[Dict]` (fetch=True) or `int` rowcount (fetch=False).
- `execute_insert(sql, params)` — INSERT with `cursor.lastrowid` return (race-free).
- `execute_transaction(queries)` — Atomic multi-statement transaction. Takes list of `(sql, params)` tuples.
- `get_connection(timeout)` — Context manager for raw connection access.
- `close_pool()` — Drain and close all connections.
- `open_dedicated_connection()` — Returns a `DedicatedConnection` outside the pool, used only by the worker's heartbeat thread. `execute(sql, params)` runs one autocommit statement (rowcount returned); `close()` drops it. Same config as the pool; it connects on first use, reconnects when older than `MAX_CONNECTION_AGE`, closed, or broken mid-statement (one retry), and raises `DatabaseError`. Not thread-safe (one per thread) and **not counted in the pool size**, so it needs one extra ToolsDB connection.
- `DatabaseError`, `PoolExhaustedError`, `ConfigurationError` — Custom exception hierarchy.

Pool size configurable via `WIKIVISAGE_DB_POOL_SIZE` env var. The worker uses that env var when set; otherwise it computes `min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 5)` and warns above 5, because Toolforge ToolsDB enforces `max_user_connections=20` across every process sharing the DB user (gunicorn + all workers). Budget: 2 web workers x 3 + 2 background workers x (5 pooled + 1 dedicated heartbeat connection) = 18 of 20.

## Data Model

```
users 1──N projects 1──N images 1──N faces
  |           |                        |
  |           +-- sdc_write_requested   +-- is_target: NULL=unclassified, 1=match, 0=non-match
  |           +-- sdc_write_error       +-- classified_by: 'human' | 'model' | 'bootstrap'
  |           +-- last_inference_threshold   +-- classified_by_user_id: FK to users (human classifications)
  |           +-- last_inference_min_confirmed +-- encoding: 512D float32 embedding (2048 bytes BLOB)
  |                                      +-- embedding_model: e.g. 'scrfd500m+arcface_mbf' (NOT NULL)
                                       +-- det_score: SCRFD confidence (NULL for manual faces)
                                       +-- aligned: 1=landmark-aligned embedding, 0=unaligned (drawn box the detector missed)
                                       +-- confidence: cosine distance from target centroid
                                       +-- sdc_written: whether P180 claim was written
                                       +-- sdc_removal_pending: 1=P180 removal queued (rejected face)
                                       +-- superseded_by: FK to replacement face (after bbox edit)

                                images:
                                       +-- bootstrapped: 1=image found via P180 bootstrap

users.session_epoch INT UNSIGNED NOT NULL DEFAULT 0
  +-- Bumped by POST /account/logout-everywhere; session["epoch"] (set at login) must equal it,
      a mismatch ends the session. A session without "epoch" counts as 0

project_members (project_id, user_id, role, joined_at)
  +-- Tracks additional collaborators beyond the project owner
  +-- role: 'owner' | 'member' (default 'member')
  +-- PK: (project_id, user_id) — composite, no duplicates
  +-- FKs cascade on DELETE from projects and users

sdc_claims (commons_page_id, wikidata_qid, project_id, face_id, claimed_at, written_at)
  +-- Cross-project deduplication: UNIQUE(commons_page_id, wikidata_qid)
  +-- Prevents two projects writing the same P180 claim to the same Commons page
  +-- Worker does INSERT IGNORE — first project to claim wins
  +-- written_at set when API write succeeds

worker_heartbeats (worker_id PK, state ENUM('running','stopping'), last_seen, expected_back_until,
                   last_progress, phase, started_at, version)
  +-- One row per worker process, written by the worker's heartbeat thread every 30 s
  +-- state 'stopping' + expected_back_until = announced planned stop (deploy, restart)
  +-- Read by the web app only (offline banner, /health `workers`)

worker_heartbeat (legacy single-row: id=1, last_seen DATETIME)
  +-- No longer written or read; still created by schema.sql/migrate.py and dropped by
      --reset, to be removed in a later release

service_status (name PK, status ENUM('up','down','unknown'), consecutive_failures,
                last_checked, last_ok, last_error)
  +-- Last-known reachability of optional external services; only 'liftwing' is seeded
  +-- Worker probe writes it; the web app only reads it (never a live call in a request)
  +-- Not touched by the embedding-model wipe
```

### Pre-filter columns (experimental)

```
projects.prefilter_mode ENUM('off','shadow','enforce') DEFAULT 'off'
  +-- 'enforce' exists for forward compatibility only: settings reject it and the
      worker treats any non-'off' value as shadow

images.prefilter_keep TINYINT(1) NULL      -- 1=keep, 0=drop, NULL=not checked
images.prefilter_reason VARCHAR(160) NULL  -- short reason (model or deterministic)
images.prefilter_checked_at DATETIME NULL  -- when the verdict was stored
  +-- Set by prefilter_images(); bootstrapped images always bypass the filter,
      and shadow mode checks images of any status (a backlog is worked off later)
```

**Project lifecycle:** `active` -> `paused`/`completed`. Worker only processes `active` for full pipeline. Inference runs on `active` + `completed`.

## Constants & Limits

### app.py
| Constant | Value | Purpose |
|----------|-------|---------|
| `APP_VERSION` | Read from `pyproject.toml` | Displayed in footer |
| `LANGUAGES` | `en, nb, es, fr` | Supported locales |
| `RTL_LANGUAGES` | `ar, he, fa, ur` | RTL layout support |
| `MAX_IMAGE_DOWNLOAD_BYTES` | 50 MB | Image download size cap (shared with worker) |
| `_THUMB_STEPS` | `config.COMMONS_THUMB_STEPS` = `(20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)` | Commons-enforced thumbnail widths |
| `FILE_PATH_URL` | width = `config.DETECTION_WIDTH` (960) | Manual-draw download; same width as the worker's detection |
| `DEFAULT_DISTANCE_THRESHOLD` | 0.5 | New-project cosine-distance threshold (provisional until calibration); keep equal to `schema.sql` / `migrate.py` |
| `MAX_BBOX_PX` | 10000 | Maximum bounding box coordinate value |
| `MIN_BBOX_AREA` | 100 | Minimum bounding box area in pixels |
| `_LIFTWING_STALE_SECONDS` | `config.LIFTWING_PROBE_INTERVAL * 3` = 900 | A `service_status` row older than this reads as `unknown` (no banner) |
| `PERMANENT_SESSION_LIFETIME` | `config.SESSION_IDLE_DAYS` = 30 days | Sliding idle lifetime of the `session` cookie (`SESSION_REFRESH_EACH_REQUEST = True`: every response re-sends it) |
| `_SESSION_MAX_AGE_SECONDS` | `config.SESSION_MAX_AGE_DAYS` = 90 days | Absolute session age measured from `session["auth_at"]` (set at login) |
| `LOCALE_COOKIE_MAX_AGE` | 1 year | `locale` cookie lifetime when the visitor accepted cookies |
| `_SESSION_LOSS_SAMPLE_RATE` | 0.05 | Fraction of anonymous requests that get the `session loss:` log line (`WIKIVISAGE_LOG_SESSION_LOSS` turns it off) |

### worker.py
| Constant | Value | Purpose |
|----------|-------|---------|
| `POLL_INTERVAL` | 60s | DB polling frequency |
| `BATCH_SIZE` | 50 | Images per processing batch (`WIKIVISAGE_WORKER_BATCH_SIZE`) |
| `_PER_FACE_CANCEL_CHECK_EVERY` | 5 | SDC write-loop cadence: re-check `sdc_write_requested` and refresh liveness/claim via `_keepalive_sdc_write` (also once per removal page) |
| `MAX_CONCURRENT_PROJECTS` | 4 | Parallel project processing |
| `IMAGE_THREADS` | 4 | Parallel image download threads per project |
| `DETECTION_WAVE_SIZE` | `max(IMAGE_THREADS, FACE_SERVICE_BATCH_SIZE)` = 8 | Images held in memory at once — this, not `BATCH_SIZE`, bounds worker peak memory |
| `MAX_IMAGE_DOWNLOAD_BYTES` | 50 MB | Image download size cap |
| `MAX_IMAGE_PIXELS` | 100M | Pixel area limit before shipping to the face service |
| `MAX_IMAGES_PER_PROJECT` | 9000 | Category traversal cap |
| `FILE_PATH_URL` | width = `config.DETECTION_WIDTH` (960, `WIKIVISAGE_DETECTION_WIDTH`) | Detection thumbnail; must be a standard Commons step (config.py raises otherwise) |
| `DEFAULT_DISTANCE_THRESHOLD` | 0.5 | Used when a project row carries none |
| `EMBEDDING_NORM_MIN` / `EMBEDDING_NORM_MAX` | 0.9 / 1.1 | Accepted L2-norm band for stored embeddings |
| `LIFTWING_BATCH_SIZE` | 20 | Files per pre-filter completion; measured ~20 s per batch vs the 90 s timeout (`WIKIVISAGE_LIFTWING_BATCH_SIZE`) |
| `LIFTWING_MAX_IMAGES` | 250 | Pre-filter cap per `prefilter_images()` call — a project cycle or a sweep pass (`WIKIVISAGE_LIFTWING_MAX_IMAGES`) |
| `LIFTWING_SWEEP_INTERVAL` | 60 s | Pause between background shadow-sweep passes; `0` disables the sweep (`WIKIVISAGE_LIFTWING_SWEEP_INTERVAL`) |
| `SWEEP_JOIN_TIMEOUT` | `REQUEST_TIMEOUT + 15` (105 s) | Shutdown join window for the sweep thread before claims and the DB pool / HTTP session are closed |
| `LIFTWING_FAIL_THRESHOLD` | 3 | Consecutive probe **or filter-batch** failures before `service_status.status` flips to `down` (`WIKIVISAGE_LIFTWING_FAIL_THRESHOLD`) |
| `HEARTBEAT_INTERVAL` | 30 s | Heartbeat thread tick (file touch + `worker_heartbeats` upsert) |
| `STALL_LIMIT` | 900 s | No `_mark_progress` for this long = stuck: the heartbeat thread logs once and stops touching the file/row so Toolforge restarts the worker (`WIKIVISAGE_WORKER_STALL_LIMIT`) |
| `HEARTBEAT_WARN_INTERVAL` | 60 s | At most one `Heartbeat write failed` warning per this interval |
| `HEARTBEAT_JOIN_TIMEOUT` | 10 s | Shutdown waits this long for the heartbeat thread to write the planned-stop row |

### config.py (shared)
| Constant | Default | Purpose |
|----------|---------|---------|
| `SESSION_IDLE_DAYS` | 30 | Sliding idle session lifetime in days (`WIKIVISAGE_SESSION_IDLE_DAYS`) |
| `SESSION_MAX_AGE_DAYS` | 90 | Absolute session age cap in days from login (`WIKIVISAGE_SESSION_MAX_AGE_DAYS`) |
| `WORKER_STALE_SECONDS` | 300 | A `worker_heartbeats` row (or a project claim) older than this no longer counts as alive; the web banner needs every signal stale |
| `WORKER_RESTART_GRACE_SECONDS` | 1200 | How long a worker's `stopping` row keeps the offline banner away (`WIKIVISAGE_WORKER_RESTART_GRACE`) |
| `SDC_AUTH_EXPIRED_MESSAGE` / `SDC_REAUTH_MESSAGES` | text / set | What the worker stores in `projects.sdc_write_error` when the user's authorization is refused, and the set (it plus older wordings) the web app turns into a `reauth_url` |

### liftwing_client.py (worker-only, experimental)
| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_URL` | `https://api.wikimedia.org/service/lw/inference/v1/models/llm-qwen36-27b/openai/v1/chat/completions` | OpenAI-compatible endpoint (`WIKIVISAGE_LIFTWING_URL`) |
| model | `llm-qwen36-27b` | Sent in the request body (`WIKIVISAGE_LIFTWING_MODEL`) |
| `REQUEST_TIMEOUT` / `PROBE_TIMEOUT` | 90 s / 20 s | Whole-batch timeout / one tiny probe (`WIKIVISAGE_LIFTWING_TIMEOUT`, `WIKIVISAGE_LIFTWING_PROBE_TIMEOUT`) |
| `PROBE_MAX_TOKENS` | 8 | The probe is a minimal **real completion**, not `GET /models` (`WIKIVISAGE_LIFTWING_PROBE_MAX_TOKENS` is not read; constant only) |
| `MAX_TOKENS` | 2000 | Room for ~50 short verdicts plus a reasoning preamble (`WIKIVISAGE_LIFTWING_MAX_TOKENS`) |
| `MIN_CALL_INTERVAL` | 2.0 s | Minimum quiet gap between outbound requests, plus a jitter of up to half the interval (`WIKIVISAGE_LIFTWING_MIN_INTERVAL`; `0` disables) |
| `CIRCUIT_FAILURE_THRESHOLD` / `CIRCUIT_COOLDOWN` | 3 / 600 s | Per-process circuit breaker (`WIKIVISAGE_LIFTWING_CIRCUIT_FAILURES`, `WIKIVISAGE_LIFTWING_CIRCUIT_COOLDOWN`) |
| `MAX_REASON_CHARS` / `MAX_DESCRIPTION_CHARS` / `MAX_CATEGORIES_PER_FILE` | 160 / 250 / 15 | Truncation of what is sent and stored |
| `NON_VISUAL_MIME_PREFIXES` / `NON_VISUAL_MIME_TYPES` | `audio/`, `video/` / `application/pdf`, `application/ogg`, `image/svg+xml`, `image/vnd.djvu` | Settled as `keep=false` without a model call (`is_visual_media()`; unknown MIME types still go to the model) |
| `api_token` | `""` | Optional Wikimedia personal API token; sends `Authorization: Bearer` (`WIKIVISAGE_LIFTWING_API_TOKEN`) |
| `USER_AGENT` | `WikiVisage/<version> (https://github.com/DiFronzo/WikiVisage)` | Wikimedia API etiquette; version from `config.PROJECT_VERSION` (`WIKIVISAGE_LIFTWING_USER_AGENT`) |

### face_client.py (shared)
| Constant | Default | Purpose |
|----------|---------|---------|
| `FACE_SERVICE_BATCH_SIZE` | 8 | Max images per request |
| `FACE_SERVICE_MAX_BATCH_BYTES` | 24 MiB | Max raw image bytes per request |
| `FACE_SERVICE_TIMEOUT` | 30s | Base read timeout |
| `FACE_SERVICE_TIMEOUT_PER_IMAGE` | 15s | Added per image in the batch |
| `EXPECTED_MODEL_ID` | `scrfd500m+arcface_mbf` | The only `model_id` accepted from the service (`WIKIVISAGE_FACE_MODEL_ID`) |
| `EMBEDDING_DIM` | 512 | Embedding length; each encoding must be `EMBEDDING_DIM * 4` bytes (`WIKIVISAGE_FACE_EMBEDDING_DIM`) |

### model-server/model.py
| Constant | Default | Purpose |
|----------|---------|---------|
| `FACE_WORKERS` | 2 | Executor threads per instance; ONNX sessions are shared and thread-safe (match the job's vCPUs) |
| `MAX_REQUEST_BYTES` | 40 MiB | Body cap enforced by `guard.py` before parsing |
| `MAX_BATCH_SIZE` | 32 | Reject requests with more instances |
| `MAX_IMAGE_BYTES` | 20 MB | Per-image decoded size ceiling |
| `MAX_IMAGE_PIXELS` | 100M | Pixel area ceiling before decoding |
| `MAX_BOXES_PER_INSTANCE` | 64 | Max boxes per `encode` instance |
| `MAX_BOX_SPACE_PX` | 65535 | Largest side a `box_space` may claim |

### model-server/backend.py
| Constant | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` / `EMBEDDING_DIM` | `scrfd500m+arcface_mbf` / 512 | Stored in `faces.embedding_model`, returned per prediction |
| `WEIGHT_SHA256` | two pinned hashes | Verified at `load()`; must equal `weights/SHA256SUMS` |
| `BackendConfig.det_max_side` | 1280 | Longest side fed to the detector (`WIKIVISAGE_DET_MAX_SIDE`) |
| `BackendConfig.score_threshold` / `nms_threshold` | 0.50 / 0.40 | SCRFD decoding |
| `BackendConfig.min_face_px` | 20 | Drop detected faces whose short side is smaller (`WIKIVISAGE_MIN_FACE_PX`); not applied to `encode` |
| `BackendConfig.max_faces` | 300 | Per-image cap, best score first (`WIKIVISAGE_MAX_FACES`) |
| `BackendConfig.coarse_min_side` | 640 | Run the extra half-scale pass for very large faces when the working image is at least this long (0 disables; `WIKIVISAGE_DET_COARSE_MIN_SIDE`) |
| `COARSE_MIN_FACE_PX` | 240 | Smallest face taken from the half-scale pass |
| `EMBED_CHUNK` | 32 | Faces per embedder call |
| `ENCODE_BOX_EXPANSION` / `ENCODE_MIN_IOU` | 0.5 / 0.3 | Landmark recovery for drawn boxes |


## Environment Variables

**Required (no defaults — will raise `ConfigurationError`):**
- `TOOL_TOOLSDB_USER` — MariaDB username
- `TOOL_TOOLSDB_PASSWORD` — MariaDB password
- `WIKIVISAGE_DB_NAME` — Database name (e.g., `s12345__wikiface`)

**Required for OAuth (empty string default = broken):**
- `OAUTH_CLIENT_ID`
- `OAUTH_CLIENT_SECRET`
- `OAUTH_REDIRECT_URI`

**Optional:**
- `TOOL_TOOLSDB_HOST` — Default: `tools.db.svc.wikimedia.cloud`
- `FLASK_SECRET_KEY` — Default: random hex (regenerates on restart — sessions lost)
- `WIKIVISAGE_SESSION_IDLE_DAYS` — Default: `30` (sliding idle lifetime of the `session` cookie; read by `config.py`)
- `WIKIVISAGE_SESSION_MAX_AGE_DAYS` — Default: `90` (absolute session age from login; a session past it is cleared with a flash)
- `WIKIVISAGE_LOG_SESSION_LOSS` — Default: on (`0`/`false`/`no`/`off` disables the sampled `session loss:` INFO line, see Cookies)
- `WIKIVISAGE_INSECURE_HTTP` — Unset by default. Local development only: `1` drops `Secure` from the `session` and `locale` cookies so a phone or another device can use the app over plain http on the LAN. Run with `FLASK_DEBUG` unset (debug always binds `127.0.0.1` because of the Werkzeug debugger) and set `OAUTH_REDIRECT_URI` to the LAN address registered on the OAuth consumer. Never set it in production: `_insecure_http_enabled()` ignores it (and logs an error) when `TOOL_TOOLFORGE_API_URL` is set, because Toolforge envvars are tool-wide. The cookie `Secure` flag comes from `_SessionInterface.get_cookie_secure()` / `_cookie_secure()` per request, not from a config mutation
- `PORT` — Default: `8000`
- `WIKIVISAGE_WEB_DB_POOL_SIZE` — Default: `3` (connections per gunicorn web worker; 2 workers x 3 + 2 background workers x (5 pooled + 1 dedicated heartbeat connection) = 18 of ToolsDB's 20). Separate from the next variable because Toolforge envvars are tool-wide.
- `WIKIVISAGE_DB_POOL_SIZE` — Background worker pool size. The worker reads it; when unset it computes `min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 5)` and warns above 5 (ToolsDB shares a 20-connection cap across all processes; each worker also holds one extra, unpooled connection for its heartbeat thread).
- `WIKIVISAGE_WORKER_POLL_INTERVAL` — Default: `60` seconds
- `WIKIVISAGE_WORKER_BATCH_SIZE` — Default: `50` (images per processing batch)
- `WIKIVISAGE_WORKER_MAX_PROJECTS` — Default: `4` (concurrent projects processed by worker)
- `WIKIVISAGE_WORKER_STALL_LIMIT` — Default: `900` (seconds without `_mark_progress` before the heartbeat thread stops touching the liveness file so Toolforge restarts the worker)
- `WIKIVISAGE_WORKER_RESTART_GRACE` — Default: `1200` (seconds a worker's planned-stop row keeps the offline banner away; read by `config.py`)
- `WIKIVISAGE_WORKER_IMAGE_THREADS` — Default: `4` (parallel image download/detection threads per project)
- `WIKIVISAGE_TOKEN_KEY` — Fernet key for encrypting OAuth tokens at rest. If unset, tokens are stored as plaintext (backward compatible). Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `WIKIVISAGE_REDIS_URL` — Redis URL for shared rate limiter storage. Default: `redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379`. Falls back to `memory://` if Redis is unreachable. Key prefix: `wikivisage:`.
- `OAUTHLIB_INSECURE_TRANSPORT=1` — Required for local dev (OAuth over HTTP)

**Face service client** (read by `face_client.py`, used by both `app.py` and `worker.py`):
- `WIKIVISAGE_FACE_SERVICE_URL` — Default: `http://localhost:8080`
- `WIKIVISAGE_FACE_SERVICE_MODEL` — Default: `wikivisage` (the name in the URL path)
- `WIKIVISAGE_FACE_SERVICE_TOKEN` — Bearer token; must equal the face tool's. Required in production (the service is public).
- `WIKIVISAGE_FACE_SERVICE_BATCH_SIZE` — Default: `8` (max images per request)
- `WIKIVISAGE_FACE_SERVICE_MAX_BATCH_BYTES` — Default: `25165824` (24 MiB raw image bytes per request)
- `WIKIVISAGE_FACE_SERVICE_TIMEOUT` — Default: `30` (base read timeout, seconds)
- `WIKIVISAGE_FACE_SERVICE_TIMEOUT_PER_IMAGE` — Default: `15` (added per image in the batch)
- `WIKIVISAGE_FACE_MODEL_ID` — Default: `scrfd500m+arcface_mbf` (the only model_id the client accepts)
- `WIKIVISAGE_FACE_EMBEDDING_DIM` — Default: `512`
- `WIKIVISAGE_DETECTION_WIDTH` — Default: `960` (read by `config.py`; the Commons thumbnail width the worker detects on and the app downloads for manual drawing; must be a standard Commons step)

**LiftWing metadata pre-filter (experimental; worker only)** — read by `liftwing_client.py`, `worker.py` and `config.py`:
- `WIKIVISAGE_LIFTWING_URL` — Default: the LiftWing `llm-qwen36-27b` OpenAI-compatible chat-completions URL
- `WIKIVISAGE_LIFTWING_MODEL` — Default: `llm-qwen36-27b`
- `WIKIVISAGE_LIFTWING_TIMEOUT` — Default: `90` (whole-batch read timeout, seconds)
- `WIKIVISAGE_LIFTWING_PROBE_TIMEOUT` — Default: `20` (the tiny probe request)
- `WIKIVISAGE_LIFTWING_MAX_TOKENS` — Default: `2000`
- `WIKIVISAGE_LIFTWING_MIN_INTERVAL` — Default: `2` (minimum quiet gap between calls, in seconds; a jitter of up to half the interval is added; `0` disables)
- `WIKIVISAGE_LIFTWING_CIRCUIT_FAILURES` / `WIKIVISAGE_LIFTWING_CIRCUIT_COOLDOWN` — Default: `3` / `600` (per-process circuit breaker; a 429 opens it immediately and `Retry-After` extends the cooldown)
- `WIKIVISAGE_LIFTWING_API_TOKEN` — Unset by default. A Wikimedia personal API token: sent as `Authorization: Bearer <token>` and moves calls from the anonymous to the authenticated quota.
- `WIKIVISAGE_LIFTWING_USER_AGENT` — Default: `WikiVisage/<version> (https://github.com/DiFronzo/WikiVisage)`
- `WIKIVISAGE_LIFTWING_BATCH_SIZE` — Default: `20` (files per completion)
- `WIKIVISAGE_LIFTWING_MAX_IMAGES` — Default: `250` (pre-filter cap per `prefilter_images()` call — a project cycle or a sweep pass)
- `WIKIVISAGE_LIFTWING_SWEEP_INTERVAL` — Default: `60` (seconds between background shadow-sweep passes; each pass covers up to `WIKIVISAGE_LIFTWING_MAX_IMAGES` files of one project; `0` disables the sweep)
- `WIKIVISAGE_LIFTWING_FAIL_THRESHOLD` — Default: `3` (consecutive probe or filter-batch failures before `down`; also the worker's hysteresis)
- `WIKIVISAGE_LIFTWING_PROBE_INTERVAL` — Default: `300` seconds (`0` disables probing; read by `config.py`, shared with the web app for staleness)

**Face service server** (read by `model-server/model.py`, set as envvars on the face tool):
- `WIKIVISAGE_FACE_SERVICE_TOKEN` — Required bearer token (>= 32 chars). Mandatory on Toolforge: the service will not start without it.
- `WIKIVISAGE_MAX_REQUEST_BYTES` — Default: `41943040` (40 MiB request body cap)
- `WIKIVISAGE_FACE_WORKERS` — Default: `2` (executor threads running inference on the shared ONNX sessions; match the job's vCPUs)
- `WIKIVISAGE_WEIGHTS_DIR` — Default: `<model-server>/weights`
- `WIKIVISAGE_DET_MAX_SIDE` — Default: `1280`
- `WIKIVISAGE_MIN_FACE_PX` — Default: `20`
- `WIKIVISAGE_MAX_FACES` — Default: `300`
- `WIKIVISAGE_DET_COARSE_MIN_SIDE` — Default: `640` (`0` disables the half-scale pass for very large faces)
- `WIKIVISAGE_MAX_BATCH_SIZE` — Default: `32`
- `WIKIVISAGE_MAX_IMAGE_BYTES` — Default: `20971520` (20 MiB)
- `WIKIVISAGE_MAX_IMAGE_PIXELS` — Default: `100000000`
- `WIKIVISAGE_MAX_BOXES` — Default: `64`


## Conventions

### Python
- **Linter/formatter**: Ruff configured in `pyproject.toml` (line-length 120, target py312). Selects E/W/F/I/UP/B/SIM/S rule sets with project-specific ignores. Run: `ruff check .` and `ruff format --check .`.
- **Test framework**: pytest configured in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["."]`). Integration tests marked with `@pytest.mark.integration`, skipped unless `WIKIVISAGE_TEST_DB=1` env var is set. See **Testing** section below.
- Code uses type hints, f-strings, DictCursor everywhere.
- `execute_query()` is the universal DB read/write interface. `execute_insert()` for INSERTs needing lastrowid. `execute_transaction()` for atomic multi-step mutations.
- All DB queries use `%s` parameterized placeholders (PyMySQL). Never interpolate **values** into SQL via f-strings. F-strings are acceptable for structural SQL (e.g., building `IN (%s, %s, %s)` placeholder lists).
- All mutation endpoints (classify, reclassify, manual-face, update-face-bbox, write-sdc) use `execute_transaction` for atomicity.
- `execute_query()` returns empty **tuple** `()` not `[]` when no rows found (PyMySQL DictCursor behavior).

### Templates
- All templates extend `base.html`. Three blocks: `title`, `extra_head` (CSS/JS), `content`.
- CSS is embedded in `base.html` `<style>` tag (CSS custom properties) + per-page `{% block extra_head %}`. No external CSS files.
- Reusable CSS classes live in `base.html`: `.decorated-card` (scanline + corner overlay pattern, accent color via `--card-accent` CSS variable), unified pagination styles (`.pagination a, .pagination button`).
- Shared JS helpers are in `static/wikivisage-common.js` (thumbnail snapping, Commons URL building, gallery face view). Templates that need them add `<script nonce="{{ csp_nonce }}" src="{{ url_for('static', filename='wikivisage-common.js', v=app_version) }}">`: the `v=app_version` query is required (see *Shared scripts and the service worker* under Known Issues) and `test_templates_version_the_shared_script_url` fails without it.
- No JavaScript build system. Inline `<script>` tags in templates for page-specific logic. **Every inline `<script>` must carry `nonce="{{ csp_nonce }}"`** (the CSP has no `'unsafe-inline'` for scripts) and **inline event-handler attributes (`onclick=`, `onsubmit=`, `onchange=` …) and `javascript:` URLs are forbidden** (`script-src-attr 'none'` blocks them silently). Wire behaviour with `addEventListener` from the nonce'd script, using ids or `data-*` attributes, and delegate for markup built at runtime (build DOM with `createElement`/`textContent`, not `innerHTML` with handlers). Page-wide helpers live in the first script of `base.html`: `<button data-dismiss-parent>` removes its parent (banners, flashes), `<form data-confirm="message">` asks `confirm()` before submitting (put the message in an autoescaped attribute, e.g. `data-confirm="{{ _('...') }}"`; interpolate names with `.replace('{username}', name)`), and `showInlineAlert(message, actionUrl, actionLabel)` fills the `role="alert"` region (`#inline-alert`) instead of `window.alert()` (an `actionUrl` shows the action button, `#inline-alert-action`, default label `REAUTH_LABEL`, but only when `sameOriginHref()` accepts it: a path on this origin, never another site, a `//` path or a script/data URL; `.btn` sets its own `display`, so hidden children of the alert need the `.inline-alert [hidden]` rule). `tests/test_security_round2.py::TestTemplatesAreCSPClean` fails on a handler attribute, an un-nonced `<script>`, a `javascript:` URL or an `alert(` call.
- Title format: `Page Name - WikiVisage BETA`
- Dark brutalist/industrial theme with CSS variables: `--bg: #090e17`, `--surface: #141f33`, `--primary: #14b8a6`, etc.

### Internationalization
- All user-facing strings wrapped with `_()` (Python) or `{{ _('...') }}` (Jinja2).
- JS strings passed via `|tojson` filter into JS objects — never use `_()` in raw `<script>` blocks.
- JS i18n strings use `{placeholder}` style for `.replace()` substitution (NOT `%(name)d` which causes KeyError at render time).
- Python/Jinja2 i18n strings use `%(name)s` named placeholders (Flask-Babel convention).
- Do NOT translate: worker log messages, health endpoint JSON values, technical terms (Wikidata, Q-ID, Commons, SDC, P180, OAuth, CSRF, WikiVisage, BETA).

### Security
- Open redirect protection: `_is_safe_url()` validates all redirect targets. Additional `urlparse` guard in `set_language` rejects referrers with scheme/netloc/`//` prefix.
- Rate limiting: Global 200/hour default. `10/min` on `api_manual_face` and `api_update_face_bbox`. Uses Redis for shared storage across gunicorn workers (`WIKIVISAGE_REDIS_URL`). Falls back to `memory://` if Redis is unreachable.
- CSRF: All POST routes protected via Flask-Session tokens (`POST /cookie-consent` and `POST /account/logout-everywhere` included; the token is generated once per session, never rotated per request).
- Sessions: signed cookie, persistent, 30-day sliding idle lifetime + 90-day absolute cap, revocable per user via `users.session_epoch` ("Log out everywhere"); see *Sessions are signed cookies, revocable per user*.
- Bbox validation: All face bounding box inputs validated against `MAX_BBOX_PX` and `MIN_BBOX_AREA`.
- Security headers set on all responses: `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`.
- Token encryption at rest: OAuth access/refresh tokens can be Fernet-encrypted in the DB via `WIKIVISAGE_TOKEN_KEY` env var (opt-in). Handled by `token_crypto.py`. Decryption gracefully falls back to plaintext for legacy tokens.
- SDC writes include `maxlag=5` parameter for Wikimedia API compliance.
- Commons thumbnail proxy: URL validated via `urlparse` (scheme must be `https`, netloc must be `upload.wikimedia.org`).

### Error handling
- `app.py`: Custom error handlers for 400/403/404/429/503/500 render `error.html`. Route handlers use try/except returning flash + redirect.
- `database.py`: Custom exceptions (`DatabaseError`, `PoolExhaustedError`, `ConfigurationError`). Retry with exponential backoff on `OperationalError`/`InterfaceError`.
- `worker.py`: Each pipeline stage catches its own exceptions, logs, and returns 0 on failure (no crash propagation). Main loop catches `DatabaseError` and sleeps 10s.

### Thumbnail URLs
- Commons enforces standard thumbnail step sizes (`$wgThumbnailSteps`). Non-standard widths return 429.
- Python: `_snap_thumb_width(w)` rounds to nearest allowed step from `_THUMB_STEPS`.
- JavaScript: `snapThumbWidth(w)` mirrors the same logic in `static/wikivisage-common.js`.
- All thumbnail URL generation must go through these snapping functions.
- Standard steps: 20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px.
- Gallery cards: the server still sends 330 px as `thumb_url`, plus `thumb_urls` (`{"500": url, "960": url}`, built by `_gallery_hires_steps()` in `app.py` as direct `upload.wikimedia.org` URLs, like `thumb_url`; `{}` when the detection width is unknown). A card may swap its thumbnail **once** to one of those, at most `min(960, largest step <= detection width)` (Commons rejects thumbnails wider than the original, and a detection width below 960 is the original's width), and only when the face crop needs it (`faceViewSourceWidth()`). The swap looks the width up in the card's `data-thumb-urls` and never builds a URL itself: it must not go through `/commons-thumb`, which is login-gated and counts against the 200/hour limit (a gallery page used to cost one proxy hit per card). A missing width keeps the 330 px source, and a failed swap reverts to it. `_GALLERY_HIRES_MAX` (`app.py`) and `FACE_VIEW_MAX_SOURCE_WIDTH` (`wikivisage-common.js`) must stay equal (a test pins it).

### Shared Helpers

Extracted common patterns to reduce duplication across routes, templates, and JS.

**Python (app.py):**

| Helper | Signature | Purpose | Call sites |
|--------|-----------|---------|------------|
| `_wikimedia_api_get` | `(url, params, timeout=10) → dict` | Standardizes User-Agent header, `raise_for_status()`, `.json()` parsing for Wikimedia API GET requests | 10: `_is_human_entity`, `_commons_category_exists`, `_commons_category_has_files`, `_check_p180_exists`, `_fetch_p18_thumb_url`, `_fetch_wikidata_label`, + 4 calls in `api_category_info` |
| `_validate_bbox` | `(top, right, bottom, left) → str \| None` | Validates bounding box coordinates against `MAX_BBOX_PX` and `MIN_BBOX_AREA`. Returns error message or `None` | 2: `api_manual_face`, `api_update_face_bbox` |
| `_get_face_stats` | `(project_id) → dict[str, Any]` | Single SQL query returning 13-column face/image stats (including `model_matches`) with NULL→0 coercion | 2: `project_detail`, `api_progress` |
| `_end_session` | `(message=None) → None` | Clears the session, keeps the cookie persistent (so the flash survives on iOS home-screen apps), optionally flashes why | 4: `before_request` (absolute age), `_load_session_user` (epoch mismatch, token decryption failure), `account_logout_everywhere` |
| `_set_locale_cookie` | `(resp, lang, *, persistent) → None` | Writes the `HttpOnly`, `SameSite=Lax`, `Secure` (not in debug) `locale` cookie; 1-year lifetime only when `persistent` | 2: `set_language`, `cookie_consent` |
| `_token_unavailable_error` | `() → ApiError` | 401 "re-authorize" with `reauth_url` for a permanent refresh failure, retryable 503 for a transient one (reads `g.token_refresh_transient`) | 3: `_remove_claims_before_undo`, `_settle_claim_for_rejection`, `api_write_sdc` |
| `_reauth_url` | `() → str` | `url_for("login", reauth=1)`: the value of `reauth_url` in API answers; it only tells the page to offer the button | 2: `_token_unavailable_error`, `api_sdc_status` |
| `_reauth_login_url` | `() → str` | The re-authorize link for the page being rendered, `/login?reauth=1&next=<request.full_path>`; exposed to every template as `reauth_login_url` by `inject_reauth_login_url()` (empty outside a request) | 2: `base.html` (`REAUTH_URL`), `project_detail.html` (stored worker error) |
| `_service_unavailable` | `() → response` | 503 JSON body for `/api/*` and JSON requests, else `abort(503)` (the 503 page) | 2: `login_required`, `login` |
| `_log_session_loss` | `() → None` | Sampled INFO line `session loss: reason=... had_cookie=... client=... path=...` for anonymous requests | 1: `before_request` |

**CSS (base.html):**

| Class | Purpose | Used in |
|-------|---------|---------|
| `.decorated-card` | Scanline + corner overlay pattern. Accent color via `--card-accent` CSS variable (defaults to `--primary`) | `project_detail.html` (SDC section with `--card-accent: var(--info)`, progress card), `leaderboard.html` |
| `.pagination-controls` | Unified pagination button/link styles | `project_detail.html`, `dashboard.html` |

**JavaScript (`static/wikivisage-common.js`):**

| Export | Purpose |
|--------|---------|
| `THUMB_STEPS` | Array of Commons-allowed thumbnail widths |
| `snapThumbWidth(w)` | Snaps arbitrary width to nearest allowed Commons step |
| `commonsThumbUrl(filename, width)` | Builds proxied Commons thumbnail URL with step-snapped width |
| `sameOriginHref(url)` / `reauthHref(data)` (`base.html`) | `sameOriginHref` returns a same-origin `pathname + search` or `null` (other origins, `//` paths and script/data URLs); `reauthHref` returns the server-rendered `REAUTH_URL` when the API answer carries a `reauth_url`, else `null` |
| `postForm(url, csrfToken, fields)` | CSRF-enabled POST helper — builds FormData, returns parsed JSON Promise |
| `bboxToDisplayRect(bbox, scaleX, scaleY, offsetX, offsetY)` | Converts detection-space bbox `{top,right,bottom,left}` to display-space `{left,top,width,height}` |
| `displayRectToDetectionBbox(rect, scaleX, scaleY, offsetX, offsetY)` | Inverse of `bboxToDisplayRect` — display rect to rounded detection bbox |
| `MAX_DEVICE_UPSCALE` | `2.0`: the single tuning knob, the most device pixels per source pixel the zoom into the face may magnify a gallery thumbnail to (an original so small that it is shown whole, in contain mode, can still be magnified past it to fill the card). Companion `FACE_VIEW_*` constants: `FACE_VIEW_PAD` 0.8 (crop = box + 0.8 x box size per side), `FACE_VIEW_MIN_ZOOM` 1.1 (below it show the whole image), `FACE_VIEW_MAX_ASPECT_DRIFT` 0.03, `FACE_VIEW_MARKER_MAX_COVER` 0.7, `FACE_VIEW_MAX_SOURCE_WIDTH` 960 (equal to `_GALLERY_HIRES_MAX` in `app.py`) |
| `computeFaceView(bbox, detW, detH, natW, natH, cW, cH, dpr, opts)` | Pure (no DOM) geometry of a gallery thumbnail: `{mode: 'crop'\|'contain', imgLeft, imgTop, imgWidth, imgHeight, box, markerVisible, needNatW}`, or `null` when the geometry cannot be trusted (missing detection size, degenerate box, natural vs detection aspect off by more than 3 %). `opts.zoom === false` forces `contain`; `opts.maxDeviceUpscale` overrides the cap. `needNatW` is the natural source width the crop needs at the capped upscale (from the crop's width **and** height), `0` when zoom is off or the crop would be below `FACE_VIEW_MIN_ZOOM` (a larger source would not change that) |
| `faceViewSourceWidth(needNatW, natW, detW)` | Width of a larger Commons thumbnail worth requesting, or `0` to keep the current one; never above `min(960, largestThumbStepAtMost(detW))`. The caller then looks the width up in the server-supplied `thumb_urls`; `commonsThumbUrl()` (the `/commons-thumb` proxy) is only used by the bbox editor's 1280 px image |
| `largestThumbStepAtMost(w)` | Largest standard Commons thumbnail step <= `w` (`0` when none) |

## Known Issues & Gotchas

### faces_confirmed counter is unreliable
The `projects.faces_confirmed` column only increments on target MATCH clicks (Yes button in classify UI) and bootstrap. It does NOT reflect total classifications. The worker's inference gate now uses a direct `COUNT(*)` query instead of this counter. **Do not use `faces_confirmed` for logic — always count `is_target=1` from the faces table.**

### dlib and `face_recognition` are gone — SCRFD + ArcFace on ONNX Runtime
The face service used to run dlib HOG + `face_recognition` (128D float64). It now runs SCRFD-500M + ArcFace MobileFaceNet (512D float32) on ONNX Runtime (`model-server/backend.py`). **Do not add dlib or `face_recognition` back.** Nothing needs a system package, so there is no `project.toml`.

**The weights are third-party and non-commercial research only** (InsightFace `buffalo_sc`). They are committed as ordinary git blobs in `model-server/weights/` with `LICENSE-NOTICE` stating the terms; the project owner accepted this for the experiment. Wikimedia admins could object, so keep the `FaceBackend` protocol pluggable (YuNet + SFace is the fallback). **Never use Git LFS or submodules for them**: the Toolforge build service does a plain clone, and a pointer file would build "successfully" and then fail. Three guards enforce it: `scripts/check_no_lfs.sh` in the CI lint job, `sha256sum -c SHA256SUMS` in every face-service CI job, and `backend.verify_weights` at startup (fail-closed). `scripts/fetch_weights.sh` re-creates and verifies the files from the official release URL.

### The embedding-model wipe (one-time, in `migrate.py`)
Old 128D dlib encodings and new 512D embeddings are not comparable, so `migrate.apply_embedding_model_upgrade()` wipes derived data exactly once. It is gated on `information_schema.COLUMNS`: if `faces.embedding_model` is absent it runs `DELETE FROM images` (cascades to faces and sdc_claims; never `TRUNCATE`, InnoDB refuses it on FK-referenced tables) and `DELETE FROM sdc_claims`, resets every project (counters, inference settings, completion reason, SDC flags, worker claim, `distance_threshold` = the new default), reactivates `completed` projects, then applies the schema changes (`det_score`, comments, default) and adds `embedding_model` **last**, so an interrupted run repeats the wipe instead of skipping it. The ALTER list that runs *before* this step must never anchor a column on `det_score` or `embedding_model` — they do not exist yet on a legacy database (error 1054): `faces.aligned` is positioned after the old `confidence` column instead, and `test_run_migration_upgrades_a_legacy_faces_schema_end_to_end` exercises the full entry point against a simulated 0.10.x table. Later runs and fresh databases (whose `schema.sql` already has the column) skip it. It keeps users, projects, project_members, user_stats and worker_heartbeat. Wiping `sdc_claims` cannot create duplicate P180 claims because `write_sdc_claims` checks Commons first, and `bootstrap_from_sparql` re-seeds from existing claims. Take a DB dump before the first run on real data. `migrate.py --reset` is a different, harsher tool (it also drops users and tokens).

### The LiftWing pre-filter is shadow-only and fails open
`projects.prefilter_mode` accepts `off` and `shadow`; `enforce` is in the ENUM so adding it later needs no schema change, but settings reject it and the worker treats any non-`off` value as shadow. Whatever the model says, every image is still downloaded and face-detected — the verdicts only land on `images.prefilter_keep/_reason/_checked_at` for `scripts/eval_prefilter.py`. Do not wire the verdicts into `process_images` before that readout shows a keep recall of at least 99.5% on human-confirmed target images (the readout also reports dropped images that had detected faces, and keeps with none).

Availability is equally soft: `liftwing_client` never raises, the worker's probe is best-effort, `/health`'s `liftwing` field never sets `degraded`, and a `service_status` row older than 3 probe intervals reads as `unknown` (banner hidden) rather than showing stale data. The probe and every real batch update `service_status`, so a batch that fails while the models endpoint is fine still flips the banner after `LIFTWING_FAIL_THRESHOLD` consecutive failures.

### SCRFD-500M misses very large faces
Measured on a 960x1307 official portrait: face of 532 px found nothing at full size (faces up to ~430 px were fine), but was found at 0.78x. Tight portraits are common on Commons, so `backend.detect` adds a half-scale pass (see Face Service). A 1024 px cap on the detector input (as in the validation smoke test) would also have found it but costs small-face recall; the two-pass approach was chosen instead.

### Commons serves thumbnails from thumb.wikimedia.org
`Special:FilePath/<file>?width=N` redirects
`commons.wikimedia.org` → `commons.wikimedia.org/w/index.php` → **`thumb.wikimedia.org`**.
The original file (no `?width=`) still resolves to `upload.wikimedia.org`. Both
`worker.py` and `app.py` build thumbnail URLs (`FILE_PATH_URL`, both at
`config.DETECTION_WIDTH` = 960), so both must allowlist `thumb.wikimedia.org` in
`_ALLOWED_DOWNLOAD_HOSTS`.

Omitting it fails every download with `Redirect to untrusted host:
thumb.wikimedia.org` — and because a failed download is recorded as
`status='error'`, which `images_processed` counts as progress, the UI shows
partial completion rather than an error. This took out 103 images across 5
projects when Wikimedia rolled the change out.

The UI's own thumbnails are unaffected: `commons_thumb_url()` builds
`upload.wikimedia.org` URLs directly and does not follow redirects.

### Auto-completion must never fire while images are pending
`process_project` step 3b writes `completion_reason` of `no_faces` /
`insufficient_faces`. The batch loop above it exits as soon as one batch yields
zero *successes*, which also happens when every download fails. Without the
`pending_images > 0` guard, a transient outage is recorded as a permanent "no
faces in this category" verdict — and since `_claim_active_projects` filters
`WHERE p.status = 'active'`, the project is never picked up again and its
remaining images are orphaned. `app.py` has always had this guard; `worker.py`
did not.

### Embeddings are not bit-identical across machines
The ONNX models run in float32, so the same image embedded on different CPU / ONNX Runtime combinations can differ in the last digits (the old dlib path measured ~7.7e-07 L2; the new one has not been measured across machines). The path is deterministic against itself on one machine (a test asserts it). What *does* invalidate stored embeddings is changing the weights, the alignment template, or the preprocessing in `backend.py` — which is why every face records `embedding_model`.

Stored embeddings can also drift because **Commons re-renders thumbnails**: with the old pipeline three March-era faces re-encoded 0.06–0.2 L2 away from their stored vectors, and one was no longer detected. The input bytes changed, not the code.

### OAuth scope & token handling
SDC writes require the `editpage` OAuth grant. The access token is stored per-user in the `users` table. Token refresh is handled in `app.py` `@before_request`. Access tokens from the DB may be `bytes` — normalized to `str` at read time in `before_request`.

### Worker must be restarted after code changes
The worker is a long-running `python worker.py` process. Code changes require manual restart. On Toolforge, redeploy the continuous job. The face service is a separate container with its own lifecycle — changing `model-server/model.py` requires rebuilding and restarting *it*, not the worker.

### Shared scripts and the service worker
A page template and `static/wikivisage-common.js` are released together, but a browser can hold an old copy of the script. `static/sw.js` used to serve every `/static/*` file stale-while-revalidate, so the first page load after an update paired the **new** template with the **old** cached script: the gallery lost its zoom and face outline (`computeFaceView is not defined`) until a second reload. Three rules prevent it: (1) templates load shared scripts with `v=app_version`, so every release asks for a new URL that no cache can hold; (2) `sw.js` serves `/static/*.js` network-first (cache only as a fallback when the network fails or answers 5xx) and does not precache scripts, which also covers local development where the version does not change; (3) the cache name moves on (`wikivisage-v2`) whenever the service worker's caching rules change, because `activate` deletes every other cache. The browser tests cannot see this class of bug (they block service workers and route requests through the Flask test client), so it is covered by `test_service_worker_serves_scripts_network_first` and `test_templates_version_the_shared_script_url`, and was verified by planting the old script in both caches of a real Chromium and reloading.

### Commons thumbnail proxy
`/commons-thumb/<path>` redirects to Commons thumbnail URLs server-side. This exists because Commons thumbnails can't be directly embedded due to referrer policies on Toolforge. The route enforces standard thumbnail step sizes to avoid 429 errors from Commons.

### Commons thumbnail step sizes
Commons enforces `$wgThumbnailSteps` — only standard sizes (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px) are allowed. Non-standard widths return HTTP 429. Both Python (`_snap_thumb_width`) and JavaScript (`snapThumbWidth`) snap requested widths to the nearest allowed step.

### Shutdown handling
Worker uses `signal.SIGTERM`/`SIGINT` handlers setting `shutdown_requested = True`. Each pipeline stage and the main loop check this flag for graceful shutdown.

### Worker heartbeat & downtime banner
Each worker's heartbeat thread upserts its own row in `worker_heartbeats` every 30 s (see Worker). The old single-row `worker_heartbeat` table is no longer written or read; it stays in `schema.sql`/`migrate.py` for existing databases and `--reset` and is dropped in a later release. The web app's `inject_worker_status()` sets `worker_down` from `_workers_alive()`, and `base.html` shows the amber "The background worker appears to be offline." banner only when **none** of these holds: (1) a worker row has `last_seen` newer than `WORKER_STALE_SECONDS` (300 s); (2) a worker row has an unexpired `expected_back_until` (announced planned stop); (3) a project claim (`projects.worker_claimed_at`) was refreshed within the same 300 s (checked only when (1) and (2) fail; it still proves work is going on when heartbeat writes fail). An empty `worker_heartbeats` table (a fresh install, or the minutes between a deploy's migration and the workers' first beat) and any error while checking also count as "not down", so an unreadable database never paints the banner. Helpers: `_worker_heartbeat_counts()`, `_recent_project_claim()`, `_workers_alive()`, `_workers_health_value()`. `/health` reports `"workers": {"fresh": n, "stopping": k, "stale": m}` (`"unknown"` when unreadable); it is informational and never sets `degraded`. The banner is hidden on the landing page and leaderboard (not relevant there).

**Deploys do not show the banner, and the deploy order is unchanged.** `deploy.yml` still stops the workers before migrating and starts them only after `/health` reports `face_service: reachable` (that order is a documented safety property). What keeps the banner away is the planned-stop marker: `toolforge jobs delete` sends SIGTERM, the worker writes `state='stopping'` with `expected_back_until` 20 minutes ahead, and the restarted worker's first beat clears it. A SIGKILL or OOM kill writes nothing, so the banner appears after 5 minutes. The same holds for a worker that dies before `_startup_done` (no `running` row is written during startup, so a crash loop shows the banner) and for a worker the stall watchdog stopped (no marker after a reported stall). `/health` reports rows of retired workers as `stale` until the next worker start deletes rows older than 7 days.

The LiftWing pre-filter has its own status path: the worker's probe writes `service_status`, and the web app's `inject_liftwing_status` context processor reads it (no network call). `base.html` shows a dismissible banner on the project detail/settings pages when the filter is down **and** the project uses it; the settings page also renders `reachable`/`unreachable`/`unknown` with the last check time. A row older than 3 probe intervals reads as `unknown`.

### The inference-status card follows cycles by polling
`project_detail.html`'s `#inferenceBanner` tracks the whole processing → inference lifecycle. It used to be a page-load snapshot. The card renders whenever inference can still happen: the viewer crossed `min_confirmed` and (eligible faces > 0 or the project is still processing images), or the page came from a re-run (`?inference_triggered=1`). With nothing eligible at render time it starts hidden (`style="display:none"`), and the `/api/progress` poller is what makes it live: it reveals the card when a model cycle begins, hides it again between cycles, and shows "Model inference complete — N new matches found" only once `complete` **and** `pending_images == 0`, then stops. `N` is the growth of `model_matches` (target faces classified by the model); `by_model` also counts non-matches. A quiescent page stops after the first poll. Do not re-gate the card or its poller on "running right now". A page loaded while nothing was eligible used to show no live status at all, and once `images_processed == images_total` the progress poller is off too. The poller's live `SDC Pending` update must add `sdc_removal_pending_faces` to match the server-rendered combined figure. Two review-driven rules: the poller stops itself the moment the banner element is gone (a dismissed card must not leave a 5 s `/api/progress` loop), and the observed-work marker persists across the hidden gaps (eligible faces seen in any poll, or `by_model` growth since page load), so a cycle that finishes between two polls still finalizes the card instead of stopping silently.

### Toolforge health check (file-based liveness)
Separate from the DB heartbeat above (though written by the same thread), each worker also touches a per-worker file at `$HOME/.wikivisage-worker-alive-{worker_id}` via `_touch_heartbeat_file()` on every 30 s tick, **before** the database write. The `healthcheck.sh` script accepts a worker ID as `$1`, checks the file exists and was modified within 5 minutes (`find -mmin +5`). Toolforge's `--health-check-script` runs this every 10s; 3 consecutive failures trigger automatic pod restart. Because the file no longer depends on the database or the main loop, a DB outage does not restart workers; a worker with no progress for `STALL_LIMIT` (15 min) stops touching the file and is restarted. Per-worker files are critical because `$HOME` is shared NFS on Toolforge — a single file would be kept fresh by any surviving worker, hiding a dead one.

### Image download limits
Both `app.py` and `worker.py` enforce a 50MB download size cap (`MAX_IMAGE_DOWNLOAD_BYTES`) via streaming download with early abort. The worker additionally validates image pixel dimensions before shipping bytes to the face service (`MAX_IMAGE_PIXELS = 100M pixels`), so oversized images are rejected before they are base64-inflated and sent over the wire. The service enforces its own independent ceilings.

### Category traversal limits
`MAX_IMAGES_PER_PROJECT = 9000` caps how many images the worker will insert per project during category traversal. The worker checks existing image count before starting and calculates remaining capacity. Uses batch `INSERT IGNORE` instead of per-file SELECT+INSERT.

### SDC writes are user-triggered only
The "Send Edits to Wikimedia Commons" button on the project detail page is the ONLY way SDC claims are written. The button sets `sdc_write_requested=1` on the project; the background worker picks this up on the next poll cycle and writes claims in batches. The web UI polls `/api/sdc-status/<id>` for progress. SDC writes run in the worker process, not the web process. Writes include `maxlag=5` for Wikimedia API compliance.

Rejections and undos must never leave a stale P180 claim behind, including when they race a write run that already selected the face:
- The worker re-reads each face's `is_target`/`superseded_by` right before the `wbeditentity` call and skips faces that changed; the same run's removal phase then clears `sdc_removal_pending` (and `sdc_written`) for those faces. After a successful write it sets `sdc_written=1` only `WHERE is_target = 1`; if the face was reset or undone in the sub-second window after the re-check, it queues `sdc_removal_pending=1` instead of leaving the freshly added claim orphaned. That recovery covers both non-target states: `is_target IS NULL` (classify undo) and `is_target = 0` (review undo), including a stop-then-undo interleaving where the undo route sees no in-flight write request and queues nothing itself. The idempotency-hit path (`already_exists`) marks `sdc_written=1` only for a still-target face and never queues a removal, because the claim predates this run and may not be ours to delete. The re-run route also refuses while `sdc_write_requested=1`.
- The web side queues `sdc_removal_pending=1` for every rejection where no sibling target confirms the image (bootstrap faces and `sdc_written=0` faces included) and removes on-record claims for non-bootstrap written faces immediately (502 if that removal fails, so the rejection is not recorded).
- Any transition back to `is_target = 1` (classify target, review approve, bbox redraw, worker bootstrap auto-classify and reconciliation) clears `sdc_removal_pending`, so a queued removal cannot strip the claim of a face that has become a target again. Model inference never creates such a transition: `run_autonomous_inference` skips faces with a queued removal, so a pending face stays unclassified until the queue is resolved and only then becomes eligible.
- Undo removes the claim for faces with one on record (queueing on failure), queues unwritten targets only while a write run is active (`sdc_write_requested=1`, the only window where a claim can be in flight), and refuses to delete a manual face whose claim could not be removed first. Undoing a rejection ("none") is the opposite case and never deletes a claim: it cancels the removal the rejection queued (rejections only set the flag on bootstrapped images; on other images a queued flag predates the rejection — an earlier undo racing a write run — so it is kept for the worker to verify). Accepted residual: a manual face deleted in the same sub-second window between the worker's re-check and its `wbeditentity` call can still strand one claim; the re-check closes the realistic window.
- A long write run keeps itself alive: `_keepalive_sdc_write` reports progress (so the heartbeat thread does not see a stall and `healthcheck.sh` stays green) and refreshes the project claim every 5 faces and once per removal page, so runs longer than `CLAIM_EXPIRY_MINUTES` are not picked up by the other worker.

### Model Results validation UI
The project detail page includes approve/reject/edit-bbox controls on each Model Results gallery card. Key behaviors:
- **Approve** (checkmark): Sets `is_target=1, classified_by='human'` via `/api/reclassify`. Updates card badge in-place.
- **Reject** (X): Sets `is_target=0, classified_by='human'`. If the face had `sdc_written=1`, also removes the P180 depicts claim from Commons via `_remove_sdc_claim()` (uses `wbgetclaims` -> `wbremoveclaims`).
- **Edit bbox** (pencil): Opens a modal with a 1280px image. User draws a new bounding box. Old face is kept; a new face row is inserted via `/api/update-face-bbox` (face encoding recomputed server-side; `faces.aligned` records whether the detector confirmed the box or the embedding fell back to an unaligned resize). Drawing works with mouse **and touch**: the drawing area listens to `touchstart`/`touchmove`/`touchend` on top of the mouse handlers and sets `touch-action: none`, so a finger drag draws instead of scrolling the page (it was desktop-only before); the geometry (scale + offsets) is measured from the live layout on every event and re-placed on `resize`/`orientationchange`, so rotating the phone mid-modal does not skew the box. `touchcancel` aborts the drag and restores the previous box instead of committing a partial one. The modal body scrolls and the image is capped at 50vh below 600px so the image, hint and Save button fit on a phone.
- **Unreadable image**: if the face service rejects the image itself (`FaceServiceRejected` with `error_code: invalid_image`, which `encode_known_face` re-raises), both `/api/manual-face` and `/api/update-face-bbox` return **422** with "This image could not be read by the face detection service." (retryable service failures stay 503; box-level rejections still answer "draw a slightly larger box").
- **Gallery thumbnails** (`cropToFace()` in `project_detail.html`, geometry from `computeFaceView()`): cards zoom into the classified face but the zoom never exceeds `MAX_DEVICE_UPSCALE` (2.0) device pixels per source pixel — when the 330 px source is too small, more surroundings are shown instead of a blurry close-up, or one larger thumbnail is requested (see Thumbnail URLs; no larger thumbnail is requested when the card stays uncropped anyway: `needNatW` is `0` below `FACE_VIEW_MIN_ZOOM`). The classified face is outlined by `.face-marker` (decorative, `aria-hidden`, at least 10 px, hidden when the box covers over 70 % of the card). Bootstrap faces are not zoomed but are outlined. Rows with no detection size, or whose thumbnail aspect differs from the detection aspect by more than 3 % (rotated small originals), show the plain image with no marker. The layout re-runs on resize through a `ResizeObserver` on `#resultsGallery` (`resize`/`orientationchange` listeners when it is missing).
- **Filter interaction**: After reclassification, `data-source` is NOT changed — the card remains visible under its original source filter (Model/Bootstrap). Only the visible method label updates to "human".
- **State sync**: the reclassify success handler refreshes `data-sdc-written` from the response's `sdc_written` (the authoritative post-action value — reject always clears it, approve may set it when the P180 check finds the claim already on Commons) and computes the "send pending" badge from it, so a reject-then-approve cannot keep a stale written flag. The removal counter is decremented only when the card actually carried a queued removal; a synchronous removal (`sdc_removed`) shows the note without touching the counters, which keeps them consistent with the server stats between reloads.

### Sessions are signed cookies, revocable per user
Sessions are still Flask signed cookies, with no server-side session store (the unused `sessions` table was dropped: `schema.sql` no longer creates it and `_ALTER_MIGRATIONS` runs `DROP TABLE IF EXISTS sessions` on existing databases; `_ALL_TABLES` still lists it so `--reset` cleans older databases). What changed is revocation and lifetime:
- **Lifetime.** Idle lifetime is 30 days (`WIKIVISAGE_SESSION_IDLE_DAYS`, was 8 h), sliding: `SESSION_REFRESH_EACH_REQUEST = True` is explicit, so every response re-sends the cookie. The absolute cap is 90 days from login (`WIKIVISAGE_SESSION_MAX_AGE_DAYS`): `session["auth_at"]` is set at login, a session without it (created before this change) is backfilled with "now" so nobody is logged out on deploy, and one past the cap is cleared with a flash (`_session_within_max_age()`, `_end_session()`).
- **`users.session_epoch`** (INT UNSIGNED, default 0). Login stores `session["epoch"]`; `before_request` compares it with the user's row (a missing key counts as 0 on both sides) and a mismatch clears the session. `POST /logout` ends **this device only** (clears the cookie, epoch untouched). `POST /account/logout-everywhere` bumps the epoch and clears the session, which ends the user's sessions on every browser and device.
- **Residual.** A stolen signed cookie stays valid until the idle lifetime or the 90-day cap runs out unless the user presses "Log out everywhere" (which makes it invalid at once); a plain logout does not revoke a copy of the cookie. Deleting the user row also revokes it, because `before_request` loads the user on every request.
- **Every `session` cookie is persistent** (`session.permanent = True` in `login()`, `_csrf_token()` and after each `session.clear()`, plus an `after_request` safety net `_keep_session_cookie_persistent` that makes any non-empty session persistent; an empty session gets no cookie). An anonymous page view creates no session: `base.html` renders a CSRF token for the cookie banner only when the visitor is logged in or already has a `locale` cookie (the only case where `POST /cookie-consent` has anything to change), and the banner skips the POST without one. Reason: iOS home-screen apps reset session-scoped cookies (WebKit bug 272325), which dropped `oauth_state`, the CSRF token and flashes. The CSRF token is never rotated per request.
- **Database errors keep the session**: see the 503 handling under Routes.

### Cookies
Only 2 cookies: `session` (strictly necessary, managed by Flask's built-in signed session cookie mechanism, persistent, `HttpOnly`, `SameSite=Lax`, `Secure` unless debug) and `locale` (functional, language preference, `HttpOnly`, `SameSite=Lax`, `Secure` unless debug; set by `_set_locale_cookie()`). No tracking cookies. The old `jimmy_pet` cookie is gone: the mascot preference lives in `localStorage` only and `base.html` copies a leftover cookie's value into `localStorage` (when none is set) before deleting it by script. A non-blocking consent banner is shown; the choice lives in `localStorage.cookie_consent`. Without consent the `locale` cookie is session-scoped: switching language still works (`/set-language/<lang>?nocookie=1`), but the choice is only persisted (1 year) once the visitor accepts. Because the cookie is `HttpOnly`, JS cannot change it, so the banner buttons `POST /cookie-consent` (fire and forget): Accept upgrades an existing `locale` cookie to 1 year, Decline downgrades it to session-scoped. The footer "Cookie settings" button clears `localStorage.cookie_consent` and shows the banner again; the banner text describes what is stored. The banner is a `role="region"` labelled "Cookie consent", and "Cookie settings" moves focus to its Accept button.

**Session-loss log line.** To make "I have to log in again" countable, `before_request` writes a sampled INFO line for anonymous requests: `session loss: reason=<no cookie sent|expired|bad signature|valid but empty> had_cookie=<bool> client=<ios|android|desktop|other> path=<path>`. `_SESSION_LOSS_SAMPLE_RATE = 0.05`; `WIKIVISAGE_LOG_SESSION_LOSS=0` disables it; it skips `/static/`, `/health`, `/sw.js`, `/robots.txt`, `/sitemap.xml`, `/llms.txt`, `/commons-thumb/` and `/.well-known/`. `no cookie sent` also counts first-time visitors and bots, so compare counts over time rather than reading single lines. iPadOS sends a Macintosh user agent, so iPads land in `client=desktop`. Helpers: `_log_session_loss()`, `_session_loss_reason()`, `_client_bucket()`.

**Unverified platform risk:** iOS home-screen apps keep their own cookie jar and open the Wikimedia OAuth page outside the app shell, so a login started inside the installed app may not return into it. `test_session_persistence_browser.py` proves the persistent cookie and the OAuth state survive a full browser restart in Chromium and WebKit, but it cannot reproduce iOS home-screen cookie-jar isolation. This has not been checked on a real device; the platform matrix in TESTING.md lists every client to test.

### Worker memory is bounded by DETECTION_WAVE_SIZE, not BATCH_SIZE
`process_images` works in waves. A wave's downloaded image bytes are all resident at once while its detection request is in flight, so `DETECTION_WAVE_SIZE` (8) is what caps peak memory — not `BATCH_SIZE` (50). Downloading a whole batch up front would be 50 x 50 MB worst case. Inference itself is cheap: each encoding is 2048 bytes, so even 10K faces is ~20 MB.

### Wikidata label fetching
`_fetch_wikidata_label(qid)` in `app.py` calls the Wikidata `wbgetentities` API with `languages=en&languagefallback=1` to retrieve human-readable labels. The `languagefallback=1` parameter makes the API automatically fall back through language variants (e.g., `mul` → `en`, `en-ca` → `en`) when no exact `en` label exists. Used during project creation to auto-fill empty project labels. Without this parameter, entities like Q153694 (Michael Bublé) return empty labels because they only have `mul`/`en-ca`/`en-gb` labels, not `en`.

## Testing

Hybrid test suite: unit tests (run in CI), integration tests (require local Docker MariaDB) and contract tests (require a running face service).

### Architecture

- **Unit tests**: Pure mocks, no DB, no network. Run everywhere (CI, local). Cover thumb snapping, URL safety, CSRF validation, error classes, migration parsing, route logic, token encryption, CSP hardening, PKCE, single-flight Redis lock, idempotent SDC removal, mid-batch cancel, the face service client (wire parsing, batching, error classification), and the LiftWing pre-filter client (verdict parsing, fail-open, circuit breaker, probe).
- **Integration tests**: Hit a real MariaDB via Docker. Marked with `@pytest.mark.integration`. Run in the `integration` CI job (MariaDB service container) and locally when `WIKIVISAGE_TEST_DB=1` is set.
- **Contract tests**: Run the real `face_client` against a real running face service container. Marked with `@pytest.mark.contract`, skipped unless `WIKIVISAGE_FACE_SERVICE_CONTRACT=1`. This is the only place both sides of the HTTP boundary are exercised together — everything else mocks it — so it is what catches a broken Dockerfile, an unresolvable dependency, or wire-format drift. Auth tests additionally need `WIKIVISAGE_FACE_SERVICE_TOKEN` set on both the container and the client. Tests that need a real face read `tests/data/portrait_sample.jpg` (override: `WIKIVISAGE_TEST_SAMPLE_IMAGE`); they skip when it is absent unless `WIKIVISAGE_REQUIRE_SAMPLE` is an explicit affirmative (`1`/`true`/`yes`/`on`; `"0"`/`"false"`/unset skip). CI sets it. The sample **is committed**: a close-cropped public-domain White House portrait whose ~514x756 px face at 960 px exercises the half-scale pass; `tests/data/README.md` records its provenance. CI runs the contract tests in both face-service jobs, with a token, and fails if the service log contains ORT `VerifyOutputSizes` warnings.
- **Face-service unit tests**: `test_face_service_backend.py` (fake ONNX sessions for decode/scaling/alignment/chunking/box recovery, plus the real weights) and `test_face_service_model.py` (the KServe envelope with a scripted backend) import `model-server/` directly and need `onnxruntime`, `opencv` and `kserve`; they `importorskip` in the plain environment and run in the `face-service-unit` CI job.
- **Guard tests**: `test_face_service_guard.py` loads `model-server/guard.py` by file path (it is stdlib-only), so the public-facing auth and body-cap rules are unit-tested without KServe or the ML stack.
- **Test DB**: `wikiface_test` — created fresh per pytest session, dropped on teardown. Never touches `wikiface_dev`.
- **Config**: `pyproject.toml` has `testpaths = ["tests"]`, `pythonpath = ["."]`, and integration marker.
- **Coverage**: the baseline when the gates were added was app 95%, worker 70% (its main loop and SDC writer were the least covered code in the repository), total 86%. After splitting `write_sdc_claims` and `main` it is worker 91%, total 93%. Floors live in `scripts/check_coverage_floors.py` and `fail_under`; raise them when coverage rises, never lower them to pass.
- **Complexity ratchet**: Ruff's `C90` rule with `max-complexity = 20` in `pyproject.toml` (it was 70, then 30). It is set to the worst function that exists, so it only blocks new giants; lower it after each refactor, target 15.
- **Test order independence**: every `test_app_*.py` passes on its own and in reverse order. Tests that change shared app config must restore it (`flask_app.config["SERVER_NAME"] = None`, never `pop`: a missing key breaks Flask's URL adapter for whatever runs next).

### Test files

The `test_app_*.py` files (`auth`, `classify_api`, `commons`, `face_drawing`, `gallery_thumbs`, `helpers`, `membership`, `misc`, `project_pages`, `sdc`, `system`, `undo_reclassify`), `test_config.py`, `test_gallery_filter.py`, `test_oauth_refresh.py`, `test_worker_loop.py`, `test_worker_sdc.py`, `test_bbox_editor_browser.py`, `test_gallery_thumb_browser.py`, `test_session_persistence_browser.py`, `test_database.py`, `test_face_client.py`, `test_face_service_backend.py`, `test_face_service_model.py`, `test_face_service_contract.py`, `test_face_service_guard.py`, `test_liftwing_client.py`, `test_migrate.py`, `test_security_round2.py`, `test_token_crypto.py`, `test_translations.py`, `test_worker.py`, `test_worker_heartbeat.py`, `test_cookie_consent.py`. Counts are not tracked here: run `pytest tests/ -q` for the current numbers (unit tests that need the ML stack or a database show as skipped in the plain environment).

`test_security_round2.py` no longer contains the `TestSubprocessHardening` tests, which covered `_harden_face_detect_subprocess()`; those guarantees are now container-level (see Architecture) and are not unit-testable.

Phase 2 added: the LiftWing client suite (`test_liftwing_client.py`); `prefilter_images` mode/fail-open/cap tests, probe claim/hysteresis tests and `_fetch_metadata` tests in `test_worker.py`; status-snapshot, banner, `/health` and settings-POST tests in `test_app.py`; migration/schema assertions in `test_migrate.py`, including `test_schema_semicolons_are_only_statement_terminators` — `load_schema` splits `schema.sql` naively on `;`, so a semicolon inside a comment or a quoted `COMMENT` would silently chop a statement in half and only fail on a real database.

The follow-up round added: `test_eval_prefilter.py` (the readout math under mock rows), auth-header/429-`Retry-After`/minimal-completion-probe tests in `test_liftwing_client.py`, prefilter success/failure `service_status` wiring and the "prefilter runs after the image loop" ordering test in `test_worker.py`, 422 `FaceServiceRejected` tests for both bbox routes plus the `aligned` flag assertions in `test_app.py`, `faces.aligned` migration/schema assertions in `test_migrate.py`, and a `require_sample_env` truthiness test pinning that `WIKIVISAGE_REQUIRE_SAMPLE=0` means "not required".

The Copilot-review round added: `test_run_migration_upgrades_a_legacy_faces_schema_end_to_end` (full migration entry point against a simulated 0.10.x `faces` table), the `encode` rejection-classification tests (`invalid_image` re-raised vs an unusable region staying `None`) in `test_face_client.py`, both bbox routes (through the real client parsing path — only `_post` stubbed) and the live contract suite, `error_code` assertions on per-instance error entries in `test_face_service_model.py` (plus a parity check against `face_client`'s constant), and the LiftWing breaker-serialization (`test_queued_calls_see_the_breaker_the_running_call_opens`) and 429-deadline probe tests.

The inference-card round added: four real-template `test_project_detail_inference_banner_*` cases in `test_app_project_pages.py` — hidden while processing (banner present with `style="display:none"` plus the poller), absent when idle, visible while running (with the combined `sdc_removal_pending_faces` stat expression pinned), and the `?inference_triggered=1` completion state.

The stale-claim round added: worker tests for the pre-write live-state re-check (face rejected / superseded / vanished / unreadable, and the whole-batch guard), `_keepalive_sdc_write` statement coverage, the every-5-faces keepalive cadence, and the removal pass clearing `sdc_written`; app tests for undo claim handling (immediate removal, removal failure queues, missing token queues, sibling guard, in-flight write queues, no-in-flight does not queue, manual-face deletion blocked on failed removal and allowed after removal) and the reclassify `sdc_written=0` queue path.

The review-follow-up round added: the reclassify `sdc_written` echo test (approve with an existing P180 echoes 1), the post-action `sdc_written == 0` assertions on the reject paths, and flash assertions on the undo tests (removed / queued flashes set only on success, empty otherwise or after a blocked manual-face undo).

The undo-semantics round added: undo-of-rejection tests (a `sdc_removal_pending=1` state row cancels the queue instead of removing the claim; the written marker is preserved; a non-bootstrapped pre-existing queue is kept for the worker) and the model-inference exclusion test (`run_autonomous_inference`'s candidate query requires `sdc_removal_pending = 0`, plus an integration case proving a pending face is left unclassified).

The DevOps-follow-up round added: `test_translations.py` (the committed `.mo` catalogs must serve exactly what their `.po` sources declare — catches a forgotten `pybabel compile` locally and in CI) and a pinned Ruff (`ruff==0.17.2`) in the CI lint job so a new Ruff release cannot turn it red on unchanged code.

The session/heartbeat/thumbnail round (0.12.0) added: `tests/test_cookie_consent.py` (27: `POST /cookie-consent` choices, CSRF, invalid choice, rate limit, `locale` cookie flags incl. `Secure` off in debug and in the LAN dev mode, no session cookie for anonymous page views, banner and footer markup); `tests/test_app_auth.py` grew to 107 (the locale-cookie sink guard included; same-account re-authorize keeps the classify state, a different account drops it, `Sec-Fetch-Site` on `?reauth=1`, logout and `/` during a database outage, session lifetime and `auth_at` backfill/cap, `session_epoch` mismatch, "Log out everywhere" vs plain logout, 503 instead of a login redirect on `DatabaseError`, 401 vs 503 from `_token_unavailable_error()`, session-loss log reasons and sampling); `tests/test_oauth_refresh.py` is now 44 (the `on_failure` callback and `is_permanent_refresh_error()`); `tests/test_worker_heartbeat.py` (36: tick order file-then-row, DB failure never blocks the file, the stall watchdog, the startup gate, the exempt sweep thread, no planned stop after a stall, the planned-stop upsert, plus integration tests against the real `worker_heartbeats` table); rewritten heartbeat assertions in `test_worker.py`/`test_worker_loop.py`/`test_worker_sdc.py` (incl. the heartbeat joined first in `_release_resources` and the pool of 5 + 1 connection budget) and `DedicatedConnection` tests in `test_database.py`; `_worker_status_db`-based banner tests in `test_app_system.py` (one per signal and combination: fresh, stale, stopping, recent claim, empty table, error) and the `/health` `workers` field; `worker_heartbeats` in the `conftest.py` truncation list and the schema/migration assertions in `test_migrate.py`; and `tests/test_gallery_thumb_browser.py` (Chromium + WebKit, 52 runs: zoom cap, face marker, bootstrap outline, aspect-mismatch fallback, the hi-res swap from server-supplied direct URLs and its fallback, no swap for a card that stays uncropped, resize relayout, the Re-authorize button, a hostile `reauth_url` in the API answer, and the same-origin guard on its action URL) and `tests/test_session_persistence_browser.py` (2 runs: a persistent profile over real HTTP keeps the session cookie across a browser restart, and the callback still reads `oauth_state`), and `tests/test_app_gallery_thumbs.py` (16: `thumb_urls` per detection width and file type, the 960 limit pinned against the JS constant, the service worker's 5xx fallback in node). The review-fix round added a rendered-HTML guard that `#sdcResult` is closed before the action buttons (a missing `</div>` once nested them inside the area `sdcShowMessage` empties). The full suite was 1576 passed / 31 skipped with the database at 93.9 % coverage, all 11 per-module floors unchanged (1509 passed / 98 skipped without it).

### Commands

```bash
# Full suite (unit + integration) — requires Docker MariaDB running
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# CI mode (unit only — integration tests auto-skipped)
pytest tests/ -v

# Single test file
pytest tests/test_app_auth.py -v

# With coverage
WIKIVISAGE_TEST_DB=1 pytest tests/ --cov=. --cov-report=term-missing
```

### Fixture Hierarchy (`tests/conftest.py`)

```
test_db (session) → creates/drops wikiface_test DB
├── db_conn (function) → raw pymysql connection, truncates all tables after test
├── db_pool (function) → initializes database.py pool for test DB
├── integration_app (function) → Flask app connected to test DB
│   └── integration_client (function) → logged-in test client with session
├── seed_user → inserts test user, depends on db_conn
├── seed_project → inserts test project, depends on db_conn + seed_user
├── seed_images → inserts 5 test images, depends on db_conn + seed_project
├── seed_faces → inserts target + non-target faces, depends on db_conn + seed_images + seed_user
├── seed_unclassified_faces → inserts unclassified faces, depends on db_conn + seed_images
└── seed_bootstrap_image → inserts bootstrapped image + face, depends on db_conn + seed_project
```

### Conventions

- **Embedding helper**: `_make_embedding(seed, dim=512)` generates deterministic unit-norm float32 embeddings (as bytes) for face rows.
- **Import pattern**: `conftest.py` helpers use try/except: `try: from conftest import X` / `except: from tests.conftest import X` for compatibility.
- **DB connection**: `host=127.0.0.1, user=root, password=devpass, port=3306`.
- **Assertion gotcha**: `execute_query()` returns empty **tuple** `()` not `[]` — use `len(result) == 0` not `result == []`.
- **Integration test isolation**: Each `db_conn` fixture truncates all tables after the test via `SET FOREIGN_KEY_CHECKS=0`, then re-inserts the `service_status` seed row (`liftwing`) that `schema.sql` created.
- **Rate limiting is disabled in tests**: `tests/_app_preamble.py` disables the limiter at import, and `tests/conftest.py` disables it in both the `flask_app` and `integration_app` fixtures (each reloads the module, so each needs its own disable). The limiter keys on the client address, which is always `127.0.0.1`, so a full run would exhaust the global 200/hour quota half-way through and turn unrelated assertions into 429s. Limiting itself is covered by `test_error_handler_429` (which fakes the abort, so it needs no limiter).
- **App auth simulation**: Set `session["user_id"]` in test client session transaction.

## CI/CD

### CI (`.github/workflows/ci.yml`)

Runs on push/PR to `main`. Seven jobs:

1. **Lint**: Ruff check + format check, then **mypy** (pinned `mypy==2.4.0`; scope is the small modules listed in `[tool.mypy]`: config, token_crypto, redis_lock, database, face_client, liftwing_client, oauth_refresh, api_errors) (Python 3.12; Ruff is version-pinned in the workflow so a new Ruff release cannot turn the job red on unchanged code), then `scripts/check_no_lfs.sh` — fails if `.gitmodules` exists, a weight file is routed through the LFS filter, or a weight file is an LFS pointer.
2. **Test**: `pytest --tb=short -q` on Python 3.12 and 3.15.1 matrix, installing the pinned root lock. Integration tests auto-skipped (no `WIKIVISAGE_TEST_DB` env var in CI); the face-service unit and contract files skip too (no ML stack here). No system deps needed — the ML stack lives in the face service, and the suite mocks `face_client` rather than reaching the network. Caches pip dependencies.
3. **Integration tests (MariaDB) + coverage** (`integration`): a `mariadb:10.11` service container, then `pytest --cov` with `WIKIVISAGE_TEST_DB=1`, so the tests that guard the race-condition fixes (SDC writes, inference persistence, claims, migrations) run in CI and not only locally. `[tool.coverage.report] fail_under` gates the whole project and `scripts/check_coverage_floors.py` gates each module; both are ratchets.
4. **Face service (backend + model unit tests)** (`face-service-unit`): verifies `model-server/weights/SHA256SUMS`, installs `model-server/requirements.txt` + dev requirements on Python 3.12, and runs `test_face_service_backend.py` and `test_face_service_model.py` with `WIKIVISAGE_REQUIRE_SAMPLE=1`.
5. **Face service** (`face-service`): verifies `SHA256SUMS`, builds `model-server/` with buildx (gha layer cache), starts the container **with a random token**, waits for readiness, runs the contract tests through the guard, and fails if the service log contains ONNX Runtime `VerifyOutputSizes` warnings. It builds the Dockerfile.
6. **Face service (Toolforge buildpack)** (`face-service-buildpack`): verifies `SHA256SUMS`, builds `model-server/` with `pack` and Toolforge's own builder (`tools-harbor.wmcloud.org/toolforge/heroku-builder:24_20261023`, buildpacks `heroku/python` + `heroku/procfile` only — no deb packages are needed; keep the builder tag in step with Toolforge's current default, published at `Module:Toolforge images/buildpack data.json`) — the production image — starts it via the `web` Procfile entry on port 8000 with `TOOL_TOOLFORGE_API_URL` set (so the fail-closed token check is live), and runs the same contract tests and log check. Catches `requirements.txt` / `Procfile` breakage that only affects the buildpack path.
7. **Browser regressions** (`browser`): installs the optional `requirements-browser.txt` and Chromium/WebKit, then runs `test_bbox_editor_browser.py` (20 runs), `test_gallery_thumb_browser.py` (52 runs) and `test_session_persistence_browser.py` (2 runs) across Chromium and WebKit with `WIKIVISAGE_BROWSER_TESTS=1`. The first two render the real Flask page with mocked DB/gallery/thumbnail responses and exercise the DOM events and layout, including multi-touch cancellation, rotation and landscape scrolling for the bbox editor and the zoom cap, face marker and hi-res thumbnail swap for the gallery. The session test starts the app on a loopback socket and loads its pages in a persistent browser profile, so the browser itself accepts the `Set-Cookie` header and keeps the cookie across a restart, rather than a mocked transport or Playwright's HTTP client. The plain suite skips these tests; production never installs Playwright.

The contract tests need `tests/data/portrait_sample.jpg` (CI sets `WIKIVISAGE_REQUIRE_SAMPLE=1`). The sample is committed, so the three face-service jobs run the real-face tests rather than failing by design.

Concurrency: `ci-${{ github.ref }}` with cancel-in-progress.

### Dependency audit (`.github/workflows/audit.yml`)

Runs on push/PR to `main` when a requirements file changes, every Monday at 06:17 UTC, and on `workflow_dispatch`. Python 3.12 (the project floor: `numpy` 2.5 requires `>=3.12`; the face service additionally caps `kserve` at `<3.14`).

1. `pip-audit -r requirements.txt` and `pip-audit -r model-server/requirements.txt`; any known vulnerability fails the run. pip-audit is version-pinned like Ruff.
2. `uv pip compile` (uv pinned to 0.13.5, the version Dependabot ships) regenerates `model-server/requirements.txt` in place and `git diff --exit-code` fails when the committed lock is not what `requirements.in` produces — a hand-edit, an unresolved merge, or a Dependabot PR that did not regenerate. The recipe is in `TESTING.md` §7.

A CVE with no fixed release yet is suppressed with `--ignore-vuln <ID>` on the affected `pip-audit` line, with the ID and a review date in a comment; remove it when a fix lands.

**Dependabot** (`.github/dependabot.yml`) watches root `uv`, `model-server` `uv` and `github-actions`, weekly. The root lock is generated from `requirements.in` with `uv pip compile --universal --python-version 3.12 -c model-server/requirements.txt requirements.in -o requirements.txt` (run from the repository root, uv 0.13.5); the `-c` keeps the packages both locks pin on the same versions, because the face-service CI job installs both. `audit.yml` regenerates it and fails on any diff, which is also the safety net if a Dependabot PR does not regenerate it with that command. The face service entry is grouped into one PR because its transitive lock has to move as a unit, and the uv ecosystem reads `requirements.in` and regenerates `requirements.txt`. `protobuf` is deliberately not pinned in `model-server/requirements.in`: `kserve 0.21.0` already requires `>=6.33.5,<7`, and the exact `protobuf==` line gave Dependabot a requirement to rewrite to 7.36.2, which can never resolve (the weekly run failed with an unsatisfiable kserve/protobuf conflict). The lock keeps the tested 6.33.6 pin, and `protobuf >=7.0.0` is also ignored in `dependabot.yml` as a second layer; drop that rule when a kserve release widens the cap. The Dockerfile is deliberately not watched: production builds from the Toolforge buildpack (which resolves the latest Python 3.12 patch on every build), so the Dockerfile only affects local development and the CI image job.

### CD (`.github/workflows/deploy.yml`)

Triggered on GitHub release publish or manual `workflow_dispatch`. Inputs:
- `tag` (required): Git tag to deploy (e.g. `v0.7.6`).
- `db-reset` (optional): Choice `true`/`false` (default `false`). Runs `migrate.py --reset` to wipe and recreate all tables.
- `db-reset-confirm` (optional): Must type `WIPE` to confirm when `db-reset` is `true`. Deploy fails without confirmation.

Configuration: `toolforge` environment secrets `HOST` / `USERNAME` / `KEY`; the SSH user must maintain both tools. Optional repository variable `FACE_TOOL` overrides the face tool name (default `wikivisage-face`). Every release deploys both tools.

Steps (one job, `contents: write` for the tag push):
1. Validate inputs, checkout the release tag, configure SSH.
2. **Publish `face-service-<tag>`** — `git commit-tree HEAD:model-server` so the tag's root *is* `model-server/`; buildpacks cannot build from a subdirectory. An existing tag is reused only if its tree matches.
3. Upload `lib.sh` (shared `build_and_wait`, prints `toolforge build quota` on failure), `deploy-face.sh`, `deploy.sh` to a per-run directory on the bastion.
4. **Check configuration** — `toolforge envvars show` on both tools (face token; main URL + token) before either is touched.
5. **`become $FACE_TOOL`** — builds `face-service-<tag>`, recreates the `face-service` job (`--command web --continuous --port 8000 --publish --mount=none --mem 3Gi --cpu 2 --health-check-http /v1/models/wikivisage`), waits up to 10 min for public readiness.
6. **`become wikivisage`** — builds the release tag, deletes both workers, runs `migrate`, restarts the webservice, then the **preflight**: polls `/health` until `face_service: reachable` (URL + token checked). Only then starts `ml-worker` / `ml-worker-2` (`--mem 2Gi --cpu 2`, `healthcheck.sh`). `dologmsg` to SAL, best-effort: both deploy scripts call `log_to_sal()`, which warns and continues when `dologmsg` is missing, fails or hangs (20 s cap) — a SAL outage must not fail a deploy that already succeeded.
7. **Summary** (`if: always()`) — writes the job result (`✅ success` / `❌ failed` / `⚠️ cancelled` from `job.status`), the per-step outcomes of `deploy_face` and `deploy_web` (`steps.<id>.outcome`; a step is `skipped` when an earlier step failed, `not run` when the job never reached it), plus tag, DB-reset flag, face tool/ref and actor to `$GITHUB_STEP_SUMMARY`.

Concurrency: `deploy-production` with `cancel-in-progress: false`.

## Internationalization (i18n)

Uses Flask-Babel with gettext `.po`/`.mo` files. 4 supported locales.

### File Layout

```
WikiVisage/
├── babel.cfg                          # pybabel extraction config
├── messages.pot                       # Extracted message template (source of truth)
└── translations/
    ├── en/LC_MESSAGES/
    │   ├── messages.po                # English (identity: msgstr = msgid)
    │   └── messages.mo                # Compiled binary
    ├── nb/LC_MESSAGES/
    │   ├── messages.po                # Norwegian Bokmal
    │   └── messages.mo
    ├── es/LC_MESSAGES/
    │   ├── messages.po                # Spanish
    │   └── messages.mo
    └── fr/LC_MESSAGES/
        ├── messages.po                # French
        └── messages.mo
```

### How It Works

- **Locale selection**: Cookie (`locale`) -> `Accept-Language` header -> default (`en`).
- **Language picker**: Dropdown in nav bar. Sets the (`Secure` unless debug) cookie via `/set-language/<lang>` route; the cookie is session-scoped until the visitor accepts cookies, and `POST /cookie-consent` upgrades (Accept) or downgrades (Decline) an existing one.
- **RTL support**: `<html dir="{{ text_direction }}">` set by context processor. `RTL_LANGUAGES = {"ar", "he", "fa", "ur"}` in `app.py`.
- **Config**: `LANGUAGES = {"en": "English", "nb": "Norsk bokmal", "es": "Espanol", "fr": "Francais"}` in `app.py`. `BABEL_DEFAULT_LOCALE = "en"`.

### Translation Conventions

- **Python strings**: `_("text")` (imported as `from flask_babel import gettext as _`)
- **Jinja2 templates**: `{{ _('text') }}`
- **Plurals**: `ngettext('%(num)d item', '%(num)d items', count, num=count)` with named `%(var)s` placeholders and explicit `num=` kwarg
- **JS strings in templates**: Pass via `|tojson` filter into a JS object, never use `_()` in raw `<script>` blocks
- **`|tojson` inside an HTML attribute**: always single-quote the attribute (`onsubmit='return confirm({{ _('...')|tojson }})'`) and double-quote the JS placeholder (`replace("{username}", ...)`); `|tojson` emits raw `"` (it escapes `'` as `\u0027`), so a double-quoted attribute lets the value close the attribute and inject markup
- **JS parameterized strings**: Use `{placeholder}` style for `.replace()` substitution — NOT `%(name)d` (causes KeyError)
- **Python/Jinja2 parameterized strings**: Always use named placeholders `%(name)s`, never positional `%s`
- **Do NOT translate**: Worker log messages, health endpoint JSON values, technical terms (Wikidata, Q-ID, Commons, SDC, P180, OAuth, CSRF, WikiVisage, BETA)
- **Note**: French plural rule differs from English/Spanish: `nplurals=2; plural=(n > 1)` vs `nplurals=2; plural=(n != 1)`

### Adding a New Language

```bash
# 1. Add locale code + display name to LANGUAGES dict in app.py
# 2. Initialize .po file from template
source venv/bin/activate
pybabel init -i messages.pot -d translations -l de

# 3. Translate all msgstr entries in translations/de/LC_MESSAGES/messages.po
# 4. Compile
pybabel compile -d translations

# 5. If RTL language, add code to RTL_LANGUAGES set in app.py
```

### pybabel Workflow

```bash
source venv/bin/activate

# Extract new/changed strings from source files
pybabel extract -F babel.cfg -o messages.pot .

# Update existing .po files with new strings (preserves existing translations)
pybabel update -i messages.pot -d translations

# Compile .po -> .mo (required after any .po change)
pybabel compile -d translations
```

**Important**: `babel.cfg` lists source files explicitly (not `**.py`) to avoid scanning the `venv/` directory.

**Compiled catalogs are verified**: `tests/test_translations.py` fails whenever a committed `.mo` no longer matches its `.po` source (it runs in the plain `pytest` suite and in CI), so the compile step above is not optional.

### English .po File

The English `.po` file uses identity translations (`msgstr` = `msgid`). This ensures Flask-Babel always has a translation to serve and allows the English text to be edited in one place (the `.po` file) without changing source code.

## Commands

```bash
# Local development — the face service must be running first
cd model-server && docker compose up --build   # Face service on :8080
export WIKIVISAGE_FACE_SERVICE_URL=http://localhost:8080
python app.py                    # Web app on http://localhost:8000
python worker.py --worker-id local-1  # Background worker (separate terminal)
python migrate.py                # Run schema migrations (idempotent)
python migrate.py --reset        # Drop all tables and recreate from scratch

# Experimental LiftWing shadow verdicts (needs the worker to have run with a
# project's Metadata Pre-Filter set to 'shadow')
python scripts/eval_prefilter.py [--project-id N] [--reasons]

# Linting
ruff check .                     # Lint (errors, warnings, security)
ruff format --check .            # Format check (dry-run)
ruff format .                    # Auto-format

# Testing
pytest tests/ -v                                          # Unit tests only (CI mode)
WIKIVISAGE_TEST_DB=1 pytest tests/ -v                     # Full suite (unit + integration)
WIKIVISAGE_TEST_DB=1 pytest tests/ --cov=. --cov-report=term-missing  # With coverage

# i18n
source venv/bin/activate
pybabel extract -F babel.cfg -o messages.pot .    # Extract strings
pybabel update -i messages.pot -d translations    # Update .po files
pybabel compile -d translations                   # Compile .mo files

# Face service (see TESTING.md for the full guide)
cd model-server
docker build -t wikivisage-face-service:local .
curl -s http://localhost:8080/v1/models/wikivisage   # readiness

# Model weights (non-commercial; see model-server/weights/LICENSE-NOTICE)
scripts/fetch_weights.sh --check     # verify the committed files, no network
scripts/fetch_weights.sh             # re-create them from the official release (verifies zip + file hashes)
scripts/check_no_lfs.sh              # what the CI lint job runs

# Toolforge deployment (manual — normally done via .github/workflows/deploy.yml)
toolforge build start https://github.com/DiFronzo/WikiVisage.git
toolforge webservice buildservice start
```
