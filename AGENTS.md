# AGENTS.md — WikiVisage

## Overview

Active-learning Flask app for Wikimedia Commons. Users classify faces via yes/no UI, training a centroid-distance model that auto-classifies remaining faces. Approved matches are written as P180 (depicts) SDC claims to Commons via OAuth — triggered manually by the user from the project detail page. Hosted on Wikimedia Toolforge (Kubernetes, no GPU).

Face detection and 512D embedding (SCRFD-500M + ArcFace MobileFaceNet on ONNX Runtime) run in a **separate face service** (`model-server/`), deployed as its own Toolforge tool and reached over HTTPS. The web and worker processes carry no ML dependencies at all.


## Structure

```
WikiVisage/
├── app.py              # Flask web app: OAuth (PKCE), routes, classification API, CSP (~4500 lines)
├── worker.py           # Background pipeline: crawl, remote detect, infer (~3400 lines)
├── face_client.py      # HTTP client for the face service. ZERO ML deps — shared by app.py + worker.py
├── liftwing_client.py  # Experimental LiftWing Qwen metadata pre-filter client (fail-open, circuit breaker, completion probe, optional bearer token) — worker-only
├── config.py           # Shared config: WAKE_FILE_PATH, HEARTBEAT_FILE_DIR, COMMONS_THUMB_STEPS, DETECTION_WIDTH, LIFTWING_PROBE_INTERVAL, PROJECT_VERSION
├── token_crypto.py     # Fernet encrypt/decrypt helpers for OAuth tokens at rest (~125 lines)
├── redis_lock.py       # Best-effort Redis single-flight lock; used to serialize OAuth refresh (~100 lines)
├── database.py         # MariaDB connection pool with retry logic (~510 lines)
├── model-server/       # Face detection service — the ONLY place ML runs
│   ├── model.py        # KServe V1 predictor: WikiVisageModel(kserve.Model), async predict, box_space scaling
│   ├── backend.py      # FaceBackend protocol + SCRFD-500M/ArcFace-MBF on ONNX Runtime; weight SHA-256 pins. No KServe dependency
│   ├── guard.py        # Stdlib-only ASGI middleware: bearer token + 40 MiB body cap (public-facing)
│   ├── weights/        # det_500m.onnx, w600k_mbf.onnx (plain git blobs, NOT LFS), SHA256SUMS, LICENSE-NOTICE (non-commercial terms)
│   ├── requirements.in # Top-level pins (kserve, onnxruntime, opencv-headless, Pillow, numpy); the source of requirements.txt
│   ├── requirements.txt # Fully pinned transitive lock (uv pip compile, Python 3.11/x86_64-manylinux_2_28) — plain pip, no system packages
│   ├── Procfile        # Toolforge buildpack entrypoint: `web: python model.py ... --http_port 8000`
│   ├── .python-version # 3.11 (Heroku Python buildpack)
│   ├── Dockerfile      # Local dev + CI only. Two-stage, python:3.11-slim-bookworm, uid 10001
│   ├── blubber.yaml    # Forward-looking only — NOT deployable (see Known Issues)
│   ├── docker-compose.yml # Local harness with memory/CPU ceilings + read-only rootfs
│   ├── make_input.py   # Builds KServe request JSON from local image files
│   └── .dockerignore
├── scripts/
│   ├── fetch_weights.sh   # Re-create/verify model-server/weights/ from the official InsightFace release (zip SHA, then both file SHAs)
│   ├── check_no_lfs.sh    # CI guard: no .gitmodules, no LFS filter on the weights, no LFS pointer files
│   └── eval_prefilter.py  # Offline readout for the LiftWing shadow verdicts (keep recall, skip rate, detector signal, sample reasons)
├── healthcheck.sh      # Toolforge liveness health check script (per-worker heartbeat file age check)
├── schema.sql          # DDL for 10 tables: users, sessions, projects, images, faces, user_stats, sdc_claims, project_members, worker_heartbeat, service_status
├── migrate.py          # Idempotent schema migration with --reset flag; one-time embedding-model wipe (apply_embedding_model_upgrade)
├── pyproject.toml      # Project config: Ruff linter/formatter rules, pytest config, markers, version
├── requirements.txt    # Python 3.11+, NO ML dependencies (see model-server/ for those)
├── requirements-dev.txt # Dev/test deps: pytest, pytest-cov, ruff (includes requirements.txt)
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
│   ├── test_app.py     # Unit + integration tests
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
│   ├── test_worker.py  # Unit + integration tests
│   └── test_security_round2.py # Round-2 security regression tests (CSP, PKCE, single-flight, /health, M3/M4)
├── static/             # Static assets
│   ├── wikivisage-logo.svg        # Full logo with text
│   ├── wikivisage-logo-notext.svg # Logo icon only
│   ├── wikivisage-common.js       # Shared JS helpers: snapThumbWidth(), commonsThumbUrl()
│   └── view-it-tool.png           # View it! Tool icon (local copy)
├── .github/
│   └── workflows/
│       ├── ci.yml      # CI: Ruff lint + no-LFS check, pytest on Python 3.11/3.13 (integration tests skipped), face-service unit/contract jobs
│       └── deploy.yml  # CD: Release-triggered Toolforge deploy via SSH (2 workers + health checks)
├── Procfile            # web: gunicorn (4 workers, app factory), worker: python -u worker.py
├── jobs.yaml           # Toolforge jobs definition (2 ml-worker instances, health check scripts)
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
silently processing nothing.

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
  worker process, requests serialised (concurrency 1), no streaming.
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
  regardless of `status`, capped at `LIFTWING_MAX_IMAGES` (250) per cycle, in
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
  outer poll loop **and** the two 60 s heartbeat loops, so a long project cycle
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
  their own request. The User-Agent is `WikiVisage/<version>`
  (`config.PROJECT_VERSION`).
- **Web side:** context processor `inject_liftwing_status` reads the row (no
  network call; failures read as "no message") and `base.html` shows a
  dismissible `.system-banner` on the project detail/settings pages only, and
  only for projects using the filter. Settings shows the status and last check
  time next to the mode selector. A status older than 3 probe intervals reads
  as `unknown` (a dead worker hides the banner instead of showing stale data).
  `/health` adds `"liftwing": "reachable" | "unreachable" | "disabled"`
  (`disabled` when unused or unknown) and it **never** sets `degraded`.

## Toolforge Processes

### Web (app.py)

Flask app served by gunicorn via app factory (`create_app()`). Handles OAuth 2.0 login, project CRUD, face classification UI, and a Commons thumbnail proxy. SDC write requests are queued via a flag; the background worker performs the actual API writes.

**Security middleware:**
- Open redirect protection on login (`_is_safe_url()`)
- CSRF protection on all POST routes (per-request CSRF token stored in Flask's signed session cookie)
- PKCE (RFC 7636, S256) on the OAuth authorization code flow as defense-in-depth against intercepted authorization codes
- Rate limiting via Flask-Limiter (global 200/hour default, 10/min on bbox endpoints)
- Security headers: `Content-Security-Policy` (with `object-src 'none'`, `base-uri 'none'`, `frame-ancestors 'none'`), `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`
- OAuth refresh single-flight via Redis lock (`redis_lock.single_flight()`, 15s TTL, key `wikivisage:lock:oauth-refresh:<user_id>`) — prevents concurrent token-refresh requests from invalidating each other's freshly rotated refresh tokens. Degrades to no-op if Redis is unreachable; DB-level CAS on `token_expires_at` remains as a second line of defense.

**Routes (37 total):**
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Landing page |
| `/set-language/<lang>` | GET | Set locale cookie, redirect back |
| `/login` | GET | OAuth 2.0 redirect to Wikimedia |
| `/auth/callback` | GET | OAuth token exchange |
| `/logout` | POST | End session (CSRF protected) |
| `/dashboard` | GET | User's project list (paginated), includes collaborator count badges |
| `/api/category-info` | GET | Return total file count for a Commons category (BFS subcategory traversal) |
| `/project/new` | GET/POST | Create project (QID + Commons category, validates P31=Q5 and category existence). Detects cross-user duplicates and offers join option |
| `/project/<id>` | GET | Project stats, classification breakdown, model results gallery with approve/reject/edit-bbox, collaborator count badge |
| `/project/<id>/classify` | GET | Active learning face classification UI |
| `/project/<id>/classify/clear-skips` | POST | Reset skipped faces for this session |
| `/api/classify` | POST | Submit face classification (yes/no/none) |
| `/api/undo-classify` | POST | Undo last classification (reverts manual face draws too) |
| `/api/reclassify` | POST | Approve/reject model-classified face (removes SDC claim on reject if sdc_written) |
| `/api/manual-face` | POST | Manually draw face bounding box (rate limited: 10/min) |
| `/api/update-face-bbox` | POST | Redraw face bounding box from Model Results (rate limited: 10/min) |
| `/api/write-sdc/<id>` | POST | Queue P180 depicts claims for writing by background worker (sets flag, returns immediately) |
| `/api/sdc-status/<id>` | GET | Poll SDC write progress (written/pending counts, in_progress flag) |
| `/api/stop-sdc/<id>` | POST | Cancel an in-progress SDC write (clears sdc_write_requested flag) |
| `/api/project/<id>/gallery` | GET | Paginated JSON API for Classification Results gallery (filter by result/source/sdc) |
| `/api/progress/<id>` | GET | Poll image processing progress (processed/total/pending counts) |
| `/project/<id>/settings` | GET/POST | Edit project params (incl. `prefilter_mode`: `off`/`shadow`), view member list (owner-only). Shows the LiftWing status line next to the mode selector |
| `/project/<id>/settings/remove-member` | POST | Remove a member from the project (owner-only, CSRF protected) |
| `/project/<id>/settings/unban-member` | POST | Unban a member so they can rejoin the project (owner-only) |
| `/project/<id>/leave` | POST | Leave a project (non-owner members only, CSRF protected) |
| `/project/<id>/invite-code` | POST | Generate or revoke invite code for the project (owner-only) |
| `/join` | POST | Join a project via invite code |
| `/project/<id>/rerun-inference` | POST | Reset model-classified faces to re-run inference with current settings (owner-only) |
| `/project/<id>/delete` | POST | Delete project |
| `/account/settings` | GET/POST | User account settings (leaderboard opt-out) |
| `/leaderboard` | GET | Top classifiers |
| `/sw.js` | GET | Serve service worker from root scope |
| `/.well-known/appspecific/com.chrome.devtools.json` | GET | Silence Chrome DevTools auto-request |
| `/health` | GET | Health check (JSON: `status`, `database`, `limiter` (`redis`/`memory`), `face_service` (`reachable`/`unreachable`), `liftwing` (`reachable`/`unreachable`/`disabled`), `degraded` boolean). `liftwing` never affects `degraded` |
| `/robots.txt` | GET | Custom robots.txt (overrides Toolforge default Disallow: /) |
| `/sitemap.xml` | GET | XML sitemap for search engines |
| `/commons-thumb/<path>` | GET | Redirect to Commons thumbnail URL (standard step sizes enforced) |

**Error handlers:** 400, 403, 404, 429, 500 — all render `error.html` (429 returns "Rate limit exceeded").

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

**Liveness health check**: Each worker periodically touches `$HOME/.wikivisage-worker-alive-{worker_id}` (via `_touch_heartbeat_file()`). Toolforge runs `healthcheck.sh {worker_id}` every 10s; 3 consecutive failures (file missing or >5 min old) trigger automatic restart. The heartbeat directory is configured in `config.py` as `HEARTBEAT_FILE_DIR`.

Two query paths:
1. **Active projects** → full pipeline: `traverse_category` → `bootstrap_from_sparql` → `prefilter_images` (experimental, shadow) → `process_images` → `run_autonomous_inference`
2. **Active or completed projects with unclassified faces and enough human-confirmed targets** → inference-only: `run_autonomous_inference`

**Scheduling:** the inference-only queue is drained **before** the active batch each cycle (waves of `MAX_CONCURRENT_PROJECTS`), and the wake handler plus the 60 s heartbeat checks of the running loops call `_run_new_inference_projects()` so a project whose classifications just crossed `min_confirmed` is picked up mid-cycle (measured: 31 s from gate crossing to classification while the active batch was mid-run). Inference costs ~1 s per project while an active project can hold all slots for a long time looping its image backlog, so the original order (one inference batch *after* the active batch) left ready projects showing "Model inference in progress" at a standstill. `seen_ids` bounds the cycle-start drain: a project that stays eligible (e.g. faces with unusable encodings) is only attempted once per cycle there.

**Note:** SDC claim writing is user-triggered via the "Send Edits to Wikimedia Commons" button on the project detail page (calls `/api/write-sdc/<id>`). The worker no longer writes SDC claims automatically.

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
| `prefilter_images` | Experimental shadow-mode LiftWing stage, run after the image loop. Caps at `LIFTWING_MAX_IMAGES` (250) per cycle, batches of `LIFTWING_BATCH_SIZE` (20), skips bootstrapped images, selects any unchecked image regardless of `status`, stores `(keep, reason)` per image and records success/failure into `service_status`. Fail-open throughout |
| `_store_prefilter_verdicts` | Atomic batch UPDATE of `prefilter_keep` / `prefilter_reason` / `prefilter_checked_at` (`WHERE prefilter_checked_at IS NULL`) |
| `_liftwing_service_down` | True when `service_status` says LiftWing is down; the pre-filter skips the project entirely |
| `_record_liftwing_failure` | Increments `consecutive_failures`; flips to `down` at `LIFTWING_FAIL_THRESHOLD` (3). A real failed pre-filter batch lands here too |
| `_record_liftwing_success` | Resets `consecutive_failures`, sets `status='up'`, stamps `last_ok`; called after each stored pre-filter batch and by the probe |
| `_probe_liftwing_if_due` | Atomic once-per-`LIFTWING_PROBE_INTERVAL` claim + real probe; runs only while a project uses the filter. Called from the outer poll loop and both 60 s heartbeat loops |
| `_load_embedding` | Validates a `faces.encoding` BLOB: exactly `dim * 4` bytes, little-endian float32, finite, L2 norm in `[0.9, 1.1]`; returns a renormalised unit vector or `None` (rejects stale 1024-byte dlib encodings) |
| `_face_distance` | Cosine distance `1 - dot(centroid, embedding)`, clamped at 0; stored in `faces.confidence` |
| `_convert_box_to_stored_space` | Re-expresses a box from one pixel space into the stored `detection_width/height` space (for re-detection on images whose boxes live in another size) |
| `run_autonomous_inference` | Cosine-distance classification on unclassified faces against the renormalised mean centroid (needs >= `min_confirmed` target faces). Only reads faces whose `embedding_model` equals `face_client.EXPECTED_MODEL_ID` |
| `write_sdc_claims` | Writes P180 claims to Commons SDC via Wikibase API (idempotent). Triggered by `sdc_write_requested` flag set from web UI. Re-checks `sdc_write_requested` every 5 faces (`_PER_FACE_CANCEL_CHECK_EVERY`) so the user can stop a running batch quickly. Treats `no-such-entity` / `no-such-claim` / `no-such-statement` / `notfound` as already-gone successes (`_ALREADY_GONE_CODES`). |
| `_api_request` | Wrapper for Commons/Wikidata API calls with maxlag, retry, and User-Agent |
| `_get_csrf_token` | Fetches CSRF token for Wikibase API writes |

### Shared (database.py)

Both processes import from `database.py`. Thread-safe connection pool (`Queue`-based) with exponential backoff retry (3 attempts).

**Exports:**
- `init_db(pool_size)` — Initialize connection pool. Web default: 5 connections. Worker calls with explicit `pool_size=15`.
- `execute_query(sql, params, fetch)` — Universal read/write. Returns `List[Dict]` (fetch=True) or `int` rowcount (fetch=False).
- `execute_insert(sql, params)` — INSERT with `cursor.lastrowid` return (race-free).
- `execute_transaction(queries)` — Atomic multi-statement transaction. Takes list of `(sql, params)` tuples.
- `get_connection(timeout)` — Context manager for raw connection access.
- `close_pool()` — Drain and close all connections.
- `DatabaseError`, `PoolExhaustedError`, `ConfigurationError` — Custom exception hierarchy.

Pool size configurable via `WIKIVISAGE_DB_POOL_SIZE` env var. The worker uses that env var when set; otherwise it computes `min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 6)` and warns above 6, because Toolforge ToolsDB enforces `max_user_connections=20` across every process sharing the DB user (gunicorn + all workers).

## Data Model

```
users 1──N projects 1──N images 1──N faces
  |           |                        |
  |           +-- sdc_write_requested   +-- is_target: NULL=unclassified, 1=match, 0=non-match
  |           +-- sdc_write_error       +-- classified_by: 'human' | 'model' | 'bootstrap'
  |           +-- last_inference_threshold   +-- classified_by_user_id: FK to users (human classifications)
  |           +-- last_inference_min_confirmed +-- encoding: 512D float32 embedding (2048 bytes BLOB)
  +---- sessions (Flask-Session)       +-- embedding_model: e.g. 'scrfd500m+arcface_mbf' (NOT NULL)
                                       +-- det_score: SCRFD confidence (NULL for manual faces)
                                       +-- aligned: 1=landmark-aligned embedding, 0=unaligned (drawn box the detector missed)
                                       +-- confidence: cosine distance from target centroid
                                       +-- sdc_written: whether P180 claim was written
                                       +-- sdc_removal_pending: 1=P180 removal queued (rejected face)
                                       +-- superseded_by: FK to replacement face (after bbox edit)

                                images:
                                       +-- bootstrapped: 1=image found via P180 bootstrap

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

worker_heartbeat (single-row: id=1, last_seen DATETIME)

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

### worker.py
| Constant | Value | Purpose |
|----------|-------|---------|
| `POLL_INTERVAL` | 60s | DB polling frequency |
| `BATCH_SIZE` | 50 | Images per processing batch (`WIKIVISAGE_WORKER_BATCH_SIZE`) |
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
| `LIFTWING_MAX_IMAGES` | 250 | Pre-filter cap per worker cycle (`WIKIVISAGE_LIFTWING_MAX_IMAGES`) |
| `LIFTWING_FAIL_THRESHOLD` | 3 | Consecutive probe **or filter-batch** failures before `service_status.status` flips to `down` (`WIKIVISAGE_LIFTWING_FAIL_THRESHOLD`) |

### liftwing_client.py (worker-only, experimental)
| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_URL` | `https://api.wikimedia.org/service/lw/inference/v1/models/llm-qwen36-27b/openai/v1/chat/completions` | OpenAI-compatible endpoint (`WIKIVISAGE_LIFTWING_URL`) |
| model | `llm-qwen36-27b` | Sent in the request body (`WIKIVISAGE_LIFTWING_MODEL`) |
| `REQUEST_TIMEOUT` / `PROBE_TIMEOUT` | 90 s / 20 s | Whole-batch timeout / one tiny probe (`WIKIVISAGE_LIFTWING_TIMEOUT`, `WIKIVISAGE_LIFTWING_PROBE_TIMEOUT`) |
| `PROBE_MAX_TOKENS` | 8 | The probe is a minimal **real completion**, not `GET /models` (`WIKIVISAGE_LIFTWING_PROBE_MAX_TOKENS` is not read; constant only) |
| `MAX_TOKENS` | 2000 | Room for ~50 short verdicts plus a reasoning preamble (`WIKIVISAGE_LIFTWING_MAX_TOKENS`) |
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
- `PORT` — Default: `8000`
- `WIKIVISAGE_DB_POOL_SIZE` — Default: `5` (web process). The worker reads the same var; when unset it computes `min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 6)` and warns above 6 (ToolsDB shares a 20-connection cap across all processes).
- `WIKIVISAGE_WORKER_POLL_INTERVAL` — Default: `60` seconds
- `WIKIVISAGE_WORKER_BATCH_SIZE` — Default: `50` (images per processing batch)
- `WIKIVISAGE_WORKER_MAX_PROJECTS` — Default: `4` (concurrent projects processed by worker)
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
- `WIKIVISAGE_LIFTWING_CIRCUIT_FAILURES` / `WIKIVISAGE_LIFTWING_CIRCUIT_COOLDOWN` — Default: `3` / `600` (per-process circuit breaker; a 429 opens it immediately and `Retry-After` extends the cooldown)
- `WIKIVISAGE_LIFTWING_API_TOKEN` — Unset by default. A Wikimedia personal API token: sent as `Authorization: Bearer <token>` and moves calls from the anonymous to the authenticated quota.
- `WIKIVISAGE_LIFTWING_USER_AGENT` — Default: `WikiVisage/<version> (https://github.com/DiFronzo/WikiVisage)`
- `WIKIVISAGE_LIFTWING_BATCH_SIZE` — Default: `20` (files per completion)
- `WIKIVISAGE_LIFTWING_MAX_IMAGES` — Default: `250` (pre-filter cap per worker cycle)
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
- **Linter/formatter**: Ruff configured in `pyproject.toml` (line-length 120, target py311). Selects E/W/F/I/UP/B/SIM/S rule sets with project-specific ignores. Run: `ruff check .` and `ruff format --check .`.
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
- Shared JS helpers are in `static/wikivisage-common.js` (thumbnail snapping, Commons URL building). Templates that need them add `<script src="{{ url_for('static', filename='wikivisage-common.js') }}">` in `{% block extra_head %}`.
- No JavaScript build system. Inline `<script>` tags in templates for page-specific logic.
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
- CSRF: All POST routes protected via Flask-Session tokens.
- Bbox validation: All face bounding box inputs validated against `MAX_BBOX_PX` and `MIN_BBOX_AREA`.
- Security headers set on all responses: `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`.
- Token encryption at rest: OAuth access/refresh tokens can be Fernet-encrypted in the DB via `WIKIVISAGE_TOKEN_KEY` env var (opt-in). Handled by `token_crypto.py`. Decryption gracefully falls back to plaintext for legacy tokens.
- SDC writes include `maxlag=5` parameter for Wikimedia API compliance.
- Commons thumbnail proxy: URL validated via `urlparse` (scheme must be `https`, netloc must be `upload.wikimedia.org`).

### Error handling
- `app.py`: Custom error handlers for 400/403/404/500 render `error.html`. Route handlers use try/except returning flash + redirect.
- `database.py`: Custom exceptions (`DatabaseError`, `PoolExhaustedError`, `ConfigurationError`). Retry with exponential backoff on `OperationalError`/`InterfaceError`.
- `worker.py`: Each pipeline stage catches its own exceptions, logs, and returns 0 on failure (no crash propagation). Main loop catches `DatabaseError` and sleeps 10s.

### Thumbnail URLs
- Commons enforces standard thumbnail step sizes (`$wgThumbnailSteps`). Non-standard widths return 429.
- Python: `_snap_thumb_width(w)` rounds to nearest allowed step from `_THUMB_STEPS`.
- JavaScript: `snapThumbWidth(w)` mirrors the same logic in `static/wikivisage-common.js`.
- All thumbnail URL generation must go through these snapping functions.
- Standard steps: 20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px.

### Shared Helpers

Extracted common patterns to reduce duplication across routes, templates, and JS.

**Python (app.py):**

| Helper | Signature | Purpose | Call sites |
|--------|-----------|---------|------------|
| `_wikimedia_api_get` | `(url, params, timeout=10) → dict` | Standardizes User-Agent header, `raise_for_status()`, `.json()` parsing for Wikimedia API GET requests | 10: `_is_human_entity`, `_commons_category_exists`, `_commons_category_has_files`, `_check_p180_exists`, `_fetch_p18_thumb_url`, `_fetch_wikidata_label`, + 4 calls in `api_category_info` |
| `_validate_bbox` | `(top, right, bottom, left) → str \| None` | Validates bounding box coordinates against `MAX_BBOX_PX` and `MIN_BBOX_AREA`. Returns error message or `None` | 2: `api_manual_face`, `api_update_face_bbox` |
| `_get_face_stats` | `(project_id) → dict[str, Any]` | Single SQL query returning 12-column face/image stats with NULL→0 coercion | 2: `project_detail`, `api_progress` |

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
| `VIDEO_EXTENSIONS`, `TIF_EXTENSIONS` | File extension sets for media type detection |
| `postForm(url, csrfToken, fields)` | CSRF-enabled POST helper — builds FormData, returns parsed JSON Promise |
| `bboxToDisplayRect(bbox, scaleX, scaleY, offsetX, offsetY)` | Converts detection-space bbox `{top,right,bottom,left}` to display-space `{left,top,width,height}` |
| `displayRectToDetectionBbox(rect, scaleX, scaleY, offsetX, offsetY)` | Inverse of `bboxToDisplayRect` — display rect to rounded detection bbox |

## Known Issues & Gotchas

### faces_confirmed counter is unreliable
The `projects.faces_confirmed` column only increments on target MATCH clicks (Yes button in classify UI) and bootstrap. It does NOT reflect total classifications. The worker's inference gate now uses a direct `COUNT(*)` query instead of this counter. **Do not use `faces_confirmed` for logic — always count `is_target=1` from the faces table.**

### dlib and `face_recognition` are gone — SCRFD + ArcFace on ONNX Runtime
The face service used to run dlib HOG + `face_recognition` (128D float64). It now runs SCRFD-500M + ArcFace MobileFaceNet (512D float32) on ONNX Runtime (`model-server/backend.py`). **Do not add dlib or `face_recognition` back.** Nothing needs a system package, so there is no `project.toml`.

**The weights are third-party and non-commercial research only** (InsightFace `buffalo_sc`). They are committed as ordinary git blobs in `model-server/weights/` with `LICENSE-NOTICE` stating the terms; the project owner accepted this for the experiment. Wikimedia admins could object, so keep the `FaceBackend` protocol pluggable (YuNet + SFace is the fallback). **Never use Git LFS or submodules for them**: the Toolforge build service does a plain clone, and a pointer file would build "successfully" and then fail. Three guards enforce it: `scripts/check_no_lfs.sh` in the CI lint job, `sha256sum -c SHA256SUMS` in every face-service CI job, and `backend.verify_weights` at startup (fail-closed). `scripts/fetch_weights.sh` re-creates and verifies the files from the official release URL.

### The embedding-model wipe (one-time, in `migrate.py`)
Old 128D dlib encodings and new 512D embeddings are not comparable, so `migrate.apply_embedding_model_upgrade()` wipes derived data exactly once. It is gated on `information_schema.COLUMNS`: if `faces.embedding_model` is absent it runs `DELETE FROM images` (cascades to faces and sdc_claims; never `TRUNCATE`, InnoDB refuses it on FK-referenced tables) and `DELETE FROM sdc_claims`, resets every project (counters, inference settings, completion reason, SDC flags, worker claim, `distance_threshold` = the new default), reactivates `completed` projects, then applies the schema changes (`det_score`, comments, default) and adds `embedding_model` **last**, so an interrupted run repeats the wipe instead of skipping it. The ALTER list that runs *before* this step must never anchor a column on `det_score` or `embedding_model` — they do not exist yet on a legacy database (error 1054): `faces.aligned` is positioned after the old `confidence` column instead, and `test_run_migration_upgrades_a_legacy_faces_schema_end_to_end` exercises the full entry point against a simulated 0.10.x table. Later runs and fresh databases (whose `schema.sql` already has the column) skip it. It keeps users, sessions, projects, project_members, user_stats and worker_heartbeat. Wiping `sdc_claims` cannot create duplicate P180 claims because `write_sdc_claims` checks Commons first, and `bootstrap_from_sparql` re-seeds from existing claims. Take a DB dump before the first run on real data. `migrate.py --reset` is a different, harsher tool (it also drops users and tokens).

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

### model-server/blubber.yaml is not deployable
Wikimedia LiftWing builds production images exclusively with Blubber, so the file exists to keep the service in the right shape. It **cannot be deployed today**: LiftWing is not self-service (it requires membership in the internal `deploy-ml-service` LDAP group, and Toolforge tools are classified as external API consumers), it hosts no biometric models, and the committed weights are non-commercial. Production uses Toolforge buildpacks; local development uses the `Dockerfile`.

### OAuth scope & token handling
SDC writes require the `editpage` OAuth grant. The access token is stored per-user in the `users` table. Token refresh is handled in `app.py` `@before_request`. Access tokens from the DB may be `bytes` — normalized to `str` at read time in `before_request`.

### Worker must be restarted after code changes
The worker is a long-running `python worker.py` process. Code changes require manual restart. On Toolforge, redeploy the continuous job. The face service is a separate container with its own lifecycle — changing `model-server/model.py` requires rebuilding and restarting *it*, not the worker.

### Commons thumbnail proxy
`/commons-thumb/<path>` redirects to Commons thumbnail URLs server-side. This exists because Commons thumbnails can't be directly embedded due to referrer policies on Toolforge. The route enforces standard thumbnail step sizes to avoid 429 errors from Commons.

### Commons thumbnail step sizes
Commons enforces `$wgThumbnailSteps` — only standard sizes (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px) are allowed. Non-standard widths return HTTP 429. Both Python (`_snap_thumb_width`) and JavaScript (`snapThumbWidth`) snap requested widths to the nearest allowed step.

### Shutdown handling
Worker uses `signal.SIGTERM`/`SIGINT` handlers setting `shutdown_requested = True`. Each pipeline stage and the main loop check this flag for graceful shutdown. Uses `multiprocessing.set_start_method("spawn")` to avoid fork-safety issues.

### Worker heartbeat & downtime banner
The worker writes `REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())` at the start of each poll cycle. The web app checks `last_seen < NOW() - INTERVAL 5 MINUTE` via a context processor (`inject_worker_status`). If stale, `base.html` displays an amber banner: "The background worker appears to be offline." The banner is hidden on the landing page and leaderboard (not relevant there). Graceful on fresh installs — returns `worker_down=False` if no heartbeat row exists.

The LiftWing pre-filter has its own status path: the worker's probe writes `service_status`, and the web app's `inject_liftwing_status` context processor reads it (no network call). `base.html` shows a dismissible banner on the project detail/settings pages when the filter is down **and** the project uses it; the settings page also renders `reachable`/`unreachable`/`unknown` with the last check time. A row older than 3 probe intervals reads as `unknown`.

### Toolforge health check (file-based liveness)
Separate from the DB heartbeat above, each worker also writes a per-worker file at `$HOME/.wikivisage-worker-alive-{worker_id}` via `_touch_heartbeat_file()`. The `healthcheck.sh` script accepts a worker ID as `$1`, checks the file exists and was modified within 5 minutes (`find -mmin +5`). Toolforge's `--health-check-script` runs this every 10s; 3 consecutive failures trigger automatic pod restart. Per-worker files are critical because `$HOME` is shared NFS on Toolforge — a single file would be kept fresh by any surviving worker, hiding a dead one.

### Image download limits
Both `app.py` and `worker.py` enforce a 50MB download size cap (`MAX_IMAGE_DOWNLOAD_BYTES`) via streaming download with early abort. The worker additionally validates image pixel dimensions before shipping bytes to the face service (`MAX_IMAGE_PIXELS = 100M pixels`), so oversized images are rejected before they are base64-inflated and sent over the wire. The service enforces its own independent ceilings.

### Category traversal limits
`MAX_IMAGES_PER_PROJECT = 9000` caps how many images the worker will insert per project during category traversal. The worker checks existing image count before starting and calculates remaining capacity. Uses batch `INSERT IGNORE` instead of per-file SELECT+INSERT.

### SDC writes are user-triggered only
The "Send Edits to Wikimedia Commons" button on the project detail page is the ONLY way SDC claims are written. The button sets `sdc_write_requested=1` on the project; the background worker picks this up on the next poll cycle and writes claims in batches. The web UI polls `/api/sdc-status/<id>` for progress. SDC writes run in the worker process, not the web process. Writes include `maxlag=5` for Wikimedia API compliance.

### Model Results validation UI
The project detail page includes approve/reject/edit-bbox controls on each Model Results gallery card. Key behaviors:
- **Approve** (checkmark): Sets `is_target=1, classified_by='human'` via `/api/reclassify`. Updates card badge in-place.
- **Reject** (X): Sets `is_target=0, classified_by='human'`. If the face had `sdc_written=1`, also removes the P180 depicts claim from Commons via `_remove_sdc_claim()` (uses `wbgetclaims` -> `wbremoveclaims`).
- **Edit bbox** (pencil): Opens a modal with a 1280px image. User draws a new bounding box. Old face is kept; a new face row is inserted via `/api/update-face-bbox` (face encoding recomputed server-side; `faces.aligned` records whether the detector confirmed the box or the embedding fell back to an unaligned resize).
- **Unreadable image**: if the face service rejects the image itself (`FaceServiceRejected` with `error_code: invalid_image`, which `encode_known_face` re-raises), both `/api/manual-face` and `/api/update-face-bbox` return **422** with "This image could not be read by the face detection service." (retryable service failures stay 503; box-level rejections still answer "draw a slightly larger box").
- **Filter interaction**: After reclassification, `data-source` is NOT changed — the card remains visible under its original source filter (Model/Bootstrap). Only the visible method label updates to "human".

### Cookies
Only 2 cookies: `session` (strictly necessary, managed by Flask's built-in signed session cookie mechanism) and `locale` (functional, language preference). No tracking cookies. A non-blocking consent banner is shown. Without consent the `locale` cookie is session-scoped: switching language still works, but the choice is only persisted (1 year) once the visitor accepts the banner.

### Worker memory is bounded by DETECTION_WAVE_SIZE, not BATCH_SIZE
`process_images` works in waves. A wave's downloaded image bytes are all resident at once while its detection request is in flight, so `DETECTION_WAVE_SIZE` (8) is what caps peak memory — not `BATCH_SIZE` (10). Downloading a whole batch up front would be 50 x 50 MB worst case. Inference itself is cheap: each encoding is 1024 bytes, so even 10K faces is ~10 MB.

### Wikidata label fetching
`_fetch_wikidata_label(qid)` in `app.py` calls the Wikidata `wbgetentities` API with `languages=en&languagefallback=1` to retrieve human-readable labels. The `languagefallback=1` parameter makes the API automatically fall back through language variants (e.g., `mul` → `en`, `en-ca` → `en`) when no exact `en` label exists. Used during project creation to auto-fill empty project labels. Without this parameter, entities like Q153694 (Michael Bublé) return empty labels because they only have `mul`/`en-ca`/`en-gb` labels, not `en`.

## Testing

Hybrid test suite: unit tests (run in CI), integration tests (require local Docker MariaDB) and contract tests (require a running face service).

### Architecture

- **Unit tests**: Pure mocks, no DB, no network. Run everywhere (CI, local). Cover thumb snapping, URL safety, CSRF validation, error classes, migration parsing, route logic, token encryption, CSP hardening, PKCE, single-flight Redis lock, idempotent SDC removal, mid-batch cancel, the face service client (wire parsing, batching, error classification), and the LiftWing pre-filter client (verdict parsing, fail-open, circuit breaker, probe).
- **Integration tests**: Hit a real MariaDB via Docker. Marked with `@pytest.mark.integration`. Skipped in CI (GitHub Actions) — only run locally when `WIKIVISAGE_TEST_DB=1` is set.
- **Contract tests**: Run the real `face_client` against a real running face service container. Marked with `@pytest.mark.contract`, skipped unless `WIKIVISAGE_FACE_SERVICE_CONTRACT=1`. This is the only place both sides of the HTTP boundary are exercised together — everything else mocks it — so it is what catches a broken Dockerfile, an unresolvable dependency, or wire-format drift. Auth tests additionally need `WIKIVISAGE_FACE_SERVICE_TOKEN` set on both the container and the client. Tests that need a real face read `tests/data/portrait_sample.jpg` (override: `WIKIVISAGE_TEST_SAMPLE_IMAGE`); they skip when it is absent unless `WIKIVISAGE_REQUIRE_SAMPLE` is an explicit affirmative (`1`/`true`/`yes`/`on`; `"0"`/`"false"`/unset skip). CI sets it. The sample **is committed**: a close-cropped public-domain White House portrait whose ~514x756 px face at 960 px exercises the half-scale pass; `tests/data/README.md` records its provenance. CI runs the contract tests in both face-service jobs, with a token, and fails if the service log contains ORT `VerifyOutputSizes` warnings.
- **Face-service unit tests**: `test_face_service_backend.py` (fake ONNX sessions for decode/scaling/alignment/chunking/box recovery, plus the real weights) and `test_face_service_model.py` (the KServe envelope with a scripted backend) import `model-server/` directly and need `onnxruntime`, `opencv` and `kserve`; they `importorskip` in the plain environment and run in the `face-service-unit` CI job.
- **Guard tests**: `test_face_service_guard.py` loads `model-server/guard.py` by file path (it is stdlib-only), so the public-facing auth and body-cap rules are unit-tested without KServe or dlib.
- **Test DB**: `wikiface_test` — created fresh per pytest session, dropped on teardown. Never touches `wikiface_dev`.
- **Config**: `pyproject.toml` has `testpaths = ["tests"]`, `pythonpath = ["."]`, and integration marker.

### Test files

`test_app.py`, `test_database.py`, `test_face_client.py`, `test_face_service_backend.py`, `test_face_service_model.py`, `test_face_service_contract.py`, `test_face_service_guard.py`, `test_liftwing_client.py`, `test_migrate.py`, `test_security_round2.py`, `test_token_crypto.py`, `test_worker.py`. Counts are not tracked here: run `pytest tests/ -q` for the current numbers (unit tests that need the ML stack or a database show as skipped in the plain environment).

`test_security_round2.py` no longer contains the `TestSubprocessHardening` tests, which covered `_harden_face_detect_subprocess()`; those guarantees are now container-level (see Architecture) and are not unit-testable.

Phase 2 added: the LiftWing client suite (`test_liftwing_client.py`); `prefilter_images` mode/fail-open/cap tests, probe claim/hysteresis tests and `_fetch_metadata` tests in `test_worker.py`; status-snapshot, banner, `/health` and settings-POST tests in `test_app.py`; migration/schema assertions in `test_migrate.py`, including `test_schema_semicolons_are_only_statement_terminators` — `load_schema` splits `schema.sql` naively on `;`, so a semicolon inside a comment or a quoted `COMMENT` would silently chop a statement in half and only fail on a real database.

The follow-up round added: `test_eval_prefilter.py` (the readout math under mock rows), auth-header/429-`Retry-After`/minimal-completion-probe tests in `test_liftwing_client.py`, prefilter success/failure `service_status` wiring and the "prefilter runs after the image loop" ordering test in `test_worker.py`, 422 `FaceServiceRejected` tests for both bbox routes plus the `aligned` flag assertions in `test_app.py`, `faces.aligned` migration/schema assertions in `test_migrate.py`, and a `require_sample_env` truthiness test pinning that `WIKIVISAGE_REQUIRE_SAMPLE=0` means "not required".

The Copilot-review round added: `test_run_migration_upgrades_a_legacy_faces_schema_end_to_end` (full migration entry point against a simulated 0.10.x `faces` table), the `encode` rejection-classification tests (`invalid_image` re-raised vs an unusable region staying `None`) in `test_face_client.py`, both bbox routes (through the real client parsing path — only `_post` stubbed) and the live contract suite, `error_code` assertions on per-instance error entries in `test_face_service_model.py` (plus a parity check against `face_client`'s constant), and the LiftWing breaker-serialization (`test_queued_calls_see_the_breaker_the_running_call_opens`) and 429-deadline probe tests.

### Commands

```bash
# Full suite (unit + integration) — requires Docker MariaDB running
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# CI mode (unit only — integration tests auto-skipped)
pytest tests/ -v

# Single test file
pytest tests/test_app.py -v

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
- **Rate limiting is disabled in tests**: `tests/test_app.py` disables the limiter at import, and `tests/conftest.py` disables it in both the `flask_app` and `integration_app` fixtures (each reloads the module, so each needs its own disable). The limiter keys on the client address, which is always `127.0.0.1`, so a full run would exhaust the global 200/hour quota half-way through and turn unrelated assertions into 429s. Limiting itself is covered by `test_error_handler_429` (which fakes the abort, so it needs no limiter).
- **App auth simulation**: Set `session["user_id"]` in test client session transaction.

## CI/CD

### CI (`.github/workflows/ci.yml`)

Runs on push/PR to `main`. Five jobs:

1. **Lint**: Ruff check + format check (Python 3.11), then `scripts/check_no_lfs.sh` — fails if `.gitmodules` exists, a weight file is routed through the LFS filter, or a weight file is an LFS pointer.
2. **Test**: `pytest --tb=short -q` on Python 3.11 and 3.13 matrix. Integration tests auto-skipped (no `WIKIVISAGE_TEST_DB` env var in CI); the face-service unit and contract files skip too (no ML stack here). No system deps needed — the ML stack lives in the face service, and the suite mocks `face_client` rather than reaching the network. Caches pip dependencies.
3. **Face service (backend + model unit tests)** (`face-service-unit`): verifies `model-server/weights/SHA256SUMS`, installs `model-server/requirements.txt` + dev requirements on Python 3.11, and runs `test_face_service_backend.py` and `test_face_service_model.py` with `WIKIVISAGE_REQUIRE_SAMPLE=1`.
4. **Face service** (`face-service`): verifies `SHA256SUMS`, builds `model-server/` with buildx (gha layer cache), starts the container **with a random token**, waits for readiness, runs the contract tests through the guard, and fails if the service log contains ONNX Runtime `VerifyOutputSizes` warnings. It builds the Dockerfile.
5. **Face service (Toolforge buildpack)** (`face-service-buildpack`): verifies `SHA256SUMS`, builds `model-server/` with `pack` and Toolforge's own builder (`tools-harbor.wmcloud.org/toolforge/heroku-builder:24_0.21.8`, buildpacks `heroku/python` + `heroku/procfile` only — no deb packages are needed) — the production image — starts it via the `web` Procfile entry on port 8000 with `TOOL_TOOLFORGE_API_URL` set (so the fail-closed token check is live), and runs the same contract tests and log check. Catches `requirements.txt` / `Procfile` breakage that only affects the buildpack path.

The contract tests need `tests/data/portrait_sample.jpg` (CI sets `WIKIVISAGE_REQUIRE_SAMPLE=1`). The sample is committed, so the three jobs run the real-face tests rather than failing by design.

Concurrency: `ci-${{ github.ref }}` with cancel-in-progress.

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
6. **`become wikivisage`** — builds the release tag, deletes both workers, runs `migrate`, restarts the webservice, then the **preflight**: polls `/health` until `face_service: reachable` (URL + token checked). Only then starts `ml-worker` / `ml-worker-2` (`--mem 2Gi --cpu 2`, `healthcheck.sh`). `dologmsg` to SAL.

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
- **Language picker**: Dropdown in nav bar. Sets cookie via `/set-language/<lang>` route.
- **RTL support**: `<html dir="{{ text_direction }}">` set by context processor. `RTL_LANGUAGES = {"ar", "he", "fa", "ur"}` in `app.py`.
- **Config**: `LANGUAGES = {"en": "English", "nb": "Norsk bokmal", "es": "Espanol", "fr": "Francais"}` in `app.py`. `BABEL_DEFAULT_LOCALE = "en"`.

### Translation Conventions

- **Python strings**: `_("text")` (imported as `from flask_babel import gettext as _`)
- **Jinja2 templates**: `{{ _('text') }}`
- **Plurals**: `ngettext('%(num)d item', '%(num)d items', count, num=count)` with named `%(var)s` placeholders and explicit `num=` kwarg
- **JS strings in templates**: Pass via `|tojson` filter into a JS object, never use `_()` in raw `<script>` blocks
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
