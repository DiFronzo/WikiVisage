# AGENTS.md — WikiVisage

## Overview

Active-learning Flask app for Wikimedia Commons. Users classify faces via yes/no UI, training a centroid-distance model that auto-classifies remaining faces. Approved matches are written as P180 (depicts) SDC claims to Commons via OAuth — triggered manually by the user from the project detail page. Hosted on Wikimedia Toolforge (Kubernetes, no GPU).

## Structure

```
WikiVisage/
├── app.py              # Flask web app: OAuth, routes, classification API (~3870 lines)
├── worker.py           # Background ML pipeline: crawl, detect, infer (~2930 lines)
├── token_crypto.py     # Fernet encrypt/decrypt helpers for OAuth tokens at rest (~110 lines)
├── database.py         # MariaDB connection pool with retry logic (~510 lines)
├── schema.sql          # DDL for 9 tables: users, sessions, projects, images, faces, user_stats, sdc_claims, project_members, worker_heartbeat
├── migrate.py          # Idempotent schema migration with --reset flag (~430 lines)
├── pyproject.toml      # Project config: Ruff linter/formatter rules, pytest config, markers
├── requirements.txt    # Python 3.11+, dlib-bin fork (no source compilation)
├── requirements-dev.txt # Dev/test deps: pytest, pytest-cov, ruff (includes requirements.txt)
├── babel.cfg           # pybabel extraction config (explicit file list, excludes venv)
├── messages.pot        # Extracted translatable strings template
├── CONTRIBUTING.md     # Contributor guide: setup, conventions, i18n, schema changes
├── translations/       # i18n translation files (Flask-Babel / gettext)
│   ├── en/LC_MESSAGES/ # English (identity: msgstr = msgid)
│   ├── nb/LC_MESSAGES/ # Norwegian Bokmål
│   ├── es/LC_MESSAGES/ # Spanish
│   └── fr/LC_MESSAGES/ # French
├── tests/              # Hybrid test suite: 544 unit + 34 integration tests
│   ├── __init__.py
│   ├── conftest.py     # Integration fixture infrastructure (~450 lines)
│   ├── test_app.py     # 453 unit + 11 integration tests (~9640 lines)
│   ├── test_database.py # 14 unit + 9 integration tests (~360 lines)
│   ├── test_migrate.py # 15 unit + 8 integration tests (~471 lines)
│   ├── test_token_crypto.py # 22 unit tests (~175 lines)
│   └── test_worker.py  # 40 unit + 6 integration tests (~1240 lines)
├── templates/          # Jinja2 templates (10 files, all extend base.html)
│   ├── base.html       # Layout: nav, flash messages, CSS variables. Blocks: title, extra_head, content
│   ├── classify.html   # Active learning UI: face image, yes/no/skip/none buttons, keyboard shortcuts, undo
│   ├── project_detail.html  # Stats, classification breakdown, model results gallery, validation UI, SDC write button
│   ├── account_settings.html  # User account settings (leaderboard opt-out)
│   └── ...             # dashboard, index, leaderboard, project_new, project_settings, error
├── static/             # Static assets
│   ├── wikivisage-logo.svg        # Full logo with text
│   ├── wikivisage-logo-notext.svg # Logo icon only
│   ├── wikivisage-common.js       # Shared JS helpers: snapThumbWidth(), commonsThumbUrl()
│   └── view-it-tool.png           # View it! Tool icon (local copy)
├── .github/
│   └── workflows/
│       ├── ci.yml      # CI: Ruff lint + pytest on Python 3.11/3.13 (integration tests skipped)
│       └── deploy.yml  # CD: Release-triggered Toolforge deploy via SSH
├── Procfile            # web: gunicorn (4 workers, app factory), worker: python -u worker.py
├── Aptfile             # System deps: libopenblas0, liblapack3 (dlib runtime)
├── jobs.yaml           # Toolforge jobs definition (ml-worker continuous job)
├── how-to-run-it.md    # Toolforge deployment guide
├── test-local.md       # Local development setup guide
├── LICENSE             # MIT license
└── .env                # Local dev env vars (gitignored)
```

## Architecture — Two Processes

### Web (app.py)

Flask app served by gunicorn via app factory (`create_app()`). Handles OAuth 2.0 login, project CRUD, face classification UI, and a Commons thumbnail proxy. SDC write requests are queued via a flag; the background worker performs the actual API writes.

**Security middleware:**
- Open redirect protection on login (`_is_safe_url()`)
- CSRF protection on all POST routes (per-request CSRF token stored in Flask's signed session cookie)
- Rate limiting via Flask-Limiter (global 200/hour default, 10/min on bbox endpoints)
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`

**Routes (36 total):**
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
| `/project/<id>/settings` | GET/POST | Edit project params, view member list (owner-only) |
| `/project/<id>/settings/remove-member` | POST | Remove a member from the project (owner-only, CSRF protected) |
| `/project/<id>/settings/unban-member` | POST | Unban a member so they can rejoin the project (owner-only) |
| `/project/<id>/invite-code` | POST | Generate or revoke invite code for the project (owner-only) |
| `/join` | POST | Join a project via invite code |
| `/project/<id>/rerun-inference` | POST | Reset model-classified faces to re-run inference with current settings (owner-only) |
| `/project/<id>/delete` | POST | Delete project |
| `/account/settings` | GET/POST | User account settings (leaderboard opt-out) |
| `/leaderboard` | GET | Top classifiers |
| `/sw.js` | GET | Serve service worker from root scope |
| `/.well-known/appspecific/com.chrome.devtools.json` | GET | Silence Chrome DevTools auto-request |
| `/health` | GET | Health check (JSON) |
| `/robots.txt` | GET | Custom robots.txt (overrides Toolforge default Disallow: /) |
| `/sitemap.xml` | GET | XML sitemap for search engines |
| `/commons-thumb/<path>` | GET | Redirect to Commons thumbnail URL (standard step sizes enforced) |

**Error handlers:** 400, 403, 404, 500 — all render `error.html`.

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
- **Project-level**: Up to `MAX_CONCURRENT_PROJECTS` (default 3) projects processed simultaneously
- **Image-level**: Within each project, up to `IMAGE_THREADS` (default 4) images downloaded and face-detected in parallel

Two query paths:
1. **Active projects** → full pipeline: `traverse_category` → `process_images` → `bootstrap_from_sparql` → `run_autonomous_inference`
2. **Completed projects with unclassified faces** → inference-only: `run_autonomous_inference`

**Note:** SDC claim writing is user-triggered via the "Send Edits to Wikimedia Commons" button on the project detail page (calls `/api/write-sdc/<id>`). The worker no longer writes SDC claims automatically.

**Pipeline stages:**
| Function | What it does |
|----------|-------------|
| `traverse_category` | Crawls Commons category API, inserts image rows (batch INSERT IGNORE). Caps at `MAX_IMAGES_PER_PROJECT` (9000). Filters out video/audio (keeps images only). |
| `_download_image` | Downloads image with streaming 50MB size cap (`MAX_IMAGE_DOWNLOAD_BYTES`) |
| `_validate_image_dimensions` | Checks image pixel area before face detection (rejects >100 megapixels) |
| `_detect_faces_in_subprocess` | Runs dlib face detection in isolated subprocess (survives segfaults) |
| `_run_face_detection` | Spawns subprocess, handles timeout/crash, returns locations + encodings |
| `_process_single_image` | Downloads one image, validates dimensions, runs HOG face detection in subprocess, stores encoding (thread-safe) |
| `process_images` | Spawns `IMAGE_THREADS` parallel threads to process pending images in a batch |
| `bootstrap_from_sparql` | Seeds model from existing P180 depicts claims via SPARQL |
| `run_autonomous_inference` | Centroid-distance classification on unclassified faces (needs >= `min_confirmed` target faces) |
| `write_sdc_claims` | Writes P180 claims to Commons SDC via Wikibase API (idempotent). Triggered by `sdc_write_requested` flag set from web UI. |
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

Pool size configurable via `WIKIVISAGE_DB_POOL_SIZE` env var. Should be >= `MAX_CONCURRENT_PROJECTS` x `IMAGE_THREADS` for the worker (3 x 4 = 12 concurrent DB users + main thread).

## Data Model

```
users 1──N projects 1──N images 1──N faces
  |           |                        |
  |           +-- sdc_write_requested   +-- is_target: NULL=unclassified, 1=match, 0=non-match
  |           +-- sdc_write_error       +-- classified_by: 'human' | 'model' | 'bootstrap'
  |           +-- last_inference_threshold   +-- classified_by_user_id: FK to users (human classifications)
  |           +-- last_inference_min_confirmed +-- encoding: 128D float64 numpy array (1024 bytes BLOB)
  +---- sessions (Flask-Session)       +-- confidence: face distance from target centroid
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
| `_THUMB_STEPS` | `(20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)` | Commons-enforced thumbnail widths |
| `MAX_BBOX_PX` | 10000 | Maximum bounding box coordinate value |
| `MIN_BBOX_AREA` | 100 | Minimum bounding box area in pixels |

### worker.py
| Constant | Value | Purpose |
|----------|-------|---------|
| `POLL_INTERVAL` | 60s | DB polling frequency |
| `BATCH_SIZE` | 10 | Images per processing batch |
| `MAX_CONCURRENT_PROJECTS` | 3 | Parallel project processing |
| `IMAGE_THREADS` | 4 | Parallel image download/detection per project |
| `MAX_IMAGE_DOWNLOAD_BYTES` | 50 MB | Image download size cap |
| `MAX_IMAGE_PIXELS` | 100M | Pixel area limit before face detection |
| `MAX_IMAGES_PER_PROJECT` | 9000 | Category traversal cap |

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
- `WIKIVISAGE_DB_POOL_SIZE` — Default: `5` (web process). Worker explicitly calls `init_db(pool_size=15)`.
- `WIKIVISAGE_WORKER_POLL_INTERVAL` — Default: `60` seconds
- `WIKIVISAGE_WORKER_MAX_PROJECTS` — Default: `3` (concurrent projects processed by worker)
- `WIKIVISAGE_WORKER_IMAGE_THREADS` — Default: `4` (parallel image download/detection threads per project)
- `WIKIVISAGE_TOKEN_KEY` — Fernet key for encrypting OAuth tokens at rest. If unset, tokens are stored as plaintext (backward compatible). Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `WIKIVISAGE_REDIS_URL` — Redis URL for shared rate limiter storage. Default: `redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379`. Falls back to `memory://` if Redis is unreachable. Key prefix: `wikivisage:`.
- `OAUTHLIB_INSECURE_TRANSPORT=1` — Required for local dev (OAuth over HTTP)

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
- Open redirect protection: `_is_safe_url()` validates all redirect targets.
- Rate limiting: Global 200/hour default. `10/min` on `api_manual_face` and `api_update_face_bbox`. Uses Redis for shared storage across gunicorn workers (`WIKIVISAGE_REDIS_URL`). Falls back to `memory://` if Redis is unreachable.
- CSRF: All POST routes protected via Flask-Session tokens.
- Bbox validation: All face bounding box inputs validated against `MAX_BBOX_PX` and `MIN_BBOX_AREA`.
- Security headers set on all responses: `Content-Security-Policy`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`.
- Token encryption at rest: OAuth access/refresh tokens can be Fernet-encrypted in the DB via `WIKIVISAGE_TOKEN_KEY` env var (opt-in). Handled by `token_crypto.py`. Decryption gracefully falls back to plaintext for legacy tokens.
- SDC writes include `maxlag=5` parameter for Wikimedia API compliance.

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

### face_recognition fork
Uses a custom fork of `face-recognition` that depends on `dlib-bin` (pre-compiled wheels) instead of `dlib` (source-only). This avoids OOM during compilation on Toolforge. The fork URL is pinned to a specific commit in `requirements.txt`. **Do not replace with `pip install face-recognition`.**

### OAuth scope & token handling
SDC writes require the `editpage` OAuth grant. The access token is stored per-user in the `users` table. Token refresh is handled in `app.py` `@before_request`. Access tokens from the DB may be `bytes` — normalized to `str` at read time in `before_request`.

### Worker must be restarted after code changes
The worker is a long-running `python worker.py` process. Code changes require manual restart. On Toolforge, redeploy the continuous job.

### Commons thumbnail proxy
`/commons-thumb/<path>` redirects to Commons thumbnail URLs server-side. This exists because Commons thumbnails can't be directly embedded due to referrer policies on Toolforge. The route enforces standard thumbnail step sizes to avoid 429 errors from Commons.

### Commons thumbnail step sizes
Commons enforces `$wgThumbnailSteps` — only standard sizes (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px) are allowed. Non-standard widths return HTTP 429. Both Python (`_snap_thumb_width`) and JavaScript (`snapThumbWidth`) snap requested widths to the nearest allowed step.

### Shutdown handling
Worker uses `signal.SIGTERM`/`SIGINT` handlers setting `shutdown_requested = True`. Each pipeline stage and the main loop check this flag for graceful shutdown. Uses `multiprocessing.set_start_method("spawn")` to avoid fork-safety issues.

### Worker heartbeat & downtime banner
The worker writes `REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())` at the start of each poll cycle. The web app checks `last_seen < NOW() - INTERVAL 5 MINUTE` via a context processor (`inject_worker_status`). If stale, `base.html` displays an amber banner: "The background worker appears to be offline." The banner is hidden on the landing page and leaderboard (not relevant there). Graceful on fresh installs — returns `worker_down=False` if no heartbeat row exists.

### Image download limits
Both `app.py` and `worker.py` enforce a 50MB download size cap (`MAX_IMAGE_DOWNLOAD_BYTES`) via streaming download with early abort. The worker additionally validates image pixel dimensions before face detection (`MAX_IMAGE_PIXELS = 100M pixels`).

### Category traversal limits
`MAX_IMAGES_PER_PROJECT = 9000` caps how many images the worker will insert per project during category traversal. The worker checks existing image count before starting and calculates remaining capacity. Uses batch `INSERT IGNORE` instead of per-file SELECT+INSERT.

### SDC writes are user-triggered only
The "Send Edits to Wikimedia Commons" button on the project detail page is the ONLY way SDC claims are written. The button sets `sdc_write_requested=1` on the project; the background worker picks this up on the next poll cycle and writes claims in batches. The web UI polls `/api/sdc-status/<id>` for progress. SDC writes run in the worker process, not the web process. Writes include `maxlag=5` for Wikimedia API compliance.

### Model Results validation UI
The project detail page includes approve/reject/edit-bbox controls on each Model Results gallery card. Key behaviors:
- **Approve** (checkmark): Sets `is_target=1, classified_by='human'` via `/api/reclassify`. Updates card badge in-place.
- **Reject** (X): Sets `is_target=0, classified_by='human'`. If the face had `sdc_written=1`, also removes the P180 depicts claim from Commons via `_remove_sdc_claim()` (uses `wbgetclaims` -> `wbremoveclaims`).
- **Edit bbox** (pencil): Opens a modal with a 1280px image. User draws a new bounding box. Old face is kept; a new face row is inserted via `/api/update-face-bbox` (face encoding recomputed server-side).
- **Filter interaction**: After reclassification, `data-source` is NOT changed — the card remains visible under its original source filter (Model/Bootstrap). Only the visible method label updates to "human".

### Cookies
Only 2 cookies: `session` (strictly necessary, managed by Flask's built-in signed session cookie mechanism) and `locale` (functional, language preference). No tracking cookies. A non-blocking consent banner is shown.

### Inference RAM
Each face encoding is 1024 bytes (128 float64). Even 10K faces ~ 10MB. No RAM concern for inference.

### Wikidata label fetching
`_fetch_wikidata_label(qid)` in `app.py` calls the Wikidata `wbgetentities` API with `languages=en&languagefallback=1` to retrieve human-readable labels. The `languagefallback=1` parameter makes the API automatically fall back through language variants (e.g., `mul` → `en`, `en-ca` → `en`) when no exact `en` label exists. Used during project creation to auto-fill empty project labels. Without this parameter, entities like Q153694 (Michael Bublé) return empty labels because they only have `mul`/`en-ca`/`en-gb` labels, not `en`.

## Testing

Hybrid test suite: **544 unit tests** (run in CI) + **34 integration tests** (require local Docker MariaDB).

### Architecture

- **Unit tests**: Pure mocks, no DB. Run everywhere (CI, local). Cover thumb snapping, URL safety, CSRF validation, error classes, migration parsing, route logic, token encryption.
- **Integration tests**: Hit a real MariaDB via Docker. Marked with `@pytest.mark.integration`. Skipped in CI (GitHub Actions) — only run locally when `WIKIVISAGE_TEST_DB=1` is set.
- **Test DB**: `wikiface_test` — created fresh per pytest session, dropped on teardown. Never touches `wikiface_dev`.
- **Config**: `pyproject.toml` has `testpaths = ["tests"]`, `pythonpath = ["."]`, and integration marker.

### Test Counts

| File | Unit | Integration | Total |
|------|------|-------------|-------|
| `test_app.py` | 453 | 11 | 464 |
| `test_database.py` | 14 | 9 | 23 |
| `test_migrate.py` | 15 | 8 | 23 |
| `test_token_crypto.py` | 22 | 0 | 22 |
| `test_worker.py` | 40 | 6 | 46 |
| **Total** | **544** | **34** | **578** |

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

- **Encoding helper**: `_make_encoding(seed)` generates deterministic 128D float64 numpy arrays for face encodings.
- **Import pattern**: `conftest.py` helpers use try/except: `try: from conftest import X` / `except: from tests.conftest import X` for compatibility.
- **DB connection**: `host=127.0.0.1, user=root, password=devpass, port=3306`.
- **Assertion gotcha**: `execute_query()` returns empty **tuple** `()` not `[]` — use `len(result) == 0` not `result == []`.
- **Integration test isolation**: Each `db_conn` fixture truncates all tables after the test via `SET FOREIGN_KEY_CHECKS=0`.
- **App auth simulation**: Set `session["user_id"]` in test client session transaction.

## CI/CD

### CI (`.github/workflows/ci.yml`)

Runs on push/PR to `main`. Two jobs:

1. **Lint**: Ruff check + format check (Python 3.11).
2. **Test**: `pytest --tb=short -q` on Python 3.11 and 3.13 matrix. Integration tests auto-skipped (no `WIKIVISAGE_TEST_DB` env var in CI). Installs system deps (`libopenblas0`, `liblapack3`) for dlib. Caches pip dependencies.

Concurrency: `ci-${{ github.ref }}` with cancel-in-progress.

### CD (`.github/workflows/deploy.yml`)

Triggered on GitHub release publish or manual `workflow_dispatch` (with a `tag` input). Steps:
1. Checkout repo + configure SSH to Toolforge bastion.
2. Generate a deploy script locally, `scp` it to the bastion.
3. Execute via `become wikivisage bash /tmp/deploy.sh '<tag>'`.
4. Deploy script stages:
   - `toolforge build start --ref "$TAG"` — rebuild container image from the given git ref.
   - Poll `toolforge build show --json | jq -r '.build.status // empty'` every 15s (up to 600s timeout). Expects `ok`; fails on `error`/`timeout`/`cancelled`.
   - `toolforge jobs run migrate` — run schema migration (`python migrate.py`).
   - `toolforge jobs delete ml-worker` + `toolforge jobs run ml-worker` — restart background worker (delete+run because `jobs load` does NOT restart if the definition is unchanged).
   - `toolforge webservice buildservice restart` — restart web.

Concurrency: `deploy-production` with cancel-in-progress.

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
# Local development
python app.py                    # Web app on http://localhost:8000
python worker.py                 # Background worker (separate terminal)
python migrate.py                # Run schema migrations (idempotent)
python migrate.py --reset        # Drop all tables and recreate from scratch

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

# Toolforge deployment
toolforge build start https://github.com/DiFronzo/WikiVisage.git
toolforge webservice buildservice start
toolforge jobs load jobs.yaml                                 # Start/update background worker from jobs.yaml
```
