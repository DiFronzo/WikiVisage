# WikiVisage - Project Instructions

## Purpose

Active-learning Flask app for Wikimedia Commons. Users classify faces via yes/no UI, training a centroid-distance model that auto-classifies remaining faces. Approved matches are written as P180 (depicts) SDC claims to Commons via OAuth. Hosted on Wikimedia Toolforge (Kubernetes, no GPU).

### Non-negotiable principles:

- **Toolforge-compatible.** No GPU, no heavy deps. `dlib-bin` fork (pre-compiled wheels), not `dlib` (source-only). Never replace with `pip install face-recognition`.
- **User-triggered SDC writes only.** The "Send Edits to Wikimedia Commons" button is the only way claims are written. Never auto-write.
- **Parameterized SQL.** All DB queries use `%s` placeholders (PyMySQL). Never interpolate values into SQL. F-strings acceptable for structural SQL only (e.g., building `IN (%s, %s)` placeholder lists).
- **No type suppression.** No `as any`, `@ts-ignore`, empty `catch` blocks.

## Architecture

Two long-running processes sharing a MariaDB connection pool (`database.py`):

```
Flask Web App (app.py)          Background Worker (worker.py)
├─ OAuth 2.0 login              ├─ Category traversal (Commons API)
├─ Project CRUD                 ├─ Image download (50MB cap, 100MP limit)
├─ Active learning UI           ├─ HOG face detection (subprocess pool)
├─ Classification API           ├─ SPARQL bootstrapping
├─ Model Results validation     ├─ Centroid-distance inference
├─ SDC write queueing           ├─ SDC claim writing (user-triggered)
└─ Commons thumbnail proxy      └─ Distributed locking (SELECT FOR UPDATE)
```

Worker uses `ThreadPoolExecutor` at two levels: up to 3 projects concurrently, up to 4 image threads per project. Polls DB every 60s.

## Current Status

- **Version:** `0.9.0` (source of truth: `pyproject.toml`)
- **Tests:** 741 total (707 unit + 34 integration)
- **Python:** 3.11+
- **Tables:** 9 (`users`, `sessions`, `projects`, `images`, `faces`, `user_stats`, `sdc_claims`, `project_members`, `worker_heartbeat`)
- **Templates:** 10 files (all extend `base.html`)
- **Locales:** en, nb, es, fr

### Test counts by file

| File | Unit | Integration | Total |
|------|------|-------------|-------|
| `test_app.py` | 498 | 11 | 509 |
| `test_worker.py` | 110 | 6 | 116 |
| `test_database.py` | 27 | 9 | 36 |
| `test_migrate.py` | 15 | 8 | 23 |
| `test_security_round2.py` | 31 | 0 | 31 |
| `test_token_crypto.py` | 26 | 0 | 26 |

## Build & Test

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Lint (must pass before submitting)
ruff check .
ruff format --check .

# Unit tests only (CI mode - integration tests auto-skipped)
pytest tests/ -v

# Full suite (requires local Docker MariaDB)
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# With coverage
WIKIVISAGE_TEST_DB=1 pytest tests/ --cov=. --cov-report=term-missing

# Run the app locally
python app.py                              # Web on http://localhost:8000
python -u worker.py --worker-id local-1    # Worker (separate terminal)

# i18n (after changing user-facing strings)
pybabel extract -F babel.cfg -o messages.pot .
pybabel update -i messages.pot -d translations
pybabel compile -d translations
```

Integration tests require Docker MariaDB (`docker run -d --name wikivisage-db -e MARIADB_ROOT_PASSWORD=devpass -e MARIADB_DATABASE=wikiface_dev -p 3306:3306 mariadb:10.11`). They use a separate `wikiface_test` database, created and dropped automatically.

## Key Files

| Area | Files |
|------|-------|
| Web app | `app.py` (~4279 lines) - Flask routes, OAuth (PKCE), classification API, CSP, /health |
| Worker | `worker.py` (~3329 lines) - ML pipeline, RLIMIT-hardened face detection subprocess, inference, mid-batch SDC cancel |
| Database | `database.py` (~510 lines) - Connection pool, `execute_query`/`execute_insert`/`execute_transaction` |
| Token encryption | `token_crypto.py` (~125 lines) - Fernet encrypt/decrypt for OAuth tokens at rest |
| OAuth refresh lock | `redis_lock.py` (~100 lines) - Best-effort Redis single-flight (`single_flight()` ctx manager) |
| Schema | `schema.sql` - DDL for 9 tables |
| Migrations | `migrate.py` (~490 lines) - Idempotent schema migration with `--reset` flag |
| Config | `pyproject.toml` - Ruff + pytest config, version |
| Templates | `templates/base.html` (layout + CSS variables), `templates/classify.html` (active learning UI), `templates/project_detail.html` (stats + model results gallery) |
| Shared JS | `static/wikivisage-common.js` - `snapThumbWidth()`, `commonsThumbUrl()`, `postForm()`, `bboxToDisplayRect()` |
| Tests | `tests/conftest.py` (~450 lines) - Integration fixture hierarchy; `tests/test_security_round2.py` (~594 lines) - CSP, PKCE, single-flight, /health, idempotent SDC removal regression tests |
| CI | `.github/workflows/ci.yml` - Ruff lint + pytest on Python 3.11/3.13 |
| CD | `.github/workflows/deploy.yml` - Release-triggered Toolforge deploy via SSH |
| Jobs | `jobs.yaml` - 2 worker instances (`ml-worker`, `ml-worker-2`) |

## Conventions

### Python

- Ruff configured in `pyproject.toml` (line-length 120, target py311). Rule sets: E/W/F/I/UP/B/SIM/S.
- Type hints and f-strings everywhere. `DictCursor` for all queries.
- `execute_query()` returns empty **tuple** `()` not `[]` when no rows found.
- All mutation endpoints use `execute_transaction()` for atomicity.
- `faces_confirmed` column is **unreliable** - always count `is_target=1` from the faces table directly.

### Templates & CSS

- All templates extend `base.html`. Three blocks: `title`, `extra_head`, `content`.
- Dark brutalist theme with CSS variables: `--bg: #090e17`, `--surface: #141f33`, `--primary: #14b8a6`.
- CSS embedded in `<style>` tags (base + per-page). No external CSS files.
- JS inline in `<script>` tags. No build system.
- Title format: `Page Name - WikiVisage BETA`

### Thumbnail URLs

Commons enforces standard step sizes. Non-standard widths return 429.
- Python: `_snap_thumb_width(w)` in `app.py`
- JavaScript: `snapThumbWidth(w)` in `static/wikivisage-common.js`
- Steps: 20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840px

### Internationalization

- Python: `_("text")` / Jinja2: `{{ _('text') }}`
- JS strings: pass via `|tojson` filter into JS objects; use `{placeholder}` for `.replace()` substitution
- Python/Jinja2 strings: use `%(name)s` named placeholders
- Do NOT translate: worker log messages, health endpoint JSON, technical terms (Wikidata, Q-ID, Commons, SDC, P180, OAuth, CSRF, WikiVisage, BETA)

### Security

- CSRF tokens on all POST routes (Flask-Session)
- Rate limiting: global 200/hour, 10/min on bbox endpoints. Redis shared storage, falls back to in-memory.
- Security headers: `Content-Security-Policy` (with `object-src 'none'`, `base-uri 'none'`, `frame-ancestors 'none'`), `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`
- Open redirect protection on login (`_is_safe_url()`)
- Bbox validation: `MAX_BBOX_PX` (10000) and `MIN_BBOX_AREA` (100)
- OAuth token encryption at rest via `WIKIVISAGE_TOKEN_KEY` env var (opt-in Fernet)
- PKCE (RFC 7636, S256) on the OAuth authorization code flow as defense-in-depth
- OAuth refresh single-flight via Redis lock (`redis_lock.single_flight()`, 15s TTL) — prevents concurrent token refreshes from invalidating each other's rotated refresh tokens
- Worker face-detection subprocess hardening: env scrubbed via preexec hook; `RLIMIT_AS=2 GiB`, `RLIMIT_CPU=180s`, `RLIMIT_FSIZE=1 MiB`
- SDC writes treat `no-such-entity` / `no-such-claim` / `no-such-statement` / `notfound` as already-gone successes (idempotent removals)
- SDC write batches re-check `sdc_write_requested` every 5 faces so the user can stop a running batch quickly
- `/health` reports `degraded: true` (200 OK) when the rate limiter fell back to in-memory storage (Redis unreachable) — so the OAuth refresh single-flight is also a no-op
- `maxlag=5` on all Wikimedia API writes

## Schema Changes

1. Edit `schema.sql` with new DDL.
2. Add an idempotent migration in `migrate.py` (check-then-alter pattern).
3. Add any new tables to `_ALL_TABLES` and `expected_tables` in `migrate.py`.
4. Run `python migrate.py` to verify. Migrations must be safe to re-run.

## Environment Variables

**Required (no defaults):**
- `TOOL_TOOLSDB_USER`, `TOOL_TOOLSDB_PASSWORD`, `WIKIVISAGE_DB_NAME`

**Required for OAuth:**
- `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_REDIRECT_URI`

**Optional:**
- `FLASK_SECRET_KEY` - default: random hex (sessions lost on restart)
- `PORT` - default: `8000`
- `WIKIVISAGE_DB_POOL_SIZE` - default: `5` (web), worker uses `15`
- `WIKIVISAGE_TOKEN_KEY` - Fernet key for token encryption at rest
- `WIKIVISAGE_REDIS_URL` - Redis for shared rate limiting
- `OAUTHLIB_INSECURE_TRANSPORT=1` - required for local dev (OAuth over HTTP)

Worker-specific: `WIKIVISAGE_WORKER_POLL_INTERVAL` (60s), `WIKIVISAGE_WORKER_MAX_PROJECTS` (3), `WIKIVISAGE_WORKER_IMAGE_THREADS` (4)

## CI/CD

**CI** (`.github/workflows/ci.yml`): Runs on push/PR to `main`. Ruff lint + pytest on Python 3.11 and 3.13 matrix. Integration tests auto-skipped in CI.

**CD** (`.github/workflows/deploy.yml`): Triggered on GitHub release publish or manual workflow_dispatch. Builds container image from tag, runs migration, restarts both workers, restarts web service. Supports optional database wipe (type `WIPE` to confirm).

## Gotchas

- **`execute_query()` returns tuples, not lists.** Use `len(result) == 0` not `result == []`.
- **`faces_confirmed` is unreliable.** Always `COUNT(*)` from faces table where `is_target=1`.
- **Worker must be restarted after code changes.** It's a long-running process.
- **Commons thumbnail proxy** (`/commons-thumb/<path>`) exists because Commons thumbnails can't be embedded directly due to referrer policies. Always snap widths to standard steps.
- **OAuth tokens from DB may be `bytes`** - normalized to `str` in `before_request`.
- **face_recognition fork** uses `dlib-bin` (pre-compiled). Do not replace with `pip install face-recognition`.
- **Cookies:** Only `session` (Flask signed) and `locale` (language pref). No tracking cookies.
