<p align="center">
        <img src="static/wikivisage-logo.svg" alt="WikiVisage" width="420" />
</p>

<p align="center">
        <a href="https://www.python.org/">
                <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge" />
        </a>
        <a href="https://flask.palletsprojects.com/">
                <img alt="Framework" src="https://img.shields.io/badge/flask-web-black?style=for-the-badge" />
        </a>
        <a href="https://github.com/DiFronzo/WikiVisage/actions/workflows/ci.yml">
                <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/DiFronzo/WikiVisage/ci.yml?branch=main&label=CI&style=for-the-badge" />
        </a>
        <a href="https://github.com/DiFronzo/WikiVisage/releases">
                <img alt="Release" src="https://img.shields.io/github/v/release/DiFronzo/WikiVisage?label=release&style=for-the-badge" />
        </a>
        <a href="https://toolsadmin.wikimedia.org/tools/id/wikivisage">
                <img alt="Build" src="https://img.shields.io/badge/build-Toolforge-success?style=for-the-badge" />
        </a>
        <a href="https://toolhub.wikimedia.org/tools/toolforge-wikivisage">
                <img alt="Hosting" src="https://img.shields.io/badge/hosted%20on-Toolforge-green?style=for-the-badge" />
        </a>
        <a href="https://wikivisage.toolforge.org/">
                <img alt="Live site" src="https://img.shields.io/badge/live-wikivisage.toolforge.org-blue?style=for-the-badge" />
        </a>
        <a href="LICENSE">
                <img alt="License" src="https://img.shields.io/github/license/DiFronzo/WikiVisage?label=license&style=for-the-badge" />
        </a>
</p>

<p align="center">
Active learning facial recognition for Wikimedia Commons. Train an ML model to recognize specific people and automatically add <a href="https://www.wikidata.org/wiki/Property:P180">P180 (depicts)</a> Structured Data to matching images.
</p>

<p align="center" width="100%">
<video src="https://github.com/user-attachments/assets/46c05ee5-4393-406f-a41d-0b8f331e66fc" width="80%" controls></video>
</p>

## 🔗 Quick links

- 📖 Local dev guide: [test-local.md](test-local.md)
- 🧪 Face service guide: [TESTING.md](TESTING.md)
- 🚀 Toolforge deploy guide: [how-to-run-it.md](how-to-run-it.md)
- 🤝 Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- 🔒 Security policy: [SECURITY.md](SECURITY.md)
- 📜 Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## ✨ Highlights

- 🧠 **Active learning UI**: fast yes/no classification with keyboard shortcuts, undo, skip, and manual face drawing
- 🧵 **Background worker**: crawls Commons categories, downloads images, and stores the 128D encodings the face service returns
- 🔀 **Multi-instance workers**: distributed locking lets multiple workers process projects concurrently without conflicts
- 🧩 **Separate face service**: dlib lives in its own KServe service, deployed as a second Toolforge tool — the web and worker processes ship with zero ML dependencies
- 🛡️ **Secret isolation**: the face service parses untrusted image bytes in a tool that holds no OAuth, database, or token-encryption secret, behind a bearer token and a request-size cap
- 📦 **Batched detection**: images are sent in size-aware batches, and a service outage leaves work `pending` for the next cycle instead of failing it permanently
- 🧷 **Bootstrap from existing tags**: seeds the model via SPARQL when P180 depicts claims already exist on Commons
- 🤖 **Autonomous inference**: centroid-distance classification once you have enough confirmed examples
- ✍️ **User-triggered Commons edits**: click "Send Edits to Wikimedia Commons" to write depicts claims via the Wikibase API (interruptible mid-batch; idempotent on already-removed claims)
- 🔐 **Hardened OAuth flow**: PKCE (S256) plus Redis-backed single-flight token refresh to survive concurrent gunicorn workers
- 🌍 **i18n-ready**: translations included (en, nb, es, fr)

## 🧭 How it works

1. 🆕 **Create a project** — pick a Wikidata entity (e.g., `Q42`) and a Commons category
2. 🔎 **Discover images** — the worker traverses the category and detects faces
3. 🧷 **Bootstrap (optional)** — if Commons already has depicts claims, seed the model from them
4. ✅❌ **Classify** — review faces one-by-one with Yes/No (keyboard shortcuts: `Y` / `N`)
5. 🤖 **Infer** — after enough confirmed faces (default `5`), classify remaining faces automatically
6. ✍️ **Write to Commons** — send approved matches as depicts claims (OAuth)

## 🏗️ Architecture

```
+---------------------------+      +---------------------------+
|       Flask Web App       |      |  Background Worker(s)     |
|          (app.py)         |      |       (worker.py)         |
|      no ML deps           |      |      no ML deps           |
|---------------------------|      |---------------------------|
| OAuth 2.0 login           |      | Category traversal        |
| Project CRUD              |      | Image download            |
| Active learning UI        |      | SPARQL bootstrapping      |
| Classification UI         |      | Autonomous inference      |
| Queue SDC writes          |      | Write SDC claims          |
| Approve/reject/edit bbox  |      | Distributed claim locking |
+------------+--------------+      +------------+--------------+
             |                                  |
             |   face_client.py (HTTP, base64 image bytes)
             +----------------+-----------------+
                              v
                 +----------------------------+
                 |       Face Service         |
                 |      (model-server/)       |
                 |----------------------------|
                 | KServe V1 predictor        |
                 | dlib HOG detection         |
                 | 128D face encoding         |
                 +----------------------------+
             |                                  |
             +----------------+-----------------+
                              v
                        +------------+
                        |   MariaDB  |
                        |  (ToolsDB) |
                        +------------+
```

- 🧰 **Stack**: Python 3.11+, Flask, gunicorn, PyMySQL, requests-oauthlib — plus KServe + dlib, isolated in the face service
- ☁️ **Hosted on**: [Wikimedia Toolforge](https://wikitech.wikimedia.org/wiki/Help:Toolforge) — `wikivisage` (web + workers) and `wikivisage-face` (face service)
- 🔀 **Workers**: Multiple instances run concurrently — each claims projects via `SELECT … FOR UPDATE` with automatic stale-claim expiry (15 min)

## 🗂️ Project layout

```
WikiVisage/
├── app.py               # Flask app: OAuth, routes, classification API
├── worker.py            # Background pipeline: crawl, remote detect, infer, write (multi-instance)
├── face_client.py       # HTTP client for the face service (zero ML deps)
├── model-server/        # Face detection service — the only place dlib lives
│   ├── model.py         # KServe V1 predictor
│   ├── guard.py         # Bearer-token + body-size middleware
│   ├── Procfile         # Toolforge buildpack entrypoint (production)
│   ├── Dockerfile       # Local development and CI
│   └── requirements.txt # kserve, dlib-bin, face_recognition_models
├── token_crypto.py      # Fernet encrypt/decrypt helpers for OAuth tokens at rest
├── database.py          # MariaDB connection pool with retry logic
├── schema.sql           # Database schema (9 tables + indices)
├── migrate.py           # Idempotent migration script with --reset flag
├── jobs.yaml            # Toolforge jobs definition (2 worker instances)
├── templates/           # Jinja2 templates (10 files, all extend base.html)
├── static/              # Logos + screenshots
├── translations/        # i18n: en, nb, es, fr
├── requirements.txt     # Runtime dependencies (no ML libraries)
├── requirements-dev.txt # Dev/test deps (pytest, ruff)
└── tests/               # 823 tests (775 unit + 34 integration + 14 contract)
```

## 🧑‍💻 Setup

### ✅ Prerequisites

- A [Toolforge](https://toolsadmin.wikimedia.org/) tool account
- An [OAuth 2.0 consumer](https://meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose) registered on Meta with grants:
        - `Basic rights`
        - `Edit existing pages`
        - Callback URL: `https://<toolname>.toolforge.org/auth/callback`

### 1) 🔐 Environment variables (Toolforge)

```bash
# Database credentials (find yours in ~/replica.my.cnf on Toolforge)
toolforge envvars create TOOL_TOOLSDB_USER      "s<NNNNN>"
toolforge envvars create TOOL_TOOLSDB_PASSWORD  "<password>"
toolforge envvars create WIKIVISAGE_DB_NAME     "s<NNNNN>__wikiface"

# OAuth 2.0
toolforge envvars create OAUTH_CLIENT_ID        "<client-id>"
toolforge envvars create OAUTH_CLIENT_SECRET    "<client-secret>"
toolforge envvars create OAUTH_REDIRECT_URI     "https://<toolname>.toolforge.org/auth/callback"

# Flask
toolforge envvars create FLASK_SECRET_KEY "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

# Token encryption (optional — tokens stored as plaintext if unset)
toolforge envvars create WIKIVISAGE_TOKEN_KEY "$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

### 2) 🗄️ Create database

```bash
mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud
```

```sql
CREATE DATABASE s<NNNNN>__wikiface;
```

### 3) 🧱 Run migration

```bash
python3 migrate.py
```

### 4) 🚀 Build & deploy

```bash
# Build container image
toolforge build start https://github.com/DiFronzo/WikiVisage.git

# Start web service
toolforge webservice buildservice start

# Start background workers (uses jobs.yaml — 2 worker instances)
toolforge jobs load jobs.yaml
```

The app will be live at `https://<toolname>.toolforge.org`.

## 🧪 Local development

Start the face service first — the app and worker call it over HTTP:

```bash
cd model-server && docker compose up --build   # http://localhost:8080
```

Then, in another terminal:

```bash
pip install -r requirements-dev.txt

export WIKIVISAGE_FACE_SERVICE_URL=http://localhost:8080
export TOOL_TOOLSDB_USER=root
export TOOL_TOOLSDB_PASSWORD=yourpassword
export TOOL_TOOLSDB_HOST=127.0.0.1
export WIKIVISAGE_DB_NAME=wikiface_dev
export OAUTH_CLIENT_ID=<client-id>
export OAUTH_CLIENT_SECRET=<client-secret>
export OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
export FLASK_SECRET_KEY=dev-secret-key
export OAUTHLIB_INSECURE_TRANSPORT=1

mysql -u root -p -e "CREATE DATABASE wikiface_dev"
python migrate.py
python app.py                                   # Web app on http://localhost:8000
python worker.py --worker-id local-1            # Background worker (separate terminal)
```

Confirm everything is wired up:

```bash
curl -s http://localhost:8000/health   # expects "face_service": "reachable"
```

For local OAuth you'll need a separate consumer with `http://localhost:8000/auth/callback` as the callback URL. Set `OAUTHLIB_INSECURE_TRANSPORT=1` to allow OAuth over HTTP.

## ⚙️ Configuration

Each project has a couple of tunables:

| Parameter | Default | Description |
|---|---:|---|
| `distance_threshold` | `0.6` | Face-distance cutoff for autonomous classification (lower = stricter). |
| `min_confirmed` | `5` | Minimum confirmed matches before autonomous inference starts. |

### 🔧 Worker environment variables (optional)

| Variable | Default | Description |
|---|---:|---|
| `WIKIVISAGE_WORKER_POLL_INTERVAL` | `60` | Seconds between poll cycles |
| `WIKIVISAGE_WORKER_MAX_PROJECTS` | `3` | Max projects processed concurrently per worker |
| `WIKIVISAGE_WORKER_IMAGE_THREADS` | `4` | Parallel image download/detection threads per project |
| `WIKIVISAGE_WORKER_BATCH_SIZE` | `50` | Images per processing batch |
| `WIKIVISAGE_DB_POOL_SIZE` | auto | DB connection pool size (auto = `max_projects × image_threads + 3`) |
| `COMMONS_DOWNLOAD_THROTTLE_SECONDS` | `0` | Delay between image downloads (seconds) |

### 🧩 Face service variables

Read by both the web app and the worker (see [TESTING.md](TESTING.md) for the full list):

| Variable | Default | Description |
|---|---:|---|
| `WIKIVISAGE_FACE_SERVICE_URL` | `http://localhost:8080` | Base URL of the face service |
| `WIKIVISAGE_FACE_SERVICE_TOKEN` | *(unset)* | Bearer token; must match the face tool's (required in production) |
| `WIKIVISAGE_FACE_SERVICE_BATCH_SIZE` | `8` | Max images per detection request |

## ✅ Testing

```bash
# Unit tests (CI mode — integration tests auto-skipped)
pytest tests/ -v

# Unit + integration tests (requires local MariaDB)
WIKIVISAGE_TEST_DB=1 pytest tests/ -v

# With coverage
WIKIVISAGE_TEST_DB=1 pytest tests/ --cov=. --cov-report=term-missing
```

## 📝 License

[MIT](LICENSE)
