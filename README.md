# WikiVisage

Active-learning web app for Wikimedia Commons. Users train an ML model via simple yes/no prompts to recognize specific people, automatically adding [P180 (depicts)](https://www.wikidata.org/wiki/Property:P180) Structured Data claims to matching images via OAuth.

## How It Works

1. **Create a project** — specify a Wikidata entity (e.g., Q42 for Douglas Adams) and a Commons category to search
2. **Discover images** — the background worker crawls the category and detects faces using HOG-based face recognition
3. **Bootstrap** — if the entity already has depicts claims on Commons, the worker uses SPARQL to find them and seed the model automatically
4. **Classify** — you review detected faces one by one with simple Yes/No buttons (keyboard shortcuts: Y/N)
5. **Autonomous inference** — after enough confirmed faces (default 5), the model classifies remaining faces automatically using centroid distance
6. **Write to Commons** — click "Send Edits to Wikimedia Commons" to write P180 depicts claims for all approved matches via the Wikibase API

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐
│   Flask Web App     │     │   Background Worker      │
│   (app.py)          │     │   (worker.py)            │
│                     │     │                          │
│  - OAuth 2.0 login  │     │  - Category traversal    │
│  - Project CRUD     │     │  - Image download        │
│  - Active learning  │     │  - HOG face detection    │
│    classification   │     │  - SPARQL bootstrapping  │
│    UI               │     │  - Autonomous inference  │
│                     │     │  - SDC P180 writes       │
└────────┬────────────┘     └────────┬─────────────────┘
         │                           │
         └─────────┬─────────────────┘
                   │
           ┌───────▼───────┐
           │   MariaDB     │
           │   (ToolsDB)   │
           └───────────────┘
```

**Stack:** Python 3.11+, Flask, gunicorn, face_recognition (dlib HOG), PyMySQL, requests-oauthlib

**Hosted on:** [Wikimedia Toolforge](https://wikitech.wikimedia.org/wiki/Help:Toolforge) (Kubernetes Build Service)

## Project Structure

```
WikiVisage/
├── Aptfile              # Runtime system dependencies (libopenblas, liblapack)
├── Procfile             # Process types: web (gunicorn) + worker
├── requirements.txt     # Python dependencies
├── database.py          # MariaDB connection pool with retry logic
├── schema.sql           # DDL for 6 tables: users, sessions, projects, images, faces, worker_heartbeat
├── migrate.py           # Schema migration script
├── app.py               # Flask app: OAuth 2.0, routes, classification API
├── worker.py            # Background ML pipeline: crawl, detect, infer
├── whitelist.txt        # Allowed usernames (one per line)
├── templates/           # Jinja2 templates (9 files)
│   ├── base.html            # Base layout with embedded CSS
│   ├── index.html           # Landing page
│   ├── dashboard.html       # Project list
│   ├── project_new.html     # Create project form
│   ├── project_detail.html  # Project stats, model results gallery, SDC write button
│   ├── project_settings.html# Edit project settings
│   ├── classify.html        # Active learning face classification UI
│   ├── leaderboard.html     # Top classifiers
│   └── error.html           # Error page
├── static/              # Logo SVGs
└── translations/        # i18n: en, nb, es, fr
```

## Setup

### Prerequisites

- A [Toolforge](https://toolsadmin.wikimedia.org/) tool account
- An [OAuth 2.0 consumer](https://meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose) registered on Meta with grants: `Basic rights`, `Edit existing pages` and callback URL `https://<toolname>.toolforge.org/auth/callback`

### 1. Environment Variables

```bash
# Database credentials (find yours in ~/replica.my.cnf on Toolforge)
toolforge envvars create TOOL_TOOLSDB_USER     "s<NNNNN>"
toolforge envvars create TOOL_TOOLSDB_PASSWORD  "<password>"
toolforge envvars create WIKIVISAGE_DB_NAME     "s<NNNNN>__wikiface"

# OAuth 2.0
toolforge envvars create OAUTH_CLIENT_ID        "<client-id>"
toolforge envvars create OAUTH_CLIENT_SECRET    "<client-secret>"
toolforge envvars create OAUTH_REDIRECT_URI     "https://<toolname>.toolforge.org/auth/callback"

# Flask
toolforge envvars create FLASK_SECRET_KEY       "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

### 2. Create Database

```bash
mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud
```

```sql
CREATE DATABASE s<NNNNN>__wikiface;
```

### 3. Run Migration

```bash
python3 migrate.py
```

### 4. Build and Deploy

```bash
# Build container image
toolforge build start https://github.com/DiFronzo/WikiVisage.git

# Start web service
toolforge webservice buildservice start

# Start background worker (uses jobs.yaml)
toolforge jobs load jobs.yaml
```

The app will be live at `https://<toolname>.toolforge.org`.

### Local Development

```bash
pip install -r requirements.txt

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
python app.py        # Web app on http://localhost:8000
python worker.py     # Background worker (separate terminal)
```

For local OAuth you need a separate consumer with `http://localhost:8000/auth/callback` as the callback URL. Set `OAUTHLIB_INSECURE_TRANSPORT=1` to allow OAuth over HTTP.

## Configuration

Each project has tunable parameters:

| Parameter | Default | Description |
|---|---|---|
| `distance_threshold` | 0.6 | Face distance threshold for autonomous classification. Lower = stricter matching. |
| `min_confirmed` | 5 | Minimum human-confirmed faces before the model classifies autonomously. |

## Key Design Decisions

- **HOG over CNN** — CPU-only face detection for Toolforge's constrained environment (~0.5-2s per image, no GPU needed)
- **BLOB encoding storage** — 128D float64 numpy arrays stored as raw bytes (1024 bytes per face, 3x smaller than JSON)
- **dlib-bin** — Pre-compiled wheels to avoid compiling dlib from source (would OOM on Toolforge)
- **Configurable threshold** — Default 0.6 per community convention; 0.38 from the original spec caused excessive false negatives
- **Idempotent SDC writes** — Checks existing P180 claims before writing to avoid duplicates
- **SDC bootstrapping** — Seeds the model from files already tagged with the target entity via `haswbstatement:P180` search, reducing cold-start human labeling

## License

[MIT](LICENSE)
