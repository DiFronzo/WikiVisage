# Local Testing

Step-by-step guide to running WikiVisage locally for development and testing.

## 1. Install Dependencies

```bash
# macOS — install dlib build deps first
brew install cmake

# Create a virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # Includes runtime deps + pytest, ruff
```

> **macOS note:** `dlib-bin` doesn't have ARM wheels — it falls back to compiling dlib from source. This needs cmake and takes a few minutes. If it fails, install dlib separately: `pip install dlib` then `pip install face-recognition`.

## 2. Local MariaDB

Easiest via Docker:

```bash
docker run -d \
  --name wikivisage-db \
  -e MARIADB_ROOT_PASSWORD=devpass \
  -e MARIADB_DATABASE=wikiface_dev \
  -p 3306:3306 \
  mariadb:10.11
```

Or if you have Homebrew: `brew install mariadb && brew services start mariadb`

## 3. Set Environment Variables

```bash
export TOOL_TOOLSDB_USER=root
export TOOL_TOOLSDB_PASSWORD=devpass
export TOOL_TOOLSDB_HOST=127.0.0.1
export WIKIVISAGE_DB_NAME=wikiface_dev
export FLASK_SECRET_KEY=dev-secret-key-not-for-production
# OAuth — leave empty for now, we'll handle this below
export OAUTH_CLIENT_ID=""
export OAUTH_CLIENT_SECRET=""
export OAUTH_REDIRECT_URI="http://localhost:8000/auth/callback"
export OAUTHLIB_INSECURE_TRANSPORT=1
```

## 4. Run Migration

```bash
python migrate.py
```

This creates all 7 tables (`users`, `sessions`, `projects`, `images`, `faces`, `user_stats`, `worker_heartbeat`) and their indexes. Safe to re-run.

## 5. Run the Web App

```bash
python app.py
# → http://localhost:8000
```

The landing page and health check (`/health`) will work immediately. OAuth login won't work without real credentials.

## 6. Run the Worker (separate terminal)

```bash
source venv/bin/activate
# Same exports as above
python -u worker.py --worker-id local-1
```

The worker will start polling but do nothing until there are active projects in the database.

> **Tip:** You can run a second worker with `--worker-id local-2` in another terminal. Workers use distributed locking to avoid conflicts.

---

## Testing Without OAuth

OAuth is the biggest friction point for local dev. To bypass it for testing, you can insert a fake user and session directly:

```bash
mysql -u root -pdevpass wikiface_dev
```

```sql
-- Create a test user
INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at)
VALUES (1, 'TestUser', 'fake-token', 'fake-refresh', '2099-01-01 00:00:00');

-- Create a test project (e.g., Q42 = Douglas Adams, in a real Commons category)
INSERT INTO projects (user_id, wikidata_qid, commons_category, label, distance_threshold, min_confirmed)
VALUES (1, 'Q42', 'Douglas Adams', 'Douglas Adams', 0.6, 5);
```

Then restart the worker — it'll start crawling the "Douglas Adams" category on Commons, downloading images, and detecting faces.

You can watch it work:

```bash
# In another terminal
mysql -u root -pdevpass wikiface_dev -e "SELECT id, file_title, status, face_count FROM images LIMIT 20;"
mysql -u root -pdevpass wikiface_dev -e "SELECT id, image_id, is_target, classified_by, confidence FROM faces LIMIT 20;"
```

> **Note:** SDC writes will fail with the fake token — that's expected. The worker logs the error and moves on. Everything else (category traversal, face detection, encoding, SPARQL bootstrapping, autonomous inference) works without a real token.

---

## Testing OAuth Locally (Optional)

If you want the full flow:

1. Register a new OAuth 2.0 consumer at [meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose](https://meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose)
   - Set callback URL to `http://localhost:8000/auth/callback`
   - It needs manual approval (can take hours/days)
2. Once approved, set `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` to the real values
3. Set `OAUTHLIB_INSECURE_TRANSPORT=1` to allow OAuth over HTTP

---

## Running Tests

Install dev dependencies if you haven't already:

```bash
pip install -r requirements-dev.txt
```

### Unit tests (no DB required)

```bash
pytest tests/ -v
```

### Full suite (unit + integration)

Requires the local MariaDB to be running (see step 2 above):

```bash
WIKIVISAGE_TEST_DB=1 pytest tests/ -v
```

Integration tests use a separate `wikiface_test` database that is created and dropped automatically. They never touch `wikiface_dev`.

### With coverage

```bash
WIKIVISAGE_TEST_DB=1 pytest tests/ --cov=. --cov-report=term-missing
```

### Linting

```bash
ruff check .              # Lint
ruff format --check .     # Format check (dry-run)
ruff format .             # Auto-format
```

---

## Quick Smoke Test Checklist

| What | How | Works without OAuth? |
|---|---|---|
| Health check | `curl http://localhost:8000/health` | Yes |
| Landing page | Browser → `http://localhost:8000` | Yes |
| Migration | `python migrate.py` (check exit code) | Yes |
| Category traversal | Insert test project, run worker, check `images` table | Yes |
| Face detection | Same — check `faces` table after worker processes images | Yes |
| SPARQL bootstrap | Worker logs will show bootstrap attempts | Yes |
| Autonomous inference | Manually insert 5+ confirmed faces, worker auto-classifies rest | Yes |
| Leaderboard | Browser → `http://localhost:8000/leaderboard` | Yes |
| Classification UI | Needs a logged-in session (use OAuth or hack the session) | No |
| Model Results / validation | Needs a logged-in session with classified faces | No |
| SDC writes | Needs real OAuth token | No |
