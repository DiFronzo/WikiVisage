# Local Testing

Step-by-step guide to running WikiVisage locally for development and testing.

## 1. Install Dependencies

```bash
# Create a virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # Includes runtime deps + pytest, ruff
```

No compiler needed — `requirements.txt` contains no ML libraries. ONNX Runtime lives
only in the face service container.

## 1b. Start the Face Service

The worker cannot process images without it. Detection failures are treated as
transient, so a missing face service shows up as images stuck `pending` rather
than as an error.

```bash
cd model-server && docker compose up --build   # http://localhost:8080
```

Leave it running and set `WIKIVISAGE_FACE_SERVICE_URL=http://localhost:8080` in
the environment below. See [TESTING.md](TESTING.md) for details.

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
# Optional: encrypt OAuth tokens at rest (leave unset for plaintext)
# export WIKIVISAGE_TOKEN_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

## 4. Run Migration

```bash
python migrate.py
```

This creates all 10 tables (`users`, `projects`, `images`, `faces`, `user_stats`, `sdc_claims`, `project_members`, `worker_heartbeats`, `service_status`, and the legacy `worker_heartbeat`, which nothing reads any more) and their indexes. Safe to re-run.

## 5. Run the Web App

```bash
python app.py
# → http://localhost:8000
```

The landing page and health check (`/health`) will work immediately. OAuth login won't work without real credentials.

### Testing from another device (phone, tablet, second laptop)

`python app.py` binds to `127.0.0.1` when `FLASK_DEBUG` is on, so only this machine can reach it. To open the app from a phone on the same network:

```bash
# FLASK_DEBUG must be off: the debug server stays on loopback on purpose (the Werkzeug
# debugger allows code execution). Shell variables win over .env because load_dotenv()
# does not override what is already set.
FLASK_DEBUG=0 WIKIVISAGE_INSECURE_HTTP=1 \
OAUTH_REDIRECT_URI="http://192.168.1.126:3001/auth/callback" \
PORT=3001 python app.py
```

- `WIKIVISAGE_INSECURE_HTTP=1` drops `Secure` from the `session` and `locale` cookies. Without it, browsers discard them over plain http and login bounces straight back to signed out.
- `OAUTH_REDIRECT_URI` must point at the same host the phone uses, and that exact URL must be registered on the OAuth consumer. `localhost` on the phone is the phone, not your laptop.
- Find your address with `ipconfig getifaddr en0`, then open `http://192.168.1.126:3001/`.
- macOS may ask whether to allow incoming connections for Python; accept it.
- Never set `WIKIVISAGE_INSECURE_HTTP` outside local development: anyone on the network can read the cookies.

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
VALUES (1, 'Q42', 'Douglas Adams', 'Douglas Adams', 0.5, 5);
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
| Health check | `curl http://localhost:8000/health` (`workers` shows the heartbeat rows of running local workers) | Yes |
| Landing page | Browser → `http://localhost:8000` | Yes |
| Migration | `python migrate.py` (check exit code) | Yes |
| Category traversal | Insert test project, run worker, check `images` table | Yes |
| Face detection | Same — check `faces` table after worker processes images | Yes |
| SPARQL bootstrap | Worker logs will show bootstrap attempts | Yes |
| Autonomous inference | Manually insert 5+ confirmed faces, worker auto-classifies rest | Yes |
| Inference status card | Log in, open a project that is mid-processing or has eligible faces; the card follows each cycle live | No |
| Leaderboard | Browser → `http://localhost:8000/leaderboard` | Yes |
| Classification UI | Needs a logged-in session (use OAuth or hack the session) | No |
| Model Results / validation | Needs a logged-in session with classified faces | No |
| Gallery face zoom and marker | Same; the cards zoom into the classified face and outline it; the zoom never exceeds 2 device pixels per source pixel | No |
| Log out everywhere | Account settings → Sessions; a second browser logged in as the same user is logged out on its next request | No |
| Cookie banner | Landing page; Accept/Decline posts `/cookie-consent`, footer "Cookie settings" shows it again | Yes |
| SDC writes | Needs real OAuth token | No |
