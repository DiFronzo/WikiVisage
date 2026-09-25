# WikiVisage — Toolforge Setup & Deployment

Complete guide to deploying WikiVisage on Wikimedia Toolforge using Build Service.

## Prerequisites

1. A [Toolforge account](https://toolsadmin.wikimedia.org/) with a tool created (e.g., `wikivisage`)
2. An [OAuth 2.0 consumer](https://meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose) registered on Meta-Wiki with:
   - **OAuth version**: 2.0 (not 1.0a)
   - **Applicable grants**: `Basic rights` + `Edit existing pages`
   - **Callback URL**: `https://wikivisage.toolforge.org/auth/callback`

> **Note**: OAuth consumer registration requires approval. This can take hours or days. Start this step first.

---

## 1. SSH into Toolforge

```bash
# Connect to the Toolforge bastion
ssh <username>@login.toolforge.org

# Switch to the tool account
become wikivisage
```

All subsequent commands assume you're running as the tool account.

---

## 2. Set Environment Variables

Environment variables are how Build Service apps receive configuration (NFS home directory is not available inside containers).

### Find your database credentials

```bash
cat ~/replica.my.cnf
# user = s<NNNNN>
# password = <password>
```

### Set the variables

```bash
# Database
toolforge envvars create TOOL_TOOLSDB_USER    "s<NNNNN>"
toolforge envvars create TOOL_TOOLSDB_PASSWORD "<your-tools-db-password>"
toolforge envvars create WIKIVISAGE_DB_NAME    "s<NNNNN>__wikiface"

# OAuth 2.0 (from your consumer registration on Meta-Wiki)
toolforge envvars create OAUTH_CLIENT_ID       "<client-id>"
toolforge envvars create OAUTH_CLIENT_SECRET   "<client-secret>"
toolforge envvars create OAUTH_REDIRECT_URI    "https://wikivisage.toolforge.org/auth/callback"

# Flask secret key (generate a strong random one)
toolforge envvars create FLASK_SECRET_KEY      "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

# Token encryption (optional — encrypts OAuth tokens at rest in the DB)
# If unset, tokens are stored as plaintext (backward compatible)
toolforge envvars create WIKIVISAGE_TOKEN_KEY  "$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# Redis for shared rate limiting (optional — falls back to in-memory if unavailable)
# Default: redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379
# Only needed if you want to customize the Redis URL
# toolforge envvars create WIKIVISAGE_REDIS_URL "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379"

# Face service (REQUIRED — image processing does nothing without it).
# The token must be the same value as on the face tool.
toolforge envvars create WIKIVISAGE_FACE_SERVICE_URL   "https://wikivisage-face.toolforge.org"
toolforge envvars create WIKIVISAGE_FACE_SERVICE_TOKEN "<same token as the face tool>"
```

> The face service runs as a **second tool** — see
> [Face Service Deployment](#face-service-deployment-second-tool) below. If
> `WIKIVISAGE_FACE_SERVICE_URL` or the token is wrong, or the service is down,
> the workers do not crash: they leave images `pending` and retry forever. The
> deploy preflight is what turns that silent stall into a failed release.

Verify with:

```bash
toolforge envvars list
```

---

## 3. Create the Database

```bash
# Connect to ToolsDB
mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud
```

```sql
CREATE DATABASE s<NNNNN>__wikiface;
exit;
```

> **Important**: The database name must follow the pattern `<tool-db-user>__<name>`. ToolsDB only allows you to create databases prefixed with your credential username.

---

## 4. Build the Container Image

```bash
# Build from your GitHub repo
toolforge build start https://github.com/DiFronzo/WikiVisage.git

# Check build status (wait for "ok")
toolforge build show
```

The build uses `Procfile` and `requirements.txt` to create the container image. The root `project.toml` was removed — it only existed to install dlib's system libraries, which now live with the face service in `model-server/project.toml`. The image name is automatically `tool-wikivisage/tool-wikivisage:latest`.

---

## 5. Run the Schema Migration

Run as a one-off job:

```bash
toolforge jobs run migrate \
  --command "python migrate.py" \
  --image tool-wikivisage/tool-wikivisage:latest \
  --mem 512Mi
```

This creates all 9 tables (`users`, `sessions`, `projects`, `images`, `faces`, `user_stats`, `sdc_claims`, `project_members`, `worker_heartbeat`) and their indexes. Safe to re-run — migrations are idempotent.

Check the migration completed:

```bash
toolforge jobs logs migrate
```

---

## 6. Start the Web Service

```bash
toolforge webservice buildservice start --mount none
```

The `Procfile` runs gunicorn with 2 workers on port 8000.

Your app will be live at: **https://wikivisage.toolforge.org**

---

## 7. Start the Background Workers

WikiVisage uses 2 concurrent worker instances for distributed processing. Workers claim projects via `SELECT … FOR UPDATE` with automatic stale-claim expiry (15 min).

Load both workers from `jobs.yaml`:

```bash
toolforge jobs load jobs.yaml
```

This starts two continuous jobs (`ml-worker` and `ml-worker-2`) that crawl Commons categories, download images, detect faces, and run the classification model. Each worker polls for new work every 60 seconds.

If `jobs.yaml` loading fails, start workers manually:

```bash
toolforge jobs run ml-worker \
  --command 'python -u worker.py --worker-id ml-worker-1' \
  --image tool-wikivisage/tool-wikivisage:latest \
  --continuous --mem 1Gi --cpu 2

toolforge jobs run ml-worker-2 \
  --command 'python -u worker.py --worker-id ml-worker-2' \
  --image tool-wikivisage/tool-wikivisage:latest \
  --continuous --mem 1Gi --cpu 2
```

Check worker status:

```bash
toolforge jobs list
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2
```

---

## 8. Verify Everything

1. **Health check**: Visit `https://wikivisage.toolforge.org/health` — should return `{"status": "healthy", "database": "connected"}`
2. **Login**: Click "Log in with Wikimedia" — should redirect to Meta for OAuth, then back to the dashboard
3. **Create a project**: Enter a Wikidata Q-ID (e.g., `Q42` for Douglas Adams) and a Commons category
4. **Worker activity**: Check `toolforge jobs logs ml-worker` — should show category traversal starting within 60 seconds

---

## Automated Deployment (CD)

Publishing a GitHub release deploys **both tools** via
`.github/workflows/deploy.yml`, with no manual steps. Order matters: the face
service is updated and proven healthy *before* the workers are allowed to start.

0. **Check configuration** on both tools before touching either: the face tool
   must have `WIKIVISAGE_FACE_SERVICE_TOKEN`, the main tool
   `WIKIVISAGE_FACE_SERVICE_URL` and `WIKIVISAGE_FACE_SERVICE_TOKEN`. A missing
   envvar fails the release with both tools still on the previous version.
1. **Publish the face service source.** Toolforge builds from the root of a git
   ref, and the face service lives in `model-server/`. The workflow pushes a tag
   `face-service-<release>` whose commit tree *is* `model-server/` at that
   release (reused if it already exists and matches).
2. **Deploy the face tool** (`become wikivisage-face`): builds
   `face-service-<release>`, recreates the `face-service` job, and waits for
   `https://wikivisage-face.toolforge.org/v1/models/wikivisage` to report ready.
3. **Deploy the main tool** (`become wikivisage`):
   1. Builds the image from the release tag
   2. **Stops** both workers
   3. Runs schema migration
   4. Restarts the web service
   5. **Preflight** — polls `/health` until `face_service` is `reachable`. That
      check uses the web app's own URL and token, and probes a token-protected
      path, so a wrong or missing token fails here
   6. Starts both workers, but *only* if the preflight passed

If the preflight fails the deploy fails and the workers stay down. That is
deliberate: workers running against a dead face service look healthy while
processing nothing, whereas a failed deploy is visible.

To deploy manually, use the **workflow_dispatch** trigger on the Actions tab
with a git tag. The workflow also supports an optional database wipe (requires
typing `WIPE` as confirmation). Rolling back either tool is re-running the
workflow with an older tag.

### Required GitHub configuration

| Kind | Name | Environment | Value |
|---|---|---|---|
| Variable | `FACE_TOOL` | — | *optional* — defaults to `wikivisage-face` |
| Secret | `HOST` / `USERNAME` / `KEY` | `toolforge` | *(existing)* — the SSH user must maintain **both** tools |

---

## Face Service Deployment (second tool)

Face detection runs on Toolforge as its own tool, not inside `wikivisage`.
Toolforge envvars are injected into every job of a tool, so a face-service job
in the main tool would receive the ToolsDB password, the OAuth secret and
`WIKIVISAGE_TOKEN_KEY` — in the one process that parses untrusted image bytes. A
second tool holds none of them. If
[T405022](https://phabricator.wikimedia.org/T405022) (per-job envvars) ships,
it could move back into the main tool as an internal job.

The tool is built with buildpacks from `model-server/`: `requirements.txt`
(dlib via the pre-built `dlib-bin` wheel), `project.toml` (OpenBLAS/LAPACK via
`heroku/deb-packages`), `.python-version` and `Procfile`. The `Dockerfile` is
only for local development and CI.

### One-time setup

**1. Create the tool** `wikivisage-face` in
[toolsadmin](https://toolsadmin.wikimedia.org/tools/) and add the same
maintainers as `wikivisage` (the deploy SSH user must be one of them).

**2. Generate the shared token** and set the *same* value on both tools:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # >= 32 chars

become wikivisage-face
toolforge envvars create WIKIVISAGE_FACE_SERVICE_TOKEN          # paste the token

become wikivisage
toolforge envvars create WIKIVISAGE_FACE_SERVICE_TOKEN          # the same token
toolforge envvars create WIKIVISAGE_FACE_SERVICE_URL "https://wikivisage-face.toolforge.org"
```

**3. Publish a release.** The deploy workflow builds and starts the face tool
automatically. To start it by hand instead:

```bash
become wikivisage-face
toolforge build start --ref face-service-<release> https://github.com/DiFronzo/WikiVisage.git
toolforge jobs run face-service \
  --command web \
  --image tool-wikivisage-face/tool-wikivisage-face:latest \
  --continuous --port 8000 --publish --mount=none \
  --mem 3Gi --cpu 2 \
  --health-check-http /v1/models/wikivisage
curl -s https://wikivisage-face.toolforge.org/v1/models/wikivisage   # {"name":"wikivisage","ready":true}
```

### How requests are secured

```
wikivisage job ──HTTPS──> wikivisage-face.toolforge.org ──> model server :8000
                                                             (RequestGuard: token, 40 MiB body cap)
```

`model-server/guard.py` rejects any request without `Authorization: Bearer
<token>` before it reaches KServe, including KServe's own admin routes such as
`POST /v2/repository/models/<name>/unload`. Only `GET /v1/models/wikivisage`
(name + boolean) is open, because the Toolforge health check cannot send
headers. Bodies over 40 MiB get `413` before being buffered, gRPC is disabled
so there is no second unguarded port, and `model.py` refuses to start on
Toolforge without a token of at least 32 characters.

### Operations

```bash
become wikivisage-face
toolforge jobs logs face-service -f        # logs
toolforge jobs restart face-service        # restart
toolforge jobs show face-service           # status, published URL

# Confirm the main tool agrees it is usable (URL and token)
curl -s https://wikivisage.toolforge.org/health | jq .face_service
```

Throughput: dlib is not thread-safe, so each replica detects one image at a
time. To scale, add `--replicas N` to the job rather than raising
`WIKIVISAGE_FACE_WORKERS`.

---

## Common Operations

### View logs

```bash
# Web service logs
toolforge webservice logs

# Worker logs (both instances)
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2
```

### Rebuild after code changes

The recommended approach is to create a GitHub release, which triggers the CD workflow automatically. For manual rebuilds:

```bash
# Rebuild the image
toolforge build start https://github.com/DiFronzo/WikiVisage.git

# Wait for build to finish
toolforge build show

# Restart web service
toolforge webservice restart

# Restart both workers (delete + run because jobs load doesn't restart unchanged jobs)
toolforge jobs delete ml-worker || true
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2
```

### Restart web service only

```bash
toolforge webservice restart
```

### Stop everything

```bash
toolforge webservice stop
toolforge jobs delete ml-worker
toolforge jobs delete ml-worker-2
```

### Re-run migration (after schema changes)

```bash
toolforge jobs run migrate \
  --command "python migrate.py" \
  --image tool-wikivisage/tool-wikivisage:latest \
  --mem 512Mi
```

### Update environment variables

```bash
# Delete and recreate (there's no "update" command)
toolforge envvars delete FLASK_SECRET_KEY
toolforge envvars create FLASK_SECRET_KEY "<new-value>"

# Restart services to pick up changes
toolforge webservice restart
toolforge jobs delete ml-worker || true
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2
toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2
```

---

## Troubleshooting

### "The background worker appears to be offline" banner

The web app checks if a worker has sent a heartbeat in the last 5 minutes. If you see this banner:

```bash
# Check if workers are running
toolforge jobs list

# Check worker logs for errors
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2

# Restart workers
toolforge jobs delete ml-worker || true
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 1Gi --cpu 2

# Or reload from jobs.yaml
toolforge jobs load jobs.yaml
```

### OAuth login fails

- Verify the callback URL in your OAuth consumer matches exactly: `https://wikivisage.toolforge.org/auth/callback`
- Check that `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, and `OAUTH_REDIRECT_URI` are set: `toolforge envvars list`
- Confirm the OAuth consumer has been approved on Meta-Wiki

### Build fails

```bash
# Check build logs
toolforge build show

# Common causes:
# - requirements.txt has a broken dependency
# - a VCS (git+https://) requirement whose repository no longer resolves,
#   which surfaces as "could not read Username for 'https://github.com'"
```

### Images stay "pending" and never process

Almost always the face service. The worker treats an unreachable service as
transient and leaves images `pending` for the next poll cycle rather than
erroring them, so a misconfigured endpoint stalls silently instead of failing
loudly.

```bash
# What the web app thinks
curl -s https://wikivisage.toolforge.org/health   # expects "face_service": "reachable"

# What the workers are pointed at
toolforge envvars list | grep FACE_SERVICE

# The worker logs its verdict once at startup
toolforge jobs logs ml-worker | grep -i "face service"

# "rejected WIKIVISAGE_FACE_SERVICE_TOKEN (HTTP 401)" in the web or worker logs
# means the two tools hold different tokens. Is the face tool itself up?
curl -s https://wikivisage-face.toolforge.org/v1/models/wikivisage
become wikivisage-face && toolforge jobs logs face-service | tail -50
```

### Database connection errors

- Verify credentials: `toolforge envvars list`
- Verify the database exists: `mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud -e "SHOW DATABASES LIKE '%wikiface%'"`
- Check that migration has run: look for 9 tables in the database
