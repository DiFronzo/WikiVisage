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

# Token encryption (REQUIRED in production — the app refuses to start on
# Toolforge without it). Encrypts OAuth tokens at rest in the DB.
toolforge envvars create WIKIVISAGE_TOKEN_KEY  "$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# Redis for shared rate limiting (optional — falls back to in-memory if unavailable)
# Default: redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379
# Only needed if you want to customize the Redis URL
# toolforge envvars create WIKIVISAGE_REDIS_URL "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379"

# Face service (REQUIRED — image processing does nothing without it).
# The token must be the same value as on the face tool.
toolforge envvars create WIKIVISAGE_FACE_SERVICE_URL   "https://wikivisage-face.toolforge.org"
toolforge envvars create WIKIVISAGE_FACE_SERVICE_TOKEN "<same token as the face tool>"

# LiftWing metadata pre-filter (optional, experimental, worker-only).
# Anonymous access works (rate limited per IP); a Wikimedia personal API token
# from api.wikimedia.org moves calls to the authenticated quota. Only public
# Commons metadata is sent — never image bytes. Calls are serialised and spaced
# by a jittered minimum interval (default 2 s; WIKIVISAGE_LIFTWING_MIN_INTERVAL).
# toolforge envvars create WIKIVISAGE_LIFTWING_API_TOKEN "<personal-api-token>"

# Sessions (optional; the defaults are what production runs).
# Idle lifetime is sliding: every response re-sends the cookie, so it only runs
# down while the app is closed. The maximum age counts from login.
# toolforge envvars create WIKIVISAGE_SESSION_IDLE_DAYS    "30"
# toolforge envvars create WIKIVISAGE_SESSION_MAX_AGE_DAYS "90"
# Sampled "session loss:" log line in the web logs (default on; 0 turns it off)
# toolforge envvars create WIKIVISAGE_LOG_SESSION_LOSS     "0"

# Worker liveness (optional; the defaults are what production runs).
# Seconds without any worker progress before the heartbeat stops and Toolforge
# restarts the worker (default 900).
# toolforge envvars create WIKIVISAGE_WORKER_STALL_LIMIT   "900"
# Seconds a planned stop (deploy, restart) keeps the offline banner away (default 1200).
# toolforge envvars create WIKIVISAGE_WORKER_RESTART_GRACE "1200"
```

> **Database connections.** ToolsDB allows 20 per user across every process of
> the tool. The web app uses 2 gunicorn workers x 3 (`WIKIVISAGE_WEB_DB_POOL_SIZE`);
> each of the two background workers uses a pool of at most 5
> (`WIKIVISAGE_DB_POOL_SIZE`, default `min(max_projects x image_threads + 3, 5)`)
> **plus one dedicated connection** for its heartbeat thread:
> 2 x 3 + 2 x (5 + 1) = 18 of 20. Do not raise the pool sizes without lowering
> something else.

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

The build uses `Procfile` and `requirements.txt` to create the container image. There is no `project.toml`: nothing in the stack needs system packages any more (the face service's ONNX Runtime and OpenCV wheels are self-contained). The image name is automatically `tool-wikivisage/tool-wikivisage:latest`.

---

## 5. Run the Schema Migration

Run as a one-off job:

```bash
toolforge jobs run migrate \
  --command "python migrate.py" \
  --image tool-wikivisage/tool-wikivisage:latest \
  --mem 512Mi
```

This creates all 10 tables (`users`, `projects`, `images`, `faces`, `user_stats`, `sdc_claims`, `project_members`, `worker_heartbeats`, `service_status`, and the legacy single-row `worker_heartbeat`, which nothing reads any more and a later release drops) and their indexes. Safe to re-run — migrations are idempotent.

Check the migration completed:

```bash
toolforge jobs logs migrate
```

---

## 6. Start the Web Service

```bash
toolforge webservice buildservice start --mount none
```

The `Procfile` runs gunicorn with 2 `gthread` workers (4 threads each) on port 8000, so a slow request (face service, Wikidata) occupies one thread instead of a whole worker.

Your app will be live at: **https://wikivisage.toolforge.org**

---

## 7. Start the Background Workers

WikiVisage uses 2 concurrent worker instances for distributed processing. Workers claim projects via `SELECT … FOR UPDATE` with automatic stale-claim expiry (15 min).

Start both workers:

```bash
toolforge jobs run ml-worker \
  --command 'python -u worker.py --worker-id ml-worker-1' \
  --image tool-wikivisage/tool-wikivisage:latest \
  --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-1'

toolforge jobs run ml-worker-2 \
  --command 'python -u worker.py --worker-id ml-worker-2' \
  --image tool-wikivisage/tool-wikivisage:latest \
  --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-2'
```

This starts two continuous jobs (`ml-worker` and `ml-worker-2`) that crawl Commons categories, download images, detect faces, and run the classification model. Each worker polls for new work every 60 seconds. `deploy.yml` starts them the same way.

Each worker also runs a heartbeat thread: every 30 seconds it touches `$HOME/.wikivisage-worker-alive-<worker-id>` (what `healthcheck.sh` checks) and upserts its row in `worker_heartbeats` over its own database connection. If the worker makes no progress for 15 minutes (`WIKIVISAGE_WORKER_STALL_LIMIT`) the thread stops touching the file, so Toolforge restarts the stuck worker. A database outage alone no longer fails the health check.

Check worker status:

```bash
toolforge jobs list
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2
```

---

## 8. Verify Everything

1. **Health check**: Visit `https://wikivisage.toolforge.org/health` — should return `"status": "healthy"`, `"database": "connected"` and `"face_service": "reachable"`. `"workers"` shows `{"fresh": 2, "stopping": 0, "stale": 0}` once both workers have beaten (informational; it never makes the check `degraded`)
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

The deploy order does not change for the offline banner. Stopping the workers
sends them SIGTERM; each one writes a "stopping" row with an
`expected_back_until` 20 minutes ahead (`WIKIVISAGE_WORKER_RESTART_GRACE`)
before it releases its claims, and the restarted worker's first beat clears it.
Users therefore do not see "The background worker appears to be offline" during
a normal deploy. If the workers stay down longer than the grace period (for
example the preflight failed), the banner appears. A SIGKILL or OOM kill writes
no marker, so the banner appears 5 minutes after the last beat.

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
(KServe, ONNX Runtime, OpenCV headless), `.python-version`, `Procfile` and the
committed model weights in `weights/`. No system packages are installed. The
`Dockerfile` is only for local development and CI.

The weights (`det_500m.onnx`, `w600k_mbf.onnx`, about 16 MB) are **ordinary git
blobs**, not Git LFS or submodules, because the build service does a plain
clone. They are InsightFace models licensed for **non-commercial research use
only** — see `model-server/weights/LICENSE-NOTICE`. The service verifies their
SHA-256 at startup and exits on a mismatch, so a broken build never becomes ready
and the deploy's readiness wait fails before any worker is restarted. If Toolforge
administrators object to hosting them, the `FaceBackend` interface in
`model-server/backend.py` is where a permissively licensed pair (YuNet + SFace)
would go.

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

Throughput: one ONNX session per model is shared by `WIKIVISAGE_FACE_WORKERS`
threads (default 2, matching the job's `--cpu 2`). To scale beyond that, add
`--replicas N` to the job. After a deploy, `toolforge jobs logs face-service`
shows a `Weights verified: ...` line with both SHA-256 values and a
`Model 'wikivisage' loaded (model_id=scrfd500m+arcface_mbf, dim=512, ...)` line.

### Upgrading the embedding model (one-time wipe) and rollback

The switch from dlib (128D) to ArcFace (512D) embeddings is **destructive**:
the `migrate` job's `apply_embedding_model_upgrade()` deletes every image, face
and SDC claim row exactly once (it is gated on the absence of
`faces.embedding_model`), resets every project's progress counters and
distance threshold, and reactivates `completed` projects. Users, projects,
project members, sessions and the leaderboard stats are kept, and the worker
re-crawls each project. Safe for Commons: `write_sdc_claims` checks Commons for
an existing P180 claim before writing, and bootstrap re-seeds from claims that
already exist.

Before the first deploy of this version:

1. **Take a database dump and store it outside the repository** (it holds user
   classifications and OAuth tokens):
   `mysqldump --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud --single-transaction <db> > ~/wikivisage-pre-wipe.sql`
2. Export the labelled data if you want to calibrate the thresholds afterwards.

Deploy order is already safe: the face tool is rebuilt and ready before the
workers are stopped, and a worker that reaches an old face service (or a new one
with an old worker) refuses vectors of the wrong model/width, leaving images
`pending`.

Rollback after the wipe: the old rows exist only in the dump. Restore it into
the database, redeploy the previous release tag (`deploy.yml` with the old
`tag`, which rebuilds the previous `face-service-<tag>` too) and re-run
`migrate` — the restored `faces` table has no `embedding_model`, so with the
*old* code nothing wipes it; with the *new* code it would be wiped again, so do
not run the new `migrate` against a restored dump.

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

# Restart both workers (delete + run; there is no "update" command)
toolforge jobs delete ml-worker || true
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-1'

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-2'
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
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-1'
toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-2'
```

---

## Troubleshooting

### "The background worker appears to be offline" banner

The banner shows only when **none** of these is true: a worker row in `worker_heartbeats` was seen in the last 5 minutes; a worker announced a planned stop that has not run out (20 minutes); a project claim was refreshed in the last 5 minutes. An empty `worker_heartbeats` table (a fresh install, or the minutes between a deploy's migration and the workers' first beat) and an unreadable database do not count as an outage. If you see this banner:

```bash
# Check if workers are running
toolforge jobs list

# Check worker logs for errors
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2

# Restart workers
toolforge jobs delete ml-worker || true
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-1'

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 2Gi --cpu 2 --health-check-script './healthcheck.sh ml-worker-2'
```

### Diagnosing "worker offline" reports

When users report the banner (or workers seem to restart), find out whether the
worker is down, restarting, or only failing to write its heartbeat:

```bash
become wikivisage

# State, restarts and the job definition
toolforge jobs list
toolforge jobs show ml-worker

# Restart count and node; then the reason for the last restart
kubectl get pods -o wide
kubectl describe pod <ml-worker-pod>      # look for OOMKilled or "Liveness probe failed"

# What the worker itself logged
toolforge jobs logs ml-worker | grep -E "Starting WikiVisage Worker|Database error in main loop|Pool exhausted|Heartbeat write failed|No worker progress for|Announced planned stop"
```

What the log lines mean:

| Line | Meaning |
|---|---|
| `Starting WikiVisage Worker` | A (re)start. Many of them in a row = restart loop |
| `Heartbeat write failed` | The heartbeat thread could not write its row (at most one warning per minute). The file is still touched, so Toolforge does not restart the worker; the banner may show if the project claims also go stale |
| `No worker progress for ... (phase=...)` | Nothing reported progress for 15 minutes. It comes with the main thread's stack. The thread then stops touching the liveness file, and Toolforge restarts the worker about 5 minutes later |
| `Announced planned stop` | SIGTERM arrived (deploy, restart, `jobs delete`); the banner stays hidden for the grace period |
| `Database error in main loop` / `Pool exhausted` | The database is unreachable or the pool is full |

What the database says (ToolsDB shell, see step 3):

```sql
SELECT worker_id, state, last_seen, expected_back_until, phase, version,
       TIMESTAMPDIFF(SECOND, last_seen, NOW()) AS age
FROM worker_heartbeats;
```

A healthy worker has `state = 'running'` and an `age` under about 35 seconds. `state = 'stopping'` with `expected_back_until` in the future is a planned stop. Also check what the web app reports:

```bash
curl -s https://wikivisage.toolforge.org/health | jq .workers
```

`fresh` counts rows seen in the last 5 minutes, `stopping` planned stops that are still within their grace, `stale` everything else.

### Users report being logged out

Sessions last 30 days of inactivity (90 days at most from login, sliding idle
lifetime), the `session` cookie is persistent, and a database error shows a
"temporarily unavailable" page instead of ending the session. Reasons that
still end one: the user pressed "Log out everywhere" (account settings), the
browser dropped the cookie, the 90-day cap, or a changed `FLASK_SECRET_KEY` (the
default is random per start, so **set it**). To see which one, read the web
log. Anonymous requests write a sampled (5 %) line:

```
session loss: reason=<no cookie sent|expired|bad signature|valid but empty> had_cookie=<bool> client=<ios|android|desktop|other> path=<path>
```

```bash
toolforge webservice logs | grep "session loss"
```

| `reason` | Usually means |
|---|---|
| `no cookie sent` | The browser sent no `session` cookie: it was never set, was cleared, or was dropped by the client. Also counts first-time visitors and bots, so compare counts over time (per `client`) instead of reading single lines |
| `expired` | A valid cookie older than the idle lifetime |
| `bad signature` | The cookie is not signed with the current `FLASK_SECRET_KEY` (key changed, or a forged cookie) |
| `valid but empty` | A valid cookie with no user in it (for example after logout) |

`WIKIVISAGE_LOG_SESSION_LOSS=0` turns the line off.

**Known platform caveat (unverified).** iOS home-screen apps keep their own
cookie jar and open the Wikimedia OAuth page outside the app shell, so logging
in from inside the installed app may not return into it. This has not been
checked on a real device; the platform matrix in TESTING.md lists every client to
test, and TESTING.md §10 describes the check.

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

### One project stops part-way ("224 / 264 images")

The face service is up and everything else processes, but one project stays at
the same count. A few images are failing **on their own**. There are two ways,
and the cause of the one production report was never confirmed, so look at the
logs first:

- **A download keeps failing** (`Transient error downloading image` in the log): Commons
  answers 5xx or times out for that one file's thumbnail. It earns a strike
  whenever another download of the same wave worked (or a small canary thumbnail
  downloads), and never while Commons serves nothing or answers 429. The third
  strike marks it `error`.
- **The face service chokes on it** (a huge or damaged file hangs or crashes the
  service and takes the other images of its request down with it).

The worker handles the second in three steps (strikes are shared):

1. After a failed request it waits up to `WIKIVISAGE_DETECT_RECOVERY_WAIT`
   (60 s) for the face service to **detect** again (a tiny real detection, not
   just "ready"), then re-sends the images one at a time, testing the service
   before each one.
2. An image that fails alone right after a working test is sent alone once more
   (another thread or worker may have hurt the shared service meanwhile); one that
   fails both times gets a strike
   (`images.detect_attempts`). The third strike (`WIKIVISAGE_MAX_DETECT_ATTEMPTS`)
   marks it `error`, so the project can finish.
3. If the service does not recover in time, nobody is blamed: the images are
   flagged `detect_isolate` and go alone next cycle.

It takes about three poll cycles for a bad image to be written off, so wait a few
minutes before digging.

```bash
# Which images are holding it up
toolforge jobs logs ml-worker   | grep -E "Transient error downloading|Detection failed for|Giving up on|Face service batch of|stopped early"
toolforge jobs logs ml-worker-2 | grep -E "Transient error downloading|Detection failed for|Giving up on|Face service batch of|stopped early"
```

```sql
-- in the tool's database: the stragglers of one project, worst first
SELECT id, file_title, status, detect_attempts, detect_isolate, error_message
FROM images WHERE project_id = <ID> AND (status = 'pending' OR detect_attempts > 0 OR detect_isolate = 1)
ORDER BY detect_attempts DESC, id;
```

If `detect_attempts` stays 0 while images stay pending and the log says `Face
service unreachable` or `batch ... failed` for every image, it is the service
itself (the previous section, or a face service that answers readiness but cannot
detect: look at `toolforge jobs logs face-service`), not the images. Images are
never struck while a test detection fails, so this cannot write good images off.

A project that never leaves "Waiting for worker..." is the wake-up or the
workers: the web app wakes them through the `worker_wake` table (they also poll
every minute), and `/health` shows how many workers have a fresh heartbeat. A
project that is paused or completed with no images says "No images have been
found" instead, because no worker will come for it.

### Database connection errors

- Verify credentials: `toolforge envvars list`
- Verify the database exists: `mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud -e "SHOW DATABASES LIKE '%wikiface%'"`
- Check that migration has run: look for 10 tables in the database (`worker_heartbeats` among them)
