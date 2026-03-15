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
```

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

The build uses `Procfile`, `requirements.txt`, and `Aptfile` to create the container image. The image name is automatically `tool-wikivisage/tool-wikivisage:latest`.

---

## 5. Run the Schema Migration

Run as a one-off job:

```bash
toolforge jobs run migrate \
  --command "python migrate.py" \
  --image tool-wikivisage/tool-wikivisage:latest \
  --mem 512Mi
```

This creates all 7 tables (`users`, `sessions`, `projects`, `images`, `faces`, `user_stats`, `worker_heartbeat`) and their indexes. Safe to re-run — migrations are idempotent.

Check the migration completed:

```bash
toolforge jobs logs migrate
```

---

## 6. Configure the Whitelist

WikiVisage restricts access to whitelisted usernames. Edit `whitelist.txt` in your repo — one Wikimedia username per line:

```
YourUsername
AnotherUser
```

The file is checked on every request (no restart needed after changes). To update the whitelist, commit the change, rebuild, and restart the web service.

> **Note**: If `whitelist.txt` is empty or missing, the app is open to all authenticated users.

---

## 7. Start the Web Service

```bash
toolforge webservice buildservice start --mount none
```

The `Procfile` runs gunicorn with 2 workers on port 8000.

Your app will be live at: **https://wikivisage.toolforge.org**

---

## 8. Start the Background Workers

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
  --continuous --mem 3Gi

toolforge jobs run ml-worker-2 \
  --command 'python -u worker.py --worker-id ml-worker-2' \
  --image tool-wikivisage/tool-wikivisage:latest \
  --continuous --mem 3Gi
```

Check worker status:

```bash
toolforge jobs list
toolforge jobs logs ml-worker
toolforge jobs logs ml-worker-2
```

---

## 9. Verify Everything

1. **Health check**: Visit `https://wikivisage.toolforge.org/health` — should return `{"status": "healthy", "database": "connected"}`
2. **Login**: Click "Log in with Wikimedia" — should redirect to Meta for OAuth, then back to the dashboard
3. **Create a project**: Enter a Wikidata Q-ID (e.g., `Q42` for Douglas Adams) and a Commons category
4. **Worker activity**: Check `toolforge jobs logs ml-worker` — should show category traversal starting within 60 seconds

---

## Automated Deployment (CD)

Releases trigger an automated deployment via `.github/workflows/deploy.yml`. The CD workflow:

1. Builds a new container image from the release tag
2. Runs schema migration
3. Restarts both workers (`ml-worker` and `ml-worker-2`)
4. Restarts the web service

To deploy manually, use the **workflow_dispatch** trigger on the Actions tab with a git tag. The workflow also supports an optional database wipe (requires typing `WIPE` as confirmation).

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
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi
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
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi
toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi
```

### Update the whitelist

Edit `whitelist.txt` in the repo, then rebuild and restart:

```bash
toolforge build start https://github.com/DiFronzo/WikiVisage.git
toolforge webservice restart
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
toolforge jobs run ml-worker --command 'python -u worker.py --worker-id ml-worker-1' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi

toolforge jobs delete ml-worker-2 || true
toolforge jobs run ml-worker-2 --command 'python -u worker.py --worker-id ml-worker-2' --image tool-wikivisage/tool-wikivisage:latest --continuous --mem 3Gi

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
# - Aptfile references a package not in Ubuntu 22.04 repos
```

### Database connection errors

- Verify credentials: `toolforge envvars list`
- Verify the database exists: `mariadb --defaults-file=$HOME/replica.my.cnf -h tools.db.svc.wikimedia.cloud -e "SHOW DATABASES LIKE '%wikiface%'"`
- Check that migration has run: look for 7 tables in the database
