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
toolforge envvars create TOOL_TOOLSDB_USER    --value "s<NNNNN>"
toolforge envvars create TOOL_TOOLSDB_PASSWORD --value "<your-tools-db-password>"
toolforge envvars create WIKIVISAGE_DB_NAME    --value "s<NNNNN>__wikiface"

# OAuth 2.0 (from your consumer registration on Meta-Wiki)
toolforge envvars create OAUTH_CLIENT_ID       --value "<client-id>"
toolforge envvars create OAUTH_CLIENT_SECRET   --value "<client-secret>"
toolforge envvars create OAUTH_REDIRECT_URI    --value "https://wikivisage.toolforge.org/auth/callback"

# Flask secret key (generate a strong random one)
toolforge envvars create FLASK_SECRET_KEY      --value "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
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
toolforge build start https://github.com/<you>/WikiVisage.git

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

This creates all 6 tables (`users`, `sessions`, `projects`, `images`, `faces`, `worker_heartbeat`). Safe to re-run — uses `CREATE TABLE IF NOT EXISTS`.

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
toolforge webservice buildservice start
```

The `Procfile` tells it to run gunicorn with 4 workers on port 8000.

Your app will be live at: **https://wikivisage.toolforge.org**

---

## 8. Start the Background Worker

The worker job is defined in `jobs.yaml`:

```bash
toolforge jobs load jobs.yaml
```

This starts a continuous job (`ml-worker`) that crawls Commons categories, downloads images, detects faces, and runs the classification model. It polls for new work every 60 seconds.

Check worker status:

```bash
toolforge jobs list
toolforge jobs logs ml-worker
```

---

## 9. Verify Everything

1. **Health check**: Visit `https://wikivisage.toolforge.org/health` — should return `{"status": "healthy", "database": "connected"}`
2. **Login**: Click "Log in with Wikimedia" — should redirect to Meta for OAuth, then back to the dashboard
3. **Create a project**: Enter a Wikidata Q-ID (e.g., `Q42` for Douglas Adams) and a Commons category
4. **Worker activity**: Check `toolforge jobs logs ml-worker` — should show category traversal starting within 60 seconds

---

## Common Operations

### View logs

```bash
# Web service logs
toolforge webservice logs

# Worker logs
toolforge jobs logs ml-worker
```

### Rebuild after code changes

```bash
# Rebuild the image
toolforge build start https://github.com/<you>/WikiVisage.git

# Wait for build to finish
toolforge build show

# Restart web service
toolforge webservice restart

# Restart worker
toolforge jobs restart ml-worker
```

### Restart web service only

```bash
toolforge webservice restart
```

### Stop everything

```bash
toolforge webservice stop
toolforge jobs delete ml-worker
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
toolforge envvars create FLASK_SECRET_KEY --value "<new-value>"

# Restart services to pick up changes
toolforge webservice restart
toolforge jobs restart ml-worker
```

### Update the whitelist

Edit `whitelist.txt` in the repo, then rebuild and restart:

```bash
toolforge build start https://github.com/<you>/WikiVisage.git
toolforge webservice restart
```

---

## Troubleshooting

### "The background worker appears to be offline" banner

The web app checks if the worker has sent a heartbeat in the last 5 minutes. If you see this banner:

```bash
# Check if the worker is running
toolforge jobs list

# Check worker logs for errors
toolforge jobs logs ml-worker

# Restart the worker
toolforge jobs restart ml-worker

# Or reload from jobs.yaml (e.g., after changing memory/image settings)
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
- Check that migration has run: look for 6 tables in the database
