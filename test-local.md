Local Testing
1. Install Dependencies
# macOS — install dlib build deps first
brew install cmake
# Create a virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
> On macOS, dlib-bin doesn't have wheels — it'll fall back to compiling dlib from source. This needs cmake and takes a few minutes. If it fails, install dlib separately: pip install dlib then pip install face-recognition.
2. Local MariaDB
Easiest via Docker:
docker run -d \
  --name wikivisage-db \
  -e MARIADB_ROOT_PASSWORD=devpass \
  -e MARIADB_DATABASE=wikiface_dev \
  -p 3306:3306 \
  mariadb:10.11
Or if you have Homebrew: brew install mariadb && brew services start mariadb
3. Set Environment Variables
export TOOL_TOOLSDB_USER=root
export TOOL_TOOLSDB_PASSWORD=devpass
export WIKIVISAGE_DB_NAME=wikiface_dev
export FLASK_SECRET_KEY=dev-secret-key-not-for-production
# OAuth — leave empty for now, we'll handle this below
export OAUTH_CLIENT_ID=""
export OAUTH_CLIENT_SECRET=""
export OAUTH_REDIRECT_URI="http://localhost:8000/auth/callback"
4. Run Migration
python migrate.py
5. Run the Web App
python app.py
# → http://localhost:8000
The landing page and health check (/health) will work immediately. OAuth login won't work without real credentials.
6. Run the Worker (separate terminal)
source venv/bin/activate
# Same exports as above
python -u worker.py
The worker will start polling but do nothing until there are active projects in the database.
---
Testing Without OAuth
OAuth is the biggest friction point for local dev. To bypass it for testing, you can insert a fake user and session directly:
mysql -u root -pdevpass wikiface_dev
-- Create a test user
INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at)
VALUES (1, 'TestUser', 'fake-token', 'fake-refresh', '2099-01-01 00:00:00');
-- Create a test project (e.g., Q42 = Douglas Adams, in a real Commons category)
INSERT INTO projects (user_id, wikidata_qid, commons_category, label, distance_threshold, min_confirmed)
VALUES (1, 'Q42', 'Douglas Adams', 'Douglas Adams', 0.6, 5);
Then restart the worker — it'll start crawling the "Douglas Adams" category on Commons, downloading images, and detecting faces.
You can watch it work:
# In another terminal
mysql -u root -pdevpass wikiface_dev -e "SELECT id, file_title, status, face_count FROM images LIMIT 20;"
mysql -u root -pdevpass wikiface_dev -e "SELECT id, image_id, is_target, classified_by, confidence FROM faces LIMIT 20;"
> Note: SDC writes will fail with the fake token — that's expected. The worker logs the error and moves on. Everything else (category traversal, face detection, encoding, SPARQL bootstrapping, autonomous inference) works without a real token.
---
Testing OAuth Locally (Optional)
If you want the full flow:
1. Register a new OAuth 2.0 consumer at meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose (https://meta.wikimedia.org/wiki/Special:OAuthConsumerRegistration/propose)
   - Set callback URL to http://localhost:8000/auth/callback
   - It needs manual approval (can take hours/days)
2. Once approved, set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET to the real values
3. You'll also need to disable the secure cookie for local HTTP:
   
   In app.py, temporarily change:
      app.config["SESSION_COOKIE_SECURE"] = True
      to:
      app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") != "development"
      Then export FLASK_ENV=development before running.
---
Quick Smoke Test Checklist
| What | How | Works without OAuth? |
|---|---|---|
| Health check | curl http://localhost:8000/health | Yes |
| Landing page | Browser → http://localhost:8000 | Yes |
| Migration | python migrate.py (check exit code) | Yes |
| Category traversal | Insert test project, run worker, check images table | Yes |
| Face detection | Same — check faces table after worker processes images | Yes |
| SPARQL bootstrap | Worker logs will show bootstrap attempts | Yes |
| Autonomous inference | Manually insert 5+ confirmed faces, worker auto-classifies rest | Yes |
| Classification UI | Needs a logged-in session (use OAuth or hack the session) | No |
| SDC writes | Needs real OAuth token | No |