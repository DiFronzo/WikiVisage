# Contributing to WikiVisage

Thanks for your interest in contributing to WikiVisage! This guide covers how to set up the project locally, make changes, and submit them.

## Local Development Setup

### Prerequisites

- Python 3.11+
- MariaDB (via Docker or Homebrew)
- cmake (macOS only, for dlib compilation)

### 1. Clone and install dependencies

```bash
git clone https://github.com/DiFronzo/WikiVisage.git
cd WikiVisage
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **macOS note:** `dlib-bin` doesn't have macOS ARM wheels. It falls back to compiling from source, which requires cmake (`brew install cmake`) and takes a few minutes.

### 2. Start a local MariaDB

```bash
docker run -d \
  --name wikivisage-db \
  -e MARIADB_ROOT_PASSWORD=devpass \
  -e MARIADB_DATABASE=wikiface_dev \
  -p 3306:3306 \
  mariadb:10.11
```

Or via Homebrew: `brew install mariadb && brew services start mariadb`

### 3. Set environment variables

```bash
export TOOL_TOOLSDB_USER=root
export TOOL_TOOLSDB_PASSWORD=devpass
export TOOL_TOOLSDB_HOST=127.0.0.1
export WIKIVISAGE_DB_NAME=wikiface_dev
export FLASK_SECRET_KEY=dev-secret-key
export OAUTH_CLIENT_ID=""
export OAUTH_CLIENT_SECRET=""
export OAUTH_REDIRECT_URI="http://localhost:8000/auth/callback"
export OAUTHLIB_INSECURE_TRANSPORT=1
```

### 4. Run migrations and start

```bash
python migrate.py          # Create/update database tables
python app.py              # Web app on http://localhost:8000
python -u worker.py        # Background worker (separate terminal)
```

The landing page and `/health` endpoint work without OAuth. See [test-local.md](test-local.md) for testing without OAuth credentials and a full smoke test checklist.

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Flask web app: OAuth, routes, classification API |
| `worker.py` | Background ML pipeline: crawl, detect, infer, write SDC |
| `database.py` | MariaDB connection pool with retry logic |
| `schema.sql` | DDL for all tables |
| `migrate.py` | Idempotent schema migrations |
| `templates/` | Jinja2 templates (all extend `base.html`) |
| `translations/` | i18n files (en, nb, es, fr) |

## Code Conventions

### Python

- Use type hints and f-strings.
- All database queries use `%s` parameterized placeholders (PyMySQL). Never interpolate **values** into SQL via f-strings. F-strings are acceptable for structural SQL (e.g., building `IN (%s, %s, %s)` placeholder lists from `",".join(["%s"] * len(ids))`).
- Use `execute_query()` for reads and simple writes, `execute_insert()` for INSERTs needing `lastrowid`, and `execute_transaction()` for atomic multi-statement mutations.
- Ruff is configured in `pyproject.toml` for linting and formatting (line-length 120, target py311). Run `ruff check .` and `ruff format --check .` before submitting.

### Templates

- All templates extend `base.html` with three blocks: `title`, `extra_head`, `content`.
- CSS is embedded in `<style>` tags (CSS custom properties in `base.html`, per-page styles in `extra_head`). No external CSS files.
- JavaScript is inline in `<script>` tags. No build system.

### Security

- All POST routes require CSRF tokens.
- All face bounding box inputs are validated against `MAX_BBOX_PX` and `MIN_BBOX_AREA`.
- Never store secrets in code. Use environment variables.

## Internationalization (i18n)

All user-facing strings must be translatable. WikiVisage uses Flask-Babel with gettext.

### Wrapping strings

- **Python:** `_("text")` (imported as `from flask_babel import gettext as _`)
- **Jinja2:** `{{ _('text') }}`
- **JavaScript in templates:** Pass strings via `|tojson` into a JS object. Never use `_()` inside `<script>` blocks.
- **Placeholders:** Python/Jinja2 use `%(name)s` named placeholders. JS strings use `{placeholder}` for `.replace()`.

### What NOT to translate

Worker log messages, health endpoint JSON, technical terms (Wikidata, Q-ID, Commons, SDC, P180, OAuth, CSRF, WikiVisage, BETA).

### Updating translations after changing strings

```bash
source venv/bin/activate

# Extract new/changed strings
pybabel extract -F babel.cfg -o messages.pot .

# Update all .po files (preserves existing translations)
pybabel update -i messages.pot -d translations

# Edit the .po files — fix any new entries
# English .po uses identity translations (msgstr = msgid)

# Compile .po -> .mo (required after any .po change)
pybabel compile -d translations
```

### Adding a new language

1. Add the locale code and display name to the `LANGUAGES` dict in `app.py`.
2. Initialize: `pybabel init -i messages.pot -d translations -l <code>`
3. Translate all `msgstr` entries in the new `.po` file.
4. Compile: `pybabel compile -d translations`
5. If the language is RTL, add the code to `RTL_LANGUAGES` in `app.py`.

## Schema Changes

1. Edit `schema.sql` with the new DDL.
2. Add an idempotent migration in `migrate.py` (check-then-alter pattern).
3. Run `python migrate.py` to verify it applies cleanly.
4. Migrations must be safe to run repeatedly without error.

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, following the conventions above.
3. If you changed translatable strings, run the pybabel extract/update/compile workflow.
4. If you changed the schema, add a migration in `migrate.py`.
5. Test locally (see [test-local.md](test-local.md)).
6. Open a pull request with a clear description of what changed and why.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
