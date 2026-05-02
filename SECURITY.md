# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

Only the latest release deployed on Toolforge receives security updates.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report them privately:

1. **GitHub Security Advisories (preferred):** Go to [Security → Advisories → New draft advisory](https://github.com/DiFronzo/WikiVisage/security/advisories/new) and fill in the details.

Include as much of the following as possible:


- Description of the vulnerability
- Steps to reproduce
- Affected component (`app.py`, `worker.py`, `database.py`, templates, etc.)
- Potential impact
- Suggested fix (if any)

## What to Expect

- **Acknowledgment** within 72 hours.
- **Status update** within 7 days with an assessment and expected timeline.
- **Fix or mitigation** in a timely manner depending on severity.
- Credit in the release notes (unless you prefer to remain anonymous).

If the report is declined, you will receive an explanation.

## Scope

The following are in scope:

- Authentication and OAuth token handling
- CSRF protection bypass
- SQL injection
- Cross-site scripting (XSS) in templates
- Wikibase API abuse via SDC write endpoints
- Sensitive data exposure (tokens, credentials)
- Denial of service against the worker or web process

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (report those to the respective project)
- Issues requiring physical access to the Toolforge infrastructure
- Social engineering
- Rate limiting thresholds (informational, not a vulnerability)

## Security Measures in Place

- OAuth 2.0 with Wikimedia (access + refresh tokens, stored as VARBINARY in DB)
- PKCE (RFC 7636, S256) on the OAuth authorization code flow as defense-in-depth against intercepted authorization codes
- Token encryption at rest: OAuth tokens can be Fernet-encrypted via `WIKIVISAGE_TOKEN_KEY` env var (AES-128-CBC + HMAC-SHA256). Opt-in; legacy plaintext tokens are handled gracefully.
- OAuth refresh single-flight via Redis lock (`redis_lock.single_flight()`, 15s TTL) — prevents concurrent token-refresh requests from invalidating each other's freshly rotated refresh tokens. Best-effort: degrades to a no-op if Redis is unreachable; DB-level CAS on `token_expires_at` remains as a second line of defense.
- CSRF tokens on all POST routes
- Open redirect protection on login
- Parameterized SQL queries (no string interpolation of values)
- Rate limiting on sensitive endpoints (shared via Redis across gunicorn workers, falls back to in-memory)
- Security headers: `Content-Security-Policy` (with `object-src 'none'`, `base-uri 'none'`, `frame-ancestors 'none'`), `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`
- Input validation on bounding box coordinates
- Image download size cap (50 MB) and pixel dimension validation (100 megapixel limit)
- Worker face-detection subprocess hardening: env scrubbed via preexec hook; `RLIMIT_AS=2 GiB`, `RLIMIT_CPU=180s`, `RLIMIT_FSIZE=1 MiB` to contain malformed-image attacks
- Distributed worker locking (`SELECT … FOR UPDATE`) with stale-claim expiry
- Cross-project SDC write deduplication (`sdc_claims` table prevents duplicate P180 claims)
- Idempotent SDC removals: `no-such-entity` / `no-such-claim` / `no-such-statement` / `notfound` are treated as already-gone successes
- Interruptible SDC writes: re-checks `sdc_write_requested` every 5 faces so users can stop a running batch quickly
- Membership-aware access control (project owners and members via `project_members`)
- `maxlag=5` compliance on all Wikimedia API writes
- `/health` endpoint reports `degraded: true` (200 OK) when the rate limiter has fallen back to in-memory storage, signaling that the OAuth refresh single-flight is also a no-op
