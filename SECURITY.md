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
- Access control (whitelist bypass)
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
- CSRF tokens on all POST routes
- Whitelist enforcement on every request
- Open redirect protection on login
- Parameterized SQL queries (no string interpolation of values)
- Rate limiting on sensitive endpoints
- Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`)
- Input validation on bounding box coordinates
- Image download size cap (50 MB) and pixel dimension validation
- Distributed worker locking (`SELECT … FOR UPDATE`) with stale-claim expiry
- `maxlag` compliance on all Wikimedia API writes
