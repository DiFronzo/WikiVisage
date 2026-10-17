# Follow-ups: face model upgrade (Phase 1) and LiftWing pre-filter (Phase 2)

Handoff for the next worker. Written 2026-10-16 after verifying the implementation
of [`face-model-upgrade.md`](face-model-upgrade.md) in a running local stack. Read
that plan's two "Implementation notes" sections first; this file lists what still
needs fixing, finishing or checking, in priority order.

> **Status 2026-10-17: all findings fixed and re-verified.** See
> [Resolution status](#resolution-status-2026-10-17) at the end for what changed,
> the suite numbers and the small set of things that still need a push or a
> production run.

## Ground rules

- **Nothing is committed.** Everything (Phase 1 + Phase 2 + test fixes) is in the
  working tree on `release/0.11.0` (HEAD `a0bc575`). Do not commit unless the owner
  asks. Do not lose the pre-existing staged `AGENTS.md`, `CLAUDE.md`, `README.md`,
  `SECURITY.md`.
- **Never run `migrate.py` against a real (Toolforge) database. No deploys.**
- Keep Phase 2 **shadow-only and fail-open**. Do not build `enforce`.
- After every change: `ruff check . && ruff format --check .`, the unit suite
  (`venv/bin/python -m pytest tests/ -q`) and the integration suite
  (`WIKIVISAGE_TEST_DB=1 venv/bin/python -m pytest tests/ -q`, Docker MariaDB
  `wikivisage-db` on 127.0.0.1:3306, root/devpass). Update `AGENTS.md`, the plan
  notes and the translations (`pybabel extract/update/compile`; nb/es/fr; en is
  identity) whenever behaviour or strings change.

## What was verified (baseline, all green)

| Check | Result |
|---|---|
| Unit suite (venv, Python 3.14) | 940 passed, 63 skipped |
| Integration suite (MariaDB 10.11) | 976 passed, 27 skipped, 0 failed |
| Face-service unit tests (throwaway Python 3.11 venv with `model-server/requirements.txt`) | 144 passed; 2 errors only because `tests/data/portrait_sample.jpg` is missing |
| Contract tests against the running Docker face service | 18 passed, 7 skipped (no sample image, no token) |
| `ruff check` / `ruff format --check` | clean |
| `docker compose up --build` in `model-server/` | builds; weights SHA-256 verified at startup; `model_id=scrfd500m+arcface_mbf, dim=512`; ~780 MiB RSS |
| `migrate.py` on the dev DB (old dlib schema, 17,258 images) | one-time wipe ran, 10 tables verified, users/projects kept |
| Worker against the real face service and Commons | crawl, bootstrap, detect, persist all work; ~1 s/image at 2 download threads, detection ~0.08 s/image |
| LiftWing live (anonymous) | probe `GET /models` 200 in 0.2 s; real verdicts are sensible |
| `prefilter_images` live on project 12 (Q7747) | 120 verdicts stored; 2 dropped (rocket launch site, fireworks), both had 0 detected faces; 106 of 111 kept-and-processed images had faces |
| Banner / settings status / `/health` with `service_status` = up and down | correct on shadow and off projects and the dashboard |

The plan's Phase 1 "Not verified" list is now partly out of date (Docker build,
compose, the wipe on real data and the integration tests have run). Fix that in
**L6**.

## Summary

| ID | Priority | Area | Title |
|---|---|---|---|
| B1 | Blocker | CI | `tests/data/portrait_sample.jpg` is missing, so three CI jobs fail by design |
| B2 | Blocker | CI/deploy | CI, the Toolforge buildpack build and `deploy.yml` have never run |
| H1 | High | Phase 2 | Only the first 250 images of a project ever get a verdict, and the pre-filter delays detection |
| H2 | High | Phase 2 | 40-file batches take ~40 s against a 60 s timeout |
| H3 | High | Phase 2 | The probe closes the circuit breaker and reports "up" while completions may be failing |
| H4 | High | Phase 2 | Anonymous LiftWing access from shared Toolforge IPs; no token support; UA lacks contact |
| M1 | Medium | Phase 2 | The prompt asks for "photographs", but faces in paintings and drawings matter |
| M2 | Medium | Phase 2 | `scripts/eval_prefilter.py`: no `.env`, a dead metric, no pre-label metric |
| M3 | Medium | Face service | Transitive dependencies are unpinned |
| M4 | Medium | Web | A rejected image is reported as "service unavailable, try again" (503) |
| L1–L9 | Low | various | UTC timestamp, env truthiness, unaligned embeddings, per-request query, doc drift, plan notes, translations, title normalisation, enforce-time injection risk |

---

## Blockers

### B1. Commit a real portrait sample for the face-service tests

**Problem.** `tests/data/portrait_sample.jpg` does not exist. CI sets
`WIKIVISAGE_REQUIRE_SAMPLE=1` in `face-service-unit`, `face-service` and
`face-service-buildpack` (`.github/workflows/ci.yml` lines 93, 164, 248), so all
three fail until it is added. Locally these tests skip (or error, see L2).

**Fix.** Follow `tests/data/README.md`: a public-domain or freely licensed
Commons photo (an official government portrait is the easiest), downloaded as the
960 px thumbnail (`Special:FilePath/<file>?width=960`), exactly one clear,
roughly frontal face. Fill in the provenance table in the README. Prefer a
portrait whose face is large (over 430 px tall): it also exercises the half-scale
detection pass that exists because SCRFD-500M misses very large faces.

**Done when.** In a Python 3.11 env with `model-server/requirements.txt` +
`requirements-dev.txt`:
`WIKIVISAGE_REQUIRE_SAMPLE=1 pytest tests/test_face_service_backend.py tests/test_face_service_model.py`
passes, and the contract tests pass against `docker compose up` with
`WIKIVISAGE_FACE_SERVICE_CONTRACT=1 WIKIVISAGE_REQUIRE_SAMPLE=1`.

### B2. Run CI and the buildpack build once

**Problem.** None of the five CI jobs has run on this branch. The
`face-service-buildpack` job (Toolforge's `heroku-builder:24_0.21.8`,
`heroku/python` + `heroku/procfile`, no `project.toml`) is the production build
path and has never been exercised. `pack` is not installed on this machine.
`deploy.yml` (face-service tag via `git commit-tree HEAD:model-server`,
readiness wait, preflight) is also untested.

**Fix.** Only with the owner's go-ahead: push the branch (or open a draft PR) and
make all five jobs green; fix whatever breaks. Optionally install `pack` and
reproduce the buildpack job locally (the commands are in `ci.yml`). Do not run
`deploy.yml`.

---

## High

### H1. Pre-filter coverage: only the first 250 images per project, and it delays detection

**Where.** `worker.py` `prefilter_images` (around line 1427) and its call site in
`process_project` (line 2820, just before `# 3. Image Download`).

**Problem.** `prefilter_images` selects
`status = 'pending' AND bootstrapped = 0 AND prefilter_checked_at IS NULL LIMIT 250`.
Right after it, the `while not shutdown_requested: process_images(project)` loop
drains **every** pending image of the project in the same cycle. The next cycle
finds no pending images, so a project with 9,000 images gets verdicts for its
first 250 (lowest ids) and never any more. The shadow readout is then a small,
biased sample, and keep recall on human-labelled images (the go/no-go metric)
will rarely have data. Second cost: the pre-filter runs **before** detection and
takes ~1 s per image (see H2), so a new project waits up to ~4 minutes per cycle
before detection starts — for an experiment that changes nothing.

**Fix (shadow mode).** Verdicts are metadata-only, so in shadow mode the image
status does not matter:
1. Drop `status = 'pending'` from the selection in shadow mode (keep
   `bootstrapped = 0 AND prefilter_checked_at IS NULL`); keep the cap per cycle so
   a large backlog is worked off over several cycles.
2. Move the call to **after** the image loop (or after inference) so it never
   delays detection.
3. Leave a comment that a future `enforce` mode must select pending images and
   run before detection — that is the only mode where order matters.

**Done when.** Unit tests cover: processed images are selected in shadow mode; the
cap is applied per call; the call happens after `process_images`. A backlog of
more than 250 images is fully covered after enough cycles (integration test or a
manual run on the dev DB). Update `AGENTS.md` (Metadata pre-filter section,
worker table, pipeline order) and the plan's Phase 2 notes.

### H2. Batch latency is close to the request timeout

**Where.** `worker.py` `LIFTWING_BATCH_SIZE` (line 92, default 40);
`liftwing_client.py` `REQUEST_TIMEOUT` (default 60 s).

**Evidence.** Live run: 120 images in 3 batches took 119.8 s, about **40 s per
40-file batch** (one verdict and a reason per file, `temperature 0`). The
docstring assumes p50 3.6 s / p95 ~18 s, which was measured on small prompts.
Under load a 40-file batch will exceed 60 s; every timeout fails the batch open,
and three in a row open the circuit for 10 minutes, so the experiment would
quietly record nothing.

**Fix.** Measure first: log per-batch wall time and, if the response has it,
`usage.completion_tokens`. Then lower the default batch size (20 is a reasonable
start) and/or raise the timeout (e.g. 120 s), keeping
`batch time × batches ≤` a sensible per-cycle budget. Consider a per-cycle time
budget instead of only an image cap. Also check whether Qwen emits a `<think>`
block here: if it does, it eats the 2000-token budget (`MAX_TOKENS`) and
truncated JSON fails open.

**Done when.** Ten consecutive live batches complete with p95 comfortably under
the timeout; the chosen defaults and the measurement are recorded in the plan's
Phase 2 notes and `AGENTS.md`.

### H3. The probe closes the circuit breaker and can report "up" while completions fail

**Where.** `liftwing_client.py` `probe()` (line 320) calls `self._record_success()`
(lines 339, 371); the circuit breaker state is shared with `filter_metadata`.

**Problem.** The probe's normal path is `GET <base>/models`, which is cheap (0.2 s)
but does not run inference. If completions fail (429, 503, timeouts — see H2),
`filter_metadata` opens the breaker; within ≤ 5 minutes the worker's probe gets a
200 from `/models`, calls `_record_success()` and closes the breaker again, so the
cooldown is effectively ignored. At the same time `service_status` says `up`, so
the settings page shows "reachable" and the banner stays hidden while every real
request fails.

**Fix.** Keep the two signals separate:
- The probe must not touch the `filter_metadata` breaker (own counter, or none —
  the database-level hysteresis already exists).
- Feed real outcomes into the status the user sees: e.g. have `prefilter_images`
  record a failure (via the same `service_status` hysteresis) when
  `filter_metadata` failed for a transport/HTTP reason, and a success when it
  returned verdicts; or make the probe a minimal completion (`max_tokens` 1–8)
  instead of `/models`. Choose one and document it.

**Done when.** A unit test shows that a successful probe does not reset an open
filter breaker, and that repeated completion failures eventually surface as
`down` even while `/models` answers 200.

### H4. Anonymous access, rate limits and the User-Agent

**Where.** `liftwing_client.py` `_headers()` (line 239), `USER_AGENT`;
`worker.py` line 55 `USER_AGENT`.

**Problem.** LiftWing is called anonymously. The live response carried
`x-wmf-ratelimit-class: anon-browser`. On Toolforge, outbound traffic from all
tools shares a small set of egress IPs, so an anonymous per-IP quota is shared
with every other tool and 429s are likely in production. There is no way to
configure a token, and 429 is treated like any other failure (no `Retry-After`).
The UA `WikiVisage/1.0 (Wikimedia Toolforge; https://toolsadmin.wikimedia.org)`
has no real contact point (the web app's UA in `app.py` line 225 at least links
the GitHub repo); Wikimedia's User-Agent policy asks for contact information.

**Fix.**
1. Check the current Wikimedia API Gateway / LiftWing documentation for how
   authenticated calls work (personal API token / OAuth 2 client credentials,
   `Authorization: Bearer …`) and the documented rate limits — do not guess.
2. Add an optional `WIKIVISAGE_LIFTWING_API_TOKEN` (send the header only when set;
   never log it). Document it in `AGENTS.md` and `how-to-run-it.md` (set with
   `toolforge envvars`).
3. On 429, respect `Retry-After` (at least as the circuit-breaker cooldown) and
   log it distinctly.
4. Use a UA with a contact: tool URL and/or maintainer contact, and the version
   from `pyproject.toml`; use the same UA in the worker.

**Done when.** Unit tests for the auth header (present only when configured) and
for 429 handling; a live call with the token succeeds.

---

## Medium

### M1. The prompt targets photographs, but WikiVisage classifies faces in any medium

**Where.** `liftwing_client.py` `_SYSTEM_PROMPT` (line 68).

**Problem.** The prompt asks whether a file "could plausibly contain a photograph
of a person" and lists "drawings" among things to drop. The face detector also
finds faces in paintings, engravings, drawings, sculptures and banknotes, and for
historical subjects those are often the majority of a category. In shadow mode
this would show up as poor keep recall for such projects; in a future enforce mode
it would remove real targets.

**Fix.** Ask whether the file could **depict a human face** in any medium
(photo, painting, drawing, engraving, sculpture, coin…), keep whenever unsure, and
drop only things that clearly contain no face (maps, diagrams, charts, logos,
coats of arms, text scans, audio). Keep the strict-JSON instruction. Adjust any
tests that pin the prompt text.

### M2. `scripts/eval_prefilter.py` needs fixes before it can drive a decision

**Where.** `scripts/eval_prefilter.py` (`init_db` at line 80; the
"bootstrap target" line at 111).

**Problems.**
1. It does not call `load_dotenv()`, so the command documented in `AGENTS.md`
   fails locally with `ConfigurationError: Missing required environment variable:
   TOOL_TOOLSDB_USER` (`app.py` and `worker.py` load `.env`; Toolforge sets the
   variables, so it only bites locally).
2. The "bootstrap target" recall row can **never** have data: bootstrapped images
   bypass the pre-filter by design, so no checked image is ever bootstrapped.
3. Every recall row needs human or model labels, so right after a crawl the
   readout says nothing. A useful pre-label signal already exists: the detector's
   own result. Add "dropped images with ≥ 1 detected face" (and the number of faces
   on them) and "kept images with no face" over images with `status = 'processed'`.
   On the live sample this showed 0 of 2 dropped images with faces, and 106 of 111
   kept images with faces.

**Done when.** The script works with only `.env`, the dead row is removed (or
replaced by a note), the face-based metrics are printed, and there is a unit
test for the query/aggregation (the script has none).

### M3. Pin the face service's transitive dependencies

**Where.** `model-server/requirements.txt` (only six top-level pins).

**Problem.** `kserve==0.15.2` pulls FastAPI, Starlette, uvicorn, pydantic,
kubernetes, grpcio, pandas and more without pins, so every build can resolve
differently. Today's build got FastAPI 0.143, which already logs
`FastAPIDeprecationWarning: ORJSONResponse is deprecated` from inside KServe; a
later FastAPI could break KServe outright, and the production image would differ
from what CI tested.

**Fix.** Generate a fully pinned file for Linux / Python 3.11, e.g. keep the
top-level pins in `model-server/requirements.in` and write
`uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 requirements.in -o requirements.txt`
(the Toolforge buildpack only reads `requirements.txt`). Rebuild the Docker image
and rerun the face-service unit and contract tests. Note the regeneration command
in `TESTING.md`.

### M4. A rejected image is reported as "service unavailable"

**Where.** `app.py` `api_manual_face` (line 2538) and `api_update_face_bbox`
(line 3087): `except FaceServiceError:` → 503 "Face detection service is
unavailable. Please try again in a moment."

**Problem.** `FaceServiceRejected` (this image cannot be decoded or is too large —
permanent) is a subclass of `FaceServiceError`, so the user is told to retry
something that will never work.

**Fix.** Catch `FaceServiceRejected` first and return 422 with a message such as
"This image could not be read by the face detection service." (new translatable
string — translate nb/es/fr, identity in en). Keep 503 for
`FaceServiceUnavailable`. Add tests for both routes.

---

## Low

- **L1. Timestamp shown in UTC without a label.** `templates/project_settings.html`
  line 54 prints `liftwing_last_checked` from the database (`NOW()` is UTC), so a
  user in CEST sees 21:57 at 23:57. Append "UTC" or show a relative time ("3 min
  ago").
- **L2. `WIKIVISAGE_REQUIRE_SAMPLE=0` counts as set.** `tests/test_face_service_backend.py`
  line 902 and `tests/test_face_service_contract.py` line 74 test truthiness of the
  string. Accept only `1/true/yes`.
- **L3. Unaligned fallback embeddings are indistinguishable in the DB.** A drawn box
  without a recoverable landmark set is embedded unaligned (`aligned: false`,
  "noticeably weaker"), stored with the same `embedding_model`, and a human-confirmed
  one enters the centroid like any other. Measure how often it happens (the app
  already logs it); if it is not rare, add a `faces.aligned` column or exclude
  unaligned faces from the centroid.
- **L4. One extra query per page render.** `inject_liftwing_status` runs on every
  template render, though the banner and status only appear on
  `project_detail` / `project_settings`. Return defaults early for other endpoints.
- **L5. Pre-existing doc drift in `AGENTS.md`.** `BATCH_SIZE` is 50 in code (table
  says 10, line 486) and `WIKIVISAGE_WORKER_BATCH_SIZE` is undocumented;
  `WIKIVISAGE_WORKER_MAX_PROJECTS` defaults to 4 in code (env section says 3,
  line 565); the pool-size sentence uses 3 × 4; the `~N lines` counts for
  `app.py`/`worker.py` are stale.
- **L6. Refresh the plan's "Not verified" lists** in `face-model-upgrade.md` with
  what is now verified (see the baseline table above) and what still is not (B1,
  B2, H2, live production rate limits).
- **L7. One pre-existing untranslated string** in nb/es/fr: "This face was already
  classified by %(user)s. Only they can change their own classification."
- **L8. `_fetch_metadata` ignores the API's `normalized` / redirect mapping.**
  Harmless today (stored titles come from the API and are already normalised; 0
  of 405 dev titles contain `_`), but a title that differs from the canonical form
  would silently get empty metadata. Map through `query.normalized`.
- **L9. Before anyone builds `enforce`** (not now): file titles and descriptions
  are user-controlled text inside a batch prompt, so one file's description can
  try to steer the verdicts of other files in the same batch. Enforce mode would
  need smaller batches or per-file isolation, and must never drop bootstrapped or
  human-labelled images. Record this in the plan's Phase 2 section.

---

## Open work from the plan (not bugs)

- **Phase 0 calibration:** `DEFAULT_DISTANCE_THRESHOLD = 0.5` is provisional;
  calibrate on re-labelled faces (plan sections 1b and 4.4).
- **`faces.det_score` is stored but unused** (no sorting or hiding of tiny/low-score
  faces; 43% of faces at 960 px are under 20 px per the plan's measurements).
- **Phase 3 rollout:** production DB dump before the first `migrate`; the restore
  steps already exist in `how-to-run-it.md` ("Upgrading the embedding model …
  and rollback").
- **Phase 2 readout:** run shadow mode on a few real categories after H1–H4 and M1–M2,
  then decide with `scripts/eval_prefilter.py` (go/no-go threshold in `AGENTS.md`:
  keep recall ≥ 99.5% on human-confirmed target images).

## Local stack (as left running)

- Web app `venv/bin/python app.py` on http://localhost:3001 (`.env`, debug reload).
- Worker `venv/bin/python worker.py --worker-id local-1` (no hot reload — restart
  after changing `worker.py`, `liftwing_client.py` or `face_client.py`).
- Face service: `cd model-server && docker compose up --build -d` (container
  `wikivisage-face-service`, port 8080, no token). Rebuild after any
  `model-server/` change.
- MariaDB: container `wikivisage-db`, database `wikiface_dev` (already migrated;
  the old images were wiped). Projects 7 and 21 were set to
  `prefilter_mode = 'shadow'` for this review, and project 12 has 120 shadow
  verdicts from a manual `prefilter_images` run.
- Face-service unit tests need Python 3.11:
  `uv venv -p python3.11 <dir> && uv pip install -p <dir>/bin/python -r model-server/requirements.txt -r requirements-dev.txt`.

---

## Resolution status (2026-10-17)

Every finding was fixed in the same working tree and the suites were re-run.
Exact numbers:

| Check | Result |
|---|---|
| Unit suite (venv, Python 3.14) | 958 passed, 63 skipped (was 940/63; +18 tests) |
| Integration suite (Docker MariaDB 10.11) | 994 passed, 27 skipped, 0 failed (was 976/27) |
| Face-service unit tests (Python 3.11, sample committed, `WIKIVISAGE_REQUIRE_SAMPLE=1`) | 116 passed + 30 guard tests (was 144 with 2 sample errors) |
| Contract tests vs the Docker image | 21 passed, 4 skipped (no token) |
| Contract tests vs a locally built Toolforge buildpack image, with token + sample | 25 passed; 0 `VerifyOutputSizes` warnings |
| `ruff check .` / `ruff format --check .` | clean |

Per finding:

- **B1 — done.** `tests/data/portrait_sample.jpg` is in the working tree: "Joe Biden
  presidential portrait (3x4 close cropped)" (Adam Schultz / White House, public
  domain), 960x1268, 244 KB, downloaded via `Special:FilePath/...?width=960`.
  Exactly one face (514x756 px, score 0.80), so the half-scale pass is exercised.
  Provenance table in `tests/data/README.md`. (Not a git commit — the owner decides
  when to commit.)
- **B2 — verified locally as far as possible.** `pack` built the production image
  from `model-server/` with the Toolforge builder (`tools-harbor.wmcloud.org/toolforge/heroku-builder:24_0.21.8`,
  `heroku/python` + `heroku/procfile`) using the fully pinned lock; the image was
  started through its `web` Procfile entry with `TOOL_TOOLFORGE_API_URL` set and a
  token, reported ready in 4 s, and passed the contract suite through the guard
  (25 passed) with no ORT warnings. CI and `deploy.yml` still have to run on push.
- **H1 — done.** `prefilter_images` moved after the image loop; selection is now
  `bootstrapped = 0 AND prefilter_checked_at IS NULL` regardless of `status`.
  Ordering is pinned by `test_process_project_runs_prefilter_after_image_processing`;
  the selection SQL is asserted in `test_prefilter_records_verdicts_and_keeps_missing_pages`.
  An `enforce` note is in the code.
- **H2 — done.** `LIFTWING_BATCH_SIZE` default 20, `REQUEST_TIMEOUT` default 90 s.
  Measured live: 10 batches of 20, min 0.4 s (a transient 503), median 20.0 s,
  max 21.8 s. Per-batch wall time logged. One 503 was observed and failed open
  correctly.
- **H3 — done.** Probe is a minimal real completion (`PROBE_MAX_TOKENS = 8`);
  `GET /models` fallback removed. Real batch outcomes call
  `_record_liftwing_failure` / `_record_liftwing_success`, so the banner tracks
  actual completions. 429 opens the client breaker immediately and `Retry-After`
  (seconds or HTTP-date) extends the cooldown.
- **H4 — done.** `WIKIVISAGE_LIFTWING_API_TOKEN` sends `Authorization: Bearer`;
  UA is `WikiVisage/<version> (https://github.com/DiFronzo/WikiVisage)`
  (`config.PROJECT_VERSION`); `how-to-run-it.md` documents the token. Live
  header observed earlier: `x-wmf-ratelimit-class: anon-browser`.
- **M1 — done.** Prompt keeps paintings/drawings/sculptures/coins and other
  person-bearing media; uncertainty still means keep; pinned by a test.
- **M2 — done.** `scripts/eval_prefilter.py` loads `.env`, drops the dead
  bootstrap-target row, adds the detector signal, and is unit-tested
  (`tests/test_eval_prefilter.py`). Live run on project 12: 120 checked, 118 keep,
  2 drops (both 0 faces), 5 keeps with no detected face.
- **M3 — done.** `model-server/requirements.in` (top-level pins) +
  fully pinned `requirements.txt` (`uv pip compile --python-version 3.11
  --python-platform x86_64-manylinux_2_28`). Verified in the Docker image and
  the buildpack image. Regeneration documented in `TESTING.md` §7.
- **M4 — done.** Both routes return 422 for `FaceServiceRejected` with a new
  translated string; 503 remains for `FaceServiceUnavailable`. Tests added for
  both routes.
- **L1 — done.** Settings shows `... HH:MM UTC`.
- **L2 — done.** `conftest.require_sample_env()` accepts only
  `1/true/yes/on`; unit test pins `"0"`/`"false"`/unset as "not required".
- **L3 — done (measurement enabler).** `faces.aligned TINYINT(1) NOT NULL DEFAULT 1`
  in `schema.sql`, an ALTER migration (applied to the dev DB), and the three
  manual/bbox INSERTs record the flag. No centroid exclusion yet — the column
  exists to measure how often the fallback happens.
- **L4 — done.** `inject_liftwing_status` returns defaults without a query on
  endpoints other than `project_detail`/`project_settings`; two tests added.
- **L5 — done.** AGENTS.md drift fixed (BATCH_SIZE 50 + env var, worker pool
  formula and 20-connection cap, `WIKIVISAGE_WORKER_MAX_PROJECTS` 4, line counts,
  LiftWing defaults).
- **L6 — done.** Both "Not verified" lists in `face-model-upgrade.md` were
  refreshed and a follow-up implementation-notes section was appended.
- **L7 — done.** The `%(user)s` string is translated in nb/es/fr; pybabel
  extract/update/compile re-run; no empty or fuzzy entries remain in any locale.
- **L8 — done.** `_fetch_metadata` maps non-canonical titles through
  `query.normalized`; test with a rewritten title added.
- **L9 — recorded** in the plan's follow-up section and in the code comment:
  `enforce` must use smaller batches or per-file isolation (prompt injection via
  file descriptions), select pending images before detection, never drop
  bootstrapped/human-labelled images, and treat a drop as cost optimisation,
  never authorization.

Live-stack notes: the face service was rebuilt with the pinned lock; the dev DB
was migrated in place (`faces.aligned` applied, no wipe); project 12 still has
the 120 shadow verdicts from the baseline run (the H2 measurement called the
client directly and stored nothing). The worker was restarted with the new code
and a live shadow run against project 7 verified the new path: 2522 unchecked
images before, exactly the 250-per-cycle cap stored in one call (475 keep / 25
drop overall), 13 batches at 18.3-21.4 s each (first batch 44 s, cold metadata
path), and `service_status.last_ok` refreshed by `_record_liftwing_success`.

### Follow-up fix: "Model inference in progress" banner stuck (same day)

Owner report: a project that was ready for model inference (project 22: 8 human
confirmations = its `min_confirmed`, 88 unclassified faces, all with valid
current-model encodings) sat on "Model inference in progress — 88 faces
remaining…" indefinitely.

Root cause: the worker's main loop claimed inference-eligible projects each
cycle but ran them only **after** the full active batch, and a single active
project can hold all slots for a long time looping its image backlog (project 1
processed 1015 images in one run). The wake handler only claimed *active*
projects, so a classification crossing `min_confirmed` never triggered
inference either.

Fix (all in `worker.py`):

- The inference-only queue is drained **before** the active batch, in waves of
  `MAX_CONCURRENT_PROJECTS`, via a `while` loop with a `seen_ids` filter (a
  project that stays eligible because of unusable encodings is attempted once
  per cycle; claims we do not run are released immediately).
- `_run_new_inference_projects()` claims and runs eligible projects inline; it
  is called from both wake handlers and both 60 s heartbeat checks, so a gate
  crossing is picked up mid-cycle without a free executor slot.
- The old post-active inference block was removed.

Live verification (dev DB, worker restarted with the fix):

- Project 22's 88 faces were classified 66 s after the worker start, before any
  image processing began; projects 7 and 21 had already been drained in the
  same pass (7 took 0.31 s, 21 took 0.06 s for 2634 / 509 faces).
- Mid-cycle pickup: project 15's `min_confirmed` was temporarily set from 40 to
  15 at 01:07:50 (dev DB only, reverted right after). The 60 s heartbeat claimed
  and classified its 203 faces at 01:08:21 — 31 s later, while projects 1 and 7
  were mid-batch. `unclassified` went to 0 and the claim was released.

`AGENTS.md` (worker "Scheduling" note) and the `_run_new_inference_projects`
docstring carry the same information.

### LiftWing prompt v2 + deterministic media filtering (same day)

Five review-driven changes to `liftwing_client.py` (plus one small `worker.py`
change), all inside the shadow-only, fail-open rules:

- **Reason only on drops.** The prompt now asks for `"reason"` (at most four
  words) on `keep=false` entries only; keeps cost one boolean each. Dropped
  files keep their reason — that is exactly what `eval_prefilter.py --reasons`
  needs to diagnose recall misses. (The previous prompt asked for a reason per
  file; 20-file batches measured ~20 s.)
- **Biography-category negatives.** The prompt now names the noise that
  dominates Commons biography categories: signatures, autographs, handwritten
  letters, text scans, certificates, gravestones/tombs without portraits,
  memorial plaques, buildings, residences, streets, landscapes, maps, diagrams,
  charts, coats of arms, flags, logos. Busts, sculptures and engravings stay
  listed as positives.
- **Deterministic MIME filter.** `filter_metadata` settles files whose MIME
  type cannot depict a face (`audio/*`, `video/*`, `application/pdf`,
  `application/ogg`, `image/svg+xml`, `image/vnd.djvu`) as `keep=false` with
  reason `non-visual media (<type>)` and never sends them; a batch composed
  only of such files makes no HTTP call. Unknown or missing MIME types still go
  to the model — the review suggested an allow-list, but a deny-list plus
  fail-open keeps the "when uncertain, keep" rule intact. Settled drops land on
  the image rows like any other verdict, so the readout sees a dropped SVG that
  had a detected face as dropped-with-faces.
- **Input hygiene.** Descriptions are stripped of HTML and `{{…}}` wikitext
  templates and truncated to 250 chars (`MAX_DESCRIPTION_CHARS`, down from
  500); categories are capped at 15 with the `Category:` prefix stripped; the
  MIME field is gone from the JSON payload.
- **Fences.** The prompt explicitly forbids markdown code fences. The parser
  already stripped them (`_CODE_FENCE_RE`, with tests) — that part of the
  review was covered, so no fallback code was added.

Deferred: the review's optional "priority" mode (`action:
priority|keep|skip`, target-name context, priority files processed first)
changes the verdict schema, needs a migration and reorders the processing
queue — i.e. enforce-mode plumbing the project deliberately does not have yet.
It is the natural next step *after* the readout gate passes.

Comparability: verdicts stored before this change were produced by prompt v1
(with per-file reasons). For a clean readout of prompt v2, clear
`prefilter_keep` / `prefilter_reason` / `prefilter_checked_at` on the affected
projects and let the worker re-check them (dev DB only — never against
production).

New tests: `test_filter_metadata_settles_non_visual_media_without_a_model_call`,
`test_filter_metadata_non_visual_batch_makes_no_request`,
`test_filter_metadata_unknown_mime_still_goes_to_the_model`,
`test_build_messages_truncates_descriptions_and_strips_category_prefix`,
`test_system_prompt_names_the_biography_noise_and_drop_reasons` (client) and
`test_plain_text_strips_wikitext_and_truncates` (worker). `AGENTS.md`'s
pre-filter section and constants table carry the same information.
