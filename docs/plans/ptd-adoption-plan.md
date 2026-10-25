# WikiVisage → Toolforge Push-to-Deploy: migration & adoption plan

**Status:** Draft for review · **Created:** 2026-10-17 · **Owner:** WikiVisage maintainer

**Related:** [Phabricator project 8755: Push-to-Deploy](https://phabricator.wikimedia.org/project/view/8755/) · beta feedback template uses parent task [T393564](https://phabricator.wikimedia.org/T393564)

---

## Scope

Two layers:

- **Part A: WikiVisage migration (the pilot).** Pilot Toolforge Push-to-Deploy (PTD, the Toolforge Components Service) on both tools in the lab. The first production cutover moves only `wikivisage-face`; `wikivisage` keeps its current release-triggered SSH deployment until the lab proves equivalent worker lifecycle gates.
- **Part B: parent initiative workstreams.** Candidate tool / maintainer inventories, outreach, tutorials and migration guides, case studies, community feedback, and adoption metrics.

**Research basis:** [Help:Toolforge/Deploy your tool](https://wikitech.wikimedia.org/wiki/Help:Toolforge/Deploy_your_tool) (beta documentation) and [its discussion page](https://wikitech.wikimedia.org/wiki/Help_talk:Toolforge/Deploy_your_tool), [sample-complex-app-backend](https://gitlab.wikimedia.org/toolforge-repos/sample-complex-app-backend), the [tool-config JSON schema](https://gitlab.wikimedia.org/repos/cloud/toolforge/components-api/-/raw/main/openapi/tool-config-schema.json), the [Toolforge changelog](https://wikitech.wikimedia.org/wiki/Portal:Toolforge/Changelog), and this repo's `deploy.yml`, `healthcheck.sh`, `Procfile`, and `model-server/Procfile`.

**Caveat:** PTD is in open beta (announced 2025-06-30); the docs currently state it is not yet recommended for production services. This plan treats it accordingly: pilot first, decision gate before the production cutover.

---

## 1. What PTD is today (and what it cannot do yet)

PTD is the Toolforge **Components Service**: a declarative `toolforge.yaml` in the repository describes the tool's *components* (continuous jobs, scheduled jobs, and, since August 2026, a web component published to the tool URL). Builds and runs are created/replaced by *deployments*, triggered manually via the CLI or from any CI system via a per-tool deploy token.

| Capability | Status | WikiVisage use |
|---|---|---|
| Multiple continuous + scheduled components per tool | ✅ | web app, 2 workers |
| `reuse_from` (components share one build) | ✅ | both workers reuse the web build |
| `cpu`, `memory`, `mount: all\|none`, `replicas` | ✅ | workers 2Gi/2CPU; face tool 3Gi/2CPU/`mount=none` |
| `health_check_http` and `health_check_script` | ✅ | `/health`; `./healthcheck.sh ml-worker-1` |
| Publish a continuous component to `https://<tool>.toolforge.org/` | ✅ (Aug 2026) | web app and face service |
| `source_url`: config fetched from the git repo on every deploy | ✅ | self-referencing `toolforge.yaml` |
| Deploy token + deployment-status polling for CI | ✅ | GitHub Actions triggers and waits |
| Cancel a deployment; `--force-build` / `--force-run` | ✅ | runbook |
| Deployment ordering, rollback, deploy queueing, per-component deploys | ❌ | ordering emulated in CI; rollback is git-based |
| One-off components (e.g. `migrate.py`) | ❌ | keep the existing SSH `migrate` step |
| Classic "webservice" components | ❌ | use `publish: /` on a continuous component instead |

Also relevant from the changelog: procfile process arguments work out of the box (2026-04), `toolforge logs` exists (2026-10), deployment events in logs are planned.

**GitHub CI is confirmed to work.** On the guide's discussion page (2025-07-01), BryanDavis (WMCS) confirmed that GitHub CI uses the same webhook trigger and explicitly asked for a GitHub Action example to be contributed to the page: *"It would be great to see a GitHub Action example added to the page by someone who takes the time to figure that out."* This plan's CI workflow therefore doubles as a welcome upstream documentation contribution (see B3 and Stage 5).

The runtime components are expressible in PTD, but their deployment sequence needs four checks:

1. **Migrations:** no one-off components, so `migrate.py` stays an explicit step.
2. **Ordering:** no ordering within a deployment, so `face → migrate → main` sequencing must happen in CI across the two tools' deployments.
3. **Rollback:** no rollback support; it becomes git revert + redeploy.
4. **Worker lifecycle:** existing workers must stop before migration, and new workers must wait for the web app's authenticated face-service preflight. A single deployment that starts web and workers together cannot preserve those gates.

---

## 2. Current WikiVisage deployment (baseline)

- `.github/workflows/deploy.yml` (371 lines): release-published trigger → SSH to the bastion as two tool accounts:
  1. Publishes the `face-service-<tag>` subtree tag (`git commit-tree HEAD:model-server`) so the face tool can build from `model-server/`.
  2. Checks envvars on both tools (face token present, ≥ 32 chars, identical on both).
  3. Builds + restarts the face tool, waits for `/v1/models/wikivisage` to report `ready: true`.
  4. Builds the main tool; stops both workers; runs `toolforge jobs run migrate --wait`; restarts the webservice.
  5. Preflight: polls the main `/health` until `face_service: reachable` (a wrong/missing token fails here).
  6. Starts both workers; writes a SAL message.
- `deploy.yml`: two continuous workers (`ml-worker`, `ml-worker-2`), 2Gi / 2 CPU, `--health-check-script './healthcheck.sh <id>'`.
- Web: `toolforge webservice buildservice` + gunicorn (Procfile `web:`); worker process type in the Procfile (`worker: python -u worker.py`).
- Rollback: rerun `toolforge build start --ref <previous tag>` and restart roles.

This is precisely the imperative surface PTD replaces with declarative config plus CI-triggered deployments.

---

## 3. Target architecture

Two lab `toolforge.yaml` files plus a new CI workflow. The two tool accounts stay separate (the face tool holds none of the main tool's secrets). Initially only the face tool uses PTD in production; the main tool retains its existing deploy script, including worker shutdown, migration, web restart, authenticated preflight, and worker startup.

### 3.1 `toolforge.yaml` (repo root, tool `wikivisage`)

**Scratch-tool configuration only.** The example below starts web and workers together. Do not apply it to the production main tool until gate G2 (§4) passes. An HTTP health check on web does not hold the worker components back.

```yaml
# yaml-language-server: $schema=https://gitlab.wikimedia.org/repos/cloud/toolforge/components-api/-/raw/main/openapi/tool-config-schema.json
config_version: v1beta1
source_url: https://raw.githubusercontent.com/DiFronzo/WikiVisage/main/toolforge.yaml
components:
  web:
    component_type: continuous
    build:
      repository: https://github.com/DiFronzo/WikiVisage
      ref: main
    run:
      command: web
      port: 8000
      publish: /
      health_check_http: /health
      cpu: "2"
      memory: 2Gi
  worker-1:
    component_type: continuous
    build:
      reuse_from: web
    run:
      command: worker --worker-id ml-worker-1
      health_check_script: ./healthcheck.sh ml-worker-1
      cpu: "2"
      memory: 2Gi
  worker-2:
    component_type: continuous
    build:
      reuse_from: web
    run:
      command: worker --worker-id ml-worker-2
      health_check_script: ./healthcheck.sh ml-worker-2
      cpu: "2"
      memory: 2Gi
```

Exact `cpu` / `memory` / `mount` values are copied from today's effective settings before the cutover (`toolforge jobs dump`, `toolforge jobs show`, webservice settings).

### 3.2 `model-server/toolforge.yaml` (tool `wikivisage-face`)

The buildservice builds from a repository **root**, but the face service lives in `model-server/`. Keep the existing subtree trick, but as a **moving branch instead of per-release tags**:

- CI force-updates branch `face-service` to `HEAD:model-server` (the branch root is the face service).
- The face tool's config builds `ref: face-service` and fetches its own config from that branch, so config and code travel atomically:

```yaml
# yaml-language-server: $schema=https://gitlab.wikimedia.org/repos/cloud/toolforge/components-api/-/raw/main/openapi/tool-config-schema.json
config_version: v1beta1
source_url: https://raw.githubusercontent.com/DiFronzo/WikiVisage/face-service/toolforge.yaml
components:
  face-service:
    component_type: continuous
    build:
      repository: https://github.com/DiFronzo/WikiVisage
      ref: face-service
    run:
      command: web
      port: 8000
      publish: /
      health_check_http: /v1/models/wikivisage
      cpu: "2"
      memory: 3Gi
      mount: none
```

### 3.3 New `.github/workflows/ptd-deploy.yml`

Starts as `workflow_dispatch`-only against the lab tools. For the first production cutover, the workflow uses this sequence:

1. Publish/update the `face-service` branch (subtree commit).
2. `curl --fail-with-body -X POST "https://api.svc.toolforge.org/components/v1/tool/${FACE_TOOL}/deployment?token=${TOKEN}"`, then poll `.../deployment/<id>` until terminal.
3. Poll `https://<face-tool>.toolforge.org/v1/models/wikivisage` until `ready: true`.
4. Invoke the main-tool deploy script with the selected release tag, retaining its current ordering. Before this PTD cutover, add an explicit process-exit check after stopping **both** worker jobs and before migration. If shutdown cannot be confirmed, abort without migrating.
5. The same script runs `migrate`, restarts web, and polls `/health` until `face_service: reachable`. This checks the main tool's URL and bearer token, rather than only public face-service readiness. On migration or preflight failure, leave both workers stopped and fail the deployment.
6. Only after preflight succeeds, start both workers from the release image, verify their health, and write the SAL message.

Do not trigger the all-components main-tool PTD deployment in this production sequence. Gate G2 must prove both shutdown during migration and preflight-before-worker-start before that path can replace the script. Keep the existing input validation, reset confirmation, environment checks, and main-tool worker gates in the shared workflow.

Deploy tokens (`toolforge components deploy-token create` per tool) are stored as masked secrets in the existing `toolforge` GitHub environment. Keep `concurrency: deploy-production` (PTD cannot queue deployments).

### 3.4 Preserved safety properties

| Current property | How it survives the migration |
|---|---|
| Face service deployed and healthy **before** the main tool | CI waits for the face PTD deployment and public readiness before invoking the existing main-tool deploy script. |
| Workers stay stopped during migration | Keep the main-tool script's stop-before-migration order and add process-exit confirmation before the PTD cutover; a failed migration leaves workers stopped. |
| Workers start **after** authenticated preflight | The existing main-tool script restarts web and requires `/health` to report `face_service: reachable` before starting either worker. A failed preflight leaves them stopped. |
| Bad face deployment cannot look green | CI checks public face readiness, and the main-tool preflight checks the URL and bearer token before worker startup. |
| Worker liveness auto-restart | `health_check_script` parity (same `healthcheck.sh`, same 5-minute age check). |
| Secrets stay off the face tool | Unchanged (tool-level envvars, separate tool accounts). |
| Rollback | Git revert on `main` → workflow redeploys the previous code. A documented "restore legacy control plane" procedure stays available until confidence. |

### 3.5 Cadence

Recommended: **release-triggered PTD first** (workflow on release, with a `tag == main tip` guard). Release cadence fits WikiVisage's migration and DB discipline at first; flipping the trigger to every push to `main` is a one-line change once the machinery is trusted. PTD is the mechanism; cadence is a deliberate policy choice.

---

## 4. Staged plan

### Stage 0: Hands-on PTD lab (the "review and evaluate" objective)

Timebox ~1 day. On a scratch tool, walk the full PTD flow with the sample repo and a small home-made config:

- `toolforge components config generate` (it can import existing jobs) / `config create` / `config show`.
- `deployment create` / `list` / `show` (including the config shown per deployment), `--force-build` / `--force-run`, `cancel`.
- Deploy-token lifecycle: `create` / `show` / `refresh` / `delete`.
- Trigger via curl with the token; retrieve status via the token endpoint (the CI wait loop).

Fixed question list to answer (feeds the tutorial and the migration guide):

1. Exact terminal deployment statuses and polling response shape.
2. Does `publish: /` conflict with an existing `webservice` on the same tool, and what is the cleanest cutover order?
3. Does a components deployment replace a same-named job it did not create (the current `face-service` job)?
4. Launcher arg passing: `command: worker --worker-id ml-worker-1` against a Procfile `worker:` entry.
5. Raw-URL config fetch: caching behavior before a freshly pushed branch is visible.
6. Failure modes: what an aborted/failed deployment leaves behind and the repair path.
7. Resource parity: how `cpu` / `memory` / `mount` defaults compare with today's jobs and webservice.
8. `toolforge logs` usefulness for deployment debugging.
9. A supported way to quiesce existing worker components without a controller restarting them during migration, and to hold replacements until authenticated preflight passes. If PTD cannot do both, keep main-tool deployment on the existing script.

**Output:** lab notes → become tutorial T1 and the experiment section of the case study.

### Stage 1: Draft artifacts (no production impact)

- `toolforge.yaml` (main) and `model-server/toolforge.yaml` (face).
- `ptd-deploy.yml`, dispatch-only.
- Parity checklist, rollback/drill runbook, friction log template.
- Owner captures ground truth for resource values: `toolforge jobs dump`, webservice settings.
- Retain the main deploy script's worker ordering and add a verified process-exit check before migration. Test that a failed stop or migration prevents worker startup.

### Stage 2: Shadow deployment on a scratch tool (`wikivisage-ptd`)

- Scratch ToolsDB created from `schema.sql`; **never** point the lab at the production DB.
- No OAuth needed for deploy mechanics; face URL/token can reuse the production face service.
- Run the full workflow end to end, then drills:
  - confirm neither worker process exists while migration runs,
  - break the main tool's face-service token and confirm preflight fails with both workers still stopped,
  - fail migration and confirm workers remain stopped,
  - worker restart via health check (kill the process, observe restart),
  - graceful SIGTERM on redeploy,
  - failed-deploy repair (break config on purpose),
  - cancel mid-deployment,
  - deploy-token refresh,
  - rollback by reverting a commit,
  - deploy timing measurement.

**Decision gate G1:** the face-tool parity checklist is green, its drills pass, and deploy time is measured with no unexplained behavior. This permits the face-tool cutover only; the main tool keeps its current deployment path.

**Decision gate G2 (main tool, blocked until proven):** document and test a supported PTD lifecycle that stops existing worker processes before migration, prevents their controllers from restarting them during migration, and holds replacement workers until the restarted web app passes authenticated preflight. Failure and cancellation must leave workers stopped. Run the three worker-gate drills above against that lifecycle before proposing the main-tool cutover. Starting all components and checking `/health` afterwards does not satisfy G2.

### Stage 3: Production cutover

Preconditions: G1 green; beta risk accepted (watch the Phab workboard "Stabilization phase"); low-traffic window.

1. Create the production **face-tool** config from its repo file (`curl <raw> | toolforge components config create`). Leave the main tool's webservice and worker jobs under their current ownership.
2. Create the face-tool deploy token and add it as a masked GitHub environment secret. Retain SSH access for the main tool's gated deployment.
3. Run one release through the new workflow: face branch publish → face PTD deployment → public face readiness → main image build → stop both workers and confirm exit → migrate → web restart → authenticated preflight → start both workers → verify. Abort at a failed gate and leave workers stopped after shutdown.
4. Verify: `/health`, face readiness, worker heartbeats, one SDC send.
5. Keep the legacy face deployment for one or two cycles as an emergency path, with a documented procedure to restore the published face-service job. Keep the main-tool deploy script until G2 passes and a separately reviewed main-tool cutover has completed.

### Stage 4: Retire legacy

After two clean PTD releases:

- Retire the legacy face deploy script once its PTD replacement has passed two clean releases. Keep the main-tool deploy script and its worker gates until G2 and the later main-tool cutover pass.
- Update `how-to-run-it.md` and `AGENTS.md`.
- File the feature requests that would complete the migration: **one-off components / pre-deploy hooks** (for migrations), worker quiescing and gated startup, deployment ordering, and rollback. (The GitHub Actions example is ours to contribute; see Stage 5.)

### Stage 5: Case study + feedback

- Write the case study from the actual diff: workflow LOC, deploy steps, timings, incidents, friction list, before/after metrics.
- Publish tutorial T3 and contribute the GitHub Actions example to the Wikitech guide (see Part B; explicitly requested on the discussion page).
- File every gap on Phabricator project [8755](https://phabricator.wikimedia.org/project/view/8755/) using the beta template.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Beta instability / breaking changes | Pilot on a scratch tool first; documented legacy fallback until stable; monitor the Phab workboard and changelog. |
| Failed deployment leaves inconsistent state | CI polls terminal status; repair runbook (fix config + redeploy with `--force-run`); drill-tested in Stage 2. |
| No rollback support | Git revert → redeploy; practiced in the lab; previous build images still exist for the legacy path. |
| Raw-URL config fetch may serve a cached config shortly after a branch push | Verify in Stage 0; if needed, wait before triggering or accept the stale-config window (config is rarely changed). |
| `publish` is a young feature | Verify coexistence with the existing webservice early (Stage 0/2); cut over in a low-traffic window. |
| Deploy tokens are credentials | Masked secrets; rotation drill; same handling discipline as the existing SSH key. |
| No deploy queueing | `concurrency` lock in the workflow + check for a running deployment before triggering. |
| Release tag vs `ref: main` drift | Guard step: fail the workflow unless the release tag equals the current `main` tip. |
| All-components deployment starts workers before migration/preflight gates | Keep main-tool deployment on the existing script until G2 proves supported quiescing and gated startup; the combined main config is lab-only. |

---

## 6. Part B: adoption initiative plan (parent task)

### B1. Candidate tool inventory

- Method: pull tools with a public `repository` from Toolhub (toolinfo data), filter to Build Service users (buildservice webservices and build-image jobs), score by activity (commits/deploys in the last 12 months), maintainer contactability, and fit:
  - *simple*: cron/one-job bots, static or single-component tools;
  - *complex*: web + workers (WikiVisage itself, `sample-complex-app`).
- Request the components team's current PTD-user list as the baseline.
- Target: ~20 candidates identified, 5–8 serious, 2–3 pursued alongside WikiVisage.

### B2. Candidate maintainer inventory

- Sources: toolinfo `maintainer` fields, Wikitech user pages, git authors, recent #wikimedia-cloud / cloud@ activity.
- Prioritize maintainers who recently asked for help or complained about deploy friction. Contact via public channels; no mass-mail.

### B3. Tutorials and migration guides

Internal first, then Wikitech (coordinate with WMCS; the main help page is protected):

- **T1 "PTD in 30 minutes"**: from the Stage 0 lab.
- **T2 "Migrating a job-only tool"**: scheduled/continuous variants.
- **T3 "Migrating a web service + workers tool"**: the WikiVisage case, covering ordering, migrations, rollback, and health checks.
- **T4 "Troubleshooting runbook"**: failed deployments, cancel, force flags, config fetch, tokens.
- **GitHub Actions example for the Wikitech page**: explicitly requested in the guide's discussion; our verified workflow (Stages 2–3) doubles as a copy-paste example to contribute upstream.

### B4. Case studies

WikiVisage first, then 2–3 from consenting maintainers (short interview + before/after metrics).

### B5. Feedback loop

One running friction log, categorized (bug / feature / docs gap), each item promoted to a Phabricator task under project 8755 (`[components-api,beta]`, parent T393564). Periodic digest to the WMCS team; the log feeds T4 and the final report.

### B6. Metrics and reporting

Exactly the counters defined by the parent task:

- candidate tools identified · candidate maintainers identified · maintainers contacted · maintainers testing PTD · tutorials completed · tutorials published · tools migrated by Cloud Services · tools migrated by volunteer maintainers · total migrated · remaining candidates.

Plus a baseline captured at kickoff (current PTD adoption as reported by the components team). Maintained as a tracking table on the Phab task:

`Tool | Repo | Class | Maintainer | Contacted | Testing | Status | Migrated by | Notes`

Cadence: monthly update on the parent task; final report at the end.

---

## 7. Indicative timeline (part-time)

| Week | WikiVisage pilot | Initiative |
|---|---|---|
| 1 | Stage 0 lab; baseline metrics request | Draft candidate inventory |
| 2–3 | Stage 1 artifacts; Stage 2 shadow tool | Maintainer shortlist; tutorial T1 draft |
| 3–4 | Stage 2 drills; gate G1 | Outreach kickoff (cloud@ post, 1:1s) |
| 4–6 | Stage 3 cutover; verify | Case study draft; assist first migrations |
| 6–8 | Stage 4 cleanup; Stage 5 case study | Publish T1–T3; track metrics |
| ongoing | monitor, file feedback | measure, report, iterate |

---

## 8. Immediate next actions and decisions

**Decisions needed first:**

1. Cadence: release-triggered PTD (recommended) or push-to-main from day one?
2. Face tool: `face-service` branch approach (recommended) or leave it on the current flow for now?
3. Scratch tool for Stage 2: create one (recommended) and under what name?
4. Part B: draft now in parallel, or after the WikiVisage pilot?

**Owner (Toolforge-side):** confirm decisions; create the scratch tool + database; capture `toolforge jobs dump` / webservice settings; later create deploy tokens and GitHub secrets; schedule the cutover window.

**Agent-assisted (no deploy impact):** the two `toolforge.yaml` files, `ptd-deploy.yml` (dispatch-only), parity checklist, drill runbook, friction-log template, tutorial outlines, and the Toolhub-based candidate-inventory script.
