# Plan: SCRFD + ArcFace MBF in the face service, with an experimental Qwen metadata pre-filter

Status: **validated 2026-10-16, ready for a worker.** All decisions are in section 3; measured results are in section 1b.

Owner decisions (2026-10-16):
- Use **SCRFD-500M + ArcFace MobileFaceNet (512D) on ONNX Runtime**. The goal is a large quality improvement, not the licence-clean option.
- This is experimental. **Existing face data may be wiped**, so there is no in-place migration, no dual-model period and no cutover logic.
- Test LiftWing `llm-qwen36-27b` on metadata as an experiment, in shadow mode.
- Detection input: **960 px thumbnails for newly crawled images.**

This replaces the original brief. That brief was checked against `release/0.11.0` (commit `a0bc575`) and external sources. The brief was cut off partway through Phase 1, section 3; Phases 2 and 3 here are this plan's own.

---

## 1. Corrections to the original brief

| Brief says | Reality | Consequence |
|---|---|---|
| Service speaks KServe v2 | It speaks **KServe V1**: `POST /v1/models/wikivisage:predict`, readiness `GET /v1/models/wikivisage`. | **Keep V1.** A switch would touch `guard.py` `open_paths`, `face_client.py`, `deploy.yml` (`--health-check-http`), both CI readiness loops, the Dockerfile `HEALTHCHECK` and `TESTING.md` for no gain. |
| Preserve a `multiprocessing.Pipe` worker pool | No multiprocessing exists. One process: `predict()` uses a `ThreadPoolExecutor` (`FACE_WORKERS`) and a lock around dlib. | Use **threads**. ORT `InferenceSession.run()` is thread-safe and releases the GIL. The Queue-vs-Pipe semaphore concern does not apply. Do not add multiprocessing. |
| `face_recognition` / compiling dlib | `face_recognition` is already gone; `dlib-bin` is a pre-built wheel. | Nothing compiles today. The gains are detection recall and dropping `libopenblas0`/`liblapack3`. |
| Weights are fine to ship | **InsightFace `buffalo_*` weights are non-commercial research only.** WMCS Terms of Use forbid hosting non-Open-Source software; a research-only licence restricts fields of endeavour. | **Accepted risk, see section 3.** Not silently ignorable: it affects how weights are delivered. |
| Rescale to "original full resolution" | The service never sees the original, only the bytes the worker sends (500 px today; the app uses 1024 px for manual draw). | Detecting at 960 px helps only if the worker **fetches** 960 px. See 4.6. |
| `estimateAffinePartial2D(..., LMEDS)` | Not how InsightFace aligns. Reference is a least-squares similarity (Umeyama) fit. | Use Umeyama in numpy, then `cv2.warpAffine`. |
| LiftWing Qwen model | **Exists:** Qwen3.6-27B-FP8, text-only, 32k context, OpenAI-compatible, no auth, Toolforge is in the "effectively unlimited" tier. Experimental, no SLA, about 4 req/s per replica, p50 3.6 s rising to 18 s under load. | Feasible. Run as a **shadow-mode experiment** (Phase 2). |

External facts were checked 2026-10-16. The Wikitech LiftWing page is a draft last edited 2026-10-13. **Re-verify before starting.**

---

## 1b. Validation results (2026-10-16)

Checked by actually running things, not by reading docs. Throwaway scripts lived outside the repo.

**Dependencies resolve** (`uv pip compile`, Python 3.11, `x86_64-manylinux_2_28`): `kserve==0.15.2`, `onnxruntime==1.31.0`, `opencv-python-headless==5.0.0.93`, `numpy==2.4.6`, `Pillow==12.3.0`, `protobuf==4.25.9`. No conflicts. Pin these.

**Weights** (`buffalo_sc.zip` from `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip`):

| File | Bytes | SHA-256 |
|---|---|---|
| `det_500m.onnx` | 2,524,817 | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| `w600k_mbf.onnx` | 13,616,099 | `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f` |

**Model I/O:**
- `det_500m.onnx`: input `input.1`, shape `[1, 3, ?, ?]`. Height and width are **dynamic**, batch is fixed at 1, so run one image per call. Nine outputs in order: scores for strides 8/16/32, then boxes (4) for 8/16/32, then keypoints (10) for 8/16/32. Two anchors per location.
- `w600k_mbf.onnx`: input `input.1`, shape `[N, 3, 112, 112]` (dynamic batch), output `[N, 512]`. Batch all faces of one image into a single call.
- ORT logs a `VerifyOutputSizes` warning on **every** detector call whose input is not 640x640, because the graph's declared output shapes are from a 640 trace. The results are correct, but set `SessionOptions.log_severity_level = 3` or the Toolforge logs will be flooded.

**Commons:** `Special:FilePath/<file>?width=960` returns 200 and redirects to `thumb.wikimedia.org` (already allowlisted). 960 px files were 155–206 KB versus 47–71 KB at 500 px, about 3x.

**End-to-end smoke test** of the plan's exact pipeline (Pillow decode, pad to a multiple of 32, `(x-127.5)/128`, SCRFD decode + NMS 0.4 at score 0.5, Umeyama alignment to `ARCFACE_SRC`, `(x-127.5)/127.5`, L2 normalisation). Run on 5 public-domain Commons portraits at 500 and 960 px, on an Apple-silicon laptop. Expect Toolforge CPUs to be roughly 2–3x slower; this is not measured.

| Measure | Result |
|---|---|
| Faces found (easy portraits, so **not** a recall test) | SCRFD and dlib agreed on every image (1, and 4 on a family photo) |
| Detection time, 1 thread | about 16–27 ms at 500 px, 47–71 ms at 960 px |
| Embedding time | about 13 ms per face |
| Same person (3 Obama photos), cosine similarity | 0.73–0.80 |
| Different people (Obama, Biden, Harris), cosine similarity | -0.05 to 0.13 |
| Same face at 500 vs 960 px | 0.94–0.97 |
| Throughput, one shared session per model | 10.3 images/s with 1 thread, 20.5 with 2 threads (linear) |

The 500-vs-960 comparison for the family photo came out at 0.11. That was the test script picking a different "largest face" at each size, not a model problem.

What this proves: the preprocessing, decoding and alignment in 4.3 are correct; same-person and different-person scores are far apart; and the threading design in 4.2 scales. What it does **not** prove: better recall on hard Commons images (small, oblique, occluded faces). That is still Phase 0's job.

**Code facts confirmed:**
- `write_sdc_claims` checks Commons with `wbgetclaims` before writing and marks the face written if P180 already exists (`worker.py` around 1701 and 1778). Wiping `sdc_claims` therefore cannot cause duplicate claims on Commons.
- `migrate.py` makes `ALTER` migrations idempotent by ignoring MySQL error 1060 (duplicate column). It has **no** run-once mechanism, so the wipe needs its own guard (see 3.2).
- FKs: `faces` references `images` (CASCADE), `sdc_claims` references `faces` and `projects` (CASCADE), and `faces.superseded_by` references `faces` (SET NULL). InnoDB refuses `TRUNCATE` on a table referenced by a foreign key, so the wipe must use `DELETE` or disable FK checks.
- Deploy order is already safe for this change. `deploy.yml` updates the face tool **before** stopping the old workers. Any old worker that calls the new service gets 2048-byte vectors, and the old `face_client` rejects anything that is not 1024 bytes as `FaceServiceUnavailable`, so the images stay `pending` and nothing bad is written. The old web app's manual draw briefly returns 503 during that window.

**Checked with Perplexity (2026-10-16), external sources:**
- **KServe gRPC is on by default** (`ModelServer(enable_grpc=True)`), so the existing `kserve.ModelServer(enable_grpc=False).start([model])` in `model.py` is what keeps port 8081 closed and every route behind `RequestGuard`. Do not remove it during the rewrite. `kserve.model_server.parser` is the correct argparse parent, and `parse_known_args()` is already used. A model is only reported ready when its `load()` sets `self.ready = True`, which the new backend must keep doing.
- **Toolforge `--ref` accepts branches and tags, not commit hashes** (phab:T429227). `deploy.yml` uses tags, so the `face-service-<tag>` subtree tag works. The repo must be public.
- **Git LFS on the build service is unverified** and a plain clone is the common behaviour, so the recommendation to avoid LFS stands. Toolforge documents no file-size or repo-size limit; GitLab's default push limit is 100 MB, and the 16 MB weights are well under it. The per-tool build quota (`toolforge build quota`) and the clone cost are the practical limits.
- **LiftWing has no documented health endpoint.** The only documented liveness check is a real chat completion. The request is a plain OpenAI body (`{"model": ..., "messages": [...]}`); the server applies the Qwen chat template itself. The response follows the OpenAI schema (`choices[0].message.content`), and the docs confirm Qwen output may include a `<think>…</think>` block that the caller must strip. The rate-limit tiers are confirmed: 100 requests/hour shared across all `llm-*` models for anonymous callers, effectively unlimited from Toolforge and Cloud VPS. A local dev machine falls in the 100/hour tier.
- **Weights provenance is now triangulated.** The `buffalo_sc.zip` URL resolves, the pack contains exactly the two files, and independent Hugging Face mirrors publish the same SHA-256 values measured in the table above (four mirrors agree on `w600k_mbf.onnx`). InsightFace publishes no checksums of its own, and the non-commercial terms live in the README rather than a licence file inside the zip, so the `LICENSE-NOTICE` required in section 3.1 must state the terms itself.

### 1c. Former residual risks, now resolved

**1. Git LFS / submodules on the Toolforge builder: eliminated by design, guarded in CI and at startup.**
- The plan uses **neither**. The weights are ordinary git blobs (16 MB total, far under any limit), and the face service has no submodules. Whether the builder supports LFS or submodules no longer matters.
- Three guards make sure this stays true and that a mistake fails loudly instead of silently:
  1. **CI check** (new step in the lint job): fail if `.gitmodules` exists, or if `git check-attr filter -- model-server/weights/*.onnx` reports `lfs`, or if any weight file starts with `version https://git-lfs` (a pointer file).
  2. **CI SHA check:** both face-service CI jobs verify `model-server/weights/SHA256SUMS` before running the contract tests. A pointer file or truncated file fails here.
  3. **Fail-closed at runtime:** `model.py` verifies the same hashes in `load()` and exits on a mismatch. The model never reports ready, so `deploy.yml`'s existing 10-minute readiness wait fails, and the workers are never restarted against a broken service. Log one line with both hashes on success, so the first production deploy can be confirmed from `toolforge jobs logs face-service`.

**2. `det_500m.onnx` hash: confirmed from two independent sources.**
- The official zip downloaded from the GitHub release has SHA-256 `57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72`, identical to the independent SourceForge mirror's published hash. The `det_500m.onnx` extracted from that zip matches the Hugging Face mirror's hash. The official artifact and two unrelated mirrors therefore all agree.
- `scripts/fetch_weights.sh` checks the **zip** hash first, then both file hashes, and refuses to write anything on a mismatch.
- Some other repos redistribute a different `det_500m.onnx` (about 2.41 MB). Only the copy from `buffalo_sc.zip` is valid for this plan.
- The model outputs 10-value keypoints (section 1b), so it is the **SCRFD_500M_KPS** variant. Official WIDER FACE validation AP: easy 90.97, medium 88.44, hard 69.49. For reference, SCRFD_10G_KPS (`det_10g.onnx` in `buffalo_l`) scores 95.40 / 94.01 / 82.80 and is the upgrade path if recall is still short. dlib's HOG detector has no published WIDER FACE score and is trained for faces of about 80x80 px (about 40 px with the 1x upsample used today).

**3. Recall on hard images: preliminary evidence collected; Phase 0 still calibrates.**
- Ran both detectors on the first 40 files of Commons `Category:Group_photographs` (real WikiVisage-style input: crowds, conferences, team photos, newspaper scans), at both thumbnail widths, with the plan's settings (score 0.5, NMS 0.4, max side 1280). One file failed to download at 960 px.

| Width | Images | SCRFD-500M faces | dlib HOG faces | Ratio |
|---|---|---|---|---|
| 500 px | 40 | 622 | 65 | 9.6x |
| 960 px | 39 | 787 | 209 | 3.8x |

- **Spot check of the extra detections:** annotated images of the 4 photos with the biggest gap were inspected by eye. A 70-person conference photo had essentially every face boxed (dlib: 3), and a 1958 parade photo had soldiers and bystanders boxed down to about 10 px (dlib: 1). On a 1926 newspaper halftone, nearly all of the 28 boxes at 500 px were real faces, with at most one low-score (about 0.56) false positive near a ball. **The extra detections are overwhelmingly real faces.**
- **960 px is clearly worth it:** 27% more faces than at 500 px, and the faces are larger. The median face short side was 13 px at 500 px versus 21 px at 960 px.
- **The real open question is tiny faces, not missed faces.** At 960 px, 43% of detected faces are under 20 px on the short side, and 17% are under 14 px. These are hard for a human to classify and give weak embeddings. Section 4.3 now handles this with a minimum face size and a stored detection score.
- **What Phase 0 still does:** measure recall against **your own labelled target faces** (export from the DB before the wipe), check that embeddings of small and oblique faces still match the target, and calibrate the distance threshold and the minimum face size. Phase 0 is a calibration step; the go/no-go question on SCRFD itself is answered.

---

## 2. Current code touch points

**Face service (`model-server/`):** `model.py` (`_Dlib`, `WikiVisageModel`, `_load_image`, `_run_instance`, `_encode_vector`, `_dlib_lock`, `FACE_WORKERS`), `guard.py` (unchanged), `requirements.txt`, `project.toml`, `Dockerfile`, `Procfile`, `docker-compose.yml`, `blubber.yaml`, `make_input.py`.

**Client and callers:**
- `face_client.py`: `ENCODING_BYTES = 1024`, `DetectionResult`, `encode_known_face`, `is_healthy`.
- `worker.py`:
  - `_validate_encoding` (139-161): 1024 bytes, float64, `|x| <= 10`.
  - `_face_distance` (164-173): Euclidean.
  - `_persist_detection` (750-829).
  - `run_autonomous_inference` (about 1270-1460): mean centroid, `distance_threshold` default 0.6.
  - `traverse_category` (571-698).
  - `bootstrap_from_sparql` (1075+).
  - `FILE_PATH_URL` width=500 (line 50).
- `app.py`:
  - `encode_known_face` call sites at 2469 and 3008.
  - `FILE_PATH_URL` width=1024 (line 218).
  - `distance_threshold` range `0.1..1.0` at 1580 and 3583.
  - rerun-inference comparison at 3947.
- Templates and `translations/*/messages.po`: grep "distance", "threshold", "confidence".

**DB:** `schema.sql:112` (`faces.encoding`, "128D float64"), `:45` (`projects.distance_threshold DEFAULT 0.6`), `:85` (`images.status` ENUM), `faces.confidence`; `migrate.py` (idempotent ordered list, run by `toolforge jobs run migrate`).

**Tests/CI:** `tests/test_face_client.py`, `test_face_service_contract.py`, `test_worker.py`, `test_app.py`, `conftest.py` (`_make_encoding`), `.github/workflows/ci.yml` (Dockerfile and buildpack jobs), `deploy.yml`.

---

## 3. Decisions (2026-10-16)

The owner accepts the SCRFD + ArcFace licence for this experiment.

1. **Weights delivery: commit the two ONNX files.**
   - Put `det_500m.onnx` and `w600k_mbf.onnx` (about 16 MB, from `buffalo_sc`) in `model-server/weights/`, with a SHA-256 manifest and a `LICENSE-NOTICE` stating the non-commercial terms and that this is experimental. The `face-service-<tag>` subtree tag picks them up automatically, and the job needs no outbound access (`--mount=none` stays).
   - Do not use Git LFS: the build service's LFS support is unverified, a plain clone is the common behaviour, and a pointer file would break the build (section 1b).
   - Known risk, accepted: the files are public in the repo, so Wikimedia admins could object. Keep the backend pluggable (4.1) so YuNet + SFace (MIT/Apache-2.0) remains a drop-in fallback.
   - Also add `scripts/fetch_weights.sh` (official release URL; checks the zip SHA-256, then both file hashes, per section 1c) so the files can be re-created and verified, and commit `model-server/weights/SHA256SUMS`.
2. **Wipe scope: targeted wipe.**
   - **Delete:** `DELETE FROM images` (this cascades to `faces`, which cascades to `sdc_claims`). Then also `DELETE FROM sdc_claims` to catch any orphans. Do not use `TRUNCATE`, because InnoDB rejects it on FK-referenced tables.
   - **Reset on every project:** `images_total = 0`, `images_processed = 0`, `faces_confirmed = 0`, `last_inference_threshold = NULL`, `last_inference_min_confirmed = NULL`, `completion_reason = NULL`, `sdc_write_requested = 0`, `sdc_write_user_id = NULL`, `sdc_write_error = NULL`, `worker_claimed_by = NULL`, `worker_claimed_at = NULL`, `distance_threshold = <new default>`. Set `status = 'active'` where it is `completed`; leave `paused` and `deleted` alone.
   - **Keep:** `users`, `sessions`, `projects`, `project_members`, `worker_heartbeat`.
   - **`user_stats` (leaderboard):** keep it. It records past contributor effort. Resetting it is optional and should be the owner's call.
   - **Run once:** `migrate.py` has no run-once mechanism. Implement the wipe as a function that first checks `information_schema.COLUMNS` for `faces.embedding_model`. If the column is absent, run the wipe **and** add the column (plus the other 4.4 schema changes) in the same step. Every later `migrate` run sees the column and skips the wipe. Add an integration test for both the first run and a re-run.
   - The alternative is `migrate.py --reset` (deploy input `db-reset=true`, type `WIPE`). That also deletes users, tokens, projects and memberships. Only choose it if that is intended.
   - **Before wiping:** export the labelled data (Phase 0) and take a DB dump stored outside the repo.
   - **SDC:** resolved. `write_sdc_claims` checks Commons first (section 1b), so wiping `sdc_claims` cannot create duplicate P180 claims, and `bootstrap_from_sparql` re-seeds from P180 claims already on Commons.
3. **App manual-draw width: yes, 960, and fix box scaling at the same time.**
   - Verified in the code: the classify and project-detail UIs convert a drawn box into **detection space** using `images.detection_width/height` (`classify.html` lines 851-861; the page itself shows a 1280 px image, which is display only). `app.py` then re-downloads the image at `FILE_PATH_URL` **width=1024** (line 218) and passes that box straight to `face_client.encode_known_face` (lines 2469 and 3008) with **no rescaling**.
   - The worker detects at width=500, so today a box drawn in 500 px space is encoded against a 1024 px image. That looks like an **existing bug**: the embedding comes from the wrong region (roughly the top-left of the face), which would degrade every manual face and bbox edit. It has not been reproduced; confirm with one real manual-draw before assuming. Either way the fix below is correct.
   - **Fix (do it in Phase 1, not as a separate change):** add an optional `box_space: {width, height}` to the `encode` instance. The service scales the box to the decoded image, so the app passes the stored `detection_width/height` and never needs to know the download size. Also set the app's `FILE_PATH_URL` to the same `WIKIVISAGE_DETECTION_WIDTH` as the worker (960), so after the wipe the scale factor is about 1. Add a unit test for the scaling and a contract test with a deliberately different `box_space`.

---

## Phase 0: Quick evaluation (before the wipe)

Goal: confirm the gain is real on your data, and keep a test set.

1. **Export labels first.** A one-off `scripts/export_labels.py` writes a CSV of human-confirmed `is_target` 1 and 0 faces: project, `file_title`, box, `detection_width/height`, label. Store it outside the repo (it contains user classifications).
2. `scripts/eval_face_models.py` (offline, not deployed, not in CI): for about 300 labelled images plus about 200 images where dlib found 0 faces, re-download at 500 and 960 px, then run dlib and SCRFD+ArcFace locally.
3. Report:
   - Detection recall on labelled boxes (IoU ≥ 0.4).
   - Extra faces on 0-face images, spot-checked by hand.
   - Per-project verification AUC and the threshold at a fixed false-positive rate.
   - CPU time per image at 1 thread, for 500 px vs 960 px.
4. Use the results to set the default thresholds (4.4) and to confirm 960 px is worth its cost. Record the numbers in this file.

The go/no-go question is answered by section 1c: on 39 Commons group photos at 960 px, SCRFD found 3.8x as many faces as dlib, and the extras were checked by eye and are real. Phase 0 is now a calibration step: per-person recall on your labelled faces, the default distance threshold, and `WIKIVISAGE_MIN_FACE_PX`. If SCRFD+ArcFace somehow does **worse** on your labelled faces, stop and revisit; if recall is fine but still short on hard faces, try `det_10g.onnx` (SCRFD_10G_KPS) before anything else.

---

## Phase 1: SCRFD + ArcFace MBF in `wikivisage-face` (KServe V1 unchanged)

### 4.1 Dependencies and build
- `requirements.txt`: remove `dlib-bin`, `face_recognition_models`, `setuptools<81`. Add `onnxruntime==1.31.0` and `opencv-python-headless==5.0.0.93`. Keep `kserve==0.15.2`, `Pillow` and `numpy`. This set was resolved without conflicts (section 1b); re-run `uv pip compile` if any pin changes.
- `project.toml`: remove `libopenblas0` and `liblapack3`. Verify with the Toolforge builder job in CI. Delete the file if the deb list ends up empty.
- `Dockerfile`: drop the BLAS/LAPACK packages, keep `curl`, and copy the weights directory.
- Weights: `det_500m.onnx` and `w600k_mbf.onnx`, loaded from `WIKIVISAGE_WEIGHTS_DIR` (default `model-server/weights`). Verify them at `load()` against the SHA-256 values in section 1b and refuse to start on a mismatch. Delivery per section 3.1.
- `model.py` hosts a `FaceBackend` interface:

```text
class FaceBackend(Protocol):
    model_id: str   # "scrfd500m+arcface_mbf"; stored in DB and returned in responses
    dim: int        # 512
    def detect(self, bgr) -> list[Detection]            # box (t,r,b,l), score, 5 landmarks
    def embed(self, bgr, det) -> np.ndarray              # L2-normalised float32[dim]
```

### 4.2 Threading
- One shared `ort.InferenceSession` per model, created in `load()`:
  `intra_op_num_threads = 1`, `inter_op_num_threads = 1`, `execution_mode = ORT_SEQUENTIAL`, `providers = ["CPUExecutionProvider"]`, and `log_severity_level = 3` (to silence the per-call `VerifyOutputSizes` warning, see section 1b).
- Parallelism comes from `FACE_WORKERS` executor threads. Set it to the pod's vCPU count (2 with `--cpu 2`). Throughput was measured to scale linearly from 1 to 2 threads with shared sessions. Call `cv2.setNumThreads(1)` once at import to avoid thread oversubscription.
- Remove `_dlib_lock`. Keep the existing semaphore that bounds queued requests.

### 4.3 Detection and alignment
- **Decode with Pillow, not `cv2.imdecode`.** Keep the existing probe (decompression-bomb check, `MAX_IMAGE_PIXELS`, `LOAD_TRUNCATED_IMAGES`), then convert to an RGB/BGR numpy array. EXIF orientation behaviour must stay the same as today; do not change it silently.
- **Detect:** downscale so the longest side is at most `WIKIVISAGE_DET_MAX_SIDE`, preserving aspect ratio, and record `scale`. Use a default of **1280**: 960 px wide portraits are about 1200–1310 px tall, and a 1024 cap would shrink most of them. The cap mainly protects against very tall scans. Pad bottom/right with zeros to a multiple of 32; the input shape is dynamic (section 1b), so no fixed square is needed. Normalise as `(x - 127.5) / 128`, RGB order. One detector call per image (batch is fixed at 1).
- **Decode SCRFD outputs:** 3 strides (8, 16, 32), 2 anchors per location, with scores, boxes and keypoints from 9 output tensors. `score_threshold = 0.50`, `nms_threshold = 0.40`. Divide boxes and landmarks by `scale` so they refer to the bytes received. Clip to the image. **Tiny faces:** drop faces whose short side is below `WIKIVISAGE_MIN_FACE_PX`, default **20** (on the 960 px input). The group-photo check (section 1c) found 43% of detections below 20 px; these are hard for people to classify and give weak embeddings, and keeping them would flood the classify queue with crowd faces. Even with the cut, SCRFD keeps an estimated 450 faces on that set versus dlib's 209 (about 2x). Phase 0 tunes this value (try 16, 20, 24). Return each face's detection `score` so it can be stored.
- **Align:** least-squares similarity (Umeyama, numpy) from the 5 landmarks to the 112x112 template, then `cv2.warpAffine`:

```python
ARCFACE_SRC = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
```

- **Embed:** `w600k_mbf.onnx` takes 112x112 RGB with `(x - 127.5) / 127.5`, and returns 512D. Batch all of an image's aligned faces into one call. **L2-normalise explicitly** and return float32.
- **`encode` task (user-drawn box):** the manual-draw and bbox-edit routes send a box with no landmarks. Run the detector on the box expanded by 50% and clipped, and take the detection with the highest IoU if it is at least 0.3. Otherwise fall back to an unaligned resize of the box, and return `aligned: false`.
- **Response (V1, same envelope):** keep `id`, `status`, `width`, `height` (the worker stores `width`/`height` as `images.detection_width/height`). Per face: `{box, encoding, score}`. Per prediction: add `model_id` and `dim`. `encoding` is base64 of `dim` little-endian float32 values. Update the module docstring.

### 4.4 Database, client and inference math
Because the data is wiped, there is a single model and no dual columns.
- **`schema.sql` and `migrate.py`:**
  - `faces.encoding BLOB NOT NULL`, comment: "float32 embedding, 2048 bytes for 512D".
  - Add `faces.embedding_model VARCHAR(32) NOT NULL`, as cheap insurance for a future model swap.
  - Add `faces.det_score FLOAT NULL` (SCRFD confidence; NULL for manual faces). It lets the threshold and minimum face size be tuned later, and low-score faces be hidden or sorted last, without re-detecting.
  - Update the comment on `faces.confidence` (cosine distance).
  - `projects.distance_threshold`: new default from Phase 0 calibration. The smoke test gave single-image cosine distances of about 0.20–0.27 for the same person and about 0.87–1.05 for different people, so expect a default somewhere around 0.5–0.6. The existing app validation range `0.1..1.0` already covers this, so only the default and the help text need changing.
  - The wipe from section 3.2, using its `information_schema` guard so it runs exactly once.
- **`face_client.py`:**
  - Replace `ENCODING_BYTES` with a check that `dim` is 512 (configurable), `len(raw) == dim * 4`, and `model_id` is non-empty.
  - Add `model_id`, `dim` to `DetectionResult` and `score` to `DetectedFace`.
  - `encode_known_face` returns `(bytes, model_id, aligned)`; update `app.py` lines 2469 and 3008.
  - Add a startup check that the service's `model_id` matches what `faces.embedding_model` expects.
- **`worker.py`:**
  - `_validate_encoding` becomes `_load_embedding(blob, dim)`: length check, `np.frombuffer(dtype='<f4')`, finite check, norm in `[0.9, 1.1]`, renormalise.
  - `_face_distance` is cosine distance `1 - dot(c, e)`, where `c` is the **renormalised** mean of unit vectors. Store it in `faces.confidence`.
  - Per-image dedup logic is unchanged.
  - Insert chunk sizing: 2 KB per face.
- **`app.py`:** threshold default and help text (range unchanged); rerun-inference comparison; store `embedding_model` on manual faces. Wrap new strings in `_()`, then run `pybabel extract`, `update` and `compile`, and translate nb, es and fr.

### 4.5 Remove dlib entirely
Delete `_Dlib`, the `face_recognition_models` notes in `AGENTS.md`, and dlib-specific tests. There is no legacy path because there is no legacy data.

### 4.6 Detection thumbnail size
- Worker `FILE_PATH_URL`: change `width=500` to `width=960` (a standard Commons step, so it will not trigger the 429 for non-standard widths). Make it configurable via `WIKIVISAGE_DETECTION_WIDTH`.
- `images.detection_width/height` already records the real size, so boxes are always interpretable.
- **Mixed-space rule (for any retained images):** if an existing image keeps old boxes in a 500 px space, new boxes found later must be **converted into the stored detection space** (scale by stored width / new width) before insertion, and new 960 px crawls apply only to newly crawled images. After a full wipe this does not arise, but implement the conversion helper anyway, with a unit test. A re-detect on an existing image (e.g. a later "re-run detection" feature) needs it.
- Align the app's manual-draw width with the worker (section 3.3). If they stay different, the app must convert drawn boxes into the stored detection space.
- Memory: 960 px thumbnails are about 3–4x the bytes. Re-check `DETECTION_WAVE_SIZE` and `FACE_SERVICE_MAX_BATCH_BYTES` against the 3 Gi face-job limit and the 2 Gi worker limit.

### 4.7 Tests, CI and docs
- `conftest.py`: add `_make_embedding(seed, dim)` (unit-norm float32).
- `test_face_client.py`: new dim and `model_id` validation; reject 1024-byte payloads.
- `test_worker.py`: cosine centroid, `_load_embedding`, threshold gating, the box-space conversion helper.
- `test_face_service_contract.py`: assert `model_id`, `dim`, float32 width; detect on a committed public-domain sample with a known face count; cover the `encode` landmark-recovery and fallback paths.
- `test_app.py`: threshold validation, manual-face storage.
- Migration tests for the new columns and the one-time wipe gating.
- CI: both face-service jobs must still pass without the deb packages, and must verify `model-server/weights/SHA256SUMS` before the contract tests. The lint job gets the no-LFS / no-submodule / no-pointer-file check from section 1c. Add a contract-test assertion (or a log check in the CI job) that the service log contains no `VerifyOutputSizes` warnings.
- Docs: `AGENTS.md`, `TESTING.md`, `how-to-run-it.md`, `CONTRIBUTING.md`, `README.md`, and a `LICENSES` note for the weights.

**Phase 1 verification gate:** lint, `pytest`, contract tests on both images, and a detection run on a pilot project reviewed by the owner.

---

## Phase 2: Experimental Qwen metadata pre-filter (shadow mode)

Purpose: learn what a text-only LLM can tell from Commons metadata. It is not expected to replace face detection. Shadow mode changes **no** behaviour.

- **`liftwing_client.py`** (no ML dependencies, mirrors `face_client.py`):
  - Endpoint: `https://api.wikimedia.org/service/lw/inference/v1/models/llm-qwen36-27b/openai/v1/chat/completions` (override with `WIKIVISAGE_LIFTWING_URL`, `WIKIVISAGE_LIFTWING_MODEL`). Send a plain OpenAI body of `model` plus `messages`; the server applies the Qwen chat template. Do **not** stream: send `stream: false` and read the full JSON, so responses are simple to parse and to log.
  - Descriptive `User-Agent` with contact info. Concurrency 1 per worker.
  - Batch 20–50 files per request. Send title, description (plain text, at most 250 chars), categories (up to 15, `Category:` prefix stripped) — and never image bytes. Non-visual MIME types (`audio/*`, `video/*`, PDF, ogg, SVG, DjVu) are settled as drops without a call; MIME is otherwise not sent (prompt v2 — see the followups doc).
  - Ask for a strict JSON array of `{page_id, keep}`, with a short `reason` only on drops. Strip any `<think>…</think>` block or markdown code fence.
  - **Fail open:** any timeout, non-200 or parse failure means keep everything. Exponential backoff plus a circuit breaker (disable for M minutes after N consecutive failures).
- **Metadata fetch:** a batched Commons API call (`prop=imageinfo|categories`, `iiprop=extmetadata|mime`) for up to 50 titles at a time, through the existing `_api_request` wrapper (maxlag, User-Agent, retry).
- **Pipeline:** new `prefilter_images(project)` in `worker.py`, **after** `traverse_category` / `bootstrap_from_sparql` and **before** `process_images`. It selects `status='pending' AND bootstrapped=0 AND prefilter_checked_at IS NULL`. Bootstrapped images always bypass it. Do not put this inside the crawler.
- **Schema:** `images.prefilter_keep TINYINT NULL`, `images.prefilter_reason VARCHAR(160) NULL`, `images.prefilter_checked_at DATETIME NULL`, `projects.prefilter_mode ENUM('off','shadow','enforce') NOT NULL DEFAULT 'off'`. Only add the `'filtered'` status to the ENUM if enforce mode is built. In that case, grep every `status IN (...)` and `'skipped'` usage in `app.py` and `worker.py`, and make sure `filtered` counts as terminal but never triggers the "no faces" auto-completion (see "Auto-completion must never fire while images are pending" in `AGENTS.md`).
- **Health check and status message** (required, part of Phase 2):
  - **Probe:** `liftwing_client.probe()` sends one minimal real request to the configured OpenAI-compatible endpoint (a chat completion with a one-word prompt and `max_tokens` of about 8) and returns up if it gets HTTP 200 with a parseable `choices` array. LiftWing documents **no** health endpoint, so a real completion is the only supported check (section 1b). Try `GET <base>/v1/models` first and fall back to the completion if it returns 404 or 405: that keeps the probe free of the 100/hour tier on a local dev machine. Timeout 20 s, because p50 latency is 3.6 s and can reach 18 s under load. Use the same `User-Agent` as the client. It never raises, and a 429 counts as a failure with the reason recorded.
  - **Who probes:** the **worker**, not the web app. The web app renders pages on gunicorn threads and must never wait on LiftWing. The worker probes only if at least one non-deleted project has `prefilter_mode != 'off'`, so nothing calls LiftWing while the experiment is unused. The interval is `WIKIVISAGE_LIFTWING_PROBE_INTERVAL` (default 300 s).
  - **One probe per interval across both workers:** claim it atomically with `UPDATE service_status SET last_checked = NOW() WHERE name = 'liftwing' AND last_checked < NOW() - INTERVAL <interval> SECOND`, and probe only if the row count is 1. This mirrors the existing multi-worker claiming.
  - **Storage:** new table `service_status` (`name VARCHAR(32) PRIMARY KEY`, `status ENUM('up','down','unknown') NOT NULL DEFAULT 'unknown'`, `consecutive_failures INT UNSIGNED NOT NULL DEFAULT 0`, `last_checked DATETIME NULL`, `last_ok DATETIME NULL`, `last_error VARCHAR(255) NULL`). Add it to `schema.sql` and `migrate.py` (and the table list in `verify_tables()`), and seed the `liftwing` row with `INSERT IGNORE`. The table is not touched by the Section 3.2 wipe.
  - **Hysteresis:** mark `down` only after `WIKIVISAGE_LIFTWING_FAIL_THRESHOLD` consecutive failures (default 3), and mark `up` after a single success, so one slow reply does not flap the message. The circuit breaker in `liftwing_client` should read the same state, so the pre-filter backs off while the service is down.
  - **Stale data:** if `last_checked` is older than 3 probe intervals, treat the status as `unknown` and show nothing. A dead worker is already reported by the existing worker-offline banner.
  - **Web app:** add a context processor `inject_liftwing_status` next to `inject_worker_status` (`app.py` around line 661). It reads the single `service_status` row (one cheap query, wrapped in try/except that fails to "no message", like the worker check) and exposes `liftwing_down`. Do not make a network call from it.
  - **Status message:** in `templates/base.html`, directly after the worker-offline banner (line 531), show a banner when `liftwing_down` is true. Use the same `.system-banner` markup and close button. Suggested text: "The LiftWing AI service used for the metadata filter is currently unavailable. Image processing continues without it." Wrap it in `_()`, and add nb, es and fr translations.
  - **Where it shows:** only on pages where the filter matters: the project detail and project settings pages, and only when the viewed project has `prefilter_mode != 'off'`. Hide it on the landing page, leaderboard and account settings, like the worker banner. It is informational, not an error, because the filter fails open. On the settings page, also show the current status (up, down or unknown, with the time of the last check) next to the mode selector.
  - **`/health`:** add `"liftwing": "reachable" | "unreachable" | "disabled"` (`disabled` when no project uses the filter, or the status is `unknown`). It reads `service_status` and makes **no live call**. It must **not** set `degraded`, because LiftWing is an optional experiment and the face service is the only dependency that should degrade health. Do not translate the JSON values.
  - **Local dev:** calls from outside Toolforge fall into the anonymous tier (100 requests/hour shared across all `llm-*` models). Probing every 5 minutes uses about 12 per hour, which is fine, but allow `WIKIVISAGE_LIFTWING_PROBE_INTERVAL=0` to turn the probe off.
  - **Tests:** probe success, non-200, malformed JSON, timeout and 429; the atomic claim (second worker gets row count 0); hysteresis (2 failures stay `up`, 3 flip to `down`, 1 success flips back); stale status reads as `unknown`; the context processor and `/health` payload with the status table empty, `up`, `down` and `unknown`; the banner renders only on the intended pages and only when the project uses the filter; migration creates and seeds the table idempotently. No live LiftWing calls in CI.
- **Experiment readout:** `scripts/eval_prefilter.py` compares shadow verdicts with where faces were actually found and confirmed. Report recall of `keep` on images with a human-confirmed target face, the skip rate, and latency. Enforce mode is only worth building if recall is at least 99.5% with a meaningful skip rate.
- Other cheap experiments to run in shadow mode while the pipeline exists (suggested, optional): ask the model to flag likely group photos, or images of a different person than the project's subject. Treat results as hints, not filters.
- **Tests:** `liftwing_client` unit suite (parsing, `<think>` stripping, fail-open, circuit breaker), worker stage tests, migration tests. No live LiftWing calls in CI. Update `AGENTS.md` for the new `/health` field, the `service_status` table, the new environment variables (`WIKIVISAGE_LIFTWING_*`) and the status banner.
- **Privacy/policy:** only public Commons metadata is sent. Per the LiftWing docs, prompts are not logged.

---

## Phase 3: Rollout

1. Commit the weights (section 3.1) and take a DB dump.
2. Phase 0: export labels and run the evaluation.
3. Deploy Phase 1 (`deploy.yml`). Deploy order is already safe (section 1b). The `migrate` job runs the one-time wipe. Re-crawl and process a **pilot project** first, then the rest.
4. Re-label enough faces per project to pass the `min_confirmed` gate, then check inference output against the pilot.
5. SDC writes are safe to re-enable straight away (section 3.2).
6. Release the Phase 2 shadow stage separately and run it for a few categories.
7. **Rollback:** after the wipe, the old data is gone. The backup is the pre-wipe DB dump (**take one**, stored outside the repo) plus the previous `face-service-<tag>`. Document the restore steps in `how-to-run-it.md`.

---

## Out of scope
- KServe V2 or gRPC.
- GPU inference.
- Multiprocessing inside the face service.
- Keeping dlib compatibility.
- Changing the bootstrap logic, beyond bypassing the pre-filter.

---

## Implementation notes (2026-10-16): Phase 1

Phase 1 is implemented in the working tree (nothing committed). Deviations and open items:

**Deviations**
- **Extra detection pass for very large faces.** With the plan's default `det_max_side = 1280`, SCRFD-500M returned *no* face on a 960x1307 official portrait whose face was 532 px tall (faces up to about 430 px were fine; at 1024 the same image works because it is scaled to 0.78x). The validation smoke test used 1024, which hid this. `backend.detect` now also runs a half-scale pass when the working image's long side is at least 640 px (`WIKIVISAGE_DET_COARSE_MIN_SIDE`, 0 disables), taking only faces of at least 240 px from it and merging by NMS. Cost: about +25% detector time on large images. Verified on the five portraits available locally; **not** measured on hard Commons images (Phase 0).
- **`backend.py` is a separate module** (the `FaceBackend` protocol and the ONNX implementation) instead of living inside `model.py`, so it can be tested without KServe. `model.py` still hosts the KServe model and the envelope. `Dockerfile`, `.dockerignore` and `blubber.yaml` copy it.
- **`FaceBackend` takes RGB**, not BGR as written in 4.1; 4.3 and the validated smoke test are RGB. `embed` takes a list of detections (batched) and there is an extra `embed_unaligned` for the drawn-box fallback.
- **`WIKIVISAGE_MAX_FACES` (default 300)** caps faces per image, best score first, and embeddings are computed in chunks of 32 (`EMBED_CHUNK`), to bound CPU time and activation memory on crowd photos and adversarial images. Not in the plan.
- **`FACE_WORKERS` default is 2** (was 1) so no deploy change is needed for the 2-vCPU job.
- **Default `distance_threshold` is 0.5**, provisional until Phase 0 calibration (the plan said "around 0.5-0.6").
- **`det_score` is stored but not yet used** anywhere (no sorting or hiding).
- **No `project.toml` at all.** The file was deleted (the deb list is empty), and the CI buildpack job no longer requests `heroku/deb-packages`. Unverified on the real builder (see below).
- **Startup model check** lives in the worker (`face_client.check_service_model()` via a 1x1 PNG probe, fatal only on a definite model/dim mismatch). `/health` is unchanged. Inference additionally filters on `faces.embedding_model`.
- **`DEFAULT_DISTANCE_THRESHOLD`, `DETECTION_WIDTH` and the Commons step list** moved into `config.py` (shared by `app.py` and `worker.py`); a non-standard `WIKIVISAGE_DETECTION_WIDTH` fails at startup.
- Pre-existing untranslated or fuzzy entries in the `.po` files (for example "Join project", "This face was already classified by %(user)s...") were left as they are; only the new threshold help string was translated.

**Not verified**
_(Partly resolved on 2026-10-17 — Docker/compose build, the real portrait sample, the fully pinned lock and the integration suite are all verified now; see "Implementation notes (2026-10-17): follow-up round" at the end of this file for what changed and what remains.)_
- The lock (`requirements.in` → `requirements.txt`, fully pinned) installs in the Docker image (rebuilt 2026-10-17, container healthy, contract tests 21 passed/4 skipped). The CI jobs themselves still run only on push (B2).
- The integration suite has run on a Docker MariaDB (numbers in `face-model-upgrade-followups.md`), including the wipe-migration tests.
- `tests/data/portrait_sample.jpg` **is committed** — a close-cropped public-domain portrait whose ~514x756 px face exercises the half-scale pass (provenance in `tests/data/README.md`). The real-face tests pass with `WIKIVISAGE_REQUIRE_SAMPLE=1`.
- Worker memory at 960 px remains an argument (`DETECTION_WAVE_SIZE` × the 50 MB download cap), not a measurement. The LiftWing batch timing was measured instead (H2: 20-file batches, median 20 s, p95 ~21 s against the 90 s timeout).

---

## Implementation notes (2026-10-16): Phase 2

Phase 2 is implemented in the same working tree (nothing committed). The
Phase 2 section above stands; deviations and decisions:

**Deviations / decisions**
- **No `enforce` mode anywhere.** `projects.prefilter_mode` is
  `ENUM('off','shadow','enforce')` (forward compatible — adding enforce later
  needs no schema change), but the settings route rejects `enforce` and the
  worker treats any non-`off` value as shadow. `images.status 'filtered'` was
  not added: nothing writes it.
- **Pre-filter cap of 250 images per cycle** (`WIKIVISAGE_LIFTWING_MAX_IMAGES`),
  in batches of 40 (`WIKIVISAGE_LIFTWING_BATCH_SIZE`). The plan had no cap; an
  unbounded pass could otherwise run for a very long time inside one project
  cycle.
- **The probe also runs from the two inner 60 s heartbeat loops**, not only the
  outer poll loop: a project cycle can hold the outer loop for many minutes,
  and the first probe could otherwise be delayed past the interval it enforces.
- **Metadata failure leaves rows unchecked** (retried next cycle); a model
  answer that omits a file stores keep with a NULL reason. Both are fail-open
  refinements the plan did not spell out.
- **The web app shares one snapshot helper** (`_liftwing_status_snapshot`) for
  the context processor and `/health`; it catches every DB failure and reads
  `unknown`.
- **Probe endpoint:** `GET <base>/models` first, minimal completion only on
  404/405, as planned; a 200 that is not a JSON object counts as a failed probe
  attempt for that endpoint (the fallback is used only for 404/405).
- **`scripts/eval_prefilter.py`** reports the skip rate, keep recall per class
  of target (human-confirmed / model / bootstrap / any) and sample drop
  reasons. No new dependencies.

**Not verified**
_(Resolved on 2026-10-17: live calls, a real shadow run and measured batch timings — see the follow-up section at the end.)_
- Live LiftWing calls were made anonymously (probe plus a real shadow run that stored 120 verdicts on project 12), then measured: 10 batches of 20 files, median 20 s, p95 ~21 s against the 90 s timeout.
- The CI workflows were not run. Integration mode was verified locally against a Docker MariaDB 10.11.

**Repo fixes that fell out of full-suite verification**
- `schema.sql` contained semicolons inside a `--` comment and a quoted
  `COMMENT`; `load_schema` splits naively on `;`, which broke `migrate.py` and
  the integration fixtures on a real database. Both reworded, and guarded by
  `test_schema_semicolons_are_only_statement_terminators`.
- The integration lane had 44 pre-existing failures at `a0bc575`, almost all
  from the in-memory rate limiter exhausting its global 200/hour quota
  mid-run. Tests now disable the limiter (at test_app import plus both conftest
  fixtures), and the three real failures were fixed: two reload-fragile tests
  (`test_hsts_header_set_when_not_debug`, `test_create_app_returns_app`) and the
  stale `test_inference_does_not_touch_bootstrap_images`, which contradicted the
  March 2026 change that deliberately included bootstrapped images.

---

## Implementation notes (2026-10-17): follow-up round

The verification review is written up in
[`face-model-upgrade-followups.md`](face-model-upgrade-followups.md) (B1–B2,
H1–H4, M1–M4, L1–L9). This round fixed every finding and re-ran the suites. The
behaviour changes worth knowing:

**Phase 2 pre-filter**
- **Order and selection (H1).** `prefilter_images` now runs **after** the image
  loop in `process_project`, and selects `bootstrapped = 0 AND
  prefilter_checked_at IS NULL` **regardless of `status`**. Previously it ran
  first and selected `status = 'pending'`, so it delayed every download and any
  image processed before its turn was never checked — a backlog could never be
  evaluated. A comment in the code records the constraint for a future
  `enforce`: it must select `status = 'pending'` and run before detection.
- **Batch size and timeout (H2).** `LIFTWING_BATCH_SIZE` defaults to **20** and
  `REQUEST_TIMEOUT` to **90 s**. Measured live on 2026-10-17 (anonymous,
  `llm-qwen36-27b`): 10 batches of 20 files, min 0.4 s (a fast 503), median
  20.0 s, max 21.8 s — ~4x headroom vs the timeout. Per-batch wall time is
  logged.
- **Probe and status (H3).** The probe is a **minimal real completion**
  (`max_tokens = 8`), not `GET <base>/models`: the models endpoint can answer
  200 while completions fail. Real batch outcomes feed the same hysteresis via
  `_record_liftwing_failure` / `_record_liftwing_success`, so the banner cannot
  claim "up" while completions fail. A 429 opens the client circuit breaker
  immediately and honours `Retry-After` (delta-seconds or HTTP-date).
- **Auth and etiquette (H4).** `WIKIVISAGE_LIFTWING_API_TOKEN` (optional) sends
  `Authorization: Bearer`; the User-Agent is `WikiVisage/<version>` from
  `config.PROJECT_VERSION`. Live evidence: anonymous calls are classed
  `anon-browser` by Wikimedia's rate limiter.
- **Prompt (M1).** The system prompt no longer asks only about "photographs":
  paintings, drawings, sculptures, coins and other media that can show a person
  are kept, and uncertainty still means keep.
- **Readout (M2).** `scripts/eval_prefilter.py` loads `.env` like the worker,
  drops the dead bootstrap-target metric (bootstrapped images are never
  checked), adds a detector signal (dropped images that had faces; keeps with
  no faces) and is covered by `tests/test_eval_prefilter.py`.

**Face service and web**
- **Dependency lock (M3).** `model-server/requirements.in` holds the top-level
  pins; `requirements.txt` is now a fully pinned transitive lock generated with
  `uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28`.
  Regeneration is documented in `TESTING.md` §7. Verified by rebuilding the
  Docker image and re-running the contract suite.
- **422 for unreadable images (M4).** `FaceServiceRejected` is caught before
  `FaceServiceError` in `/api/manual-face` and `/api/update-face-bbox` and
  returns 422 with a translatable message; `FaceServiceUnavailable` still
  returns 503.
- **`faces.aligned` (L3).** New `TINYINT(1) NOT NULL DEFAULT 1` column
  (schema + ALTER migration + the three manual/bbox INSERTs) recording whether
  a stored embedding came from landmark alignment or the unaligned resize
  fallback. Record-only for now: it makes the frequency measurable before any
  decision about excluding those faces from the centroid.

**Low-priority items**
- L1: the settings page labels the last check time `UTC`.
- L2: `WIKIVISAGE_REQUIRE_SAMPLE` accepts only explicit affirmatives
  (`1`/`true`/`yes`/`on`); `"0"` means "not required", pinned by a unit test
  (`conftest.require_sample_env`).
- L4: `inject_liftwing_status` returns defaults without querying on endpoints
  other than `project_detail` / `project_settings`.
- L5: `AGENTS.md` drift fixed (`BATCH_SIZE` 50, worker pool formula, env
  defaults, line counts).
- L6: the two "Not verified" lists above refreshed (this section plus inline
  pointers).
- L7: the `%(user)s` classification string translated in nb/es/fr (en is
  identity; no empty or fuzzy entries remain in any catalogue).
- L8: `_fetch_metadata` maps non-canonical titles through `query.normalized`.
- L9 is recorded here and in the `prefilter_images` code comment:

> **Before `enforce` is ever built:** file titles and descriptions are
> user-controlled text inside one batch prompt, so one file can try to steer
> the verdicts of others. `enforce` needs smaller batches or per-file
> isolation; it must select pending images and run before detection; it must
> never drop bootstrapped or human-labelled images; and a drop verdict is a
> cost optimisation, never an authorization.

**Verification for this round**
- Unit suite: 958 passed, 63 skipped (was 940/63). Integration suite and the
  face-service suites were re-run; numbers are in the follow-ups file.
- Docker image rebuilt with the pinned lock; contract suite against it: 21
  passed, 4 skipped (auth tests need a token).
- Face-service unit tests with `WIKIVISAGE_REQUIRE_SAMPLE=1` and the newly
  committed sample: pass (exactly one face, score 0.80, coarse pass exercised).
- Live LiftWing: the probe and 10×20-file batches measured; one transient 503
  observed and correctly failed open.

**Still open / not verified**
- B2: CI, the Toolforge buildpack build and `deploy.yml` still have to run on
  push (a local `pack` attempt is recorded in the follow-ups file).
- Production rate limits for the pre-filter (anonymous vs authenticated quota)
  have not been exercised from a shared Toolforge IP.
- Phase 0 calibration of `distance_threshold`, and `det_score` still unused
  (unchanged from the list above).
