# Licences

## WikiVisage code

MIT — see [LICENSE](LICENSE).

## Face model weights (`model-server/weights/`)

| File | What it is | Source | Terms |
|---|---|---|---|
| `det_500m.onnx` | SCRFD-500M face detector with 5-point keypoints | InsightFace `buffalo_sc` model pack | **Non-commercial research use only** |
| `w600k_mbf.onnx` | ArcFace MobileFaceNet, 512-dimensional face embedding | InsightFace `buffalo_sc` model pack | **Non-commercial research use only** |

Pack: <https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip>
(SHA-256 of the zip, `57d31b56…47c72`, and of each file are pinned in
`scripts/fetch_weights.sh`, `model-server/backend.py` and
`model-server/weights/SHA256SUMS`).

InsightFace releases its code under the MIT licence but states that its
pre-trained models, and the data they were trained on, are available for
non-commercial research purposes only. These weights are **not** covered by this
repository's MIT licence, and nothing here grants a licence to them. The
complete notice shipped with the files is
[`model-server/weights/LICENSE-NOTICE`](model-server/weights/LICENSE-NOTICE).

This is an accepted, documented risk for an experimental deployment, not a
clearance: Wikimedia Cloud Services' Terms of Use restrict hosting
non-Open-Source software, so the files could be objected to and removed. The face
service reaches its models through the `FaceBackend` interface
(`model-server/backend.py`), so a permissively licensed pair (for example YuNet +
SFace) can replace them without changing the HTTP contract, the client or the
database schema (every face records the `embedding_model` that produced it).

## Python dependencies of the face service

`kserve`, `onnxruntime`, `opencv-python-headless`, `Pillow` and `numpy` are used
under their own (permissive) licences; see their distributions for the texts.
