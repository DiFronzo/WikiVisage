# WikiVisage face detection model server — local development and CI.
#
# Production does NOT use this file: Toolforge builds the service with Cloud
# Native Buildpacks from requirements.txt + Procfile. Both paths install the same
# requirements.txt, so keep it installable with plain pip. Neither needs system
# packages beyond curl (this image's health check).
#
# Two-stage build: the builder stage carries a compiler toolchain for sdists
# without wheels; the runtime stage does not.

# ---------- builder ----------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Needed to build sdists that have no wheel for the target architecture
# (notably psutil on arm64, pulled in transitively by kserve). Confined to the
# builder stage — the runtime image ships no compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim-bookworm AS production

# onnxruntime and opencv-python-headless ship self-contained wheels; curl is
# only for the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv/wikivisage

COPY model.py backend.py guard.py ./
# Committed ONNX weights (non-commercial; see weights/LICENSE-NOTICE). The
# service verifies their SHA-256 at startup and refuses to run on a mismatch.
COPY weights ./weights

# Run unprivileged. The service performs no outbound network I/O and writes
# nothing to disk, so it needs no writable paths beyond /tmp.
RUN useradd --system --uid 10001 --no-create-home wikivisage \
    && chown -R wikivisage:wikivisage /srv/wikivisage
USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/v1/models/wikivisage || exit 1

ENTRYPOINT ["python", "model.py"]
CMD ["--model_name", "wikivisage", "--http_port", "8080"]
