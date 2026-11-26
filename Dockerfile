# syntax=docker/dockerfile:1.7
#
# Multi-stage CPU image for the churn inference API.
#
#   builder  resolves ONLY the locked runtime dependencies (no dev / train groups, so no
#            pytest, mypy or Optuna) into /opt/venv and installs the package non-editable.
#   runtime  slim Python + the OpenMP runtime + the venv + the exported model. Runs as a
#            non-root user; nothing is compiled or downloaded at start-up.
#
# Build via `uv run python tasks.py docker-build`, which first exports the registry's
# production model (hash-verified) to build/model/.
#
# Base images are pinned by tag AND digest: the tag documents intent, the digest makes
# the build reproducible even if the tag is re-pushed.

ARG PYTHON_IMAGE=python:3.12.13-slim-trixie@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Dependencies first: this layer is reused until uv.lock or pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# LightGBM and XGBoost link against the OpenMP runtime, which slim images omit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY build/model /opt/model

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_PIPELINE_MODEL_DIR=/opt/model

USER app
WORKDIR /opt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# `ml-pipeline serve` configures structured request/latency logging, then runs uvicorn.
CMD ["ml-pipeline", "serve", "--host", "0.0.0.0", "--port", "8000"]
