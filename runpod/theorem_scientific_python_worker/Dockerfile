# syntax=docker/dockerfile:1.7
# Theorem scientific Python worker: the vendored Theorem uv environment with
# its accelerated and CUDA extras, Theorem's runner, and the RunPod handler.
# Every input is pinned: base images by digest, Python by patch version, the
# environment by uv.lock (--frozen), the handler SDK by hash.
FROM ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 AS uv

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04@sha256:1287141d283b8f06f45681b56a48a85791398c615888b1f96bfb9fc981392d98

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/theorem/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_DOWNLOADS=manual \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/theorem-uv-cache \
    THEOREM_PYTHON=3.13.12 \
    THEOREM_ENV=/opt/theorem/env \
    THEOREM_LOCK=/opt/theorem/project/uv.lock \
    THEOREM_RUNNER=/opt/theorem/theorem_compute_runner.py \
    THEOREM_CONSTRAINTS=/opt/theorem/constraints.txt \
    THEOREM_EXTRAS=accelerated,cuda

RUN uv python install "$THEOREM_PYTHON"

WORKDIR /opt/theorem/project
COPY environment/pyproject.toml environment/uv.lock ./
RUN --mount=type=cache,target=/tmp/theorem-uv-cache \
    UV_PROJECT_ENVIRONMENT="$THEOREM_ENV" uv sync \
        --frozen \
        --extra accelerated \
        --extra cuda \
        --compile-bytecode \
        --python "$THEOREM_PYTHON" \
 && uv pip freeze --python "$THEOREM_ENV/bin/python" > "$THEOREM_CONSTRAINTS"

COPY theorem_compute_runner.py /opt/theorem/theorem_compute_runner.py

COPY requirements.txt /opt/worker/requirements.txt
RUN --mount=type=cache,target=/tmp/theorem-uv-cache \
    uv venv --python "$THEOREM_PYTHON" /opt/worker/venv \
 && uv pip install --python /opt/worker/venv/bin/python --require-hashes -r /opt/worker/requirements.txt
COPY worker.py /opt/worker/worker.py

WORKDIR /opt/worker
CMD ["/opt/worker/venv/bin/python", "-u", "/opt/worker/worker.py"]
