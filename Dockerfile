# Targets: runtime (ORT_EXTRA=cpu|openvino|cuda) and exporter. Model weights are never baked in.
FROM python:3.14-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    PATH=/app/.venv/bin:$PATH MODELS_DIR=/models
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
COPY pyproject.toml uv.lock ./

FROM base AS exporter
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra exporter
COPY exporter ./exporter
ENTRYPOINT ["python", "-m", "exporter.export"]

FROM base AS runtime
ARG ORT_EXTRA=cpu
# OpenVINO's iGPU plugin needs Intel's OpenCL runtime inside the container. Debian trixie no longer packages it,
# so install Intel's release debs (same versions as Immich's ML openvino image; Gen12+ iGPUs such as the N100).
ARG IGC_VERSION=2.36.3+21719
ARG COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG GMMLIB_VERSION=22.10.0
RUN if [ "$ORT_EXTRA" = "openvino" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ocl-icd-libopencl1 wget \
      && cd /tmp \
      && wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}/intel-igc-core-2_${IGC_VERSION}_amd64.deb" \
      && wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}/intel-igc-opencl-2_${IGC_VERSION}_amd64.deb" \
      && wget -nv "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
      && wget -nv "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMMLIB_VERSION}_amd64.deb" \
      && dpkg -i *.deb && rm *.deb \
      && apt-get remove -y wget && rm -rf /var/lib/apt/lists/*; fi
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra "$ORT_EXTRA"
COPY tagger ./tagger
# Unhealthy when no pass succeeded recently or the version guard forced read-only. The first full scan can be long.
HEALTHCHECK --interval=5m --timeout=30s --start-period=60m CMD ["python", "-m", "tagger", "health"]
ENTRYPOINT ["python", "-m", "tagger"]
CMD ["run"]
