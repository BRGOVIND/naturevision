# Rasterio/GeoPandas wheels bundle GDAL, so no system GDAL install is needed.
#
# Multi-stage: dependency installation happens in a throwaway builder layer.
# Only the installed packages and application code reach the runtime image —
# apt package lists, pip metadata and build-time state never do.
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./

# torch's default PyPI wheel bundles CUDA runtime libraries this backend
# never uses — inference here is always CPU-only. Installing the CPU-only
# build first satisfies pyproject.toml's "torch>=2.5" constraint (PEP 440
# ignores the "+cpu" local version tag in that comparison) without pulling
# several GB of unused GPU libraries. No functional change: torch, PyTorch
# inference and the torch_mlp backend are untouched.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --prefix=/install .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# curl is used by the container healthcheck; the rest are runtime libs the
# scientific wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app ./app

# training/ (offline model-training entrypoint) and research/ (experiment
# framework) are never imported by the running application — app/ has no
# import of either — so neither belongs in the production runtime image.
# Both stay in git for reproducibility; nothing is deleted from the repo.

RUN useradd --create-home --uid 1000 naturevision \
    && mkdir -p /app/artifacts && chown -R naturevision:naturevision /app
USER naturevision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
