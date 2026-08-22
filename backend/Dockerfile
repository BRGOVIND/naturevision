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
# Both installs target the builder's normal site-packages (no --prefix):
# splitting them across two locations previously made the second `pip
# install .` see torch (and its own dependencies, including
# typing-extensions) as already satisfied by the first command and skip
# copying them into a separate --prefix directory, so the runtime image
# silently shipped without torch at all. Installing everything into one
# location and copying that whole location below avoids the mismatch.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RASTER_CACHE_DIR=/app/.cache/raster

WORKDIR /app

# curl is used by the container healthcheck; the rest are runtime libs the
# scientific wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY app ./app

# training/ (offline model-training entrypoint) and research/ (experiment
# framework) are never imported by the running application — app/ has no
# import of either — so neither belongs in the production runtime image.
# Both stay in git for reproducibility; nothing is deleted from the repo.

# app/core/config.py resolves raster_cache_dir's default from the repo root
# (one level above the backend package), which is correct in a local
# checkout but resolves to "/" in this image, since only app/ is copied in.
# RASTER_CACHE_DIR above overrides it to a path under the already-writable,
# already-chowned /app directory — no application code changes.
RUN useradd --create-home --uid 1000 naturevision \
    && mkdir -p /app/artifacts /app/.cache/raster \
    && chown -R naturevision:naturevision /app
USER naturevision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
