# Rasterio/GeoPandas wheels bundle GDAL, so no system GDAL install is needed.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the container healthcheck; the rest are runtime libs the
# scientific wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed from the manifest alone so this layer is cached
# across source changes.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY app ./app
COPY training ./training

RUN useradd --create-home --uid 1000 naturevision \
    && mkdir -p /app/artifacts && chown -R naturevision:naturevision /app
USER naturevision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
