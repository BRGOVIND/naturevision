# NatureVision

Geospatial environmental intelligence over Sentinel-2 satellite imagery. Select
a region and a time period, and the system retrieves real satellite
observations, computes vegetation indices, detects change between dates,
classifies land cover with a trained model, and produces an evidence-grounded
Nature Intelligence Report.

Everything the product reports is derived from real data. Measured values come
from deterministic raster processing; land-cover shares are model predictions
and are labelled as such; the written interpretation is generated only from
those measured values and is validated against them before it is shown.

---

## What it does

The public site explains the method and its limits; the workspace runs the real
pipeline. Both live in the same application.

1. Draw a bounding box or polygon on an interactive map.
2. Choose one period, or two to compare.
3. Search the Sentinel-2 catalogue and inspect candidate observations.
4. Run the analysis: imagery retrieval, cloud masking, NDVI, change detection,
   land-cover classification.
5. Inspect results as map overlays, metric cards and charts.
6. Generate and export a report.
7. Reopen any previous analysis from history.

## Architecture

```
frontend/  React + TypeScript + Vite, MapLibre GL map, Recharts
           └ design/     tokens and shared primitives (one source of truth)
             app/        router, navigation, footer, site content
             pages/      landing, workspace, methodology, reports, about
             features/   metrics, charts, panels, controls, lifecycle
backend/   FastAPI, SQLAlchemy, rasterio/shapely/pyproj, scikit-learn + PyTorch
           └ app/geospatial   geometry validation, raster grid, rendering
             app/imagery      provider-agnostic imagery layer + Sentinel-2 STAC
             app/analysis     spectral indices, statistics, change detection
             app/models_ml    features, model backends, registry, inference
             app/interpretation  evidence package, grounding, language service
             app/reports      report assembly and export
             app/orchestration   the analysis pipeline
           training/  land-cover model training against ESA WorldCover labels
```

The backend is a modular monolith. See [docs/architecture.md](docs/architecture.md).

## Data sources

| Source | Use | Access |
| --- | --- | --- |
| Sentinel-2 Level-2A surface reflectance | All imagery and indices | [Element84 Earth Search](https://earth-search.aws.element84.com/v1) STAC over AWS Open Data. No key required. |
| ESA WorldCover v200 (2021) | Reference labels for training and evaluating the land-cover model | Public COG tiles. Training only; never used at inference. |
| Esri World Imagery / OpenStreetMap | Map basemaps | Keyless raster tiles, attributed in the map. |

## Quick start

### Docker (all services)

```bash
cp .env.example .env          # optionally add GROQ_API_KEY
docker compose up --build
```

- Frontend: http://localhost:8080
- API docs: http://localhost:8000/docs

### Local development

Backend:

```bash
cd backend
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# PostgreSQL + PostGIS is the deployment target; SQLite needs no setup locally.
export DATABASE_URL="sqlite+aiosqlite:///./naturevision.db"
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev        # proxies /api to http://localhost:8000
```

### Train the land-cover model

Classification is unavailable until a model is trained; the API reports this
explicitly rather than returning placeholder classes.

```bash
cd backend
python -m training.train_land_cover --backend random_forest
# or the neural backend on the same feature contract:
python -m training.train_land_cover --backend torch_mlp
```

This downloads real Sentinel-2 scenes for ten training regions, reads
co-located ESA WorldCover labels, fits the model, and evaluates it on
geographically disjoint hold-out regions. It takes roughly 10-20 minutes
depending on network speed, and writes `artifacts/models/<backend>/` containing
the artifact, a manifest with a SHA-256 integrity digest, and `metrics.json`
with the measured hold-out scores.

## Configuration

All configuration is environment-based; see [.env.example](.env.example).
No secrets are committed and no API key is hardcoded.

### Interpretation provider

`GROQ_API_KEY` is optional and backend-only. It is never sent to the browser and
never appears in an API response. Without it, analyses run in full and reports
are generated with the interpretation section explicitly marked unavailable.

Vision interpretation additionally requires a vision-capable model on the
account. `/health` probes the provider's model list rather than assuming, so the
UI only offers the control when the model can actually be called.

Copy `.env.example` to `.env` and set the key there; `.env` is gitignored.

## Development commands

```bash
# Backend
cd backend
pytest -q                                   # 161 tests
pytest -q --cov=app --cov-report=term-missing
ruff check app training tests research && ruff format --check app training tests research
mypy app training research

# Frontend
cd frontend
npm run lint && npm run typecheck && npm run build
```

## Research

`backend/research/` is a reproducible experimental framework for the write-up.
It reads the production pipeline through its public interfaces and changes no
production behaviour. See [backend/research/README.md](backend/research/README.md).

```bash
cd backend
python -m research.run --list                 # what is available
python -m research.run --experiment smoke     # fast structural check, no network
python -m research.run --experiment dataset   # build the pixel cache (network, ~15-25 min)
python -m research.run --experiment core      # 8 experiments from the cache, no network
python -m research.run --experiment all       # core + the three network experiments
```

The core suite takes roughly 40 minutes and needs no network once the cache
exists. Each run writes `research/results/<name>/` with the configuration, the
metrics, and a metadata record carrying the git commit, seeds, runtime and
dataset digest that produced them.

## API

Interactive documentation at `/docs`. Core endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness plus which capabilities are actually available |
| `GET` | `/api/v1/methodology` | How this deployment computes what it reports |
| `GET` | `/api/v1/models` | Installed models, hold-out metrics, class definitions |
| `POST` | `/api/v1/imagery/search` | Search Sentinel-2 observations |
| `POST` | `/api/v1/ndvi` | Compute NDVI and statistics for a region and period |
| `POST` | `/api/v1/change-detection` | Compare two periods |
| `POST` | `/api/v1/land-cover` | Classify land cover |
| `POST` | `/api/v1/analysis` | Start a full analysis (202; poll for progress) |
| `GET` | `/api/v1/analysis/{id}/status` | Lifecycle state and progress |
| `GET` | `/api/v1/analysis/{id}` | Full result: metrics, observations, layers |
| `GET` | `/api/v1/analysis/{id}/layers/{key}` | Rendered map overlay (PNG) |
| `GET` | `/api/v1/analysis` | Analysis history |
| `POST` | `/api/v1/ai/report` | Generate a Nature Intelligence Report |
| `GET` | `/api/v1/analysis/{id}/report` | Fetch the report an analysis already has |
| `GET` | `/api/v1/reports/{id}/export` | Export the report as HTML or PDF |

## Scientific scope

This is the part that constrains everything else, so it is stated plainly.

**What the system measures.** Surface reflectance from Sentinel-2 Level-2A, and
indices derived from it. NDVI statistics and their change between two dates are
measurements, computed only over pixels that survive cloud, shadow, cirrus and
snow masking in both periods.

**What the system predicts.** Land-cover class per pixel, from a trained
classifier with a published hold-out accuracy. These are predictions carrying
classification error, never measurements.

**What the system does not claim.** A two-date optical index comparison cannot
establish deforestation, logging, agricultural expansion, urbanisation, fire,
drought, human causation or climate attribution, and NDVI does not measure
biomass, carbon stock, habitat quality or biodiversity. The system reports
*vegetation-index change* and stops there. Generated interpretation is checked
against the evidence and flagged when it reaches beyond it.

Every report includes a limitations section, and the limitations are specific to
that run — cloud fraction, seasonal mismatch between acquisitions, and low model
confidence are all detected and stated.

Full detail: [docs/methodology.md](docs/methodology.md).

## Documentation

- [Architecture](docs/architecture.md) — structure and the decisions behind it
- [Methodology](docs/methodology.md) — processing, thresholds, model evaluation, limitations
- [Data sources](docs/data-sources.md) — provenance, licensing, known metadata issues

## Licence

Code in this repository is available for review and reuse. Satellite and
land-cover data retain their upstream terms: Copernicus Sentinel data is free
and open; ESA WorldCover is CC BY 4.0.
