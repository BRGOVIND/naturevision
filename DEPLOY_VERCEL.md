# Deployment

## What happened, and why the architecture changed

The original plan deployed both `frontend/` and `backend/` as Vercel container
services on one domain. That failed:

```
Error: pushing image vcr.vercel.com/.../backend:...
writing blob: uploading layer chunked: StatusCode: 413, "Request Entity Too Large"
```

**Root cause, confirmed by inspecting the actual built image**
(`naturevision-backend:latest`, built from the unmodified `backend/Dockerfile`):
site-packages is ~9 GB uncompressed / **3.09 GB compressed**. Almost all of it
is one thing:

| Package | Size |
| --- | --- |
| `nvidia` (CUDA runtime libraries) | **2.7 GB** |
| `torch` (CUDA build, `2.13.0+cu130`) | 1.2 GB |
| `triton` (CUDA JIT compiler, pulled in by the CUDA torch build) | 691 MB |
| everything else (rasterio/GDAL, geopandas, scikit-learn, pandas, numpy, matplotlib, scipy, ...) | ~1 GB combined |

`pip install .` installs plain `torch>=2.5` from PyPI, which defaults to the
CUDA build and pulls the matching `nvidia-*` CUDA libraries and `triton` —
**3.4 GB this backend never uses**, since it only ever runs CPU inference
(there is no GPU on any target host here). Vercel Container Registry rejects
the resulting layer push with 413 before the image ever reaches a running
container — this is a push-time limit, not a runtime one.

## Deployment architecture

```
Vercel (frontend, static Vite build — no container)
        │  HTTPS, VITE_API_BASE
        ▼
Separate container host (backend — FastAPI, unchanged functionality)
        │
        ▼
External PostgreSQL + PostGIS
```

- **Frontend → Vercel.** Native Vite static build (`vercel.json`: `buildCommand`
  / `outputDirectory`), not a container — there is no reason to pay container
  overhead for a static SPA once the backend is elsewhere. SPA client-side
  routing is preserved via a catch-all rewrite to `index.html` (Vercel serves
  an existing static file over the rewrite when one matches, so hashed
  asset URLs are unaffected).
- **Backend → any Docker-capable container host**, unchanged application code,
  using the existing `backend/Dockerfile` (now multi-stage, see below).
  **Render** is the recommended default for a portfolio/demo: point it at this
  repo with root directory `backend/`, it builds the Dockerfile server-side
  (no client-side registry-push size limit to hit), gives you a free HTTPS
  URL, and takes env vars directly — no additional CLI or infrastructure.
  Railway or Fly.io work identically since both build from the same
  `backend/Dockerfile` with no Vercel-specific code — pick whichever you
  already have an account on.
- **Database:** unchanged — PostgreSQL + PostGIS, external (`docker-compose.yml`
  still runs `postgis/postgis:16-3.4` locally). Vercel and the chosen backend
  host both provide no database themselves; provision one externally (Neon,
  Supabase, Render's own Postgres, RDS, or any PostGIS-enabled Postgres).

## Files changed

- `vercel.json` — frontend-only static build + SPA rewrite; the `services`/
  container backend entry is removed.
- `backend/Dockerfile` — multi-stage; CPU-only torch; `training/` no longer
  copied into the runtime image (confirmed unused at runtime — `app/` has no
  import of it). No application code touched.
- `DEPLOY_VERCEL.md` — this file.

`frontend/Dockerfile` and `docker-compose.yml` are untouched and still work
for local all-in-one Docker Compose development.

## Required environment variables

**Frontend (Vercel project settings — public, build-time):**

| Variable | Value |
| --- | --- |
| `VITE_API_BASE` | `https://<your-backend-host>/api/v1` — **required now**, since frontend and backend are no longer same-origin. Leaving it unset falls back to same-origin `/api/v1`, which will 404 on Vercel once the backend isn't there. |

**Backend (on the container host — never in frontend/Vercel env, never in `vercel.json`):**

| Variable | Value |
| --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@host:5432/dbname` — external PostgreSQL+PostGIS |
| `CORS_ORIGINS` | `https://<your-vercel-domain>` — **required now**: frontend and backend are cross-origin, so the backend must explicitly allow the Vercel domain (default is `localhost` only) |
| `GROQ_API_KEY` | Optional. Backend-only. Unset → interpretation reports as unavailable, analysis still completes. |
| `PORT` | Only if the host requires it explicitly; the Dockerfile's `uvicorn` already binds `8000` and most hosts (Render, Railway, Fly) detect and route to the exposed port automatically. |
| Everything else (`LOG_FORMAT`, `LAND_COVER_BACKEND`, `TARGET_RASTER_MAX_DIM`, ...) | Optional — see `.env.example`. |

**Trained model:** `backend/artifacts/models/**/model.joblib` is `.gitignore`d,
so a fresh container build has no model until you commit one or add a
fetch-on-startup step (not implemented). Without it, `/api/v1/analysis` still
runs and reports classification as unavailable — the existing, intended
behaviour — it does not fail.

## Docker image size — before / after

| | Compressed content size | Notably includes |
| --- | ---: | --- |
| Before (unmodified `Dockerfile`, measured) | **3.09 GB** | CUDA `torch 2.13.0+cu130` + 2.7 GB `nvidia` + 691 MB `triton` |
| After (multi-stage + CPU-only torch) | **Not measured this session** — see note below | CPU-only `torch`, no `nvidia`/`triton` packages at all |

The "before" figure is a real measurement (`docker images naturevision-backend:latest`
→ 3.09 GB content size, `docker history` confirms the CUDA torch build and the
exact `nvidia`/`torch`/`triton` sizes above via `du -sh` inside the running
container). The optimized image could **not** be built in this session: Docker
Desktop's engine could not reach `auth.docker.io` to pull the `python:3.12-slim`
base image (`failed to fetch anonymous token: ... EOF`, reproduced on 5 retries
with backoff — a sandbox network restriction, not a flaky retry). Removing the
`nvidia` (2.7 GB) and `triton` (691 MB) packages entirely, plus shrinking
`torch` itself from a 1.2 GB CUDA build to a ~200–300 MB CPU wheel, is expected
to cut the compressed image by roughly 2–2.5 GB — but this is a reasoned
estimate from the measured breakdown above, **not a verified number**. Run
this to get the real figure:

```bash
cd backend
docker build -t naturevision-backend:optimized .
docker images naturevision-backend:optimized --format "{{.Size}}"
```

## Deploy steps

**Backend (Render, or any Docker host):**
1. Create a new Web Service from this repo, root directory `backend/`, Docker
   runtime (it will use `backend/Dockerfile` automatically).
2. Set `DATABASE_URL`, `CORS_ORIGINS`, and optionally `GROQ_API_KEY` in that
   service's environment variables.
3. Deploy. Note the resulting HTTPS URL.

**Frontend (Vercel):**
```bash
npm i -g vercel        # if not already installed
vercel link             # link this repo to a Vercel project
vercel env add VITE_API_BASE production   # value: https://<backend-host>/api/v1
vercel deploy --prod
```

## Verification performed this session

- `pytest -q` — 161 backend tests pass (unchanged; Dockerfile/vercel.json
  changes don't touch application code).
- `ruff check` / `ruff format --check` — clean.
- `mypy` — clean.
- Backend starts locally (`uvicorn app.main:app`) and `GET /api/v1/health` →
  `200 {"status":"ok", ...}`.
- Frontend `lint`, `typecheck`, `build` — all pass.
- Backend image build with the *unmodified* Dockerfile — succeeded, produced
  the 3.09 GB baseline measured above.
- Backend image build with the *optimized* multi-stage Dockerfile — **not
  completed**, blocked by the Docker Hub network issue described above. The
  Dockerfile was reviewed for correctness (multi-stage `--prefix=/install` +
  `COPY --from=builder /install /usr/local` is a standard, well-tested
  pattern) but its resulting size was not measured.
- No live Vercel or Render deployment was performed or claimed as successful.

## Real limitations (not worked around)

- **Background analysis jobs run via `asyncio.create_task` inside the request
  process.** On any host that scales the container to zero when idle (Vercel
  container services do this after 5 minutes; check your chosen host's
  behaviour), a long analysis with no other inbound traffic can be killed
  mid-run. Render/Railway/Fly's persistent web-service tiers keep the process
  running continuously, which avoids this specific risk, but their free tiers
  may still sleep after a period of no *requests* — the practical mitigation
  is the same: keep the service warm, or move analysis execution to a queue
  in a future change.
- **Rendered layer PNGs and evidence JSON write to local container disk**
  (`artifact_dir`). That disk is ephemeral and private to one instance — it
  does not survive a restart and is not shared across multiple instances.
  Analysis *records* persist correctly in Postgres; artifact files on disk do
  not, beyond that instance's lifetime. Not changed here — redesigning
  storage was out of scope.
- **Optimized image size is unverified** (see above) — the CPU-only-torch
  change is standard and low-risk, but confirm the actual byte count with a
  local `docker build` before trusting it fits comfortably under any given
  host's image-size limit.
