# Deploying to Vercel

Vercel Services routes one public domain to two containers built from the
existing Dockerfiles — no application code changed.

- `frontend/` → `frontend/Dockerfile` (nginx serving the Vite build), all
  non-`/api` paths.
- `backend/` → `backend/Dockerfile` (uvicorn + FastAPI, unchanged), `/api/*`.

Routing is same-origin (`vercel.json` rewrites), so no CORS configuration is
needed for the deployed frontend to reach the API.

## Required Vercel project setup

1. **Backend service port.** `backend/Dockerfile` listens on `8000`; set
   `PORT=8000` in the **backend** service's environment variables (Vercel
   defaults container services to port 80).
2. **Database.** The backend requires PostgreSQL with PostGIS
   (`docker-compose.yml` runs `postgis/postgis:16-3.4`). Vercel does not
   provide this. Provision it externally (Neon, Supabase, RDS, or any
   PostGIS-enabled Postgres) and set:
   - `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname`
3. **Trained model.** `backend/artifacts/models/**/model.joblib` is
   `.gitignore`d and is bind-mounted locally by `docker-compose.yml`. Vercel
   only sees tracked git content, so the container will build **without** a
   model unless you commit it (or add a fetch-on-startup step, not currently
   implemented). Without it, `/api/v1/analysis` runs and reports classification
   as unavailable — the existing, intended behaviour for a missing model — it
   does not fail.
4. **Language interpretation (optional).** `GROQ_API_KEY`, backend-only.
   Add it in the backend service's environment variables when ready; leave
   unset and interpretation reports as unavailable.
5. Other backend variables (`LOG_FORMAT=json`, `LAND_COVER_BACKEND`,
   `TARGET_RASTER_MAX_DIM`, etc.) are optional — see `.env.example` for the
   full list and defaults.
6. **Frontend:** no environment variables required. `VITE_API_BASE` is
   unset in production so the client calls same-origin `/api/v1` — do not
   set it unless you intentionally point at a different API host.

## Deploy steps

```bash
npm i -g vercel      # if not already installed
vercel link          # link this repo to a Vercel project
vercel env add DATABASE_URL production
vercel env add PORT production          # value: 8000, backend service
vercel env add GROQ_API_KEY production  # optional
vercel deploy --prod
```

## Real limitations (not worked around)

- **Background analysis jobs run via `asyncio.create_task` inside the
  request process.** Vercel container services scale to zero after 5 minutes
  idle in production, sending `SIGTERM` with a 30s grace period. A long
  analysis with no other inbound traffic in that window can be killed
  mid-run. This is a platform characteristic, not something patched here —
  keep the service warm (periodic health-check traffic) if this matters, or
  move analysis execution to a queue in a future change.
- **Rendered layer PNGs and evidence JSON write to local container disk**
  (`artifact_dir`), as they do today. That disk is ephemeral and private to
  one container instance: it does not survive a restart and is not shared if
  Vercel runs more than one instance. Analysis *records* persist correctly in
  Postgres; artifact files written to disk do not, beyond that instance's
  lifetime. Not changed here because doing so would mean redesigning storage,
  which was out of scope.
- **Image size.** rasterio/GDAL, geopandas, scikit-learn, torch and matplotlib
  together are far past Vercel's ~500 MB *native Python runtime* bundle
  limit — this is why the backend is deployed as a container image instead
  of a Python Vercel Function; container images are not subject to that
  limit. Not yet measured against Vercel's container size ceiling — do a
  real `vercel deploy` to confirm before relying on it.
