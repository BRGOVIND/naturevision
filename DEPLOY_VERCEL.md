# Deploy NatureVision on Vercel Hobby

NatureVision runs on one Vercel project with a Neon PostgreSQL database. The
frontend and API share one domain. The backend runs as a Vercel container
service; Vercel Queues runs analysis outside the request that creates it.

```text
Browser → Vercel Vite frontend → /api/v1 → FastAPI container
                                      │
                                      ├─ Neon PostgreSQL + PostGIS
                                      └─ queue producer → Vercel Queue
                                                             │
                                                             ▼
                                                    queue worker → FastAPI
```

## Configure

1. Connect the repository to a Vercel project. Build from the repository root;
   `vercel.json` defines all four services and the routing rules.
2. Add a Neon database through Vercel Marketplace. Connect it to this project
   for both Production and Preview, using the `NEON` environment prefix. The
   backend reads `NEON_DATABASE_URL` and enables PostGIS at startup.
3. Remove any `VITE_API_BASE` environment variable that points to an old
   backend host. Without it, the frontend uses same-origin `/api/v1`.
4. Deploy `master`. Check `/api/v1/health`, then run an analysis in the
   workspace. The health response reports database status and optional
   capabilities.

No Render service or Render PostgreSQL database is needed. Never place a
database connection string in the frontend or commit it to Git.

## Optional capabilities

- Land-cover classification needs a trained model artifact in
  `backend/artifacts/models/`. This repository does not include one, so the
  control is disabled in the hosted workspace.
- Written AI interpretation needs `GROQ_API_KEY` set on the backend. Without
  it, NDVI analysis and deterministic HTML reports still work.
- PDF export is unavailable in the slim hosted image. HTML reports work.

## Runtime behavior

The FastAPI container stores analysis records and rendered PNG layers in Neon,
so both survive container restarts. The queue worker calls the backend through
a private service binding. The backend uses temporary disk only for imagery
processing. Vercel and Neon Free plans have usage and execution limits; monitor
both dashboards as traffic grows.

For local development, the backend can use the database and in-process task
runner configured in `.env.example` and `docker-compose.yml`.

## Verified deployment

On 2026-10-27, the Production deployment at
`https://naturevision-one.vercel.app` passed `/api/v1/health`. A small
Sentinel-2 analysis reached `report_ready`, returned NDVI layers, and appeared
in the workspace. A Preview deployment generated an HTML report from stored
results. Backend test suite: 167 passing tests; Ruff clean.
