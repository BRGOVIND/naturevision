# Architecture

A modular monolith. There is no message bus, no agent framework and no service
mesh, because the problem does not need them: the analysis is a linear pipeline
over deterministic services, and keeping it linear is what makes it testable.

## Layering

```
API routes  ──▶  orchestration  ──▶  domain services  ──▶  providers
   │                                       │
   └── schemas (Pydantic)                  └── geospatial primitives
```

Dependencies point inward. `app/geospatial` and `app/analysis` know nothing
about HTTP, the database, or the imagery provider. The provider layer knows
nothing about NDVI.

## Key decisions

### The imagery layer is an abstraction, not a Sentinel-2 client

`ImageryProvider` defines search, calibration resolution and band reading in
physical terms — logical bands, wavelengths, resolutions — rather than provider
asset keys. Swapping Earth Search for another STAC API, a commercial provider or
a local archive means adding one implementation and rebinding a factory.

`SentinelHubStacProvider` is one implementation. The test suite ships another,
in-memory one, which is what lets the whole pipeline be tested without network
access.

### Every raster carries its georeferencing

`RasterGrid` bundles a masked array with its CRS, affine transform and nodata
mask. Operations return a new grid with that metadata intact. Losing it
mid-pipeline is the classic source of silently misaligned remote-sensing
results, so alignment is asserted before any two rasters are combined — a
mismatch raises rather than broadcasting to a wrong answer.

Nodata is an actual mask, not a sentinel value. Sentinels leak into arithmetic;
masks do not.

### Radiometric calibration is resolved once per observation

A spectral index is only meaningful when all its inputs share one
digital-number-to-reflectance convention. The decision is therefore made once,
before any band is read, and passed to every read. See
[methodology.md](methodology.md#radiometric-calibration) for why the catalogue's
declared value cannot be trusted.

### The feature contract is versioned and enforced

The land-cover feature vector is defined in one place, shared by training and
inference, and written into every model artifact. The registry refuses to serve
a model whose feature names do not match the running code, so a stale artifact
fails loudly instead of producing quietly wrong maps.

Artifacts are integrity-checked against a SHA-256 in their manifest before being
deserialised.

### Analyses run in the background, with a real lifecycle

A full analysis fetches and processes real imagery and takes tens of seconds.
`POST /analysis` returns 202 immediately and the client polls. Each stage
commits a status, detail string and progress fraction, so the UI shows what is
actually happening rather than an unexplained spinner. Every failure path
records a terminal state with a safe message.

The background task resolves its imagery provider through a module-level factory
rather than constructing one directly, because it runs outside the request scope
where FastAPI's dependency overrides apply. That indirection is what makes the
lifecycle testable.

### Overlays are rendered once and persisted

When an analysis completes, every available layer is rendered to PNG and written
to the artifact store, and the manifest records each layer's WGS84 bounds and
legend. Holding raster stacks in memory would not survive a restart and would
not scale; re-deriving them per map interaction would re-fetch imagery.
Rendering once avoids both, and layer requests become static file reads.

Overlays are reprojected to EPSG:4326 before encoding so their corners coincide
exactly with the bounds handed to the map client.

### Geometry is stored twice, deliberately

Canonical GeoJSON in a portable JSON column, plus — on PostgreSQL — a PostGIS
geometry column for spatial querying. The JSON copy keeps the schema usable on
SQLite for tests and preserves the exact client-submitted coordinates.

### Metrics are narrow rows, not wide columns

`EnvironmentalMetric` stores one measurement per row with its own unit, period,
category and provenance tag. New indices can be added without a migration, and
every value carries whether it was observed or predicted — a distinction the
product depends on all the way to the report.

### Capability probing over assumption

`/health` probes what this deployment can actually do: database reachable,
land-cover model installed, language provider configured, PDF toolchain
available. The frontend disables controls for genuinely unavailable features
instead of letting a user trigger a guaranteed failure.

## Request flow

```
POST /analysis
  └─ validate geometry (server-side, geodesic extent)
  └─ persist Region + Analysis, return 202
  └─ background task
       ├─ searching     catalogue query, rank, select
       ├─ acquiring     resolve calibration, windowed band reads, cloud mask
       ├─ processing    co-register, NDVI, statistics
       ├─ analyzing     change detection, land-cover inference
       ├─ interpreting  render overlays, build evidence package
       └─ report_ready  persist observations, metrics, predictions, manifest

POST /ai/report
  └─ load persisted evidence
  └─ generate interpretation → schema check → grounding check
  └─ assemble 13 provenance-tagged sections → render HTML
```

## Testing strategy

127 tests, no network access required.

- **Geospatial**: coordinate and polygon validation, CRS handling, geodesic
  area, grid alignment, clipping, windowing.
- **Analysis**: NDVI against its closed form, divide-by-zero, nodata
  propagation, misalignment rejection, threshold classification,
  comparable-pixel rules.
- **ML**: feature contract stability, probability validity, artifact
  round-trip, registry integrity and feature-mismatch rejection, and an
  explicit test that accuracy is never invented.
- **API**: validation failures, provider failures, the full lifecycle including
  the failed terminal state, layer serving, report generation and history.
- **Interpretation**: grounding acceptance and rejection, malformed and missing
  provider output, transport failure, and graceful degradation without a key.

Synthetic fixtures are used deliberately — they exercise numerical edge cases
real imagery only produces occasionally. No production code path depends on
them.
