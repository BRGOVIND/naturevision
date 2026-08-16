# Data sources

Every source is a legitimate public environmental dataset. Nothing is scraped,
and no environmental value in the product is synthesised.

## Sentinel-2 Level-2A surface reflectance

- **Provider**: Element84 Earth Search STAC API over the AWS Open Data mirror
- **Endpoint**: `https://earth-search.aws.element84.com/v1`
- **Collection**: `sentinel-2-l2a`
- **Access**: public, no account or key
- **Licence**: Copernicus Sentinel data, free and open
- **Resolution**: 10 m (B02, B03, B04, B08), 20 m (B11, B12, SCL)
- **Revisit**: ~5 days at the equator with both satellites

Used for: all imagery, all spectral indices, all change detection, and the
inputs to land-cover classification.

### Bands used

| Logical | Sentinel-2 | Centre | Native | Purpose |
| --- | --- | --- | --- | --- |
| blue | B02 | 492.4 nm | 10 m | Classifier feature, true colour |
| green | B03 | 559.8 nm | 10 m | NDWI, true colour |
| red | B04 | 664.6 nm | 10 m | **NDVI**, true colour |
| nir | B08 | 832.8 nm | 10 m | **NDVI**, NDWI, NDBI, NBR |
| swir16 | B11 | 1613.7 nm | 20 m | NDBI, classifier feature |
| swir22 | B12 | 2202.4 nm | 20 m | NBR, classifier feature |
| scl | SCL | — | 20 m | Cloud, shadow and snow masking |

### Known metadata issue

The catalogue publishes a single static asset description for the whole
collection, so **every** item is tagged `raster:bands.offset: -0.1` — including
acquisitions predating processing baseline 04.00, which introduced that offset.

Applying it blindly corrupts NDVI severely. The system verifies the declared
offset against actual pixel values before applying it; see
[methodology.md](methodology.md#radiometric-calibration).

## ESA WorldCover v200 (2021)

- **Provider**: European Space Agency
- **Access**: public COG tiles at
  `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/`
- **Licence**: CC BY 4.0
- **Resolution**: 10 m global, eleven classes

Used **only** for training and evaluating the land-cover model. It is never
consulted at inference time, and no user-facing analysis reads it.

Its eleven classes are collapsed onto this product's five. The collapse is lossy,
is documented in [methodology.md](methodology.md#labels), and is surfaced in
every report's limitations.

WorldCover is itself a model product with its own error. Agreement with it is
what the reported accuracy measures — not agreement with ground truth.

## Map basemaps

| Source | Use | Terms |
| --- | --- | --- |
| Esri World Imagery | Satellite basemap | Free tile service, attributed in-map |
| OpenStreetMap | Street basemap | ODbL, attributed in-map |

Both are keyless raster tile services. Attribution is rendered by MapLibre from
the style definition and is not removable from the UI.

## Language interpretation

- **Provider**: Groq, OpenAI-compatible chat completions
- **Models**: configurable; defaults are a Llama 3.3 70B text model and a
  Llama 4 Scout vision model
- **Access**: requires `GROQ_API_KEY`

Optional. Without a key the entire measured analysis runs normally and the
report is generated with the interpretation section explicitly marked
unavailable. The provider never sees raw imagery — only the finished evidence
package — and never contributes a number to any result.

## What is not used

- No scraped web content.
- No synthetic, simulated or placeholder environmental values in any production
  path. Synthetic data exists only in `backend/tests/` fixtures, where it
  exercises numerical edge cases, and is not reachable from the API.
- No invented accuracy figures. A model with no completed evaluation reports
  `null`, not a plausible number.
