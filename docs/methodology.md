# Methodology

How NatureVision turns satellite observations into the numbers it reports, and
what those numbers do and do not support.

---

## 1. Region validation

Geometry is validated server-side regardless of what the client checked.

- Only WGS84 (EPSG:4326) is accepted, in GeoJSON longitude-first order.
- Bounding boxes must satisfy `west < east` and `south < north`; boxes crossing
  the antimeridian are rejected rather than silently mishandled.
- Polygons are parsed with shapely; self-intersecting rings are repaired with
  `make_valid` where possible and rejected otherwise.
- Extent is computed **geodesically** on the WGS84 spheroid, not as planar
  degrees. Square degrees are not an area, and a one-degree box near 60°N covers
  less than half the ground of one at the equator.
- Regions must fall between 0.01 km² and 2,500 km².

**Known limitation.** Axis-order errors are detected only when the latitude slot
exceeds ±90°. Coordinates that are valid in both slots produce a legal geometry
in the wrong place, which cannot be caught from coordinates alone.

## 2. Imagery acquisition

Source: Sentinel-2 Level-2A surface reflectance, via the Element84 Earth Search
STAC API over the AWS Open Data mirror. No credentials are required.

Observations are ranked by usability before selection:

```
score = (1 - regional_coverage) * 2 + cloud_cover / 100
```

Coverage is weighted twice as heavily as cloud, because a partially overlapping
scene cannot be repaired, whereas moderate cloud is removed by masking.

Reads are **windowed**: only the byte ranges of the cloud-optimised GeoTIFF that
intersect the region are fetched, and the read is downsampled server-side to the
analysis grid. A typical analysis transfers a few megabytes rather than the
~1 GB a full scene would cost.

### Radiometric calibration

Sentinel-2 processing baseline 04.00 introduced a −1000 DN radiometric offset,
which catalogues advertise through `raster:bands.offset`.

**That metadata cannot be trusted on its own.** Earth Search publishes one
static asset description for the entire collection, so items acquired *before*
the baseline change are also tagged `offset: -0.1`.

This matters more than it looks. NDVI is invariant to a common *scale* factor
but not to a common *offset*: subtracting 0.1 from both bands pushes the
denominator `NIR + Red` toward zero, inflating the index toward ±1. Applying the
declared offset to a 2021 scene over the Western Ghats produced a median NDVI of
−0.08 and discarded 97% of pixels; the correct value is 0.74.

So the declared offset is treated as a hypothesis and tested against pixels. A
cheap 64×64 probe read of a low-reflectance band is taken; if applying the offset
drives more than 10% of valid pixels to negative reflectance and dropping it
does not, the offset is rejected. The decision is made **once per observation**
and applied to every band, so all inputs to an index share one radiometric
convention. Mixing conventions between bands is the subtler version of the same
bug and biases every index.

The decision, the diagnostic that produced it, and the applied offset are
recorded in each analysis's provenance.

### Quality masking

The Level-2A scene classification layer (SCL) is resampled onto the analysis
grid with **nearest-neighbour** interpolation — interpolating categorical class
codes would invent classes — and these classes are excluded:

| SCL | Class |
| --- | --- |
| 0 | No data |
| 1 | Saturated or defective |
| 3 | Cloud shadow |
| 8 | Cloud, medium probability |
| 9 | Cloud, high probability |
| 10 | Thin cirrus |
| 11 | Snow or ice |

Surface reflectance outside a tolerant [−0.2, 1.6] envelope is also dropped as
saturation or a processing artefact. An analysis that retains too few valid
pixels fails with a specific error rather than reporting statistics from a
handful of survivors.

### Grid construction

All bands are co-registered onto the **finest-resolution** band present, so 20 m
bands are resampled up onto the 10 m grid rather than discarding 10 m detail.
Every array carries its CRS, affine transform and nodata mask through the entire
pipeline; operations return a new grid with georeferencing intact.

Clipping to the selection polygon **masks** outside pixels but preserves the
grid extent, so two periods clipped to the same polygon stay pixel-comparable.

## 3. Vegetation index

```
NDVI = (NIR − Red) / (NIR + Red)
```

Sentinel-2 B08 (832.8 nm) and B04 (664.6 nm), both 10 m.

Pixels are invalidated when either input is masked, when `|NIR + Red| < 1e-6`
(the divide-by-zero guard), or when the result falls outside the mathematically
achievable [−1, 1] range.

Reported statistics: mean, median, min, max, standard deviation, 10th and 90th
percentiles, valid pixel count, and valid area in km². Dispersion statistics are
**withheld** below 30 valid pixels rather than reported as if meaningful.

Vegetation density is binned into five NDVI classes whose boundaries are
published with every result.

## 4. Change detection

Period B is reprojected onto period A's pixel grid before differencing, so the
comparison always happens on identical pixel geometry.

```
difference = NDVI(period B) − NDVI(period A)
```

Only pixels valid in **both** periods contribute. Anything cloud-masked in
either date is excluded, never treated as change. Period means are recomputed
over that shared footprint, so the reported change equals the difference of the
two means on exactly the same pixels.

### Thresholds

| Class | Condition |
| --- | --- |
| Stable | \|Δ\| < 0.10 |
| Moderate increase/decrease | 0.10 ≤ \|Δ\| < 0.20 |
| Significant increase/decrease | \|Δ\| ≥ 0.20 |

The 0.10 default sits above the commonly cited ~0.02–0.05 combined radiometric
and atmospheric-correction uncertainty of Sentinel-2 L2A NDVI, so it is unlikely
to fire on sensor noise alone. 0.20 marks change large enough to be visible in
the imagery itself.

Both are configurable (`CHANGE_MODERATE_THRESHOLD`,
`CHANGE_SIGNIFICANT_THRESHOLD`) and are reported with every result so a reader
can re-derive the classes. They were **not** tuned to make maps look eventful.

Relative change is reported only when `|mean(period A)| ≥ 0.05`; a near-zero
baseline makes a percentage change numerically unstable.

## 5. Land-cover classification

Five classes: Forest, Agriculture, Water, Urban/built-up, Bare land.

### Features

A 11-element per-pixel vector, versioned and written into every model artifact
so a model can never be served against features it was not trained on:

- Six reflectance bands: blue, green, red, NIR, SWIR-1, SWIR-2
- Five normalised-difference indices: NDVI, NDWI, NDBI, NBR, and a
  SWIR-1/blue contrast

Training and inference share one implementation of this cube, so the two cannot
drift apart.

### Labels

ESA WorldCover v200 (2021), a 10 m global product, read from public COG tiles
co-located with each training scene. Its eleven classes are collapsed onto the
five reported here. **The collapse is lossy** and bounds what the classifier can
be held accountable for:

- Shrubland and grassland fold into **Agriculture**, so that class means managed
  or herbaceous cover, not cropland specifically.
- Mangroves are reported as **Forest**.
- Snow, ice, moss and lichen are reported as **Bare land**.
- Herbaceous wetland is reported as **Water**.

### Backends

Two interchangeable implementations on the same feature contract:

- **Random forest** (scikit-learn) — the default. Needs no scaling, tolerates
  the strong correlation between Sentinel-2 bands, and its class-vote
  frequencies serve as an honest confidence signal.
- **Multilayer perceptron** (PyTorch) — standardised inputs, class-weighted
  cross-entropy, cosine-annealed AdamW. Useful where a differentiable model or
  GPU inference is wanted. Stores only tensors, so it loads with
  `weights_only=True`.

### Evaluation protocol

Accuracy is measured on **geographically disjoint hold-out regions**. The model
is fitted on ten training regions and scored on three evaluation regions that
share no pixels, scenes or landscapes with them.

A random pixel-level split would badly overstate accuracy here: adjacent 10 m
pixels are strongly spatially autocorrelated, so a held-out pixel would almost
always sit beside a training pixel from the same field or canopy. Holding out
whole geographies is the honest protocol, and it produces markedly lower — and
more truthful — numbers.

Metrics live in `backend/artifacts/models/<backend>/metrics.json` and are served
by `GET /api/v1/models`. **No accuracy figure is ever defaulted**: a model whose
evaluation did not run records `null`.

### Known model limitations

- The classifier is **per-pixel** and uses no spatial context, so it produces
  salt-and-pepper noise at class boundaries and on mixed pixels.
- Training scenes are single-date. Cropland appearance varies strongly with
  phenology, so agriculture and bare land are confusable outside the growing
  season.
- Agreement is measured against WorldCover, which is itself a model product with
  its own error. Agreement with it is not ground truth.
- Measured performance is uneven across classes. Per-class precision, recall and
  F1 are published so a reader can see exactly which classes are reliable in a
  given deployment rather than trusting a single headline number.

### Confidence

Per-pixel confidence is the model's own maximum class probability, exposed only
because both backends genuinely produce one. Pixels below 0.50 are flagged — a
five-class problem has a 0.20 chance baseline. When a backend cannot produce
probabilities, the confidence fields are omitted rather than filled in.

## 6. Interpretation

The language layer performs **no measurement**. It receives a finished evidence
package — deterministic values, provenance-tagged as observed or predicted — and
writes prose about it.

Its output is then verified, because a prompt is not an enforcement mechanism:

1. **Schema validation.** The response must match a strict structure of summary,
   observations, interpretation, uncertainty, limitations and a confidence
   qualifier.
2. **Numeric grounding.** Every number in the generated text is matched against
   the evidence package, within 2% tolerance, also accepting the percentage form
   of a stored fraction. Years and small counting numbers are excluded.
   **A number that does not correspond to a measured value is fatal** — a
   fabricated figure is indistinguishable from a measured one once printed.
3. **Claim screening.** Assertions the methodology cannot support — deforestation,
   biodiversity loss, climate attribution, specific human causes, "proves",
   "certainly" — are flagged and surfaced in the report.

On failure the model gets one correction turn with the specific offending
numbers. If it fails again the interpretation is discarded, and the report is
generated with that section explicitly marked unavailable. Measured results are
never affected.

Vision interpretation, when enabled, describes a rendered NDVI overlay. It is
prompted to state no numbers, is stored separately, and is never a source of any
value in the report.

## 7. Reporting

Thirteen sections, each tagged with one of four provenance registers that the
report never merges:

| Register | Meaning |
| --- | --- |
| **Observed data** | Measured by deterministic processing |
| **Model prediction** | Classifier output, carrying error |
| **Generated interpretation** | Written by a language model from the evidence |
| **Technical metadata** | Configuration and provenance |

Limitations are always included, and are specific to the run: heavy cloud
masking, seasonally mismatched acquisitions, a low comparable-pixel fraction, a
mean change below the detection threshold, and a high share of low-confidence
pixels are each detected and stated in the report that exhibits them.
