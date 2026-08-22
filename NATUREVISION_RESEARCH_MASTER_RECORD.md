# NatureVision — Research Master Record

**Permanent reference document.** Every numeric result in this file is copied verbatim from generated repository artifacts (`backend/research/results/*/metrics.json`, `metadata.json`, `backend/research/manifests/dataset_manifest.json`, `backend/research/tables/*`). Nothing is estimated, rounded before being stated exactly, or reconstructed from memory. Where a rounded restatement is given, the exact value is given first.

Document generated: 2026-08-19. Repository state: branch `master`, commit `f18316a` (working tree clean at time of inspection). Research artifacts inspected under commits `879e2b0`, `fe6f900`, `2ba284b`, `e2e6187`, `bffb319`, `7276ebe`.

---

## 1. PROJECT OVERVIEW

**NatureVision** is a geospatial environmental-intelligence platform. A user selects a geographic region on a map; the system retrieves real Sentinel-2 Level-2A satellite imagery for that region, computes vegetation indices (NDVI and related indices), detects change between two observation periods, runs a trained land-cover classifier, assembles a structured evidence package, and optionally generates a language-model interpretation of the evidence, validated against the evidence before being returned.

**Problem it solves:** turning raw multispectral satellite reflectance into a defensible, source-traceable environmental report — with an explicit separation between what was *measured* (reflectance, NDVI, change), what was *predicted* (land-cover class, from a trained classifier), and what was *interpreted* (language-model text, validated against the measured/predicted evidence before being shown to a user).

**Why it connects geospatial AI, remote sensing and environmental intelligence:** it is a full vertical slice — STAC-based scene discovery over a public remote-sensing archive (AWS Open Data Sentinel-2), physically-aware raster processing (co-registration, radiometric calibration, quality masking), a supervised ML classifier trained on Sentinel-2 features against a reference land-cover product, and an LLM interpretation layer constrained to the evidence it was given.

### Application contribution vs. research contribution

| | Application contribution | Research contribution |
|---|---|---|
| What | The NatureVision product: FastAPI backend, React frontend, Docker deployment, report generation | The `backend/research/` framework: 11 controlled experiments on spatial/temporal generalization |
| Purpose | Serve real analyses to a user, with scientific-integrity safeguards | Answer a methodological question about how validation protocol affects measured ML performance on this dataset |
| Data | Live Sentinel-2 queries at request time | A frozen, cached, versioned pixel dataset so every experiment reads identical data |
| Output | Analysis records, reports, map layers | Experiment records (`metadata.json`, `metrics.json`), CSV/JSON tables, PNG figures |

**The research question, as stated in `backend/research/README.md`:**

> How well do lightweight machine-learning models trained on Sentinel-2 multispectral data generalize across geographically and temporally distinct environments, and how reliably can their outputs support evidence-grounded environmental interpretation?

---

## 2. ORIGIN OF THE RESEARCH IDEA

This section documents only what the repository and project context directly support; it does not fabricate a narrative beyond that.

Prior technical background applied to this project: Python, machine learning, PyTorch, scikit-learn, feature engineering, ML pipelines, FastAPI, and general AI-systems engineering. The project's initial motivation was applying that background to climate-tech, geospatial analytics, remote sensing and environmental intelligence — an applied engineering project rather than a research project at the outset.

**The transition that produced the research question**, as reflected in the repository structure and `backend/research/README.md`:

```
Applied project (NatureVision platform)
        ↓
Sentinel-2 acquisition + processing pipeline
        ↓
Land-cover ML classifier (random_forest / torch_mlp)
        ↓
Validation methodology question:
  Sentinel-2 pixels at 10 m resolution are strongly spatially
  autocorrelated — neighbouring pixels are highly similar.
  A validation split that ignores geography can place near-duplicate
  pixels in both train and test.
        ↓
Research question: how much does validation protocol change the
measured generalization estimate for this kind of model/data?
        ↓
Controlled experiments (backend/research/), isolating validation
protocol as the single varied factor under an otherwise identical
dataset, feature set, model and seed.
```

This is stated directly in `backend/research/splits.py`:

> "A random pixel-level split would badly overstate accuracy here, because neighbouring 10 m pixels are strongly spatially autocorrelated — a held-out pixel would almost always sit next to a training pixel from the same field or canopy. Holding out whole geographies is the honest protocol."

---

## 3. RESEARCH PROBLEM

**Core problem:** satellite pixels exhibit spatial dependence. Randomly splitting individual pixels into training and test sets can place geographically adjacent, near-identical pixels on both sides of the split. A model can then achieve high test performance by recognizing a specific field, canopy, or water body it effectively already saw during training — not by learning a generalizable land-cover signal. This produces an optimistic estimate of how well the model would perform on genuinely new geography.

**Random pixel split (text diagram):**

```
Region A          Region B          Region C
[T][t][T][t]      [T][t][T][t]      [T][t][T][t]
 T = train pixel, t = test pixel — interleaved within every region.
 Every region contributes to both train and test.
```

**Spatial holdout split:**

```
TRAIN regions: A, B, ... (whole geographies)
TEST region:   D (a wholly separate geography, never seen in training)

[Region A: all train] [Region B: all train] ... | [Region D: all test]
```

**Why this matters for remote sensing specifically:** at 10 m Sentinel-2 resolution, a single field, forest stand, or urban block spans many contiguous pixels with near-identical spectral signatures. A random split effectively performs interpolation within a spatial cluster, not extrapolation to new terrain. A spatial holdout forces the model to classify pixels from land it never trained on — closer to how the model would actually be used (a new, previously unanalysed region).

This repository implements both protocols side by side, under the same dataset, features, model and seeds, so the difference measured between them is attributable to the validation protocol alone (`spatial_validation` experiment, Section 15).

---

## 4. RESEARCH OBJECTIVE

**Primary objective:** evaluate how validation protocol (random pixel split vs. spatially separated holdout) affects the measured generalization performance of lightweight machine-learning models on Sentinel-2 land-cover classification.

**Secondary objectives** (each maps to one implemented experiment):

| Secondary objective | Experiment |
|---|---|
| Compare lightweight model architectures under identical splits | `model_comparison` |
| Measure whether derived spectral indices improve performance over raw bands | `feature_ablation` |
| Measure transfer performance across observation periods (2021 vs 2024) | `temporal_transfer` |
| Measure whether prediction confidence is informative and calibrated | `confidence_analysis` |
| Rank spectral features by contribution to the classifier | `feature_importance` |
| Quantify class imbalance and its effect on headline metrics | `class_imbalance` |
| Relate scene cloud cover to usable pixels and NDVI stability | `cloud_robustness` |
| Measure sensitivity of reported change area to the change-detection threshold | `threshold_sensitivity` |
| Evaluate whether evidence-grounding reduces unsupported claims in LLM output | `llm_grounding` (infrastructure implemented; result blocked — see Section 24) |

---

## 5. RESEARCH QUESTIONS

Extracted verbatim (paraphrased minimally for numbering) from `backend/research/README.md`'s hypothesis table. Each hypothesis (H1–H7) is the question actually implemented; no additional questions are invented here.

| ID | Question / Hypothesis | Why it matters | Experiment | Result (summary) |
|---|---|---|---|---|
| RQ1 (H1) | Does random pixel validation overestimate performance relative to spatial holdout validation? | Determines whether a headline accuracy number reflects genuine geographic generalization or spatial autocorrelation | `spatial_validation` | Yes — random pixel accuracy exceeded spatial holdout accuracy by 0.3088 (Section 15) |
| RQ2 (H2) | Does adding derived spectral indices change land-cover performance relative to raw bands? | Tests whether NDVI/NDWI/NDBI/NBR/BSI-partial add signal beyond the 6 raw reflectance bands | `feature_ablation` | Small effect only: macro-F1 moved from 0.5106 (raw bands) to 0.5150 (full set), Δ ≤ 0.0044 in all cases (Section 17) |
| RQ3 (H3) | Do models lose performance when transferred across observation periods? | Tests temporal generalization independent of geography | `temporal_transfer` | Asymmetric: p2024→p2021 (accuracy 0.5118) markedly worse than p2024→p2024 (0.6382); p2021→p2024 (0.6149) exceeded same-period p2021→p2021 (0.5703) (Section 18) |
| RQ4 (H4) | Do lightweight algorithms differ in spatial-generalization behaviour? | Tests whether model choice, not just validation protocol, affects the spatial-holdout estimate | `model_comparison` | Yes — macro-F1 ranged from 0.5150 (random_forest) to 0.5585 (mlp) under identical spatial-holdout splits (Section 16) |
| RQ5 (H5) | Is prediction confidence informative about correctness? | Determines whether confidence can be trusted to flag likely-wrong predictions | `confidence_analysis` | Confidence separates correct (mean 0.7596) from incorrect (mean 0.5993) predictions, but the model is measurably overconfident (ECE 0.1541) (Section 19) |
| RQ6 (H6) | Does observation degradation (cloud cover) reduce analysis reliability? | Tests whether usable-pixel fraction and NDVI stability degrade with cloud cover | `cloud_robustness` | Valid-pixel fraction fell from 0.9433 (0–5% cloud) to 0.7175–0.7329 (10–30% cloud) across 34 real observations (Section 22) |
| RQ7 (H7) | Does evidence grounding reduce unsupported statements in generated reports? | Tests the production interpretation-validation layer against an ungrounded baseline | `llm_grounding` | **Not measured** — the Groq API returned 401 Unauthorized for every request; 0 generations succeeded in either mode (Section 24) |

---

## 6. DATASET

### Satellite source

- **Sentinel-2 Level-2A** (surface reflectance, atmospherically corrected), discovered via **STAC** (SpatioTemporal Asset Catalog).
- **Provider**: Element84 Earth Search (`https://earth-search.aws.element84.com/v1`), collection `sentinel-2-l2a`, over the AWS Open Data mirror — a public, keyless archive.
- **Access pattern**: windowed reads of cloud-optimized GeoTIFFs (COGs), not full-scene downloads (`app/imagery/` uses `/vsicurl`-style windowed access via rasterio/GDAL).
- **Cloud metadata**: per-scene `cloud_cover_percent` from the STAC item, used both for scene selection and as the independent variable of `cloud_robustness`.
- **Radiometric calibration**: resolved per-observation by a physical probe (small-window read, checks whether the catalogue-declared reflectance offset produces a physically implausible fraction of negative reflectance) rather than trusting the catalogue's declared offset blindly. Recorded per group in the dataset manifest under `radiometric_calibration`.

### Reference labels

- **ESA WorldCover 2021 v200** (10 m resolution, CC BY 4.0 license).
- **Explicitly documented as a reference product, not ground truth.** From `research/config.py`: *"Reference land-cover map used as training and evaluation labels. It is a model product with its own error, not ground truth."* This framing is repeated in `research/README.md`, the dataset manifest's `limitations` field, and `app/models_ml/labels.py`.
- Labels are fixed at the 2021 epoch. Imagery spans two periods (2021, 2024), so genuine land-cover change after 2021 appears as label noise in the 2024 period — documented as a limitation, not corrected for.

### Exact dataset size (current committed artifact — `backend/research/manifests/dataset_manifest.json`)

| Field | Value |
|---|---|
| `dataset_version` | 1.0.0 |
| `total_samples` | **313,580** pixels |
| `cache_sha256` | `8ff45eb8d1752cf8338ecde187b086c46ac1a749206c2430a272ea415cb91ee1` |
| Number of groups (region × period, present) | **23** of 26 expected (13 regions × 2 periods) |
| Regions | 13 |
| Periods | 2 (`p2021`, `p2024`) |
| `samples_per_group` (cap) | 20,000 |
| `sampling_max_dim` | 1024 |
| `max_cloud_cover` (dataset build filter) | 20.0% |
| `build_seed` | 42 |
| `built_at` | 2026-08-17T03:28:22.307168+00:00 |

The 313,580-pixel figure matches what was previously reported; it has been re-verified directly from the current committed manifest, not assumed.

### Missing groups (explicitly recorded as absent, not substituted)

Three (region, period) pairs failed to build and are absent from the manifest rather than filled with synthetic data:

- `nile_delta`, `p2021`
- `sahel_niger`, `p2024`
- `sonoran_desert`, `p2021`

This is why the baseline experiment trains on 7 regions rather than the production model's 10 (Section 14).

### Regions (13, spanning multiple biomes)

| Region key | Name | Biome |
|---|---|---|
| `western_ghats` | Western Ghats, India | Tropical moist broadleaf forest / plantation mosaic |
| `po_valley` | Po Valley, Italy | Intensive irrigated cropland, dense settlement |
| `brandenburg` | Brandenburg, Germany | Temperate mixed forest, lakes, peri-urban |
| `nile_delta` | Nile Delta margin, Egypt | (p2024 only, present) |
| `mato_grosso` | Mato Grosso, Brazil | — |
| `sonoran_desert` | Sonoran Desert | (p2024 only, present) |
| `sahel_niger` | Sahel, Niger | (p2021 only, present) |
| `kalahari_botswana` | Kalahari, Botswana | — |
| `sacramento_valley` | Sacramento Valley, USA | — |
| `finnish_lakeland` | Finnish Lakeland | — (spatial-split validation region) |
| `murray_basin` | Murray Basin, Australia | — (spatial-split test region) |
| `iberian_meseta` | Iberian Meseta, Spain | — (spatial-split test region) |
| `zambezi_miombo` | Zambezi Miombo | — (spatial-split test region) |

### Per-group scene metadata (from `dataset_manifest.json`)

Every group records: `scene_id`, `observation_date`, `cloud_cover_percent`, `platform`, `mgrs_tile`, grid geometry (CRS, width, height, resolution, valid-pixel fraction), the radiometric-calibration decision, and per-group class counts. Example (Western Ghats, 2021):

```
scene_id: S2B_43PFM_20210207_3_L2A
observation_date: 2021-02-07
cloud_cover_percent: 0.000328
platform: sentinel-2b
samples: 13,569
grid: EPSG:32643, 1014×1024, valid_fraction 0.984467
```

### Class distribution (whole dataset, 313,580 pixels — stratified sampling, not natural landscape proportions)

| Class | Count | Proportion |
|---|---:|---:|
| Forest | 80,741 | 0.257481 |
| Agriculture | 80,734 | 0.257459 |
| Water | 64,289 | 0.205016 |
| Urban / built-up | 47,584 | 0.151744 |
| Bare land | 40,232 | 0.128299 |

Dataset-level imbalance ratio (max class ÷ min non-zero class): **2.007** (Section 21 for the derived train/test-split ratios, which differ).

---

## 7. LAND-COVER CLASSES

Extracted from `app/models_ml/labels.py` — five classes, collapsed from eleven ESA WorldCover classes.

| ID | Label | Description (as coded) |
|---|---|---|
| 0 | Forest | Tree-dominated cover, including closed and open canopy |
| 1 | Agriculture | Cropland and managed herbaceous cover, including grassland and shrubland |
| 2 | Water | Permanent open water bodies and herbaceous wetland |
| 3 | Urban / built-up | Impervious and built surfaces |
| 4 | Bare land | Sparsely vegetated or unvegetated ground, including snow and ice |

**WorldCover → this product's classes** (the collapse is lossy — documented consequences):

- Shrubland and grassland are merged into Agriculture, so "Agriculture" here means managed-or-herbaceous cover, not cropland specifically.
- Mangroves are counted as Forest.
- Snow, ice, moss and lichen are counted as Bare land.
- Herbaceous wetland is counted as Water.

---

## 8. FEATURES

### Raw spectral bands (6)

`blue`, `green`, `red`, `nir`, `swir16`, `swir22` — Sentinel-2 reflectance bands used by the production feature contract (`app/imagery/bands.py` → `LAND_COVER_BANDS`).

### Derived indices (5), all normalized differences `(a − b) / (a + b)`

| Index | Formula | Role |
|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation greenness proxy |
| NDWI | (Green − NIR) / (Green + NIR) | Open-water / surface-moisture indicator (McFeeters) |
| NBR | (NIR − SWIR2) / (NIR + SWIR2) | Sensitive to canopy moisture loss / burnt surfaces |
| NDBI | (SWIR1 − NIR) / (SWIR1 + NIR) | Elevated over impervious/built-up surfaces |
| `bsi_partial` | (SWIR1 − Blue) / (SWIR1 + Blue) | Partial bare-soil contrast (named "partial" because it omits the Green/NIR terms of the full Bare Soil Index) |

Full feature vector (11 elements, `app/models_ml/features.py::FEATURE_NAMES`): `blue, green, red, nir, swir16, swir22, ndvi, ndwi, ndbi, nbr, bsi_partial`. `FEATURE_VERSION = "1.0.0"`, versioned and written into every trained model artifact so a model can never be served against a feature contract it was not trained on.

### NDVI — what it is and is not

```
NDVI = (NIR − Red) / (NIR + Red)
```

NDVI **is**: a chlorophyll-sensitive vegetation-greenness proxy — higher values indicate denser, photosynthetically active vegetation (as documented in `app/analysis/indices.py`).

NDVI **is not**, and this repository never claims it measures:
- biodiversity
- biomass
- carbon stock
- habitat quality
- deforestation (directly)
- climate change

Index computation guards divide-by-zero (`DENOMINATOR_EPSILON = 1e-6`) and enforces the physically valid [−1, 1] envelope, masking anything outside it.

---

## 9. DATA PREPROCESSING

Every step below is confirmed present in code (not assumed from documentation alone).

| Step | Implemented? | Where |
|---|---|---|
| STAC-based scene discovery | Yes | `app/imagery/stac.py`, `app/imagery/service.py` |
| Windowed / cloud-optimized raster reads | Yes | rasterio + GDAL env tuning, `AWS_NO_SIGN_REQUEST` |
| Radiometric calibration resolution | Yes | `app/imagery/stac.py::resolve_calibration` — probes a small window, rejects the declared offset if it produces >10% negative reflectance |
| Band co-registration / grid alignment | Yes | All bands resampled onto the **finest-resolution** band present (10 m); 20 m bands (SWIR, SCL) are resampled up |
| Resampling method for categorical data | Yes | SCL is resampled with **nearest-neighbour** specifically (interpolating class codes would invent classes) |
| Cloud masking | Yes | SCL-based, see exact excluded classes below |
| Cloud-shadow masking | Yes | SCL class 3 (cloud shadow) is excluded |
| Cirrus masking | Yes | SCL class 10 (thin cirrus) is excluded |
| Snow/ice masking | Yes | SCL class 11 is excluded |
| General topographic-shadow masking | **No** — SCL class 2 ("Dark area / topographic shadow") is *not* in the excluded set | `app/imagery/bands.py::INVALID_SCL_CLASSES` |
| Invalid/no-data pixel handling | Yes | SCL classes 0 (no data) and 1 (saturated/defective) excluded |
| Reflectance-envelope screening | Yes | Surface reflectance outside [−0.2, 1.6] dropped as saturation/artefact (`docs/methodology.md`) |
| Missing-band handling | Yes | Analyses fail with a specific error if required bands are absent, rather than substituting |
| Feature construction | Yes | Shared function (`stack_features`) used identically by training sampler and inference path |
| Label alignment | Yes | WorldCover pixels co-registered to the sampled Sentinel-2 grid at dataset-build time |

**Excluded SCL classes** (`app/imagery/bands.py::INVALID_SCL_CLASSES`):

| SCL code | Class |
|---|---|
| 0 | No data |
| 1 | Saturated or defective |
| 3 | Cloud shadow |
| 8 | Cloud, medium probability |
| 9 | Cloud, high probability |
| 10 | Thin cirrus |
| 11 | Snow or ice |

An analysis that retains too few valid pixels after masking fails with a specific error rather than reporting statistics from a handful of survivors (`docs/methodology.md`).

---

## 10. MODEL ARCHITECTURE

Four models compared under identical splits, seeds and features (`research/config.py::MODEL_CONFIGS`). Hyperparameters are fixed in advance and never tuned against a test split.

| Model | What it is | Why included | Key hyperparameters (fixed, from config) |
|---|---|---|---|
| `random_forest` | Bagged ensemble of decision trees (scikit-learn `RandomForestClassifier`) | Strong, low-tuning-cost baseline for tabular/pixel features; also the production model | `n_estimators=300`, `max_depth=22`, `min_samples_leaf=4`, `max_features="sqrt"`, `class_weight="balanced_subsample"` |
| `hist_gradient_boosting` | Histogram-based gradient boosting (scikit-learn `HistGradientBoostingClassifier`) | Stronger boosting baseline, fast on hundreds of thousands of rows | `max_iter=200`, `learning_rate=0.1`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.15` |
| `linear_svm` | Linear SVM wrapped in `CalibratedClassifierCV` (sigmoid calibration, 3-fold) for honest probabilities | Linear baseline; full-kernel SVC is O(n²), infeasible at this scale, so a calibrated linear SVM stands in | `C=1.0`, `max_iter=3000`, `class_weight="balanced"` |
| `mlp` | Small feed-forward network (scikit-learn `MLPClassifier`, `StandardScaler` pipeline) | Lightweight neural baseline | hidden sizes `(128, 64)`, dropout 0.15 (config only — `MLPClassifier` does not natively apply dropout, see note below), `learning_rate_init=1e-3`, `weight_decay(alpha)=1e-4`, `batch_size=1024`, `epochs(max_iter)=30`, `early_stopping=True`, `n_iter_no_change=5` |

Output classes for every model: the fixed 5-class space (`CLASS_ORDER` in `app/models_ml/labels.py`), regardless of which classes are present in a given split — confusion matrices always span the full class space so results are comparable across experiments.

**Seed handling:** every randomized model is trained and evaluated at three seeds — 42, 123, 2024 — and reported as mean/std/min/max, never a single run (`research/config.py::SEEDS`).

**Also present in the platform but not one of the four compared models:** `torch_mlp`, a PyTorch neural backend on the same feature contract, selectable in production via `LAND_COVER_BACKEND=torch_mlp`. This is not one of the four research-suite models compared in `model_comparison`; it is a separate production backend option.

---

## 11. VALIDATION STRATEGIES

Implemented in `backend/research/splits.py`.

### Random pixel split (`random_split`)

All eligible pixels are pooled, shuffled with `numpy.random.default_rng(seed)`, and divided into train/val/test fractions (val 0.15, test 0.20 of the pool) without regard to spatial origin. Every geographic block appears in all three partitions by construction. The code comments this explicitly: *"this is exactly the property that makes the estimate optimistic, so it is recorded rather than hidden."*

### Spatial holdout (`spatial_split`)

Whole regions are assigned to test/validation/train. Test regions: `murray_basin`, `iberian_meseta`, `zambezi_miombo` (`SPATIAL_TEST_REGIONS`). Validation region: `finnish_lakeland` (`SPATIAL_VAL_REGIONS`). No test pixel shares a spatial block with any training pixel.

### Temporal split / transfer (`temporal_split`)

Trains on one observation period, tests on another. By default (`spatial_holdout=True`) the transfer is *also* geographic — train excludes the spatial test regions, test uses only the spatial test regions, isolating temporal shift from unfamiliar-ground effects. A `spatial_holdout=False` mode exists to isolate temporal shift alone on the same ground (used for the `p2021→p2021`/`p2024→p2024` same-period rows, which still restrict to the spatial-test-region pixels for comparability).

### Leakage detection (explicit, tested)

- `spatial_leakage(dataset, split)` — returns the set of region blocks present in both train and test. Every spatial-holdout run in this repository asserts this set is empty.
- `temporal_leakage(dataset, split)` — returns periods present in both train and test.
- `index_overlap(split)` — returns the count of sample indices appearing in more than one partition; must always be zero.

These are asserted inside the experiment code (`research/experiments/classification.py::_assert_no_leakage`) and separately unit-tested (`tests/test_research.py`, e.g. `test_spatial_split_has_no_leakage`, `test_spatial_leakage_is_detected_when_introduced` — which deliberately contaminates a split to prove the detector actually fires).

### Why random-pixel performance should not be read as geographic generalization

The random split's own metadata records `blocks_shared_across_partitions: True` for every run — the code does not merely produce a higher number, it records the exact structural reason (shared spatial blocks) that the number is expected to be optimistic. The `spatial_validation` experiment (Section 15) quantifies the resulting gap directly, on identical data/model/seeds.

---

## 12. EXPERIMENTAL DESIGN

**Seeds:** 42, 123, 2024 (`research/config.py::SEEDS`). Used for every randomized experiment; single-seed experiments (`confidence_analysis`, `feature_importance`, `class_imbalance`) are explicitly seed-42-only and reported as such, not disguised as multi-seed.

**Why three seeds:** so a result is reported as a distribution (mean/std/min/max across 3 independent runs) rather than a single number that could be an artefact of one random draw. `research/metrics.py::aggregate` computes this and the code comment states: *"A single seed is never reported as if it expressed uncertainty."*

**Train/test protocols:** random pixel split, spatial holdout split, temporal transfer split (Section 11) — all built from the same cached 313,580-pixel dataset.

**Metrics:** accuracy, balanced accuracy, macro-precision, macro-recall, macro-F1, weighted-F1, per-class precision/recall/F1/support, full confusion matrix (Section 13).

**Repetitions:** 3 seeds × relevant protocol combinations per experiment (e.g. `spatial_validation` runs both protocols at all 3 seeds = 6 total training/evaluation runs).

**Dataset partitions:** see exact per-run sample counts in each experiment section below (train/val/test sizes are recorded per run, not just aggregated).

**Model comparison methodology:** all four models trained under the identical spatial-holdout split (same train/val/test pixel indices) at all three seeds; ranked by macro-F1, with training and inference cost reported alongside so model selection is not reduced to one number.

---

## 13. METRICS

Implemented in `backend/research/metrics.py`, using scikit-learn's `precision_recall_fscore_support`, `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `confusion_matrix`.

| Metric | Definition | Why used |
|---|---|---|
| **Accuracy** | (correct predictions) / (total predictions) | Simple, but misleading alone under class imbalance |
| **Balanced accuracy** | Mean of per-class recall | Weights every class equally regardless of its size — exposes minority-class collapse that accuracy hides |
| **Macro-F1** | Unweighted mean of per-class F1 | `2·(precision·recall)/(precision+recall)`, averaged across classes with equal weight — this repository's primary ranking metric, chosen because 5 classes are imbalanced (Section 21) and macro-F1 does not let majority classes dominate the score |
| **Weighted-F1** | Per-class F1 weighted by class support | Reported alongside macro-F1 for contrast — shows what the score would look like if majority classes were allowed to dominate |
| **Per-class precision/recall/F1/support** | Standard per-class definitions | Published for every headline result so a reader can see exactly which classes are reliable, not just a single number |
| **Confusion matrix** | Full 5×5 count matrix, fixed class order | Always spans the full class space, even for classes absent from a particular split, so matrices are comparable across experiments |
| **Expected Calibration Error (ECE)** | Bucketed |accuracy − mean confidence|, weighted by bucket size | Reported in `confidence_analysis` (Section 19) — quantifies whether predicted probabilities are trustworthy, not just whether predictions are accurate |

**Why macro-F1 matters here specifically:** the dataset has a 2.007:1 majority-to-minority class ratio at the whole-dataset level (up to 4.172:1 within the spatial-holdout test split, Section 21). Accuracy alone would let strong performance on Forest/Agriculture/Water mask weak performance on Bare land; macro-F1 does not allow this.

---

## 14. EXPERIMENT 1 — BASELINE

**File:** `research/experiments/classification.py::run_baseline`. **Output:** `research/results/baseline/`.

**Objective:** reproduce the production model's spatial-holdout accuracy figure inside the research framework, within a pre-declared tolerance, as a sanity check that the research pipeline reflects the production pipeline.

**Method:** `random_forest` (`BASELINE_MODEL`), full feature set `D_full` (`BASELINE_FEATURE_SET`), spatial-holdout split, 3 seeds.

**Published (production) baseline** (`PUBLISHED_BASELINE`, hardcoded in `classification.py` from the production model's own recorded metrics):

| | Accuracy | Macro-F1 | Evaluation samples |
|---|---:|---:|---:|
| Published | 0.5922 | 0.5268 | 82,617 |

**Reproduced (research framework, spatial holdout, 3 seeds):**

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Accuracy | 0.5325 | 0.0024 | 0.5302 | 0.5350 |
| Balanced accuracy | 0.5430 | 0.0022 | 0.5406 | 0.5448 |
| Macro-F1 | 0.5150 | 0.0024 | 0.5124 | 0.5171 |
| Weighted-F1 | 0.5349 | 0.0025 | 0.5325 | 0.5374 |

**Per-seed:**

| Seed | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Test samples |
|---|---:|---:|---:|---:|---:|
| 42 | 0.5302 | 0.5406 | 0.5124 | 0.5325 | 37,647 |
| 123 | 0.5350 | 0.5448 | 0.5171 | 0.5374 | 37,647 |
| 2024 | 0.5323 | 0.5437 | 0.5156 | 0.5349 | 37,647 |

**Gap vs. published:** accuracy_gap = **−0.0597**; macro_f1_gap = **−0.0118**. Declared tolerance: **±0.08** (`BASELINE_TOLERANCE`). `within_tolerance: true`.

**Per-class (aggregated):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Forest | 0.5093 | 0.5339 | 0.5213 | 10,302 |
| Agriculture | 0.4604 | 0.5178 | 0.4874 | 8,221 |
| Water | 0.8561 | 0.4587 | 0.5974 | 11,645 |
| Urban / built-up | 0.4417 | 0.8276 | 0.5760 | 4,688 |
| Bare land | 0.4123 | 0.3805 | 0.3958 | 2,791 |

**Interpretation:** the research framework reproduces the production model's spatial-holdout figures within the declared tolerance. The gap is attributable to a documented, non-arbitrary cause: the production model trains on 10 regions, this research cache has only 7 usable training regions because `nile_delta` (p2021) and `sonoran_desert` (p2021) failed to fetch (Section 6). This is not treated as an unexplained discrepancy — the cause is on record.

**Limitations:** single feature set/model combination; the gap's exact attribution to "missing regions" specifically (vs. other cache differences) has not been isolated by a separate controlled experiment.

**Reproducibility:** deterministic — the experiment was rerun on 2026-08-18 (after the framework's runtime-provenance fix) and reproduced these exact figures bit-for-bit against the same cached dataset and seeds.

---

## 15. EXPERIMENT 2 — SPATIAL VALIDATION (PRIMARY EXPERIMENT)

**File:** `research/experiments/classification.py::run_spatial_validation`. **Output:** `research/results/spatial_validation/`.

**Method:** same dataset, same model (`random_forest`, `D_full` features), same 3 seeds — only the split protocol changes: `random_pixel` vs `spatial_holdout`.

**Exact results:**

| Protocol | Accuracy (mean±std) | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Random pixel | 0.8413 ± 0.0019 | 0.8414 ± 0.0025 | 0.8375 ± 0.0028 | 0.8417 ± 0.0018 |
| Spatial holdout | 0.5325 ± 0.0024 | 0.5430 ± 0.0022 | 0.5150 ± 0.0024 | 0.5349 ± 0.0025 |

**Exact difference (random − spatial), as computed and stored in the artifact:**

| Metric | Difference |
|---|---:|
| Accuracy | **+0.3088** |
| Balanced accuracy | +0.2984 |
| Macro-F1 | **+0.3225** |
| Weighted-F1 | +0.3068 |

**Per-run detail (seed 42, illustrative):**

| Protocol | Accuracy | Macro-F1 | Train n | Val n | Test n | Test blocks |
|---|---:|---:|---:|---:|---:|---:|
| Random pixel | 0.8417 | 0.8374 | 100,023 | 23,082 | 30,776 | 11 |
| Spatial holdout | 0.5302 | 0.5124 | 104,121 | 12,113 | 37,647 | 3 |

**What this demonstrates:** the same dataset, model and seed produce a 0.3088 (accuracy) / 0.3225 (macro-F1) gap purely from changing how the train/test split is constructed. Both splits pool from the same underlying pixels — only which pixels are assigned to test differs.

**Scientifically appropriate framing (used deliberately here, per the repository's own stated stance):**

> Random pixel validation produced substantially higher performance on this dataset, suggesting an optimistic estimate of geographic generalization. This does not establish that random-pixel validation is "wrong" in general, or that this exact gap magnitude generalizes to other datasets, models, or resolutions — only that, for this dataset/model/feature configuration, validation protocol changed the measured result by more than the difference between any two of the four compared model architectures (Section 16).

**Limitations:** three test regions and one validation region is a small sample of the world's landscapes; the spatial-holdout figure characterizes generalization to *these* three withheld regions specifically, not "geography in general."

---

## 16. EXPERIMENT 3 — MODEL COMPARISON

**File:** `research/experiments/classification.py::run_model_comparison`. **Output:** `research/results/model_comparison/`.

**Method:** all four models (Section 10), spatial-holdout split, `D_full` features, 3 seeds each, identical train/test pixel indices across models.

**Exact results (mean ± std across 3 seeds):**

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| `random_forest` | 0.5325 ± 0.0024 | 0.5430 ± 0.0022 | 0.5150 ± 0.0024 | 0.5349 ± 0.0025 |
| `hist_gradient_boosting` | 0.5544 ± 0.0027 | 0.5702 ± 0.0013 | 0.5461 ± 0.0013 | 0.5592 ± 0.0033 |
| `linear_svm` | 0.5513 ± 0.0002 | 0.5750 ± 0.0002 | 0.5499 ± 0.0002 | 0.5545 ± 0.0002 |
| `mlp` | 0.5506 ± 0.0114 | 0.5952 ± 0.0107 | 0.5585 ± 0.0101 | 0.5518 ± 0.0099 |

**Ranking by macro-F1 (`ranked_by_macro_f1`, as stored):** `mlp` > `linear_svm` > `hist_gradient_boosting` > `random_forest`.

**Per-seed, with training/inference cost:**

| Model | Seed | Accuracy | Macro-F1 | Train (s) | Inference (s) |
|---|---:|---:|---:|---:|---:|
| random_forest | 42 | 0.5302 | 0.5124 | 14.743 | 0.7057 |
| random_forest | 123 | 0.5350 | 0.5171 | 17.016 | 0.2094 |
| random_forest | 2024 | 0.5323 | 0.5156 | 16.440 | 0.6947 |
| hist_gradient_boosting | 42 | 0.5513 | 0.5468 | 5.229 | 0.4478 |
| hist_gradient_boosting | 123 | 0.5554 | 0.5446 | 6.569 | 1.6737 |
| hist_gradient_boosting | 2024 | 0.5564 | 0.5469 | 4.181 | 0.4270 |
| linear_svm | 42 | 0.5515 | 0.5501 | 4.440 | 0.0277 |
| linear_svm | 123 | 0.5513 | 0.5498 | 6.614 | 0.0272 |
| linear_svm | 2024 | 0.5512 | 0.5497 | 2.748 | 0.0246 |
| mlp | 42 | 0.5385 | 0.5472 | 11.638 | 0.0176 |
| mlp | 123 | 0.5521 | 0.5618 | 12.317 | 0.0167 |
| mlp | 2024 | 0.5612 | 0.5665 | 10.654 | 0.0173 |

**Best-performing model (macro-F1, spatial-holdout protocol only):** `mlp` (0.5585 mean), but with markedly higher seed-to-seed variance (std 0.0101 vs. 0.0002–0.0024 for the others) — a materially less stable ranking than the number alone suggests. `linear_svm` is a close second (0.5499) with the lowest variance of the non-tree models.

**Note recorded in the artifact:** *"Ranking is by macro-F1 rather than accuracy, and training and inference cost are reported alongside, so model selection is not reduced to a single number."*

---

## 17. EXPERIMENT 4 — FEATURE ABLATION

**File:** `research/experiments/classification.py::run_feature_ablation`. **Output:** `research/results/feature_ablation/`.

**Feature sets** (`research/config.py::FEATURE_SETS`, verified verbatim):

| Set | Features (n) | Contents |
|---|---:|---|
| A — raw bands | 6 | blue, green, red, nir, swir16, swir22 |
| B — bands + NDVI | 7 | A + ndvi |
| C — bands + indices | 9 | B + ndwi, ndbi |
| D — full | 11 | C + nbr, bsi_partial |

**Model held fixed:** `random_forest`, spatial-holdout split, 3 seeds.

**Exact results:**

| Feature set | n features | Accuracy (mean) | Macro-F1 (mean) | Macro-F1 std | Δ macro-F1 vs. D (full) |
|---|---:|---:|---:|---:|---:|
| A — raw bands | 6 | 0.5217 | 0.5106 | 0.0005 | −0.0044 |
| B — bands + NDVI | 7 | 0.5263 | 0.5134 | 0.0009 | −0.0016 |
| C — bands + indices | 9 | 0.5366 | 0.5156 | 0.0023 | **+0.0006** |
| D — full | 11 | 0.5325 | 0.5150 | 0.0024 | 0.0000 |

**Interpretation:** derived spectral indices produced only small differences relative to raw bands — the largest macro-F1 delta across the four sets is 0.0044 (set A vs. D), and set C (9 features, no NBR/BSI) very slightly *exceeded* the full 11-feature set D on macro-F1 (+0.0006), though this is within noise given the ~0.002–0.003 seed-to-seed std observed elsewhere. **This does not support a claim that derived indices materially improve performance in this configuration.** It also does not support a claim that they are useless — set B and C both beat set A, and NDVI is independently the single most important feature by both impurity and permutation importance (Section 20). The correct reading: adding indices redistributes information already partly present in the raw bands (NDVI is a fixed transform of NIR and Red, both already in set A) more than it adds new information.

**Limitations:** ablation was tested only with `random_forest`; other models were not ablated identically.

---

## 18. EXPERIMENT 5 — TEMPORAL TRANSFER

**File:** `research/experiments/classification.py::run_temporal_transfer`. **Output:** `research/results/temporal_transfer/`.

**Method:** train on one period, test on another, spatial holdout applied by default (train excludes spatial-test regions; test uses only spatial-test regions) — 3 seeds per train→test combination.

**Exact results (mean ± std across 3 seeds):**

| Train → Test | Kind | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| p2021 → p2021 | same-period | 0.5703 ± 0.0007 | 0.5696 ± 0.0007 | 0.5506 ± 0.0007 | 0.5660 ± 0.0007 |
| p2021 → p2024 | forward transfer | 0.6149 ± 0.0021 | 0.5910 ± 0.0014 | 0.5794 ± 0.0016 | 0.6146 ± 0.0024 |
| p2024 → p2024 | same-period | 0.6382 ± 0.0008 | 0.6338 ± 0.0008 | 0.6076 ± 0.0008 | 0.6557 ± 0.0008 |
| p2024 → p2021 | backward transfer | 0.5118 ± 0.0020 | 0.5537 ± 0.0016 | 0.5111 ± 0.0018 | 0.5139 ± 0.0023 |

**Asymmetry observed:** transfer is *not* symmetric.
- p2021→p2024 (0.6149 acc) actually **exceeds** the p2021→p2021 same-period figure (0.5703 acc), by +0.0446.
- p2024→p2021 (0.5118 acc) is **worse** than p2024→p2024 same-period (0.6382 acc), by −0.1264 — the largest gap in the table.

**Reading, staying within what was tested:** the two periods are not interchangeable — training on 2024 and testing on 2021 costs far more accuracy than the reverse direction. Because labels are fixed at the 2021 WorldCover epoch (Section 6), the 2024 imagery is being scored against increasingly stale labels wherever real land cover changed since 2021 — this experiment **cannot separate genuine temporal domain shift from 2021-label staleness**, and the repository does not claim to have separated them. This result is reported only for the two tested periods (Jan–Apr windows of 2021 and 2024) over the same 13 regions; it does not generalize to other years or seasons untested here.

---

## 19. EXPERIMENT 6 — CONFIDENCE ANALYSIS

**File:** `research/experiments/classification.py::run_confidence_analysis`. **Output:** `research/results/confidence_analysis/`. **Single seed (42)** — explicitly not averaged across seeds.

**Base run:** `random_forest`, spatial holdout, seed 42 (accuracy 0.5302, macro-F1 0.5124 — identical to the baseline's seed-42 row, Section 14).

**Confidence buckets:**

| Bucket | n | Accuracy | Mean confidence | Gap (conf − acc) |
|---|---:|---:|---:|---:|
| 0.0–0.2 | 0 | — | — | — |
| 0.2–0.4 | 2,524 | 0.2096 | 0.3472 | 0.1376 |
| 0.4–0.6 | 11,736 | 0.3611 | 0.5054 | 0.1442 |
| 0.6–0.8 | 11,492 | 0.5049 | 0.6991 | 0.1942 |
| 0.8–1.0 | 11,895 | 0.7897 | 0.9182 | 0.1285 |

**Correct vs. incorrect predictions:**

| | Mean confidence |
|---|---:|
| Correct predictions | 0.7596 |
| Incorrect predictions | 0.5993 |

**Expected Calibration Error: 0.1541.** Every populated bucket has a **positive** gap — the model is overconfident in every bucket, not just on average.

**Interpretation:** confidence does separate correct from incorrect predictions (0.7596 vs. 0.5993 — a real, exploitable signal), and `confidence_separates_correct_and_incorrect: true` is recorded in the artifact. However, the model's stated probabilities are **not well calibrated** — a prediction at "0.70 confidence" is right only about 50% of the time in the 0.6–0.8 bucket (accuracy 0.5049 in that bucket). This is stated as poor calibration, not glossed over.

**Top confusions** (from the same run, illustrative — full list in `table09b_top_confusions`):

| True → Predicted | Count | Share of errors | Mean confidence |
|---|---:|---:|---:|
| Water → Forest | 4,390 | 24.82% | 0.5946 |
| Forest → Agriculture | 3,351 | 18.95% | 0.6612 |
| Agriculture → Urban/built-up | 2,372 | 13.41% | 0.5536 |
| Bare land → Urban/built-up | 1,596 | 9.02% | 0.6286 |
| Water → Agriculture | 1,276 | 7.22% | 0.5972 |

---

## 20. EXPERIMENT 7 — FEATURE IMPORTANCE

**File:** `research/experiments/classification.py::run_feature_importance`. **Output:** `research/results/feature_importance/`. **Single seed (42).**

**Impurity importance** (Gini importance from the trained `random_forest`):

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | ndvi | 0.13811 |
| 2 | nir | 0.12034 |
| 3 | swir16 | 0.11784 |
| 4 | ndwi | 0.09980 |
| 5 | bsi_partial | 0.09851 |
| 6 | red | 0.09663 |
| 7 | swir22 | 0.08182 |
| 8 | green | 0.07936 |
| 9 | blue | 0.06672 |
| 10 | nbr | 0.05321 |
| 11 | ndbi | 0.04766 |

**Permutation importance** (computed on the validation block, not the test block — `research/README.md` safeguard):

| Rank | Feature | Importance | Std |
|---:|---|---:|---:|
| 1 | ndvi | 0.09603 | 0.00137 |
| 2 | swir16 | 0.06883 | 0.00207 |
| 3 | bsi_partial | 0.04432 | 0.00167 |
| 4 | ndwi | 0.03328 | 0.00122 |
| 5 | blue | 0.02245 | 0.00091 |
| 6 | nir | 0.01617 | 0.00072 |
| 7 | green | 0.01228 | 0.00099 |
| 8 | ndbi | 0.01025 | 0.00092 |
| 9 | red | −0.00213 | 0.00036 |
| 10 | swir22 | −0.01212 | 0.00144 |
| 11 | nbr | −0.01318 | 0.00054 |

**Interpretation:** NDVI ranks 1st by both methods — the single most influential feature by a clear margin under permutation importance. SWIR-1 (`swir16`) ranks highly by both methods (3rd impurity, 2nd permutation). `red` and `swir22` have small **negative** permutation importance, meaning permuting them did not measurably hurt (or slightly helped) held-out performance in this run — consistent with feature redundancy (their information is partly captured elsewhere, e.g. via NDVI/NDBI which are derived from correlated bands), not evidence they are actively harmful in general.

**Feature importance is not causality.** These rankings describe which features this specific trained model relied on to reduce impurity / preserve validation accuracy — they do not establish that NDVI *causes* correct land-cover classification, only that it correlates strongly with the label in this dataset and this model's use of it.

---

## 21. EXPERIMENT 8 — CLASS IMBALANCE

**File:** `research/experiments/classification.py::run_class_imbalance`. **Output:** `research/results/class_imbalance/`. **Single seed (42).**

**Class distribution and imbalance ratio at three stages:**

| Stage | Total | Imbalance ratio (max/min class) |
|---|---:|---:|
| Whole dataset | 313,580 | 2.007 |
| Train split | 104,121 | 1.783 |
| Test split (spatial holdout) | 37,647 | **4.172** |

**Test-split class counts** (spatial holdout, seed 42):

| Class | Count | Proportion |
|---|---:|---:|
| Water | 11,645 | 30.93% |
| Forest | 10,302 | 27.36% |
| Agriculture | 8,221 | 21.84% |
| Urban / built-up | 4,688 | 12.45% |
| Bare land | 2,791 | 7.41% |

**Headline metrics (same run as baseline seed 42):** accuracy 0.5302, balanced accuracy 0.5406, macro-F1 0.5124, weighted-F1 0.5325.

**Accuracy − balanced accuracy: −0.0104.**

**Interpretation:** the negative sign (accuracy *below* balanced accuracy) means accuracy is not artificially inflated by the majority classes in this particular split — if anything, balanced accuracy is marginally higher, indicating per-class recall is relatively even rather than concentrated on the largest classes. The gap's small magnitude (−0.0104) means this particular split does not show severe accuracy-vs-balanced-accuracy divergence, even though the raw class-count imbalance ratio in the test split (4.172) is nontrivial. This is exactly why macro-F1/balanced accuracy are reported alongside accuracy throughout this framework: accuracy alone can be misleading under imbalance, and here the framework's own metrics show the direction and size of that effect rather than assuming it.

---

## 22. EXPERIMENT 9 — CLOUD ROBUSTNESS

**File:** `research/experiments/robustness.py::run_cloud_robustness`. **Output:** `research/results/cloud_robustness/`. **No model/seed** — this experiment operates on raw imagery, not the cached pixel table.

**Setup:** real Sentinel-2 scenes searched over 3 regions (`western_ghats`, `po_valley`, `brandenburg`) across 2021, up to 30% cloud cover, capped at 12 scenes per region ordered by cloud cover so the sample spans buckets. **34 real observations retrieved.**

**Tested conditions / results:**

| Cloud bucket | n observations | Status | Mean valid-pixel fraction | Mean masked fraction | Mean NDVI | Std of mean NDVI |
|---|---:|---|---:|---:|---:|---:|
| 0–5% | 21 | reported | 0.9433 | 0.0271 | 0.5506 | 0.1488 |
| 5–10% | 7 | reported | 0.8680 | 0.0975 | 0.6207 | 0.1213 |
| 10–20% | 3 | reported | 0.7175 | 0.2233 | 0.4041 | 0.3217 |
| 20–30% | 3 | reported | 0.7329 | 0.2074 | 0.5746 | 0.1669 |

All four buckets met the minimum-observations threshold (2) and were reported — none were skipped.

**Interpretation:** valid-pixel fraction fell from 0.9433 (lowest-cloud bucket) to 0.7175–0.7329 (higher-cloud buckets) — a clear, monotonic-ish drop supporting H6. Mean NDVI does *not* move monotonically with cloud bucket (0.55 → 0.62 → 0.40 → 0.57) — but this is not interpreted as contradicting H6, because NDVI differences across scenes/dates also reflect genuine seasonal and regional variation (documented limitation, not model error).

**Scene-level vs. pixel-level distinction:** `cloud_cover_percent` is a **scene-level** STAC metadata field describing the whole tile, not the analysed region specifically — a low-cloud scene can still be cloudy over the exact area of interest, and vice versa. Pixel-level masking (Section 9) is the mechanism that actually removes bad pixels; the scene-level cloud figure is only used for scene *selection* and as the bucketing variable here.

---

## 23. EXPERIMENT 10 — THRESHOLD SENSITIVITY

**File:** `research/experiments/robustness.py::run_threshold_sensitivity`. **Output:** `research/results/threshold_sensitivity/`. **No model/seed** — operates on real bi-temporal scene pairs.

**Change-detection threshold:** the absolute NDVI-difference cutoff above which a pixel is classified as "changed" (moderate) vs. "significantly changed" (2× the moderate threshold). Production defaults (0.10 / 0.20) are unchanged by this experiment; this is a research-only sweep at 4 thresholds: **0.05, 0.10, 0.15, 0.20** (`research/config.py::CHANGE_THRESHOLDS`).

**Exact results — changed area (% of comparable pixels) at each threshold, per region:**

| Region | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---:|---:|---:|---:|
| Western Ghats | 18.832% | 5.131% | 2.312% | 1.164% |
| Po Valley | 58.875% | 36.477% | 24.257% | 16.799% |
| Brandenburg | 27.448% | 9.416% | 4.186% | 1.892% |

**Example full row (Western Ghats, threshold 0.05):** scenes `S2B_43PFM_20210207_3_L2A` (2021) vs `S2B_43PFM_20240123_0_L2A` (2024), 143,100 comparable pixels, 18.832% changed area (73.499876 km²), 5.521% decreased, 13.311% increased, mean NDVI change 0.015373.

**Sensitivity/tradeoff:** reported changed area is highly sensitive to threshold choice — for Po Valley, changed area falls from 58.875% at 0.05 to 16.799% at 0.20, a >3.5× swing from threshold alone, on the exact same underlying imagery pair. A lower threshold reports more area as "changed" but includes more noise-level NDVI fluctuation; a higher threshold is more conservative but will miss smaller genuine changes. Threshold choice is not a neutral technical detail — it materially determines the headline "% area changed" figure a report would show.

**Explicit distinction (repository's own framing, `robustness.py` limitations field):** *"Reported change is change in a reflectance-derived vegetation index. It does not establish land-cover conversion, and no causal process is attributed to it."* NDVI change-detection ≠ deforestation or any other specific land-cover-conversion event; it is a change in a spectral index, which deforestation, seasonal phenology, agricultural cycles, or cloud/shadow artefacts can all produce.

---

## 24. EXPERIMENT 11 — LLM GROUNDING

**File:** `research/experiments/grounding.py::run_grounding_evaluation`. **Output:** `research/results/llm_grounding/`.

**Status: BLOCKED.** This experiment produced **zero valid generations** in either mode.

**Exact recorded result:**

| Mode | n_generated | n_rejected | Note |
|---|---:|---:|---|
| Baseline (Mode A) | 0 | 3 | "No output in this mode passed schema validation." |
| Grounded (Mode B) | 0 | 3 | "No output in this mode passed schema validation." |

All 6 individual generation attempts (3 cases × 2 modes) recorded `"status": "rejected"`, `"error": "InterpretationProviderError"`, `"detail": "The interpretation provider rejected the request."` The underlying cause, independently confirmed by directly probing the Groq API during this project's development: the configured `GROQ_API_KEY` returns **HTTP 401 Unauthorized** from `https://api.groq.com/openai/v1/models`. **No LLM evaluation result exists for H7.** This is reported as a blocked infrastructure test, not a negative result and not a positive one.

**Intended methodology** (implemented and testable once a valid key is available):

- **Two generation modes on the same evidence packages**, from real completed analyses (`build_cases_from_database`) or a saved case file (`manifests/grounding_cases.json`):
  - **Mode A (baseline):** a minimal prompt with no grounding constraints, output not validated.
  - **Mode B (grounded):** the production system prompt (`app/interpretation/prompts.py`) plus the production validators (`app/interpretation/validation.py`), unchanged from what real users receive.
- Both outputs are scored by the same deterministic checker (`score_output`), so the comparison isolates the effect of grounding, not the effect of scoring.
- Scoring checks: **unsupported numerical claims** (numbers in the LLM text not traceable to the evidence package), **invented source IDs** (scene identifiers not present in `data_sources`), and causal-overreach phrase screening.
- The production validator is explicitly never weakened to make Mode B look better (code comment, `grounding.py`).

**What is implemented vs. what has an experimental result:**

| | Status |
|---|---|
| Evidence-package structured extraction | Implemented (`app/interpretation/evidence.py`) |
| Dual-mode prompt/generation infrastructure | Implemented (`research/experiments/grounding.py`) |
| Grounding validator (numeric + source-ID checking) | Implemented and unit-tested (`tests/test_research.py::test_grounding_scorer_flags_unsupported_numbers`, `test_grounding_scorer_detects_invented_source_ids`) |
| **Successfully evaluated H7 result** | **Not obtained** — API key rejected by provider |

No numbers were fabricated to fill this gap. This is the correct, honest state of Experiment 11.

---

## 25. CENTRAL RESEARCH FINDING

**Random pixel validation substantially outperformed spatial holdout validation on identical data, model and seeds.**

Exact numbers (`research/results/spatial_validation/metrics.json`):

| | Random pixel | Spatial holdout | Difference |
|---|---:|---:|---:|
| Accuracy | 0.8413 | 0.5325 | **+0.3088** |
| Macro-F1 | 0.8375 | 0.5150 | **+0.3225** |

**What this can support:** for this dataset (313,580 Sentinel-2 pixels, 13 regions, 2 periods), this feature set (11 features), and this model (`random_forest`), measured performance is highly sensitive to validation protocol — the same trained-model-family evaluated on the same underlying pixel pool produces an accuracy figure that differs by more than 30 percentage points depending solely on whether the test set shares geography with the training set. The gap (+0.3225 macro-F1) exceeds the entire spread across four different model architectures under the (harder) spatial-holdout protocol (0.5150 to 0.5585, a 0.0435 range — Section 16). Validation protocol mattered more than model choice, in this experiment.

**What this cannot support:**
- That this exact magnitude (0.3088 / 0.3225) generalizes to other datasets, resolutions, sensors, or classifiers.
- That random-pixel validation is invalid in every remote-sensing context — only that it was measurably optimistic here.
- That the spatial-holdout number (0.5150 macro-F1) represents the ceiling of achievable geographic generalization — only three held-out regions were tested, and other regions might perform better or worse.
- Any claim about production system accuracy in the field — the production system uses its own held-out evaluation (Section 14), which this experiment reproduces within tolerance but does not replace.

---

## 26. ALL OTHER IMPORTANT FINDINGS

Every bullet below has a cited source in this document.

- **Baseline reproduction is within tolerance**: accuracy_gap −0.0597, macro_f1_gap −0.0118, against a declared ±0.08 tolerance (Section 14).
- **Model choice affects the spatial-holdout estimate**: macro-F1 ranges from 0.5150 (`random_forest`) to 0.5585 (`mlp`) under an identical spatial-holdout split (Section 16), with `mlp` showing the highest seed-to-seed variance (std 0.0101).
- **Derived spectral indices produced only small gains over raw bands**: macro-F1 deltas across feature sets A→D are all ≤ 0.0044 in magnitude; set C (9 features) marginally exceeded the full 11-feature set D (Section 17).
- **NDVI is the single most important classifier feature** by both impurity (0.1381, rank 1) and permutation importance (0.0960, rank 1) (Section 20).
- **Temporal transfer is asymmetric**: p2021→p2024 (0.6149 acc) exceeds same-period p2021→p2021 (0.5703); p2024→p2021 (0.5118 acc) is 0.1264 below same-period p2024→p2024 (0.6382) (Section 18).
- **Prediction confidence separates correct from incorrect predictions** (mean 0.7596 vs. 0.5993) but is **measurably overconfident** — ECE 0.1541, with a positive (overconfident) gap in every populated bucket (Section 19).
- **Class imbalance ratio grows under spatial holdout**: 2.007 (whole dataset) → 4.172 (spatial-holdout test split specifically), yet accuracy − balanced accuracy is only −0.0104 in that split (Section 21).
- **Usable-pixel fraction drops with scene cloud cover**: 0.9433 (0–5% cloud) → 0.7175–0.7329 (10–30% cloud), across 34 real observations (Section 22).
- **Reported changed area is highly sensitive to the NDVI change threshold**: Po Valley's reported changed area falls from 58.875% to 16.799% (a >3.5× change) moving the threshold from 0.05 to 0.20 NDVI units on the exact same scene pair (Section 23).
- **LLM grounding evaluation (H7) is unmeasured**, blocked by a provider authentication failure (Section 24) — listed here explicitly as a non-finding, not omitted.

---

## 27. FIGURES

All in `backend/research/figures/` (13 PNG files).

| Filename | Shows | Experiment | Supports |
|---|---|---|---|
| `fig01_pipeline.png` | Overview diagram of the research pipeline stages | (framework-level) | Section 30 |
| `fig02_study_regions.png` | Map of the 13 study regions, coloured by spatial-split assignment (train/val/test) | `spatial_validation` | Section 15 / 11 |
| `fig02b_class_distribution.png` | Class distribution bar chart over the cached dataset | (dataset-level) | Section 6 |
| `fig03_random_vs_spatial.png` | Grouped bars: random-pixel vs. spatial-holdout metrics | `spatial_validation` | Section 15 (central finding) |
| `fig04_model_comparison.png` | Grouped bars: all four models' headline metrics | `model_comparison` | Section 16 |
| `fig05_feature_ablation.png` | Macro-F1 across feature sets A–D | `feature_ablation` | Section 17 |
| `fig06_temporal_transfer.png` | Accuracy across the four train→test period combinations | `temporal_transfer` | Section 18 |
| `fig07_cloud_robustness.png` | Valid-pixel/masked-fraction by cloud bucket | `cloud_robustness` | Section 22 |
| `fig08_threshold_sensitivity.png` | Changed-area line chart vs. threshold, per region | `threshold_sensitivity` | Section 23 |
| `fig09_feature_importance.png` | Ranked feature importances | `feature_importance` | Section 20 |
| `fig10_confusion_random_pixel.png` | 5×5 confusion matrix, random-pixel protocol | `spatial_validation` | Section 15 |
| `fig10_confusion_spatial_holdout.png` | 5×5 confusion matrix, spatial-holdout protocol | `spatial_validation` | Section 15 |
| `fig11_confidence_reliability.png` | Reliability diagram (confidence vs. accuracy per bucket) | `confidence_analysis` | Section 19 |

---

## 28. TABLES

All in `backend/research/tables/` (CSV + matching JSON per table; per-experiment copies also live under `results/<experiment>/tables/`).

| Filename (`.csv`/`.json`) | Contents | Experiment | Purpose |
|---|---|---|---|
| `experiment_runs` | Run ledger: status/runtime/output path per experiment | (framework-level) | Reproducibility ledger — **see Section 29 discrepancy note** |
| `smoke_checks` | Pass/fail of the 12 structural CI checks | `smoke` | CI gate |
| `table02_class_distribution` | Class counts/proportions | (dataset-level) | Section 6 / 21 |
| `table03_model_comparison` | Aggregated per-model metrics | `model_comparison` | Section 16 |
| `table03a_model_runs` | Per-seed, per-model rows with train/inference cost | `model_comparison` | Section 16 |
| `table04_random_vs_spatial_runs` | All 6 individual protocol×seed runs | `spatial_validation` | Section 15 |
| `table04a_baseline_per_seed` | Baseline per-seed rows | `baseline` | Section 14 |
| `table04b_random_vs_spatial_summary` | Aggregated summary + difference | `spatial_validation` | Section 15 (central finding) |
| `table05_feature_ablation` | Aggregated per-feature-set metrics | `feature_ablation` | Section 17 |
| `table05a_feature_ablation_runs` | Per-seed, per-feature-set rows | `feature_ablation` | Section 17 |
| `table06_temporal_transfer` | Per-run temporal transfer rows | `temporal_transfer` | Section 18 |
| `table07_cloud_buckets` | Aggregated cloud-bucket statistics | `cloud_robustness` | Section 22 |
| `table07a_cloud_observations` | All 34 individual scene observations | `cloud_robustness` | Section 22 |
| `table08_threshold_sensitivity` | Per-region, per-threshold change-detection rows | `threshold_sensitivity` | Section 23 |
| `table09_confidence_buckets` | Confidence-bucket accuracy/calibration rows | `confidence_analysis` | Section 19 |
| `table09b_top_confusions` | Top confusion pairs with counts and confidence | `confidence_analysis` | Section 19 |
| `table10_llm_grounding` | Per-generation grounding attempt records (all rejected) | `llm_grounding` | Section 24 |
| `table11_feature_importance` | Impurity-importance ranking | `feature_importance` | Section 20 |
| `table11b_permutation_importance` | Permutation-importance ranking | `feature_importance` | Section 20 |

Note: `table11*` was previously numbered `table08*`, which collided with `table08_threshold_sensitivity`. Renumbered to `table11*` in commit `fe6f900` — current filenames as listed above are correct.

---

## 29. REPRODUCIBILITY

**Environment setup:**

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
```

**Commands (from `research/run.py`, verified against the current CLI):**

```powershell
python -m research.run --list                # show all experiments and groupings
python -m research.run --experiment smoke     # fast structural check — no network
python -m research.run --experiment dataset   # (re)build the pixel cache — network
python -m research.run --experiment core      # 8 experiments, cached data only — no network
python -m research.run --experiment network   # cloud_robustness, threshold_sensitivity, llm_grounding — network
python -m research.run --experiment all       # core + network
python -m research.run --experiment baseline  # any single experiment by name
```

**Which experiments need the network:** `cloud_robustness`, `threshold_sensitivity`, `llm_grounding` (`NETWORK` tuple in `run.py`) — these fetch real Sentinel-2 scenes or call the Groq API at run time. The other 8 (`baseline`, `spatial_validation`, `model_comparison`, `feature_ablation`, `temporal_transfer`, `confidence_analysis`, `feature_importance`, `class_imbalance` — the `CORE` tuple) run entirely from the cached pixel table with no network access.

**Measured runtimes** (most recently recorded per-experiment `metadata.json`, in seconds; two source generations exist — see discrepancy note below):

| Experiment | Runtime (s), most recent per-experiment `metadata.json` |
|---|---:|
| baseline | 98.97 |
| spatial_validation | 76.88 |
| model_comparison | 119.53 |
| feature_ablation | 192.43 |
| temporal_transfer | 683.48 |
| confidence_analysis | 51.62 |
| feature_importance | 69.04 |
| class_imbalance | 47.99 |
| cloud_robustness | 122.45 |
| threshold_sensitivity | 51.55 |
| llm_grounding | 9.91 |
| **Approximate full `core` suite** | ~15–20 minutes, run-to-run variable |

**⚠️ Discrepancy, explicitly flagged rather than silently resolved:** `research/tables/experiment_runs.csv` (the top-level run ledger) currently shows an **earlier** set of runtimes for `baseline` (172.28s), `spatial_validation` (310.26s), `model_comparison` (530.29s) and `feature_ablation` (494.86s), all timestamped 2026-08-17 20:xx UTC. The four experiments' own `results/<name>/metadata.json` files were subsequently updated on 2026-08-18 (commits `e2e6187`, `bffb319`) with shorter runtimes and no ledger update accompanying them — consistent with those four having been rerun individually outside the `run_named` CLI path that normally refreshes the ledger. **The measured metric *values* (accuracy/macro-F1/etc.) are identical between both generations — verified directly against the current committed `metrics.json` files** — so this is a metadata/timing discrepancy only, not a results discrepancy. The per-experiment `metadata.json` files are the authoritative runtime source; the ledger CSV is stale for these four rows.

**Where results are written:** `research/results/<experiment>/` (config.json, metadata.json, metrics.json, tables/, figures/), plus shared copies in `research/tables/` and `research/figures/`, plus `research/manifests/` for the dataset manifest and grounding cases.

**Seed/configuration control:** every experiment record's `metadata.json` stores `research_version`, `dataset_version`, `git_commit`, `seeds`, `config_hash` (SHA-256 of the exact configuration used), `dataset_manifest_hash`, `dataset_cache_sha256`, contributing `scene_ids`, `label_source`, `python`/`platform` versions, and (as of commit `fe6f900`) `runtime_seconds`. This is what makes any single number in a table traceable back to the exact code, config and data that produced it.

**Test suite:** 161 backend tests total (`pytest -q`), of which **31** are in `tests/test_research.py` specifically covering split integrity (leakage detection, reproducibility across seeds), metric correctness, and experiment-record completeness.

---

## 30. RESEARCH SOFTWARE ARCHITECTURE

`backend/research/` package, module by module:

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for seeds, feature sets, model hyperparameters, periods, cloud buckets, change thresholds, confidence buckets, and the versioned label source. A run is fully described by this module plus a seed. |
| `dataset.py` | Builds the cached pixel dataset from real Sentinel-2 scenes + WorldCover labels (`build_dataset`), loads it (`load_dataset`), and exposes `ResearchDataset` with `.columns()`/`.mask_for()` helpers. Caches to `research/cache/pixels.npz` + a written manifest. |
| `splits.py` | `random_split`, `spatial_split`, `temporal_split`, plus `spatial_leakage`, `temporal_leakage`, `index_overlap` leakage detectors. |
| `models.py` | `FittedModel` dataclass, `fit_model` (builds and trains one configured model), `feature_importances`. Hyperparameters come only from `config.py`; nothing is tuned inside this module. |
| `metrics.py` | `evaluate` (single-run metrics), `aggregate` (mean/std/min/max across seeds), `class_distribution`, `confidence_analysis`. |
| `artifacts.py` | `ExperimentRecord` (the self-describing run record), `git_commit`, `config_hash`, `file_sha256`, `write_table`, `write_confusion`. |
| `figures.py` | Shared matplotlib (Agg backend) styling and the specific figure-generation functions listed in Section 27. |
| `experiments/classification.py` | `run_baseline`, `run_spatial_validation`, `run_model_comparison`, `run_feature_ablation`, `run_temporal_transfer`, `run_confidence_analysis`, `run_feature_importance`, `run_class_imbalance`, plus `_assert_no_leakage`. |
| `experiments/robustness.py` | `run_cloud_robustness`, `run_threshold_sensitivity` — the two imagery-level (network-requiring) experiments. |
| `experiments/grounding.py` | `run_grounding_evaluation`, `score_output`, `build_cases_from_database` — the LLM grounding evaluation (Section 24). |
| `run.py` | CLI entrypoint: `--list`, `--experiment {core,network,all,smoke,dataset,<name>}`, the run ledger (`_merged_ledger`), and the smoke-test structural checks used by CI. |

**Interaction with the production application:** the research layer imports and calls production modules directly — `app.analysis.change`, `app.analysis.indices`, `app.analysis.statistics`, `app.imagery.service`, `app.imagery.stac`, `app.models_ml.labels`, `app.interpretation.*` — through their normal public interfaces. It does not fork, duplicate, or monkeypatch production logic. **The research framework does not modify production behaviour**: production defaults (e.g. the 0.10/0.20 change thresholds) are left untouched even while `threshold_sensitivity` sweeps other values in its own research-only code path; production settings, database, and API routes are entirely unaffected by anything in `backend/research/`.

---

## 31. APPLICATION ARCHITECTURE

**Frontend:** React 19, TypeScript, Vite, MapLibre GL (map rendering), Recharts (charts). Client-side routing via `window.history.pushState` (`frontend/src/app/router.tsx`), pages: Landing, Explore, Workspace (analysis), Methodology.

**Backend:** FastAPI, SQLAlchemy 2.0 (async), Alembic migrations, Rasterio, GeoPandas, Shapely, PyProj, xarray, scikit-learn, PyTorch, structlog (structured logging), Jinja2 (report templates).

**Database:** PostgreSQL + PostGIS is the declared production target (`docker-compose.yml` runs `postgis/postgis:16-3.4`; `app/core/config.py`'s default `DATABASE_URL` is a `postgresql+asyncpg://` DSN). SQLite is supported for local development (via `aiosqlite`) but is explicitly documented as insufficient for PostGIS spatial queries.

**AI (language interpretation):** Groq, via `app/interpretation/language_service.py` (`LanguageInterpretationService`). Model: `llama-3.3-70b-versatile` (text), `meta-llama/llama-4-scout-17b-16e-instruct` (vision, optional). Backend-only — the API key is never sent to the frontend, never appears in an API response (verified: `/health` reports only `"interpretation": {"ok": true, "provider": "Groq", "model": ..., "vision_enabled": false}`, no key value). Interpretation is optional: without a working key, analyses complete in full with the interpretation section explicitly marked unavailable, rather than failing.

**Infrastructure:** Docker (separate `backend/Dockerfile`, `frontend/Dockerfile`), `docker-compose.yml` (db + backend + frontend/nginx), GitHub Actions CI (`.github/workflows/ci.yml`) running backend lint/typecheck/tests/research-smoke, frontend lint/typecheck/build, and a Docker-image build-and-health-check job.

**Data flow** (user-selected region → report), as implemented across `app/api/routes/`, `app/imagery/`, `app/analysis/`, `app/models_ml/`, `app/interpretation/`:

```
User selects region + date range (frontend)
        ↓
POST /api/v1/analysis  → region validated (app/geospatial/geometry.py)
        ↓
STAC search + scene selection (app/imagery/stac.py, service.py)
        ↓
Windowed raster read + radiometric calibration + SCL cloud/shadow/cirrus/snow masking
        ↓
NDVI + derived spectral indices (app/analysis/indices.py)
        ↓
Bi-temporal change detection, if two periods given (app/analysis/change.py)
        ↓
Land-cover classification — per-pixel, trained random_forest / torch_mlp (app/models_ml/)
        ↓
Evidence package assembly — observed vs. predicted vs. (optionally) interpreted,
kept as distinct, labelled fields (app/interpretation/evidence.py)
        ↓
Optional: Groq language-model interpretation of the evidence, validated
against the evidence before being accepted (app/interpretation/validation.py)
        ↓
Report (HTML; PDF only if weasyprint extra is installed) + analysis history
```

---

## 32. ENGINEERING CONTRIBUTIONS

Only work actually present in the repository is listed.

- **API architecture:** FastAPI app with lifespan-managed startup/shutdown, structured request-ID logging middleware, centralized exception handling with a stable JSON error envelope (`{code, message, details, request_id}` — directly verified via a live 404 response), versioned `/api/v1` prefix plus an unprefixed `/health` for infra probes.
- **Geospatial validation:** region bbox validation (`app/geospatial/geometry.py`), area bounds (`min_region_area_km2`/`max_region_area_km2`), CRS/transform-aware `RasterGrid` with alignment assertions that raise rather than silently misalign.
- **Raster processing:** windowed COG reads, physically-verified radiometric calibration (`resolve_calibration`, per-observation, guards against a mis-declared catalogue offset), SCL-based masking, nearest-neighbour resampling for categorical data, finest-resolution co-registration.
- **Model registry:** feature-contract versioning (`FEATURE_VERSION`) embedded in every trained artifact; the registry refuses to serve a model whose feature names don't match the running code; SHA-256 integrity check on artifacts before deserialization.
- **Structured evidence / grounding validation:** `app/interpretation/evidence.py` (evidence package assembly, numeric-claim walking), `app/interpretation/validation.py` (unsupported-claim pattern screening, date-pattern false-positive fixes — e.g. ISO dates and "N days/months/years" phrases excluded from being flagged as unsupported numbers).
- **Report generation:** HTML reports via Jinja2, optional PDF export via `weasyprint` (an optional dependency — `pdf: false` when not installed, honestly reported via `/health`).
- **Testing:** 161 backend tests total (`pytest -q`), including 31 dedicated to the research framework; ruff (lint) and mypy (types) clean across `app`, `training`, `tests`, `research`.
- **CI:** GitHub Actions — backend lint/type/test job, frontend lint/type/build job, and a Docker-build-plus-live-healthcheck job, plus a research-smoke-test step gating every push/PR.
- **Docker:** multi-stage Dockerfiles for backend (Python 3.12-slim, rasterio/geopandas wheels bundle GDAL) and frontend (Node build → nginx runtime with SPA fallback and same-origin `/api` proxy).
- **Database migrations:** Alembic, against the async SQLAlchemy models.
- **Frontend map:** MapLibre GL region selection and result-layer rendering.
- **Analysis lifecycle/progress:** background `asyncio.create_task` execution with status polling (`analysis_status` route) and layer-by-layer PNG output.
- **Analysis history:** listable/retrievable past analyses (`list_analyses`, `get_analysis`, `get_analysis_report`).

---

## 33. SCIENTIFIC INTEGRITY

**Observed vs. predicted vs. interpreted — kept as distinct fields**, per `app/interpretation/evidence.py`'s `EvidencePackage`:

- **Observed** = measured, deterministic satellite-derived quantities (reflectance, NDVI, change statistics) — computed directly from Sentinel-2 pixels, no ML involved.
- **Model prediction** = land-cover class output of the trained classifier — explicitly labelled as a prediction, not a measurement.
- **AI interpretation** = language-model text generated strictly from the observed + predicted evidence, and validated against that evidence (numeric-claim checking, source-ID checking, causal-overreach screening) before being returned to the user.

**Explicit non-equivalences maintained throughout this repository and this document:**

- **NDVI change ≠ deforestation.** NDVI change detection reports a change in a reflectance-derived vegetation index; it does not establish land-cover conversion, and no causal process is attributed to it (`research/experiments/robustness.py` limitations field, Section 23).
- **NDVI ≠ biodiversity, biomass, carbon stock, or habitat quality.** NDVI is a chlorophyll-sensitive greenness proxy only (Section 8).
- **Classification ≠ measurement.** Land-cover class is a model prediction with its own error profile (Section 14's per-class precision/recall), distinct from the observed reflectance/NDVI values.
- **WorldCover ≠ perfect ground truth.** Stated verbatim in `research/config.py`, `app/models_ml/labels.py`, and the dataset manifest's `limitations` field: it is "a model product with its own error, not ground truth." Agreement with WorldCover is not automatically classifier correctness.
- **LLM ≠ source of numerical truth.** Interpretation text is generated only from supplied evidence and validated against it; the grounding evaluator specifically checks for numbers the LLM stated that do not trace back to the evidence package (Section 24).

---

## 34. LIMITATIONS

Extracted directly from the repository's own documented limitations (`research/README.md`, per-experiment `limitations` fields, dataset manifest, `docs/methodology.md`) — not invented here.

- **Geographic coverage:** 13 regions is a small sample of global land cover; results describe these landscapes specifically, not the planet.
- **Temporal coverage:** two ~4-month windows (Jan–Apr 2021, Jan–Apr 2024) — results do not generalize to other seasons or years untested here.
- **Missing source groups:** 3 of 26 expected (region, period) groups failed to build (`nile_delta`/p2021, `sahel_niger`/p2024, `sonoran_desert`/p2021) and are recorded as absent rather than substituted.
- **Reference-label quality:** ESA WorldCover 2021 v200 is a reference map with its own error, fixed at the 2021 epoch even though imagery spans two periods — 2024-period comparisons conflate genuine change with label staleness.
- **Cloud coverage caveats:** scene-level cloud cover describes the whole tile, not the analysed region specifically; only 34 real observations back the cloud-robustness result, with as few as 3 observations in the two highest-cloud buckets.
- **Seasonal effects:** NDVI variation across dates includes genuine seasonal change, confounding cloud-effect interpretation in `cloud_robustness`.
- **Model confidence:** informative (separates correct/incorrect) but **poorly calibrated** — ECE 0.1541, overconfident in every bucket (Section 19).
- **Dataset size / sampling:** pixels are class-stratified per region-period (capped at 20,000/group), so cached class proportions do not reflect true landscape proportions.
- **Per-pixel classification with no spatial context:** produces salt-and-pepper error at class boundaries and on mixed pixels (`docs/methodology.md`).
- **LLM evaluation status:** H7 (grounding) is entirely unmeasured — blocked by a provider authentication failure, 0 successful generations.
- **Generalizability of the central finding:** the +0.3088/+0.3225 random-vs-spatial gap is established only for this dataset/model/feature configuration — not shown to be universal.
- **Grounding evaluator is lexical, not semantic** (documented limitation, applies once H7 is unblocked): it matches numbers against evidence and screens fixed causal-phrase patterns; it cannot judge scientific soundness.
- **Language-model non-determinism:** hosted model output is non-deterministic and the provider's model may change without notice, so any future grounding figures describe one provider at one point in time.

---

## 35. WHAT HAS ACTUALLY BEEN CONTRIBUTED

**Software contribution:** a working, tested, end-to-end geospatial analysis platform (FastAPI + React) that performs real Sentinel-2 acquisition, NDVI/change computation, land-cover classification, and evidence-grounded reporting, deployed via Docker/CI.

**Research contribution:** a controlled, reproducible experimental comparison of validation protocols (random-pixel vs. spatial-holdout vs. temporal-transfer) for a lightweight land-cover classifier on real Sentinel-2 data, with a measured, artifact-backed result (Section 25), plus 7 further controlled sub-experiments (model comparison, feature ablation, confidence calibration, feature importance, class imbalance, cloud robustness, change-threshold sensitivity) run under the same dataset and reproducibility discipline (versioned config, hashed dataset, recorded seeds and git commit per result).

**Engineering contribution:** the specific safeguards listed in Section 32 — physically-verified radiometric calibration, feature-contract versioning and enforcement, evidence-grounding validation, leakage-detection unit tests, and a reproducibility-first experiment-record format (Section 29).

**Potential methodological contribution:** the explicit, artifact-quantified demonstration that validation-protocol choice can move measured performance by more than model-architecture choice does, on this dataset — a methodological point that is well established in the broader remote-sensing/spatial-statistics literature (spatial autocorrelation, Tobler's first law) but is here demonstrated with a specific, reproducible, fully-instrumented experiment rather than asserted. **No claim of novelty is made** — this repository does not establish that this specific finding is new to the literature; a literature review has not been performed as part of this work.

---

## 36. WHAT IS NOT YET PROVEN

- The observed +0.3088 accuracy / +0.3225 macro-F1 validation-protocol gap is established only for the current dataset/configuration (13 regions, 2 periods, `random_forest`, `D_full` features). It does not prove the same exact gap magnitude occurs universally across all Sentinel-2 datasets, resolutions, or classifiers.
- More geographic regions are needed before the spatial-holdout figure (0.5150 macro-F1) can be treated as representative of "global" generalization rather than generalization to the three specific tested regions.
- More temporal periods (beyond the two ~4-month 2021/2024 windows) are needed before the temporal-transfer asymmetry (Section 18) can be generalized to other year pairs or seasons.
- Reference-label uncertainty (WorldCover's own error rate) has not been separately quantified against an independent ground-truth source in this work — only documented as a caveat.
- The LLM grounding experiment (H7) is incomplete — no result exists, positive or negative.
- Causal environmental claims (e.g. "this NDVI change is deforestation") cannot be inferred from NDVI change alone, and this repository does not attempt to.
- Statistical significance testing (e.g. a formal hypothesis test on the random-vs-spatial gap, beyond reporting mean/std across 3 seeds) has not been performed.
- Whether the feature-ablation near-null result (Section 17) would hold for a different model architecture has not been tested — ablation was run only with `random_forest`.

---

## 37. FUTURE RESEARCH

Labelled explicitly as future work — none of the following is implemented in this repository.

- More geographic regions, spanning additional biomes not yet represented, to test whether the spatial-holdout figure and the random-vs-spatial gap hold at larger geographic scale.
- More temporal periods (additional years/seasons) to disentangle genuine temporal domain shift from 2021-label staleness.
- Additional Sentinel-2 scenes per region-period to reduce per-group sampling variance.
- Additional classifiers beyond the four compared here (e.g. gradient-boosted trees with spatial features, or models explicitly incorporating neighbourhood context to address the per-pixel salt-and-pepper limitation).
- Stronger spatial cross-validation protocols (e.g. spatial k-fold with multiple held-out folds, rather than one fixed 3-region test set) for a variance estimate on the spatial-holdout figure itself.
- Confidence calibration methods (e.g. temperature scaling, isotonic regression) to address the measured ECE of 0.1541.
- Uncertainty estimation beyond a single softmax-derived confidence value.
- Class-balanced training strategies evaluated specifically against the measured 4.172 test-split imbalance ratio.
- External validation against an independent, non-WorldCover reference-label source.
- Formal statistical-significance testing on the reported gaps.
- Additional spectral sensors (e.g. Landsat, Sentinel-1 SAR) for cross-sensor generalization testing.
- Vision-language-model (VLM) evaluation of the rendered map imagery, distinct from the current text-only LLM interpretation.
- Environmental-event-specific datasets (e.g. documented deforestation or fire events) to move beyond "NDVI change" toward event-attributed change detection — with the causal-attribution caveats this would require made explicit from the outset.
- Completing the LLM grounding experiment (H7) once a working provider credential is available.

---

## 38. PAPER STRUCTURE

Proposed outline, mapped to material already available in this repository.

1. **Abstract** — Section 40 draft.
2. **Introduction** — Sections 1–4 (problem, motivation, objective).
3. **Related Work** — not yet written; requires an actual literature review (spatial autocorrelation in remote sensing, spatial cross-validation methodology) not performed as part of this repository.
4. **Dataset** — Section 6, 7 (regions, periods, classes, WorldCover attribution, exact sample counts).
5. **Methodology** — Sections 8–13 (features, preprocessing, models, validation strategies, metrics).
6. **Experimental Design** — Section 12 (seeds, protocols, repetitions).
7. **Results** — Sections 14–24 (all 11 experiments, exact numbers), Section 45 (result ledger).
8. **Discussion** — Sections 25–26 (central finding + all other findings), read together with Section 36 (what is not yet proven).
9. **Limitations** — Section 34.
10. **Conclusion** — restates Section 25's central finding within the exact bounds stated there; explicitly does not overreach into Section 36's territory.
11. **References** — Sentinel-2/Copernicus data attribution and ESA WorldCover citation are already recorded (`research/README.md`'s Data Attribution section); academic literature references still need to be added via a literature review not yet performed.

---

## 39. POSSIBLE PAPER TITLE

1. "Spatial Autocorrelation and Validation Protocol in Sentinel-2 Land-Cover Classification: A Controlled Comparison"
2. "Random-Pixel vs. Spatial-Holdout Validation for Lightweight Sentinel-2 Land-Cover Models"
3. "Quantifying the Optimism of Random Pixel Splits in Multispectral Land-Cover Classification"
4. "Spatial and Temporal Generalization of Lightweight Classifiers on Sentinel-2 Reflectance Data"
5. "Validation Protocol as a Confound in Sentinel-2 Land-Cover Model Evaluation: An Empirical Study"

---

## 40. ABSTRACT DRAFT

**Preliminary abstract — subject to literature review and final experiments.**

We evaluate how validation protocol affects the measured generalization performance of lightweight machine-learning models trained on Sentinel-2 Level-2A multispectral reflectance for five-class land-cover classification. Using a cached dataset of 313,580 pixels sampled from 13 geographically and biome-diverse regions across two observation periods (2021, 2024), with reference labels from ESA WorldCover 2021 v200, we compare a random-pixel train/test split against a spatially disjoint holdout split under an identical random-forest classifier, feature set, and three random seeds. Random-pixel validation produced an accuracy of 0.8413 (macro-F1 0.8375), while spatially disjoint validation on the same underlying data produced an accuracy of 0.5325 (macro-F1 0.5150) — a difference of 0.3088 accuracy and 0.3225 macro-F1. This gap exceeds the entire spread observed across four compared model architectures (random forest, histogram gradient boosting, calibrated linear SVM, and a small MLP; macro-F1 range 0.5150–0.5585) under the spatial-holdout protocol. We further report model comparison, feature ablation (raw bands vs. derived spectral indices), temporal transfer between the two observation periods, confidence-calibration analysis (expected calibration error 0.1541, with overconfidence in every bucket), feature-importance ranking (NDVI ranked first by both impurity and permutation importance), class-imbalance characterization, and observation-quality robustness against scene cloud cover. All results are reported with explicit limitations, including the reference-label source's own uncertainty and the small number of geographic regions tested. An accompanying evidence-grounding evaluation for language-model-generated interpretation text was implemented but not completed due to a provider authentication failure, and is reported as such rather than omitted or fabricated.

---

## 41. PROFESSOR PRESENTATION EXPLANATION

**"What did you build?"**
A full geospatial-AI platform (NatureVision) that pulls real Sentinel-2 satellite imagery for a user-selected region, computes vegetation indices and change detection, runs a trained land-cover classifier, and generates an evidence-grounded report — plus a separate, reproducible research framework that runs 11 controlled experiments on that same pipeline's data.

**"Why did you build it?"**
It started as an applied project connecting my ML/backend background to remote sensing. Working on the land-cover classifier surfaced a specific methodological concern: Sentinel-2 pixels are spatially autocorrelated at 10 m resolution, so a naive random train/test split risks leaking near-duplicate pixels across the split boundary. That concern became the research question.

**"What is the research question?"**
How does validation protocol — random pixel split vs. spatially separated holdout — affect the measured generalization performance of a Sentinel-2 land-cover classifier, and how does that compare in magnitude to differences from model architecture, features, and observation period?

**"What did you test?"**
Eleven controlled experiments on a fixed, versioned, 313,580-pixel cached dataset: the core random-vs-spatial comparison, four-model comparison, feature ablation, temporal transfer between 2021 and 2024, confidence calibration, feature importance, class imbalance, cloud-cover robustness, change-threshold sensitivity, and an LLM evidence-grounding evaluation.

**"What did you find?"**
Validation protocol changed measured macro-F1 by 0.3225 (0.8375 random-pixel vs. 0.5150 spatial-holdout) — a larger swing than switching between any two of the four model architectures I compared (a 0.0435 macro-F1 range). Confidence was informative but overconfident (ECE 0.1541). NDVI was the single most important feature. Derived spectral indices added only marginal value over raw bands.

**"Why is it interesting?"**
It quantifies, on real data, exactly how much an evaluation-methodology choice can distort a headline accuracy number — a bigger effect here than the choice of model itself. That's directly relevant to how remote-sensing ML results should be reported and read critically.

**"What are the limitations?"**
Thirteen regions, two ~4-month windows, one reference label source with its own error, and a per-pixel classifier with no spatial context. The result establishes the *magnitude of the effect on this dataset*, not a universal constant.

**"What do you want to do next?"**
More regions and periods to test how stable this gap is; calibration methods to address the overconfidence; completing the blocked LLM-grounding experiment once I have a working API key; and a literature review to properly position this against existing spatial-cross-validation research before writing it up formally.

---

## 42. INTERVIEW EXPLANATION

**30-second version:**
"I built NatureVision, a Sentinel-2 satellite-analysis platform, and used it to run a controlled experiment on ML evaluation methodology: I showed that validating a land-cover classifier with a random pixel split versus a spatially separated holdout changes measured accuracy by over 30 percentage points on the same data and model — a bigger effect than switching between four different model architectures."

**1-minute version:**
"NatureVision is a full-stack geospatial platform — FastAPI backend, React/MapLibre frontend — that pulls real Sentinel-2 imagery, computes NDVI and change detection, and runs a trained land-cover classifier with an evidence-grounded LLM interpretation layer. Building the classifier raised a methodology question: satellite pixels are spatially autocorrelated, so a random train/test split can overstate how well a model generalizes to new geography. I built a separate research framework — 313,580 cached pixels across 13 regions and two years — and ran 11 controlled experiments comparing random-pixel vs. spatial-holdout validation, four model architectures, feature sets, and more. The headline result: spatial holdout dropped macro-F1 from 0.8375 to 0.5150 — a 0.32 swing — bigger than the entire spread across model architectures. Everything is reproducible: versioned config, hashed dataset, fixed seeds, git-commit-stamped result records."

**3-minute technical version:**
"NatureVision does end-to-end remote-sensing analysis: STAC-based Sentinel-2 Level-2A discovery over the AWS Open Data archive, windowed COG reads, physically-verified radiometric calibration — I found the catalogue's declared reflectance offset was actually wrong for older scenes and had to build a probe that checks it empirically rather than trusting it — SCL-based cloud/shadow/cirrus/snow masking, NDVI and four other spectral indices, bi-temporal change detection, and a per-pixel land-cover classifier trained against ESA WorldCover reference labels, explicitly documented as a reference product and not ground truth. Model output feeds an evidence package that's kept separate from any LLM-generated interpretation text, which is validated against the evidence — unsupported numeric claims and invented source IDs get rejected — before being shown to a user.

On the research side, I isolated validation protocol as a single variable: same 313,580-pixel dataset, same random-forest classifier, same 11-feature vector, three seeds, comparing a pixel-level random split against a split that holds out three entire geographic regions. Random split: 0.8413 accuracy, 0.8375 macro-F1. Spatial holdout: 0.5325 accuracy, 0.5150 macro-F1. That 0.32 macro-F1 gap is bigger than the spread I measured across four different model architectures under the harder spatial protocol — random forest, hist-gradient-boosting, a calibrated linear SVM, and a small MLP, ranging 0.515 to 0.5585. I also ran feature ablation — raw bands versus adding NDVI, NDWI/NDBI, and NBR/a bare-soil index — and found only marginal differences, even though NDVI turned out to be the single most important feature by both impurity and permutation importance, which tells you the signal in NDVI substantially overlaps with what's already in the raw NIR/Red bands it's derived from. Confidence calibration showed the model's probabilities separate correct from incorrect predictions but are meaningfully overconfident — 0.154 expected calibration error, positive in every bucket. And I was upfront about what didn't work: an LLM-grounding experiment is fully built — dual-mode generation, a deterministic scorer for unsupported numbers and invented source IDs — but the Groq API key stopped authenticating partway through, so that result is reported as blocked, zero generations, not faked."

---

## 43. TECHNICAL VIVA / QUESTION BANK

### Remote sensing

**Q1. What is Sentinel-2 Level-2A?**
A: Atmospherically corrected surface-reflectance product from the Copernicus Sentinel-2 mission — as opposed to Level-1C, which is top-of-atmosphere reflectance.
*Deeper:* Level-2A is produced by Sen2Cor atmospheric correction; it also carries the Scene Classification Layer (SCL) used here for cloud/shadow/snow masking.

**Q2. Why use Sentinel-2 rather than Landsat?**
A: Not directly tested in this repository; the project's choice was 10 m resolution and the free, keyless AWS Open Data STAC archive via Element84 Earth Search.
*Deeper:* the repository does not run a comparative Landsat experiment — this is a design choice, not a research finding.

**Q3. What is a STAC catalog?**
A: SpatioTemporal Asset Catalog — a standard JSON API for discovering geospatial assets (scenes) by bounding box, date range, and metadata filters like cloud cover, without downloading full imagery first.

**Q4. What resolution does this project work at?**
A: 10 m for the finest Sentinel-2 bands; 20 m SWIR/SCL bands are resampled up to the 10 m grid (Section 9).

### GIS

**Q5. What is a Cloud-Optimized GeoTIFF (COG) and why does it matter here?**
A: A GeoTIFF internally organized for efficient partial (windowed) HTTP range-reads, so the pipeline can read only the pixels covering a region of interest instead of downloading an entire scene.

**Q6. How is georeferencing preserved through the pipeline?**
A: Every raster (`RasterGrid`) carries its CRS, affine transform, and a real nodata mask (not a sentinel value) through every operation; alignment is asserted before combining two rasters (`docs/architecture.md`).

**Q7. Why nearest-neighbour resampling for the SCL layer specifically?**
A: SCL values are categorical class codes; any interpolating resampling method (bilinear, cubic) would invent nonexistent intermediate classes.

### Sentinel-2

**Q8. What bands does the classifier use?**
A: Blue, Green, Red, NIR, SWIR-1 (swir16), SWIR-2 (swir22) — 6 raw bands (Section 8).

**Q9. What is the radiometric-calibration issue this project found?**
A: The STAC catalogue's declared reflectance offset (−0.1 in some cases) did not hold for all scenes; applying it uniformly could push most pixels negative. `resolve_calibration` empirically probes a small window per observation and rejects the declared offset if it produces excessive negative reflectance.
*Deeper:* recorded per group in the dataset manifest under `radiometric_calibration` — e.g. Western Ghats p2021 recorded `declared_offset: -0.1`, `negative_fraction_with_offset: 1.0`, `negative_fraction_without_offset: 0.0` → decision: override to offset 0.0.

**Q10. What is the Scene Classification Layer (SCL)?**
A: A per-pixel categorical Sentinel-2 Level-2A product classifying each pixel as e.g. vegetation, water, cloud (various confidence levels), cloud shadow, cirrus, snow/ice, no-data. Used here to build the validity mask (Section 9).

### NDVI

**Q11. What is the NDVI formula?**
A: `NDVI = (NIR − Red) / (NIR + Red)`.

**Q12. What does NDVI measure, and what does it not measure?**
A: A chlorophyll-sensitive vegetation-greenness proxy. It does not measure biodiversity, biomass, carbon stock, habitat quality, deforestation, or climate change directly (Section 8, 33).

**Q13. Why is NDVI bounded to [−1, 1], and what happens outside that range?**
A: It's a normalized difference, so it is mathematically bounded; values computed outside that range indicate a numerical/data artefact and are masked out (`app/analysis/indices.py`).

**Q14. Why was NDVI the most important feature, given it's derived from bands already in the raw set?**
A: Impurity and permutation importance both ranked it 1st (Section 20) — the normalized-difference transform apparently exposes the vegetation signal in a form the tree splits exploit more efficiently than the raw NIR/Red bands alone, even though the underlying information overlaps (consistent with the feature-ablation near-null result, Section 17).

### Machine learning

**Q15. Which models were compared, and why these four?**
A: Random forest, histogram gradient boosting, a calibrated linear SVM, and a small MLP (Section 10) — lightweight, workstation-trainable, and spanning tree-ensemble, boosting, linear, and neural approaches.

**Q16. Why calibrate the linear SVM with `CalibratedClassifierCV`?**
A: `LinearSVC` has no native `predict_proba`; wrapping it with sigmoid calibration (3-fold CV) gives genuine probability estimates instead of a raw decision-function stand-in.

**Q17. Were hyperparameters tuned per experiment?**
A: No — fixed in advance in `research/config.py::MODEL_CONFIGS`, never tuned against a test split (explicit safeguard, Section 12/30).

**Q18. Which model had the best macro-F1, and is that the "right" choice?**
A: `mlp` (0.5585 mean), but with the highest seed-to-seed standard deviation (0.0101) among the four — a materially less stable result. `linear_svm` (0.5499, std 0.0002) is a lower-variance alternative (Section 16). The repository's own note states ranking is by macro-F1 with cost also reported, "so model selection is not reduced to a single number."

### Validation / spatial leakage

**Q19. What is spatial leakage in this context?**
A: A training pixel and a test pixel from the same geographic block (region) both present — meaning the "held-out" evaluation isn't testing generalization to new geography.

**Q20. How is spatial leakage detected in this codebase, and is the detector itself tested?**
A: `spatial_leakage()` returns the set of region blocks appearing in both train and test. It is unit-tested by deliberately contaminating a clean split and asserting the detector fires (`test_spatial_leakage_is_detected_when_introduced`) — proving the check isn't a no-op.

**Q21. Why does the random split still get used at all, if it's "wrong"?**
A: It's not run to be recommended — it's run specifically as the *optimistic protocol under test* (code comment in `splits.py`), to measure exactly how much it overstates performance relative to the honest spatial protocol.

**Q22. What regions are held out in the spatial-holdout protocol?**
A: Test: `murray_basin`, `iberian_meseta`, `zambezi_miombo`. Validation: `finnish_lakeland` (Section 11).

### Temporal transfer

**Q23. Why does temporal transfer use a spatial holdout too, by default?**
A: To separate "unfamiliar time" from "unfamiliar time *and* unfamiliar geography" — `spatial_holdout=True` trains on non-test regions and tests on the same three spatial-test regions across the other period, so temporal shift isn't confounded with new-geography difficulty in the transfer condition. A `spatial_holdout=False` mode isolates temporal shift alone on the same ground for the same-period comparison rows (Section 11, 18).

**Q24. Why was p2021→p2024 transfer *better* than same-period p2021→p2021?**
A: Observed, not fully explained: possible contributing factors include more/different training pixels available when the 2021-trained model is evaluated on the specific 2024 test-region scenes versus its own period's test scenes, and label staleness (labels are 2021-epoch) interacting differently with each test period. **This asymmetry is reported as observed, not causally explained** (Section 18) — the repository does not claim to know why.

### Metrics

**Q25. Why macro-F1 over accuracy as the primary metric?**
A: With 5 classes at up to a 4.172:1 imbalance ratio in the spatial-holdout test split (Section 21), accuracy alone can be dominated by majority classes; macro-F1 weights every class equally.

**Q26. What is balanced accuracy, and how does it differ from macro-F1?**
A: Balanced accuracy is the mean of per-class recall only; macro-F1 combines precision and recall per class (harmonic mean) then averages. Both counter class-size dominance, but macro-F1 also penalizes low precision (false positives), which balanced accuracy alone does not.

**Q27. What is Expected Calibration Error, and what does 0.1541 mean here?**
A: A bucketed estimate of |accuracy − mean predicted confidence|, weighted by bucket population. 0.1541 means, on average across confidence buckets, the model's stated confidence overstates its actual accuracy by about 0.154 — and this overstatement is positive (overconfident) in every populated bucket (Section 19), not just on average.

### Dataset

**Q28. How big is the research dataset, exactly?**
A: 313,580 pixels, 23 of 26 expected (region, period) groups, across 13 regions and 2 periods (Section 6).

**Q29. Why are only 23 of 26 groups present?**
A: Three (region, period) combinations failed to fetch during dataset build — `nile_delta`/p2021, `sahel_niger`/p2024, `sonoran_desert`/p2021 — and are recorded as absent rather than filled with synthetic data.

**Q30. How is the dataset cached and versioned?**
A: `research/cache/pixels.npz`, with a written manifest (`research/manifests/dataset_manifest.json`) recording a SHA-256 digest (`cache_sha256`), `dataset_version`, and per-group scene/calibration/class-count metadata — so every experiment reads identical, checksummed data.

### WorldCover

**Q31. Is WorldCover ground truth?**
A: No — explicitly documented as a reference land-cover map produced by a model, with its own error, in `research/config.py`, `app/models_ml/labels.py`, and the dataset manifest.

**Q32. How were WorldCover's 11 classes collapsed into this project's 5?**
A: Tree cover + mangroves → Forest; shrubland + grassland + cropland → Agriculture; built-up → Urban; bare + snow/ice + moss/lichen → Bare land; water + herbaceous wetland → Water (Section 7). This collapse is lossy and documented as such — e.g. "Agriculture" here includes non-cropland herbaceous cover.

### Geospatial processing

**Q33. What happens if too few valid pixels remain after masking?**
A: The analysis fails with a specific error rather than reporting statistics computed from a handful of survivors (`docs/methodology.md`).

**Q34. How are two periods kept pixel-comparable for change detection?**
A: Both are clipped to the same selection polygon — clipping masks outside pixels but preserves the grid extent, so the two periods share the same pixel grid.

### Research methodology

**Q35. Why three seeds specifically, not one or ten?**
A: A design choice recorded in `research/config.py` — enough to report a distribution (mean/std/min/max) rather than a single point estimate, while keeping the ~40-minute `core` suite runtime tractable on a workstation. Not statistically optimized for a target confidence interval.

**Q36. How is a result made traceable back to the exact code/data that produced it?**
A: Every experiment record's `metadata.json` stores git commit, config hash, dataset manifest hash, dataset cache SHA-256, seeds, contributing scene IDs, and (as of this document's inspection) runtime (Section 29).

**Q37. Was there a discrepancy found during this document's own verification process?**
A: Yes — the top-level run ledger (`experiment_runs.csv`) showed stale runtimes for 4 of 11 experiments versus their own `metadata.json` files, because those 4 were rerun outside the ledger-updating CLI path. The underlying *metric values* were verified identical across both generations. Documented explicitly in Section 29 rather than silently resolved.

### LLM grounding

**Q38. What blocked the LLM grounding experiment?**
A: The configured Groq API key returns HTTP 401 Unauthorized; all 6 generation attempts (3 cases × 2 modes) were rejected by the provider before any grounding scoring could occur (Section 24).

**Q39. What would the grounding evaluator have checked, had generations succeeded?**
A: Unsupported numerical claims (numbers in LLM text not traceable to the evidence package), invented source IDs (scene identifiers not present in the evidence's `data_sources`), and causal-overreach phrase patterns — via `score_output`, already unit-tested independent of a live API call.

### Software architecture

**Q40. Does the research framework modify production behaviour?**
A: No — it calls production modules through their normal public interfaces and leaves production defaults (e.g. change-detection thresholds) untouched even while sweeping other values in its own code path (Section 30).

**Q41. How does the API keep the LLM interpretation from being treated as ground truth by the frontend/user?**
A: The evidence package keeps `observed`, model `prediction`, and LLM `interpretation` as separate, labelled fields; the interpretation is validated against the evidence before being returned (Section 33).

### Reproducibility

**Q42. How would someone reproduce the primary (spatial-validation) result exactly?**
A: `cd backend && .\.venv\Scripts\Activate.ps1 && python -m research.run --experiment spatial_validation` (requires the cached dataset; `--experiment dataset` rebuilds it if absent) — deterministic given the same cache and seeds, as verified by the 2026-08-18 rerun reproducing the baseline figures bit-for-bit (Section 14).

**Q43. What CI gate exists for the research layer?**
A: `python -m research.run --experiment smoke` — a fast, network-free structural check (config load, split leakage checks, metric sanity on perfect predictions) run on every push/PR, distinct from the full ~15–20 minute `core` benchmark suite which is not run automatically on every PR.

---

## 44. "KNOW THIS PERFECTLY" — CHEAT SHEET

| Item | Value |
|---|---|
| **Research question** | How do spatial and temporal validation strategies affect measured generalization of lightweight ML models on Sentinel-2 land-cover classification, and how reliably can outputs support evidence-grounded interpretation? |
| **Main hypothesis (H1)** | Random pixel validation overestimates performance relative to spatial holdout validation |
| **Dataset** | Real Sentinel-2 L2A pixels + ESA WorldCover 2021 v200 reference labels |
| **Number of samples** | **313,580** pixels |
| **Groups present / expected** | 23 / 26 (3 missing, recorded not substituted) |
| **Regions** | 13 |
| **Periods** | 2 (`p2021`, `p2024`, ~4-month windows, Jan start) |
| **Classes** | 5 — Forest, Agriculture, Water, Urban/built-up, Bare land |
| **Features** | 11 — 6 raw bands + NDVI, NDWI, NDBI, NBR, bsi_partial |
| **Models compared** | random_forest, hist_gradient_boosting, linear_svm (calibrated), mlp |
| **Seeds** | 42, 123, 2024 |
| **Validation protocols** | random pixel split, spatial holdout, temporal transfer |
| **Metrics** | accuracy, balanced accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion matrix, ECE |
| **Main result** | Random pixel: 0.8413 acc / 0.8375 macro-F1. Spatial holdout: 0.5325 acc / 0.5150 macro-F1. **Diff: +0.3088 acc / +0.3225 macro-F1** |
| **Secondary result — models** | macro-F1 range 0.5150 (RF) – 0.5585 (MLP) under spatial holdout |
| **Secondary result — features** | max macro-F1 delta across A–D feature sets: 0.0044 |
| **Secondary result — temporal** | p2024→p2021 transfer costs −0.1264 acc vs. p2024→p2024 same-period |
| **Secondary result — confidence** | ECE 0.1541; correct-mean-conf 0.7596 vs. incorrect-mean-conf 0.5993 |
| **Secondary result — importance** | NDVI ranked #1 by both impurity and permutation importance |
| **Secondary result — imbalance** | test-split ratio 4.172; accuracy − balanced accuracy = −0.0104 |
| **Secondary result — cloud** | valid-pixel fraction 0.9433 (0–5% cloud) → 0.7175 (10–20% cloud) |
| **Secondary result — threshold** | Po Valley changed-area 58.875% (0.05) → 16.799% (0.20) |
| **Blocked** | H7 LLM grounding — 0/6 generations succeeded, Groq 401 |
| **Baseline reproduction** | 0.5325 acc / 0.5150 macro-F1 vs. published 0.5922 / 0.5268 — within ±0.08 tolerance |
| **Limitations (top 3)** | small region count (13), WorldCover is not ground truth, label epoch fixed at 2021 while imagery spans 2021–2024 |
| **Research contribution** | Quantified, reproducible measurement of validation-protocol effect size, exceeding model-architecture effect size, on this dataset |
| **Future work (top 3)** | more regions/periods, confidence calibration, complete the blocked LLM-grounding experiment |
| **NDVI formula** | (NIR − Red) / (NIR + Red) |
| **NDVI is NOT** | biodiversity, biomass, carbon stock, deforestation, climate change |
| **Key terminology** | "reference labels" not "ground truth"; "vegetation-index change" not "deforestation"; "model prediction" not "measurement" |

---

## 45. EXACT RESULT LEDGER

| Experiment | Purpose | Protocol | Model | Feature set | Seed(s) | Accuracy | Balanced Acc. | Macro-F1 | Other key metric | Status | Main finding |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| baseline | Reproduce production spatial-holdout figure | Spatial holdout | random_forest | D_full | 42,123,2024 | 0.5325 ±0.0024 | 0.5430 ±0.0022 | 0.5150 ±0.0024 | gap vs published: −0.0597 acc / −0.0118 F1 | Completed, within ±0.08 tolerance | Reproduces production baseline |
| spatial_validation | Compare validation protocols | Random pixel | random_forest | D_full | 42,123,2024 | 0.8413 ±0.0019 | 0.8414 ±0.0025 | 0.8375 ±0.0028 | — | Completed | Central finding source |
| spatial_validation | Compare validation protocols | Spatial holdout | random_forest | D_full | 42,123,2024 | 0.5325 ±0.0024 | 0.5430 ±0.0022 | 0.5150 ±0.0024 | Diff vs random: −0.3088 acc / −0.3225 F1 | Completed | Central finding source |
| model_comparison | Compare model architectures | Spatial holdout | random_forest | D_full | 42,123,2024 | 0.5325 ±0.0024 | 0.5430 ±0.0022 | 0.5150 ±0.0024 | train 14.7–17.0s | Completed | Lowest macro-F1 of the four |
| model_comparison | Compare model architectures | Spatial holdout | hist_gradient_boosting | D_full | 42,123,2024 | 0.5544 ±0.0027 | 0.5702 ±0.0013 | 0.5461 ±0.0013 | train 4.2–6.6s | Completed | — |
| model_comparison | Compare model architectures | Spatial holdout | linear_svm | D_full | 42,123,2024 | 0.5513 ±0.0002 | 0.5750 ±0.0002 | 0.5499 ±0.0002 | train 2.7–6.6s, lowest variance | Completed | Most stable non-tree model |
| model_comparison | Compare model architectures | Spatial holdout | mlp | D_full | 42,123,2024 | 0.5506 ±0.0114 | 0.5952 ±0.0107 | 0.5585 ±0.0101 | train 10.7–12.3s, highest variance | Completed | Highest macro-F1, least stable |
| feature_ablation | Test derived-index value | Spatial holdout | random_forest | A_raw_bands (6) | 42,123,2024 | 0.5217 | Not reported | 0.5106 ±0.0005 | Δ vs D: −0.0044 | Completed | — |
| feature_ablation | Test derived-index value | Spatial holdout | random_forest | B_bands_ndvi (7) | 42,123,2024 | 0.5263 | Not reported | 0.5134 ±0.0009 | Δ vs D: −0.0016 | Completed | — |
| feature_ablation | Test derived-index value | Spatial holdout | random_forest | C_bands_indices (9) | 42,123,2024 | 0.5366 | Not reported | 0.5156 ±0.0023 | Δ vs D: +0.0006 | Completed | Marginally exceeded full set |
| feature_ablation | Test derived-index value | Spatial holdout | random_forest | D_full (11) | 42,123,2024 | 0.5325 ±0.0024 | 0.5430 ±0.0022 | 0.5150 ±0.0024 | Δ vs D: 0.0000 | Completed | Indices add only marginal value |
| temporal_transfer | Test period generalization | p2021→p2021 (same-period) | random_forest | D_full | 42,123,2024 | 0.5703 ±0.0007 | 0.5696 ±0.0007 | 0.5506 ±0.0007 | — | Completed | Baseline for comparison |
| temporal_transfer | Test period generalization | p2021→p2024 (forward) | random_forest | D_full | 42,123,2024 | 0.6149 ±0.0021 | 0.5910 ±0.0014 | 0.5794 ±0.0016 | Exceeds same-period p2021 | Completed | Asymmetric transfer |
| temporal_transfer | Test period generalization | p2024→p2024 (same-period) | random_forest | D_full | 42,123,2024 | 0.6382 ±0.0008 | 0.6338 ±0.0008 | 0.6076 ±0.0008 | — | Completed | Highest same-period score |
| temporal_transfer | Test period generalization | p2024→p2021 (backward) | random_forest | D_full | 42,123,2024 | 0.5118 ±0.0020 | 0.5537 ±0.0016 | 0.5111 ±0.0018 | −0.1264 acc vs same-period p2024 | Completed | Largest transfer cost |
| confidence_analysis | Test confidence informativeness/calibration | Spatial holdout | random_forest | D_full | 42 | 0.5302 | 0.5406 | 0.5124 | ECE = 0.1541 | Completed | Informative but overconfident |
| feature_importance | Rank feature contribution | Spatial holdout | random_forest | D_full | 42 | Not reported (uses baseline model) | Not reported | Not reported | NDVI rank #1 (both methods) | Completed | NDVI dominant feature |
| class_imbalance | Quantify imbalance effect | Spatial holdout | random_forest | D_full | 42 | 0.5302 | 0.5406 | 0.5124 | test-split ratio 4.172; acc−bal.acc = −0.0104 | Completed | Small accuracy-inflation effect here |
| cloud_robustness | Relate cloud cover to usability | N/A (imagery-level) | N/A | N/A | N/A (no seed) | Not applicable | Not applicable | Not applicable | valid-pixel fraction 0.9433→0.7175 | Completed | Usability degrades with cloud |
| threshold_sensitivity | Change-threshold sensitivity | N/A (imagery-level) | N/A | N/A | N/A (no seed) | Not applicable | Not applicable | Not applicable | Po Valley 58.875%→16.799% (0.05→0.20) | Completed | High sensitivity to threshold |
| llm_grounding | Test evidence grounding | N/A | Groq llama-3.3-70b-versatile | N/A | N/A (no seed) | Not applicable | Not applicable | Not applicable | n_generated=0, n_rejected=6 | **Blocked** (401 Unauthorized) | No result obtained |

---

## 46. SOURCE OF TRUTH RULE

| For | Authoritative source |
|---|---|
| **Dataset** contents, size, provenance | `backend/research/manifests/dataset_manifest.json` (checksummed: `cache_sha256`) |
| **Configuration** (seeds, feature sets, models, thresholds, label source) | `backend/research/config.py` (single source; every experiment record stores a hash of the config it actually used) |
| **Experiment definitions** (what each experiment does) | `backend/research/experiments/classification.py`, `robustness.py`, `grounding.py` |
| **Results** (numeric values) | `backend/research/results/<experiment>/metrics.json` (per-experiment — authoritative over any table copy or this document if they ever diverge) |
| **Run provenance** (commit, timing, seeds actually used) | `backend/research/results/<experiment>/metadata.json` **per experiment** — authoritative over the top-level `tables/experiment_runs.csv` ledger, which was found stale for 4 of 11 rows during this document's verification (Section 29) |
| **Figures** | `backend/research/figures/*.png`, regenerated by `backend/research/figures.py` |
| **Tables** | `backend/research/tables/*.csv` / `*.json`, written by `backend/research/artifacts.py::write_table` |
| **Methodology** (production pipeline steps) | `docs/methodology.md`, cross-checked against the actual code in `app/imagery/`, `app/analysis/`, `app/models_ml/` |
| **Reproducibility** (how to rerun) | `backend/research/README.md` and `backend/research/run.py` (CLI is the executable source of truth; README describes it) |
| **This document** | A synthesis of the above, current as of the repository state described at the top of this file. If any number here ever conflicts with a freshly-read `metrics.json`, **the `metrics.json` wins** — regenerate this document rather than trusting a stale copy. |

---

## FACTS VERIFIED FROM REPOSITORY

- Dataset: 313,580 pixels, 23/26 groups present, `cache_sha256 = 8ff45eb8d1752cf8338ecde187b086c46ac1a749206c2430a272ea415cb91ee1`.
- Central finding (spatial_validation): random pixel 0.8413 acc / 0.8375 macro-F1 vs. spatial holdout 0.5325 acc / 0.5150 macro-F1; difference +0.3088 acc / +0.3225 macro-F1 — all four numbers read directly from `research/results/spatial_validation/metrics.json`.
- Baseline reproduction: 0.5325 acc / 0.5150 macro-F1 vs. published 0.5922 / 0.5268; gaps −0.0597 / −0.0118; within declared ±0.08 tolerance; reproduced bit-for-bit on rerun.
- Model comparison ranking by macro-F1: mlp (0.5585) > linear_svm (0.5499) > hist_gradient_boosting (0.5461) > random_forest (0.5150).
- Feature ablation: all macro-F1 deltas vs. full feature set ≤ 0.0044 in magnitude.
- Temporal transfer: largest asymmetry is p2024→p2021 (0.5118 acc) vs. p2024→p2024 (0.6382 acc), a −0.1264 gap.
- Confidence/calibration: ECE = 0.1541, overconfident in every populated bucket.
- Feature importance: NDVI ranked #1 by both impurity (0.13811) and permutation (0.09603) importance.
- Class imbalance: whole-dataset ratio 2.007; spatial-holdout test-split ratio 4.172; accuracy − balanced accuracy = −0.0104.
- Cloud robustness: 34 real observations; valid-pixel fraction 0.9433 (0–5% cloud) down to 0.7175 (10–20% cloud).
- Threshold sensitivity: Po Valley changed area 58.875% at threshold 0.05 vs. 16.799% at threshold 0.20.
- LLM grounding: 0 successful generations out of 6 attempted; all rejected with `InterpretationProviderError`, traced to a Groq 401 Unauthorized response.
- 161 total backend tests pass; 31 are research-specific (`tests/test_research.py`).
- A discrepancy exists between `tables/experiment_runs.csv` (stale runtimes for 4 experiments) and their own `metadata.json` files (updated runtimes) — metric *values* were verified identical across both; only timing metadata differs.

## OPEN / UNRESOLVED ITEMS

- The exact cause of the p2021→p2024 forward-transfer result exceeding same-period p2021→p2021 has not been isolated (label staleness vs. genuine seasonal/training-set-composition effects are both plausible, untested individually).
- `tables/experiment_runs.csv` has not been regenerated to match the current per-experiment `metadata.json` timings for `baseline`, `spatial_validation`, `model_comparison`, `feature_ablation` — cosmetic/provenance issue only, not a results issue.
- LLM grounding (H7) remains entirely unmeasured pending a working Groq API credential.
- No literature review has been performed; Section 38 ("Related Work") and any novelty claim remain open until one is done.
- Whether the feature-ablation near-null result would replicate under a different model architecture (only `random_forest` was ablated) is untested.
- No formal statistical-significance test has been applied to any of the reported gaps; all uncertainty reporting is limited to mean/std/min/max across 3 seeds.
