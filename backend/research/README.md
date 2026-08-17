# NatureVision research framework

A reproducible experimental layer over the NatureVision production pipeline.
It reuses the production feature contract, imagery access and change detection,
and adds controlled experiments, versioned artifacts and deterministic
evaluation on top.

This is an implementation and reproducibility document. It is not the paper,
and it contains no literature review, citations or claims of novelty.

---

## Research question

> How well do lightweight machine-learning models trained on Sentinel-2
> multispectral data generalize across geographically and temporally distinct
> environments, and how reliably can their outputs support evidence-grounded
> environmental interpretation?

## Hypotheses

| | Hypothesis | Experiment |
| --- | --- | --- |
| H1 | Random pixel validation overestimates performance relative to spatial holdout validation | `spatial_validation` |
| H2 | Adding derived spectral indices changes land-cover performance relative to raw bands | `feature_ablation` |
| H3 | Models lose performance when transferred across observation periods | `temporal_transfer` |
| H4 | Lightweight algorithms differ in spatial generalization behaviour | `model_comparison` |
| H5 | Prediction confidence is informative about correctness | `confidence_analysis` |
| H6 | Observation degradation reduces analysis reliability | `cloud_robustness` |
| H7 | Evidence grounding reduces unsupported statements in generated reports | `llm_grounding` |

Each experiment is built to be capable of contradicting its hypothesis. Results
are reported as measured, including where a hypothesis is not supported.

---

## Dataset

Real Sentinel-2 Level-2A pixels, sampled once and cached so every experiment
reads identical data.

- **Imagery**: Sentinel-2 L2A via Element84 Earth Search over AWS Open Data
- **Reference labels**: **ESA WorldCover 2021 v200** (CC BY 4.0)
- **Regions**: 13 study areas spanning humid tropical, temperate, boreal,
  Mediterranean, savanna and desert biomes
- **Periods**: two observation windows (`p2021`, `p2024`), same seasonal window
- **Features**: the production 11-element per-pixel vector (6 reflectance bands
  + NDVI, NDWI, NDBI, NBR and a SWIR-1/blue contrast)
- **Sampling**: class-stratified, capped per region-period

The manifest at `manifests/dataset_manifest.json` records, per group: scene id,
acquisition date, cloud cover, platform, MGRS tile, grid geometry, radiometric
calibration decision, class counts and the label source version.

**ESA WorldCover is a reference land-cover map produced by a model.** It is not
ground truth. Disagreement with it is not automatically classifier error, and
this framework never describes it as ground truth.

---

## Validation protocols

| Protocol | Construction | Purpose |
| --- | --- | --- |
| Random pixel | All pixels shuffled and divided | The optimistic protocol under test |
| Spatial holdout | Whole regions withheld | No test pixel shares a block with training |
| Temporal transfer | Train on one period, test on another | Measures temporal domain shift |

Spatial blocks are regions. Three regions are reserved for testing and one for
validation, so hyperparameter choices never touch the test blocks.

Leakage is asserted, not assumed: `tests/test_research.py` fails if a test
sample shares a spatial block with training, if a temporal split trains on the
target period, or if any sample appears in two partitions.

---

## Commands

```bash
cd backend

# Build the pixel cache (network; ~15-25 min for 26 groups)
python -m research.run --experiment dataset

# Core suite — runs entirely from the cache, no network
python -m research.run --experiment core

# A single experiment
python -m research.run --experiment spatial_validation

# Experiments that fetch imagery or call the language provider
python -m research.run --experiment network

# Fast structural check, safe for CI
python -m research.run --experiment smoke

python -m research.run --list
```

Grounding cases are extracted from analyses the product has actually run:

```bash
python -c "from research.experiments.grounding import build_cases_from_database as b; b()"
python -m research.run --experiment llm_grounding   # needs GROQ_API_KEY
```

If no completed analyses exist, case building fails with that message rather
than inventing environmental numbers.

---

## Experiments

| Experiment | Network | What it varies |
| --- | --- | --- |
| `baseline` | no | Reproduces the production spatial-holdout result |
| `spatial_validation` | no | Validation protocol (random vs spatial) |
| `model_comparison` | no | Estimator (RF, HistGB, calibrated linear SVM, MLP) |
| `feature_ablation` | no | Feature subset (A raw bands → D full) |
| `temporal_transfer` | no | Train/test observation period |
| `confidence_analysis` | no | Confidence against correctness, calibration error |
| `feature_importance` | no | Impurity and permutation importance |
| `class_imbalance` | no | Class distribution and its effect on headline metrics |
| `cloud_robustness` | yes | Scene cloud cover against usable pixels and NDVI |
| `threshold_sensitivity` | yes | Change threshold (0.05–0.20) |
| `llm_grounding` | yes | Generation mode (baseline vs grounded) |

Randomised experiments run at seeds **42, 123, 2024** and report mean, standard
deviation, minimum and maximum. A single seed is never presented as if it
expressed uncertainty.

---

## Outputs

```
research/
  cache/            cached pixel table (not committed — rebuildable)
  configs/          reserved for alternative configurations
  manifests/        dataset manifest, grounding cases
  results/<name>/   config.json, metadata.json, metrics.json, tables/, figures/
  figures/          publication figures
  tables/           publication tables (CSV + JSON)
```

Every `metadata.json` records the research version, dataset version, git commit,
seeds, wall-clock runtime, configuration hash, dataset cache SHA-256,
contributing scene ids, label source and platform — which is what makes a number
in a table traceable to the code and data that produced it.

`tables/experiment_runs.csv` is the run ledger. It accumulates across
invocations, so running the suite in batches still leaves one table describing
every experiment, with failures recorded as failures rather than omitted.

---

## Methodological safeguards

- Hyperparameters are fixed in advance and never tuned against the test split.
- Permutation importance is computed on the validation block, never the test
  block.
- Feature construction uses only per-pixel reflectance available at prediction
  time; labels never enter features.
- Confusion matrices always span the full class space in a fixed order, so
  matrices from different experiments are comparable.
- Accuracy is never reported alone; balanced accuracy, macro-F1 and per-class
  figures accompany every result.
- Production defaults are not modified by any experiment; the change-threshold
  sweep is research-only.

---

## Limitations

- **ESA WorldCover 2021 v200 is a reference map, not ground truth.** Its own
  error is inside every accuracy figure reported here.
- Labels are fixed at the 2021 epoch while imagery spans two periods, so genuine
  land-cover change after 2021 appears as label noise in the later period. The
  temporal experiment therefore conflates domain shift with label staleness.
- Thirteen regions is a small sample of global land cover; results describe
  these landscapes, not the planet.
- Pixels are class-stratified, so cached class proportions do not reflect true
  landscape proportions.
- Scene-level cloud cover describes the whole tile, not the analysed region.
- Per-pixel classification uses no spatial context, producing salt-and-pepper
  error at class boundaries and on mixed pixels.
- NDVI is a reflectance-derived greenness proxy, not a measure of biomass,
  carbon stock, habitat quality or biodiversity.
- The grounding evaluator is lexical, not semantic: it matches numbers against
  the evidence and screens a fixed list of causal phrasings. It cannot judge
  whether an interpretation is scientifically sound.
- Language-model output is non-deterministic and the hosted model may change
  without notice, so grounding figures describe one provider at one point in
  time.
- Not every region-period group is guaranteed to build; groups whose imagery
  cannot be retrieved are absent from the manifest and are reported as absent
  rather than substituted.

---

## Terminology

The framework uses **land-cover classification** (not "environmental truth"),
**reference labels** (not "ground truth"), **vegetation-index change** (not
"deforestation"), **model prediction** for classifier output, **language-model
interpretation** for generated text, and **observed** only for measured
satellite-derived quantities.

---

## Data attribution

- Contains modified Copernicus Sentinel data. Sentinel-2 Level-2A, free and open.
- ESA WorldCover 2021 v200, © ESA WorldCover project, licensed CC BY 4.0,
  <https://esa-worldcover.org/>.
