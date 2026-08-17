# NatureVision

**Geospatial environmental intelligence from Sentinel-2 satellite imagery.**

NatureVision is a reproducible geospatial AI platform for analyzing vegetation
indices, land-cover patterns, and environmental change from real Sentinel-2
observations.

The project combines a production-oriented analysis pipeline with a controlled
research framework investigating how remote-sensing machine-learning models
generalize across geographic and temporal conditions.

---

## Research

The central research question is:

> **How do spatial and temporal validation strategies affect the measured
> generalization performance of lightweight machine-learning models for
> Sentinel-2 land-cover classification?**

The research framework evaluates:

- Random-pixel vs spatially separated validation
- Lightweight model architectures
- Spectral feature ablation
- Temporal transfer between observation periods
- Prediction confidence and calibration
- Feature importance
- Class imbalance
- Cloud/data-quality robustness
- NDVI change-threshold sensitivity
- Evidence-grounded environmental reporting

All experiments are reproducible and record their configuration, random seeds,
dataset provenance, runtime, Git commit, and dataset digest.

### Dataset

The experiments use real Sentinel-2 Level-2A observations and ESA WorldCover
2021 v200 reference labels.

The current research cache contains **313,580 pixels** across two observation
periods.

Three source groups that could not be retrieved are recorded as missing rather
than being replaced with synthetic data.

### Validation

A central experiment compares random pixel-level validation with geographically
separated spatial holdout validation.

The current results show a substantial difference:

| Protocol | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Random pixel | ~0.84 | ~0.84 |
| Spatial holdout | ~0.53 | ~0.51 |

This gap is the primary result currently being investigated. It demonstrates
why evaluation methodology is important when measuring generalization of
remote-sensing models.

The research does not assume that any hypothesis is correct; results are
generated directly from the experiments.

### Current model experiments

The framework compares:

- Random Forest
- Histogram Gradient Boosting
- Linear SVM
- MLP

and evaluates feature configurations ranging from raw spectral bands to
derived vegetation and spectral indices.

The current feature-ablation results show only small differences between
feature configurations, with macro-F1 remaining close to the baseline across
the tested feature sets.

Temporal experiments additionally evaluate transfer between the 2021 and 2024
observation periods.

---

## Environmental analysis

The production pipeline performs:

```text
Geographic region
       ↓
Sentinel-2 observations
       ↓
Cloud / quality masking
       ↓
Spectral processing
       ↓
NDVI and environmental indices
       ↓
Temporal change detection
       ↓
Land-cover classification
       ↓
Evidence package
       ↓
Grounded environmental interpretation
