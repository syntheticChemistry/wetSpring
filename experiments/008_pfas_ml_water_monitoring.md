# Experiment 008: PFAS ML Water Monitoring

**Date**: 2026-02-19
**Status**: DONE (Phase 3 — CPU parity via synthetic proxy, sklearn model export)
**Track**: 2 (PFAS Analytical Chemistry)

---

## Objective

Build and validate a machine-learning-based PFAS water contamination predictor
using Michigan DEQ surface water monitoring data. Validate against published
accuracy metrics from Water Research 2023 studies on PFAS environmental modeling.

## Data Sources

### Primary: Michigan EGLE PFAS Surface Water Sampling

- **Source**: Michigan Department of Environment, Great Lakes, and Energy (EGLE)
- **Portal**: https://gis-egle.hub.arcgis.com/datasets/egle::pfas-surface-water-sampling
- **Records**: 3,383 surface water samples
- **Analytes**: PFOA, PFOS, PFHxS, PFBS, PFHxA, PFPeA, PFBA, and others (ppt)
- **Format**: CSV/GeoJSON via ArcGIS Hub download
- **Last updated**: April 2024 (data collected through June 2023)

### Download Instructions

1. Navigate to https://gis-egle.hub.arcgis.com/datasets/egle::pfas-surface-water-sampling
2. Click "Download" → select CSV or Spreadsheet format
3. Save to `data/michigan_deq_pfas/pfas_surface_water.csv`

### Supplementary: Jones Lab PFAS Library (Nature Scientific Data 2025)

- **DOI**: 10.1038/s41597-024-04363-0
- **Content**: 175 PFAS, 281 ion types, m/z + RT + CCS values
- **Use**: Reference library for suspect screening feature engineering

## Design

### Phase 1: Data Exploration & Feature Engineering (Python baseline)

1. Load Michigan DEQ PFAS surface water data
2. Exploratory analysis: distribution of PFAS concentrations, spatial patterns,
   co-occurrence of compounds
3. Feature engineering:
   - Geospatial features (lat/lon, distance to known PFAS sources)
   - Temporal features (sampling date)
   - Inter-analyte correlations (PFAS fingerprint patterns)
   - Total PFAS concentration metrics

### Phase 2: ML Model Training (Python baseline)

1. Task: Binary classification — PFAS above/below advisory threshold
   (e.g., EPA PFOA+PFOS 4 ppt advisory)
2. Models: Random Forest, Gradient Boosted Trees (scikit-learn/XGBoost)
3. Validation: 5-fold stratified cross-validation
4. Metrics: Accuracy, precision, recall, F1, AUC-ROC

### Phase 3: Rust Implementation

1. Port feature engineering to Rust (sovereign implementation)
2. Implement decision tree inference in Rust (no external ML crate)
3. Validate: Rust predictions match Python predictions exactly

### Phase 4: GPU Acceleration

1. Feature-level batch prediction via ToadStool GEMM/FMR
2. Benchmark Python → Rust CPU → Rust GPU inference

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Binary classification F1 | ≥ 0.80 | Water Research 2023 baseline |
| AUC-ROC | ≥ 0.85 | Conservative target |
| Rust vs Python prediction match | 100% | Exact parity |
| Feature count | ≥ 10 | Meaningful engineering |

## Validation Against Published Work

The open science approach: Michigan DEQ data is public. Our pipeline processes
it sovereignly. We validate against published environmental modeling accuracy.
If results are compelling, authors would want to validate with us.

## Dependencies

- Michigan DEQ data download (manual from ArcGIS portal)
- Jones Lab PFAS library (open access, Nature Scientific Data)
- Python: numpy, scipy, scikit-learn (baseline)
- Rust: wetspring-barracuda (sovereign implementation)
