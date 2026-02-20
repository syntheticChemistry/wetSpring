# Exp041 — EPA PFAS National-Scale ML Classification

**Date:** February 20, 2026
**Status:** COMPLETE — 14/14 checks PASS
**Track:** 2 (Analytical Chemistry / PFAS)
**Proxy for:** Paper #22, Jones PFAS fate-and-transport modeling

---

## Objective

Validate PFAS contamination classification using the Rust decision tree
module on surface water concentration data. Extends existing Michigan
EGLE dataset (3,719 samples, 30+ analytes) with EPA UCMR 5 national
data (29 PFAS analytes, thousands of water systems).

## Data Sources

- **Michigan EGLE:** 3,719 surface water samples with GPS, 30+ PFAS analytes (already available)
- **EPA UCMR 5:** National drinking water PFAS survey, 2023-2025 (download pending)
- **EPA PFOS surface water:** GPS + concentration time-series (download pending)
- **Download:** `scripts/download_public_data.sh --epa-pfas`

## Model

Decision stump (depth-1 tree) classifying samples as high/low PFAS
based on total PFAS concentration threshold (70 ng/L, EPA advisory proxy).

Features: PFOS, PFOA, PFHxS concentrations + latitude + total PFAS.

## Validation Checks (14/14)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | Tree node count | 3 | PASS |
| 2 | Tree leaf count | 2 | PASS |
| 3 | Tree depth | 1 | PASS |
| 4 | predict(low) | 0 | PASS |
| 5 | predict(high) | 1 | PASS |
| 6 | predict(boundary_below) | 0 | PASS |
| 7 | predict(boundary_above) | 1 | PASS |
| 8 | batch length | 4 | PASS |
| 9-12 | batch predictions | correct | PASS |
| 13 | predict deterministic | ✓ | PASS |
| 14 | batch deterministic | ✓ | PASS |

## GPU Promotion Path

`DecisionTree::predict_batch` → one workgroup per sample, uniform buffer
for tree structure. Already flat-array layout.

## Run

```bash
cargo run --bin validate_epa_pfas_ml
python3 scripts/epa_pfas_ml_baseline.py
```
