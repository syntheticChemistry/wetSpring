# Experiment 009: Feature Pipeline End-to-End Validation

**Track**: Track 2 (PFAS / blueFish) — Cross-track (also validates Track 1 signal processing)
**Date**: 2026-02-18
**Status**: ACTIVE
**Depends on**: Exp005 (asari baseline)

---

## Objective

Validate wetSpring's Rust feature extraction pipeline (`bio::eic`, `bio::signal`,
`bio::feature_table`) against asari's MT02 baseline from Exp005. This closes the
gap where three Rust modules have unit tests but no end-to-end comparison against
the Python baseline.

## Baseline (from Exp005)

| Field | Value |
|-------|-------|
| Tool | asari 1.13.1 |
| Dataset | MT02 demo (8 mzML files, HILIC-pos, Orbitrap HRMS) |
| Features | 5,951 (filtered from 8,659 total) |
| Compounds | 4,107 unique |
| Parameters | 5 ppm, min 6 scans, SNR >= 2, min_peak_height 100,000 |
| Feature table | `experiments/results/005_asari/preferred_Feature_table.tsv` |

## What We Validate

### Level 1: Mass track detection
- Rust `bio::eic::detect_mass_tracks` on MT02 mzML files
- Compare count and m/z coverage against asari's mass track count

### Level 2: EIC extraction + peak detection
- Rust `bio::eic::extract_eics` + `bio::signal::find_peaks` on detected tracks
- Verify peaks are found in similar RT ranges as asari features

### Level 3: Feature table comparison
- Rust `bio::feature_table::extract_features` on MT02
- Compare: feature count, m/z range, RT range, area distribution
- Match Rust features to asari features by m/z (5 ppm) + RT (50 sec)
- Report: matched %, unmatched Rust, unmatched asari

## Acceptance Criteria

1. Mass track count within 2x of asari's parent_masstrack_id range
2. Feature count within same order of magnitude as asari (thousands)
3. m/z range covers asari's m/z range (83-999)
4. RT range covers asari's RT range
5. At least 30% of asari features matched by m/z+RT to a Rust feature
   (asari uses sophisticated alignment; exact match not expected)

## Validation Binary

`cargo run --bin validate_features`

Uses `WETSPRING_MZML_DIR` env var (default: `../data/exp005_asari/MT02/MT02Dataset`).

## Outputs

- `validate_features` checks in Validator harness
- Comparison statistics printed to stdout
