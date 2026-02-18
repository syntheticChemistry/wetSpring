# Experiment 010: Peak Detection Baseline (scipy.signal.find_peaks)

**Track**: Cross-track (signal processing used by Track 1 + Track 2)
**Date**: 2026-02-18
**Status**: ACTIVE
**Depends on**: None (uses synthetic + controlled data)

---

## Objective

Validate wetSpring's `bio::signal::find_peaks` against `scipy.signal.find_peaks`
on identical input data to confirm that the Rust implementation matches scipy's
peak-finding behavior for the parameter combinations used in the pipeline.

## Baseline Generation

Script: `scripts/generate_peak_baselines.py`

| Field | Value |
|-------|-------|
| Tool | scipy.signal.find_peaks |
| scipy version | 1.14+ |
| numpy version | 2.1+ |
| Output | `experiments/results/010_peak_baselines/*.dat` |

## Test Cases

| Case | N | Peaks | Description |
|------|---|-------|-------------|
| `single_gaussian` | 200 | 1 | Gaussian + noise, height/prominence/width filters |
| `three_chromatographic` | 500 | 3 | Three LC-MS-like peaks with noise |
| `noisy_with_spikes` | 1000 | 3 | High-noise baseline with 3 injected spikes, distance filter |
| `overlapping_peaks` | 200 | 1 | Two overlapping Gaussians, prominence filter |
| `monotonic_no_peaks` | 100 | 0 | Linear ramp (no peaks expected) |

## Acceptance Criteria

1. Peak count matches scipy exactly for all 5 test cases
2. Peak indices within ±1 of scipy (boundary rounding differences)
3. Peak heights within 1% of scipy (same data, should be exact)

## Validation Binary

`cargo run --bin validate_peaks`

## Outputs

- `validate_peaks` checks in Validator harness
- Per-case peak count, index match, and height match
