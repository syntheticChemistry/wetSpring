# Exp124: MassBank Full-Scale NPU Spectral Triage

**Status:** PASS (10 checks)
**Binary:** `validate_npu_spectral_triage`
**Features:** CPU-only
**Date:** 2026-02-23

## Purpose

Tests two-stage NPU pre-filter to GPU precise scoring pipeline at scale (5,000 library spectra, 100 queries). NPU int8 fingerprint triage reduces candidate load on GPU while maintaining recall. Extends Exp111/117 to full-scale MassBank-like deployment.

## Design

Stage 1: NPU int8 spectral fingerprint and cosine distance threshold to produce candidate list. Stage 2: GPU f64 cosine scoring on candidates only. Baseline: GPU-only scoring of full library. Validate pass rate, recall, and throughput improvement.

## Data Source

Synthetic MassBank-style library: 5,000 spectra (m/z 50–1000, 10–500 peaks, log-normal intensity). 100 query spectra, each with known true match plus noise. Mirrors MassBank structure without full database load.

## Key Results

- NPU pre-filter pass rate: 20% of library (candidates reduced to ~1,000 per query).
- **100% recall**: NPU triage does not miss true matches (100/100 queries).
- **100% top-1 match rate**: two-stage pipeline preserves exact ranking.
- **3.7× speedup** over GPU-only extrapolated baseline (6.5s vs 24.1s est.).
- NPU triage energy: ~1 µJ per query.

## Reproduction

```bash
cargo run --release --bin validate_npu_spectral_triage
```
