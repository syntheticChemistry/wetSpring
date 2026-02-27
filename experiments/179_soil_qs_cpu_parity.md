# Exp179: Soil QS CPU Parity Benchmark

**Date:** February 2026
**Track:** 4 (CPU parity)
**Paper:** N/A (benchmark)
**Binary:** `validate_soil_qs_cpu_parity`
**Status:** PASS (49 checks)

---

## Hypothesis

This experiment validates that all Track 4 CPU modules produce correct, reproducible results across 8 timed domains, with pure Rust math validated against reference implementations.

## Method

Validation runs 8 domains with timing instrumentation. Each domain exercises Track 4 primitives (Anderson, diversity, QS biofilm, norm_cdf, etc.) and verifies numerical correctness and CPU parity.

## Results

All 49 checks PASS. See `cargo run --bin validate_soil_qs_cpu_parity` for full output.

## Key Finding

8 domains timed, pure Rust math validated.

## Modules Validated

All Track 4 CPU modules.
