# Exp182: Soil QS metalForge Cross-Substrate

**Date:** February 2026
**Track:** 4 (metalForge)
**Paper:** N/A (metalForge dispatch)
**Binary:** `validate_soil_qs_metalforge`
**Status:** PASS (14 checks)

---

## Hypothesis

This experiment validates that metalForge dispatch produces identical results on CPU and GPU for all Track 4 domains, enabling transparent substrate switching.

## Method

Validation runs the soil QS pipeline through metalForge with both CPU and GPU backends. Results are compared to verify CPU = GPU parity across all Track 4 domains.

## Results

All 14 checks PASS. See `cargo run --bin validate_soil_qs_metalforge` for full output.

## Key Finding

CPU = GPU for all Track 4 domains.

## Modules Validated

metalForge dispatch.
