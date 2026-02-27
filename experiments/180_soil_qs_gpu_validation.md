# Exp180: Soil QS GPU Validation

**Date:** February 2026
**Track:** 4 (GPU)
**Paper:** N/A (GPU parity)
**Binary:** `validate_soil_qs_gpu`
**Status:** PASS (23 checks)

---

## Hypothesis

This experiment validates that GPU implementations of FMR, BrayCurtisF64, Anderson 3D, and ODE solvers match CPU results for all Track 4 domains.

## Method

Validation runs FMR (finite-size scaling), Bray-Curtis diversity, Anderson 3D localization, and ODE integration on GPU. Results are compared bit-exactly or within tolerance against CPU reference implementations.

## Results

All 23 checks PASS. See `cargo run --bin validate_soil_qs_gpu` for full output.

## Key Finding

FMR + BrayCurtisF64 + Anderson 3D + ODE on GPU.

## Modules Validated

GPU parity for Track 4.
