# Exp177: Rabot 2018 Structure-Function

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Rabot et al. "Soil structure as an indicator of soil functions: A review" Geoderma 314:122-137 (2018)
**Binary:** `validate_soil_structure_function`
**Status:** PASS (16 checks)

---

## Hypothesis

This experiment validates that structural properties (porosity, aggregation, pore connectivity) map to Anderson parameters, and that those parameters correctly predict functional outcomes (nutrient cycling, habitat quality).

## Method

Validation links soil structural metrics to Anderson disorder and connectivity parameters. Functional outcomes are computed from the structure-function relationships described in the Rabot review.

## Results

All 16 checks PASS. See `cargo run --bin validate_soil_structure_function` for full output.

## Key Finding

Structural properties → Anderson parameters → functional outcomes.

## Modules Validated

- `special::norm_cdf`
