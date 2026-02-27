# Exp171: Feng 2024 Pore-Size Diversity

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Feng et al. "Composition and metabolism of microbial communities in soil pores" Nature Communications 15:3578 (2024)
**Binary:** `validate_soil_pore_diversity`
**Status:** PASS (27 checks)

---

## Hypothesis

This experiment validates that pore class maps to effective dimension, and that effective dimension drives diversity metrics (Shannon, Bray-Curtis) in soil pore microbial communities.

## Method

Validation links pore size classes to effective spatial dimension, then computes diversity indices (Shannon, Bray-Curtis) to verify composition and metabolism patterns consistent with the Feng et al. pore-scale model.

## Results

All 27 checks PASS. See `cargo run --bin validate_soil_pore_diversity` for full output.

## Key Finding

Pore class → effective dimension → diversity (Shannon, Bray-Curtis).

## Modules Validated

- `bio::diversity`
- `bio::qs_biofilm`
