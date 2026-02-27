# Exp176: Tecon & Or 2017 Biofilm-Aggregate

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Tecon & Or "Biophysics of bacterial biofilms—insights from soil" Biochimica et Biophysica Acta 1858:2774-2781 (2017)
**Binary:** `validate_soil_biofilm_aggregate`
**Status:** PASS (23 checks)

---

## Hypothesis

This experiment validates that water film thickness governs QS diffusion, and that aggregate surface area correctly predicts colonization extent in soil biofilm-aggregate systems.

## Method

Validation couples water film thickness to autoinducer diffusion and QS activation. Aggregate surface geometry is used to compute colonization capacity, matching the Tecon & Or biophysical framework.

## Results

All 23 checks PASS. See `cargo run --bin validate_soil_biofilm_aggregate` for full output.

## Key Finding

Water film thickness → QS diffusion, aggregate surface → colonization.

## Modules Validated

- `bio::qs_biofilm`
- `bio::ode`
