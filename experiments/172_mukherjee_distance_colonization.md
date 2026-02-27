# Exp172: Mukherjee 2024 Distance Colonization

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Mukherjee et al. "Manipulating the physical distance between cells during soil colonization reveals the importance of biotic interactions" Environmental Microbiome 19:14 (2024)
**Binary:** `validate_soil_distance_colonization`
**Status:** PASS (23 checks)

---

## Hypothesis

This experiment validates that autoinducer diffusion, QS biofilm ODE dynamics, and cooperation collapse correctly capture the dependence of colonization success on physical cell-to-cell distance.

## Method

Validation combines autoinducer diffusion models with QS biofilm ODE integration to test whether biotic interactions (cooperation) collapse at critical distances, matching the Mukherjee experimental design.

## Results

All 23 checks PASS. See `cargo run --bin validate_soil_distance_colonization` for full output.

## Key Finding

Autoinducer diffusion + QS biofilm ODE + cooperation collapse.

## Modules Validated

- `bio::qs_biofilm`
- `bio::cooperation`
- `bio::ode`
