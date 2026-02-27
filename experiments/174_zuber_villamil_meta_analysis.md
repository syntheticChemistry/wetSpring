# Exp174: Zuber & Villamil 2016 Meta-Analysis

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Zuber & Villamil "Meta-analysis approach to assess effect of tillage on microbial biomass and enzyme activities" Soil Biology and Biochemistry 97:176-187 (2016)
**Binary:** `validate_notill_meta_analysis`
**Status:** PASS (20 checks)

---

## Hypothesis

This experiment validates that effect sizes and confidence intervals from the Zuber & Villamil meta-analysis are correctly reproduced, and that Anderson-based predictions match MBC (microbial biomass carbon) increase under no-till.

## Method

Validation computes effect sizes and CIs for tillage effects on microbial biomass and enzyme activities. Anderson disorder parameters are used to predict MBC changes, which are compared against meta-analytic estimates.

## Results

All 20 checks PASS. See `cargo run --bin validate_notill_meta_analysis` for full output.

## Key Finding

Effect sizes, CI verification, Anderson predicts MBC increase.

## Modules Validated

- `bio::diversity`
- `special::norm_cdf`
