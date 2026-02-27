# Exp175: Liang 2015 31-Year Tillage

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Liang et al. "Long term tillage, cover crop, and fertilization effects on microbial community structure, activity" Soil Biology and Biochemistry 89:37-44 (2015)
**Binary:** `validate_notill_longterm_tillage`
**Status:** PASS (19 checks)

---

## Hypothesis

This experiment validates that the 2×2×2 factorial design (tillage × cover crop × fertilization) correctly produces Shannon and Pielou diversity patterns, and that 31-year temporal recovery trajectories are reproduced.

## Method

Validation implements the factorial design and computes diversity indices (Shannon, Pielou) across treatments. Long-term temporal dynamics are checked against the Liang et al. 31-year dataset.

## Results

All 19 checks PASS. See `cargo run --bin validate_notill_longterm_tillage` for full output.

## Key Finding

2×2×2 factorial, Shannon/Pielou, 31-year temporal recovery.

## Modules Validated

- `bio::diversity`
