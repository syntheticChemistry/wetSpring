# Exp173: Islam 2014 Brandt Farm Soil Health

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Islam et al. "No-till and conservation agriculture in the United States: An example from the David Brandt farm, Carroll, Ohio" ISWCR 2:97-107 (2014)
**Binary:** `validate_notill_brandt_farm`
**Status:** PASS (14 checks)

---

## Hypothesis

This experiment validates that published Brandt farm metrics are correctly reproduced, and that no-till management yields lower disorder and higher diversity in our Anderson-based framework.

## Method

Validation compares computed metrics (disorder, diversity) against published Brandt farm soil health indicators. No-till conditions are mapped to reduced Anderson disorder and elevated diversity indices.

## Results

All 14 checks PASS. See `cargo run --bin validate_notill_brandt_farm` for full output.

## Key Finding

Published metrics validated; no-till → lower disorder → higher diversity.

## Modules Validated

- `bio::diversity`
- `special::norm_cdf`
