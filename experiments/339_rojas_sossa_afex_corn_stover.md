# Exp339: Rojas-Sossa et al. AFEX Corn Stover Pretreatment

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC)
**Paper:** Rojas-Sossa et al. 2019 "Effect of ammonia fiber expansion (AFEX) treated corn stover on anaerobic microbes and corresponding digestion performance" (Biomass Bioenergy 127:105263)
**Binary:** `validate_anaerobic_afex_stover`
**Status:** PASS (11 checks)

---

## Hypothesis

Pretreated substrate (AFEX corn stover) provides a controlled perturbation — validates the ΔW prediction from substrate accessibility changes.

## Method

Community composition and Anderson disorder W are compared for raw vs AFEX-pretreated corn stover. Substrate accessibility metrics (hydrolysis rate, fiber digestibility) are correlated with ΔW. Methane yield and community stability indices validate the disorder reduction hypothesis.

## Results

All 13 checks PASS. See `cargo run --bin validate_anaerobic_afex_stover` for full output.

## Key Finding

AFEX pretreatment → more accessible substrate → lower disorder → improved community stability and methane yield.

## Modules Validated

- bio::diversity
- barracuda::spectral
