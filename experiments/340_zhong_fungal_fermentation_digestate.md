# Exp340: Zhong et al. Fungal Fermentation on Digestate

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC)
**Paper:** Zhong et al. 2016 "Fungal fermentation on anaerobic digestate for lipid-based biofuel production" (Biotechnol Biofuels 9:253)
**Binary:** `validate_fungal_fermentation_digestate`
**Status:** PASS (10 checks)


---

## Hypothesis

Aerobic fungal fermentation operating on anaerobic substrate tests QS dynamics at the oxygen regime boundary. Validates Monod growth kinetics and Anderson W transition at the aerobic-anaerobic interface.

## Method

Monod growth parameters (μ_max, K_s) are fitted to fungal biomass data on digestate. Anderson disorder W is computed for aerobic vs anaerobic phases and at the oxygen boundary. Community diversity tracks the regime transition.

## Results

All 10 checks PASS. See `cargo run --bin validate_fungal_fermentation_digestate` for full output.

## Key Finding

Fungal growth follows Monod kinetics; the aerobic process on anaerobic substrate shows a measurable W regime shift at the oxygen boundary.

## Modules Validated

- bio::diversity
- barracuda::spectral
