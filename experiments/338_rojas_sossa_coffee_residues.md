# Exp338: Rojas-Sossa et al. Coffee Processing Residues

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC)
**Paper:** Rojas-Sossa et al. 2017 "Effects of coffee processing residues on anaerobic microorganisms and corresponding digestion performance" (Bioresour Technol 245:714-723)
**Binary:** `validate_anaerobic_coffee_residues`
**Status:** PASS (10 checks)

---

## Hypothesis

Substrate perturbation (coffee waste addition) shifts the anaerobic community composition in predictable ways — maps to disorder perturbation in Anderson model.

## Method

Community composition shifts are quantified before and after coffee residue addition. Anderson disorder W is computed for control vs perturbed conditions. Substrate inhibition metrics are correlated with W and QS probability. Diversity indices track perturbation magnitude.

## Results

All 11 checks PASS. See `cargo run --bin validate_anaerobic_coffee_residues` for full output.

## Key Finding

Coffee waste → increased substrate inhibition → higher disorder W → reduced QS probability.

## Modules Validated

- bio::diversity
- barracuda::spectral
