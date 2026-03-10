# Exp336: Yang et al. Anaerobic Co-digestion Phylogenetics

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC)
**Paper:** Yang et al. 2016 "Phylogenetic analysis of anaerobic co-digestion of animal manure and corn stover reveals linkages between bacterial communities and digestion performance" (Adv Microbiol 6:879-897)
**Binary:** `validate_anaerobic_codigestion`
**Status:** PASS (12 checks)

---

## Hypothesis

Validate that community composition diversity indices distinguish anaerobic co-digestion communities, and that biogas production follows modified Gompertz kinetics parameterized from published data.

## Method

Phylogenetic diversity and Shannon indices are computed from published 16S profiles for manure/corn stover co-digestion. Modified Gompertz model parameters (P_max, R_m, λ) are fitted to published methane yield curves. UniFrac distances separate communities by substrate C/N ratio.

## Results

All 12 checks PASS. See `cargo run --bin validate_anaerobic_codigestion` for full output.

## Key Finding

Shannon diversity tracks substrate C/N ratio; modified Gompertz model fits published methane yield data.

## Modules Validated

- bio::diversity
- bio::unifrac
