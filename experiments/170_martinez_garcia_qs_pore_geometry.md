# Exp170: Martínez-García 2023 QS-Pore Geometry Coupling

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Martínez-García et al. "Spatial structure, chemotaxis and quorum sensing shape bacterial biomass accumulation in complex porous media" Nature Communications 14:8332 (2023)
**Binary:** `validate_soil_qs_pore_geometry`
**Status:** PASS (26 checks)

---

## Hypothesis

This experiment validates that non-linear connectivity in porous media maps to Anderson disorder W, and that P(QS) via norm_cdf correctly predicts quorum-sensing emergence in complex pore geometries.

## Method

Validation exercises the coupling between pore geometry (non-linear connectivity), Anderson localization parameters, and quorum-sensing probability via the normal CDF. Checks verify that spatial structure and chemotaxis constraints produce biomass accumulation patterns consistent with the Martínez-García model.

## Results

All 26 checks PASS. See `cargo run --bin validate_soil_qs_pore_geometry` for full output.

## Key Finding

Non-linear connectivity → Anderson disorder W; P(QS) via norm_cdf.

## Modules Validated

- `bio::qs_biofilm`
- `bio::diversity`
- `special::norm_cdf`
