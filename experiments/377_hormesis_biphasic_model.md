# Exp377: Hormesis Biphasic Dose-Response Model

**Status:** PROPOSED
**Date:** 2026-03-19
**Binary:** `validate_hormesis_biphasic` (to create)
**Feature gate:** none
**Track:** Anderson Hormesis (cross-spring)

## Purpose

Validate the biphasic dose-response model (`bio::hormesis`) against known
hormesis curves from Calabrese & Mattson (2017). Establishes that the
Hill-based stimulation × inhibition model correctly produces the canonical
J-shaped hormetic curve and that the Anderson disorder mapping connects
the hormetic zone to the near-critical regime (W ≈ W_c).

## Scientific Question

Does the hormesis model predict the correct shape and amplitude of biphasic
dose-response curves? Does mapping dose → Anderson disorder correctly place
the hormetic zone near W_c?

## Validation Targets

| Check | Expected | Tolerance | Source |
|-------|----------|-----------|--------|
| R(dose=0) = 1.0 | 1.0 | `ANALYTICAL_F64` | Model definition |
| R(dose→∞) → 0.0 | < 0.01 | `ASYMPTOTIC_LIMIT` | Hill asymptotics |
| Peak response > 1.0 | True | — | Hormesis definition |
| Peak dose between K_stim and K_inh | True | — | Model structure |
| Hormetic zone width > 0 | True | — | Model structure |
| W at hormetic peak ≈ W_c (±margin) | True | 0.1 × W_c | Anderson mapping |
| Subthreshold regime at dose ≈ 0 | `Subthreshold` | — | Classification |
| Toxic regime at high dose | `Toxic` | — | Classification |

## Chain

Anderson QS (Exp107–156) → Gonzales IC50 (Exp280) → **This** → Trophic Cascade (Exp378) → Joint Colonization (Exp379)

## Cross-Spring

- healthSpring exp099 (hormesis bonus — sub-threshold adaptive response)
- groundSpring exp_trophic_01 (soil trophic lattice)

## Modules Tested

- `bio::hormesis::response`
- `bio::hormesis::evaluate`
- `bio::hormesis::sweep`
- `bio::hormesis::find_peak`
- `bio::hormesis::hormetic_zone`
- `bio::hormesis::dose_to_disorder`
- `bio::hormesis::predict_hormetic_zone_from_wc`
