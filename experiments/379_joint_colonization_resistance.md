# Exp379: Joint Colonization Resistance Surface (healthSpring × wetSpring)

**Status:** PROPOSED
**Date:** 2026-03-19
**Binary:** `validate_colonization_resistance` (to create)
**Feature gate:** none
**Track:** Anderson Hormesis / Low-Affinity Binding (joint cross-spring)

## Purpose

Compute the colonization resistance surface — a 3D manifold in (adhesion
strength, species diversity, epithelial disorder) space — where commensal
colonization exceeds 90%. Validates the Anderson prediction that many weak
binders (delocalized regime) produce more robust resistance than few strong
binders (localized regime).

## Scientific Question

Does the shape of the colonization resistance surface confirm that diversity
of weak binders outperforms monoculture of strong binders, as predicted
by Anderson delocalization? Where is the phase boundary?

## Joint Experiment IDs

| ID | Spring | Description |
|----|--------|-------------|
| exp097 | healthSpring | Affinity landscape (18/18 PASS) |
| exp098 | healthSpring | Toxicity landscape (22/22 PASS) |
| exp_joint_01 | wetSpring | Anderson lattice with adhesion-modulated hopping |
| exp_joint_02 | joint | Full 3D surface + phase diagram |

## Validation Targets

| Check | Expected | Tolerance | Source |
|-------|----------|-----------|--------|
| More species → higher resistance fraction | True | — | Anderson delocalization |
| Single species (N=1) resistance < many (N=15) | True | — | Diversity advantage |
| IPR(uniform binding) = 1/N | 0.125 (N=8) | `ANALYTICAL_F64` | IPR definition |
| IPR(concentrated binding) = 1.0 | 1.0 | `ANALYTICAL_F64` | IPR definition |
| Localization length(uniform) = N | 8.0 | `ANALYTICAL_F64` | ξ = 1/IPR |
| Selectivity index from coincidence > 100 | True | — | Composite binding |
| Surface has 27 points (3×3×3) | 27 | `EXACT` | Sweep structure |

## Modules Tested

- `bio::binding_landscape::colonization_resistance`
- `bio::binding_landscape::resistance_surface_sweep`
- `bio::binding_landscape::composite_binding`
- `bio::binding_landscape::selectivity_index`
- `bio::binding_landscape::binding_ipr`
- `bio::binding_landscape::localization_length`
- `bio::binding_landscape::site_occupancy_profile`

## Chain

Gonzales IC50 (Exp280) → Hormesis Model (Exp377) → **This**

## Cross-Spring

- healthSpring exp097 (affinity landscape) + exp098 (colonization resistance)
- healthSpring `discovery::affinity_landscape`, `microbiome::anderson_gut_lattice`

## wateringHole Handoff

```
wateringHole/handoffs/WETSPRING_V130_HEALTHSPRING_JOINT_COLONIZATION_HANDOFF_{DATE}.md
```
