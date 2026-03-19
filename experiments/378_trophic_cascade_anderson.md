# Exp378: Trophic Cascade via Anderson Lattice

**Status:** PROPOSED
**Date:** 2026-03-19
**Binary:** `validate_trophic_cascade` (to create)
**Feature gate:** none
**Track:** Anderson Hormesis (cross-spring)

## Purpose

Model a multi-species trophic network as an Anderson lattice where
pesticide dose differentially perturbs species' effective disorder. Tests
the prediction that predators (sensitive) localize (collapse) before prey
(resistant), producing a trophic cascade that can paradoxically increase
pest fitness at sublethal doses.

## Scientific Question

Does differential sensitivity to pesticide perturbation in a trophic
Anderson lattice predict the counterintuitive hormetic effect where
sublethal pesticide doses increase target pest populations?

## Modules Tested

- `bio::hormesis::sweep_with_disorder`
- `bio::anderson_spectral::sweep`
- `bio::anderson_spectral::estimate_w_c`
- `bio::diversity::shannon`
- `bio::diversity::pielou_evenness`

## Chain

Hormesis Model (Exp377) → **This** → Joint Colonization (Exp379)

## Cross-Spring

- groundSpring exp_trophic_01 (soil food web)
- airSpring exp_dispersal_01 (atmospheric pesticide dispersal)
