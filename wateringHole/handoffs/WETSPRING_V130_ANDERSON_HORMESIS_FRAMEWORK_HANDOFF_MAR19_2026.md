# wetSpring V130 Handoff — Anderson Hormesis Framework & Absorption Candidates

**Date:** 2026-03-19
**From:** wetSpring V130
**To:** barraCuda team, toadStool team, healthSpring team, ecosystem
**License:** AGPL-3.0-or-later
**Supersedes:** WETSPRING_V129_DEEP_DEBT_EVOLUTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

wetSpring V130 introduces the Anderson Hormesis framework: biphasic dose-response
modeled through the Anderson metal-insulator transition, low-affinity binding
landscape for colonization resistance, and a new "computation as experiment
preprocessor" methodology. Two new bio modules with 31 tests, 4 new tolerance
constants, 3 new experiment protocols, and cross-spring joint experiment
architecture with healthSpring.

**Key metrics:** 1,579+ tests, 7 proptest modules, 109 bio modules, 379 experiments,
zero clippy warnings (pedantic+nursery), zero unsafe code, zero mocks in production,
zero TODO/FIXME, 218 named tolerances, 0 local WGSL.

---

## Part 1: What wetSpring V130 Added

### 1a. `bio::hormesis` — Biphasic Dose-Response via Anderson

New module implementing the biphasic dose-response model:

`R(d) = (1 + A × hill(d, K_stim, n_s)) × (1 − hill(d, K_inh, n_i))`

| Function | Purpose |
|----------|---------|
| `response(dose, params)` | Single-point biphasic response |
| `evaluate(dose, params)` | Response + regime classification |
| `sweep(doses, params)` | Batch evaluation across dose range |
| `find_peak(doses, params)` | Locate hormetic peak (max response) |
| `hormetic_zone(doses, params)` | Find dose range where R > 1.0 |
| `dose_to_disorder(dose, w_base, sens, γ)` | Map dose → Anderson disorder W |
| `sweep_with_disorder(...)` | Combined hormesis + disorder sweep |
| `predict_hormetic_zone_from_wc(...)` | Predict hormetic zone from W_c |

Depends on: `barracuda::stats::hill`, `crate::cast::*`, `crate::tolerances::*`.
14 tests, zero clippy warnings.

**barraCuda action:** Consider absorbing `hormesis::response` and `dose_to_disorder`
as `barracuda::bio::hormesis` for cross-spring use. groundSpring (pesticide
hormesis) and healthSpring (caloric restriction, immune calibration) both need
this model.

### 1b. `bio::binding_landscape` — Colonization Resistance & Binding Metrics

New module for low-affinity composite binding and colonization resistance:

| Function | Purpose |
|----------|---------|
| `fractional_occupancy(conc, params)` | Hill-based receptor occupancy |
| `composite_binding(occupancies)` | Coincidence detection from multiple weak binders |
| `selectivity_index(target, off_target)` | On-target vs off-target ratio |
| `colonization_resistance(...)` | Single-point resistance on disordered lattice |
| `resistance_surface_sweep(...)` | 3D sweep (affinity × diversity × disorder) |
| `site_occupancy_profile(...)` | Per-site occupancy on disordered lattice |
| `binding_ipr(occupancies)` | Inverse Participation Ratio of binding profile |
| `localization_length(occupancies)` | Correlation length of binding distribution |

Depends on: `barracuda::stats::hill`, `crate::cast::*`, `crate::tolerances::*`.
17 tests, zero clippy warnings. Uses internal deterministic PRNG (splitmix64)
for reproducible disorder generation.

**barraCuda action:** `binding_ipr` and `localization_length` are general-purpose
Anderson localization metrics. Consider absorbing as `barracuda::spectral::ipr`
and `barracuda::spectral::localization_length` alongside the existing
`anderson_spectral` primitives.

### 1c. Tolerance Constants

4 new named constants in `tolerances/mod.rs`:

| Constant | Value | Domain |
|----------|-------|--------|
| `HORMESIS_PEAK_MARGIN` | 0.01 | Hormetic stimulation vs noise |
| `COLONIZATION_RESISTANCE_THRESHOLD` | 0.9 | 90% occupancy = resistant |
| `BINDING_IPR_DELOCALIZED` | 0.15 | Delocalized binding safety |
| `COMPOSITE_BINDING_FLOOR` | 1e-6 | Minimum composite signal |

### 1d. Methodology: Computation as Experiment Preprocessor

Documented in STUDY.md §4.8 and METHODOLOGY.md §3b. Core claim: the Anderson
phase diagram predicts WHERE the hormetic zone is before experiments are designed.
Computation inverts the traditional experiment→analyze pipeline to
compute→design→validate.

This methodology applies across springs:
- wetSpring: microbial QS dose-response, diversity hormesis
- healthSpring: drug repurposing, immune calibration, colonization resistance
- groundSpring: pesticide hormesis, tillage recovery, trophic cascade
- airSpring: atmospheric pesticide dispersal, multi-source exposure

---

## Part 2: Absorption Candidates for barraCuda / toadStool

### P1 — Absorb into `barracuda::bio`

| Module | What | Benefit |
|--------|------|---------|
| `hormesis::response` | Biphasic dose-response | Cross-spring hormesis modeling |
| `hormesis::dose_to_disorder` | Dose → W mapping | Universal Anderson-hormesis bridge |
| `binding_landscape::binding_ipr` | Inverse Participation Ratio | General Anderson metric |
| `binding_landscape::localization_length` | Correlation length | General Anderson metric |
| `bio::kinetics::monod` | Monod growth (centralized V130) | Already requested V129 |
| `bio::kinetics::haldane` | Haldane inhibition | Already requested V129 |

### P2 — Consider for `barracuda::spectral`

| Primitive | What | Cross-Spring Users |
|-----------|------|--------------------|
| `ipr(eigenstate)` | IPR from eigenstate coefficients | wetSpring, healthSpring, groundSpring |
| `localization_length(profile)` | ξ from exponential decay fit | Same |
| `disorder_to_wc_distance(W, W_c)` | Normalized distance to critical point | Same |

### P3 — GPU Shader Candidates

| Model | GPU Path | Benefit |
|-------|----------|---------|
| Hormesis sweep | `FusedMapReduceF64` with Hill terms | 10K+ dose points, real-time |
| Resistance surface | `BatchedOdeRK4` with adhesion coupling | 3D parameter space exploration |
| Binding IPR batch | New `BindingIprF64` shader | High-throughput drug screening |

---

## Part 3: Cross-Spring Joint Experiments

### healthSpring × wetSpring: Colonization Resistance Surface

| Exp | wetSpring | healthSpring | Shared |
|-----|-----------|-------------|--------|
| 379 / hs-exp097 | Anderson lattice, diversity metrics | PK/PD, adhesion strengths | Resistance surface sweep |
| — / hs-exp098 | Disorder generation, IPR | Toxicity delocalization | Binding landscape |

**Protocol:** wetSpring generates disordered 1D epithelial lattice.
healthSpring provides species-specific adhesion strengths (Kd values).
Both springs compute colonization fraction independently; results cross-validated.

### groundSpring × airSpring: Pesticide Hormesis (Planned)

wetSpring provides biphasic model and Anderson spectral.
groundSpring provides soil transport and trophic lattice.
airSpring provides atmospheric dispersal and multi-source exposure.

---

## Part 4: Ecosystem Learnings

### 4a. Computation as Experiment Preprocessor Pattern

Springs traditionally validate against known baselines. V130 introduces a new
pattern where computational models PREDICT where to run experiments, rather than
just confirming known results. Other springs should adopt this for:
- hotSpring: predicting which material compositions are near phase transitions
- neuralSpring: predicting which ESN hyperparameters explore regime boundaries
- groundSpring: predicting which soil treatments are in the hormetic zone

### 4b. Cross-Spring Joint Experiment Protocol

When two springs need to co-validate, each spring:
1. Exports its contribution via wateringHole handoff
2. Implements its half of the experiment with its own validation binary
3. Cross-references the partner spring's experiment number
4. Results merge in a shared `work/` study document

This pattern (first used in V130 with healthSpring) should be generalized
to the wateringHole standards.

### 4c. Low-Affinity Binding = Delocalized Anderson Regime

The key insight: conventional drug design seeks "localized" (strong, specific)
binding. Low-affinity composite binding is the "delocalized" regime — weaker
per-site but safer because load is distributed. IPR < 0.15 indicates
delocalized binding. This maps directly to Anderson localization theory
and should be absorbed into `barracuda::spectral` as a first-class concept.

---

## Part 5: Metrics

| Metric | V129 | V130 | Delta |
|--------|------|------|-------|
| Tests | 1,548+ | 1,579+ | +31 |
| Bio modules | 107 | 109 | +2 |
| Experiments | 376 | 379 | +3 |
| Tolerances | 214 | 218 | +4 |
| Local WGSL | 0 | 0 | — |
| Unsafe blocks | 0 | 0 | — |
| Clippy warnings | 0 | 0 | — |
| `#[allow()]` | 0 | 0 | — |
