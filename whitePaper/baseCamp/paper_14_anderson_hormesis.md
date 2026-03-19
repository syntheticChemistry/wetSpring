# baseCamp Paper 14: Anderson Hormesis — Biphasic Dose-Response as Near-Critical Disorder

**Date:** March 19, 2026
**Status:** Framework complete — `bio::hormesis` (14 tests), `bio::binding_landscape`
(17 tests), all PASS. Validation binaries and literature baselines in progress.
**Domain:** Toxicology × condensed matter physics × ecology × pharmacology × immunology
**Novelty:** No prior work maps hormesis to the Anderson metal–insulator transition;
no prior work uses Anderson phase diagrams to preprocess dose-response experiment design
**Cross-Spring:** wetSpring (Anderson spectral, hormesis, binding) × healthSpring
(PK/PD, colonization resistance, toxicity delocalization) × groundSpring (soil
transport, trophic lattice) × airSpring (pesticide dispersal) × neuralSpring
(ESN regime classification)

---

## Abstract

Hormesis — the universal biphasic dose-response where low-dose stress stimulates
beneficial adaptation while high-dose stress causes harm — is observed across
toxicology, immunology, ecology, and metabolism. We show that Anderson
localization provides a quantitative framework: biological systems evolved to
operate near the critical disorder threshold W_c, and perturbations that push W
closer to W_c increase coordination (hormesis) while perturbations past W_c
fragment it (toxicity). The hormetic zone IS the near-critical regime.

This extends the Anderson framework from quorum sensing (Paper 01), soil ecology
(Paper 06), and immunological signaling (Paper 12) to a universal theory of
dose-response that unifies pesticide hormesis, caloric restriction, hygiene
hypothesis, mithridatism, and the low-affinity binding landscape (healthSpring
joint experiment).

---

## 1. The Unifying Claim

Biological systems are evolved to operate near the Anderson critical point W_c.
The level spacing ratio r(W) defines coordination capacity:
- r > midpoint(GOE, Poisson) → extended regime → signals propagate → coordination
- r < midpoint → localized regime → signals trapped → fragmentation

Perturbations to W produce three regimes:
1. **Subthreshold** (W << W_c): system is robust but uninspired; no adaptive response
2. **Hormetic** (W ≈ W_c): system explores new configurations → adaptive pathways
   activate → net benefit
3. **Toxic** (W >> W_c): coordination breaks → damage exceeds repair → failure

---

## 2. Phenomena Unified

| Phenomenon | System | Dose | Hormetic Mechanism |
|---|---|---|---|
| Pesticide hormesis | Trophic network | Pesticide conc. | Prunes weak competitors → mean fitness increases |
| Caloric restriction | Metabolic network | Caloric deficit | AMPK/sirtuin repair pathways activate |
| Hygiene hypothesis | Developing immune system | Antigen dose | Controlled exposure calibrates W_c |
| Peanut allergy (LEAP) | Immune tolerance | Peanut protein | Early small doses prevent Th2 skew |
| Mithridatism | Detox network | Toxin dose | Upregulates clearance; shifts W_c |
| Sub-MIC antibiotics | Microbial community | Antibiotic conc. | QS enhancement, horizontal gene transfer |
| Exercise hormesis | Musculoskeletal | Mechanical stress | Micro-damage → supercompensation |
| Low-affinity binding | Drug–tissue lattice | Drug conc. | Delocalized load stays in linear clearance |

---

## 3. Computational Models

### 3.1 Biphasic Dose-Response (`bio::hormesis`)

`R(d) = (1 + A × hill(d, K_stim, n_s)) × (1 − hill(d, K_inh, n_i))`

Validated: 14 tests, zero clippy warnings. Parameters: amplitude, K_stim, n_stim,
K_inh, n_inh. Functions: `response`, `evaluate`, `sweep`, `find_peak`,
`hormetic_zone`, `dose_to_disorder`, `predict_hormetic_zone_from_wc`.

### 3.2 Binding Landscape (`bio::binding_landscape`)

Composite binding from weak interactions (coincidence detection model).
Colonization resistance on 1D disordered epithelial lattice. IPR and
localization length for toxicity delocalization.

Validated: 17 tests, zero clippy warnings. Functions: `fractional_occupancy`,
`composite_binding`, `selectivity_index`, `colonization_resistance`,
`resistance_surface_sweep`, `binding_ipr`, `localization_length`.

### 3.3 Anderson Spectral (existing)

`bio::anderson_spectral::sweep` + `estimate_w_c` — validated across
Exp107–156 (28-biome atlas, W_c ≈ 16.26 ± 0.95).

---

## 4. Computation as Experiment Preprocessor

Traditional: experiment → analyze. We propose: compute → design experiment → validate.

The Anderson phase diagram predicts WHERE the hormetic zone is before the
experiment is designed. The plate screener / field trial becomes a validator
of computational predictions rather than an undirected discovery engine.

See STUDY.md §4.8 for the full methodological argument.

---

## 5. Cross-Spring Architecture

| Spring | Contribution |
|--------|-------------|
| **wetSpring** | Anderson eigensolver, diversity metrics, biphasic model, kinetics |
| **healthSpring** | PK/PD, colonization resistance, toxicity delocalization, iPSC validation |
| **groundSpring** | Soil transport, pesticide fate, trophic lattice, tillage recovery |
| **airSpring** | Atmospheric dispersal, multi-source exposure aggregation |
| **neuralSpring** | ESN regime classification, LSTM treatment response |

---

## 6. Connection to Other Papers

| From | Connection |
|------|-----------|
| Paper 01 (Anderson QS) | Same Hamiltonian, same eigensolver; hormesis extends W_c to dose-response |
| Paper 06 (No-till soil) | Tillage = dimensional collapse; pesticide hormesis = sublethal W perturbation |
| Paper 12 (Immunological Anderson) | AD scratching = dimensional promotion; hygiene hypothesis = insufficient W |
| Paper 13 (Sovereign health) | healthSpring joint experiment for colonization resistance |

---

## 7. Faculty Alignment

| Faculty | Track | Connection |
|---------|-------|-----------|
| **Gonzales** | Immune calibration, iPSC validation | AD dose-response, JAK/IL-31 |
| **Lisabeth** | ADDRC HTS, 8K compound screen | Full dose-response including low-affinity tail |
| **Waters** | QS dose-response | Sub-MIC antibiotic effects on biofilm |
| **Jones** | Environmental toxicology | PFAS dose-response, pesticide exposure |
| **Kachkovskiy** | Spectral theory | Anderson mathematics for all models |

---

## 8. Experiments

| Exp | Title | Status |
|-----|-------|--------|
| 377 | Hormesis biphasic model | Proposed |
| 378 | Trophic cascade Anderson | Proposed |
| 379 | Joint colonization resistance surface | Proposed |
