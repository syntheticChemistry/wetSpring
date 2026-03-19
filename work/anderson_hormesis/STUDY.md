# Anderson Hormesis: A Unified Framework for Biphasic Dose-Response Across Biological Scales

## Thesis

Hormesis — the phenomenon where low-dose stress stimulates beneficial adaptation
while high-dose stress causes harm — is a universal biphasic dose-response pattern
observed in toxicology, immunology, ecology, and metabolism. We show that Anderson
localization provides a quantitative framework for hormesis: the dose maps to
disorder strength W, and the biological response depends on whether W pushes the
system toward or away from the critical disorder threshold W_c.

**Core claim**: Biological systems evolved to operate near the Anderson critical
point W_c. Perturbations that push W closer to W_c increase signal coordination
(hormesis). Perturbations that push W past W_c fragment coordination (toxicity).
The hormetic zone IS the near-critical regime.

## Computation as Experiment Preprocessor

Traditional toxicology:
```
expose → measure → fit dose-response curve → identify NOAEL/LOAEL
```

Proposed pipeline:
```
model tissue/population as Anderson lattice → compute W_c for system →
predict hormetic zone from Anderson phase diagram → design dose-response
experiment targeting the predicted critical regime → validate predictions
```

The computation tells us WHERE to look before we look. The plate screener,
field trial, or clinical measurement becomes a validator of computational
predictions rather than an undirected discovery engine.

## Unifying Phenomena

All of these map to position relative to W_c in the Anderson framework:

| Phenomenon | System | Lattice | Disorder W | Dose | Hormetic Zone |
|---|---|---|---|---|---|
| Pesticide hormesis | Trophic network | Organisms × interactions | Trophic diversity | Pesticide concentration | Sublethal dose that prunes weak competitors |
| Caloric restriction | Metabolic network | Cell × signaling pathway | Pathway activity heterogeneity | Caloric deficit | Mild deficit activating AMPK/sirtuin repair |
| Hygiene hypothesis | Developing immune system | T-cells × antigen exposure | Repertoire diversity | Antigen dose during development | Early controlled exposure calibrating W_c |
| Mithridatism | Detox network | Tissue × CYP450 enzyme | Enzyme expression heterogeneity | Toxin dose | Sublethal dose upregulating clearance |
| Peanut allergy (LEAP) | Immune tolerance | Dendritic cells × T-cells | Th1/Th2 balance | Peanut protein dose | Early small doses preventing Th2 skew |
| Antibiotic resistance | Microbial community | Species × resistance genes | Community diversity | Antibiotic concentration | Sub-MIC promoting horizontal gene transfer |
| Exercise hormesis | Musculoskeletal system | Myocytes × repair pathways | Fiber type heterogeneity | Mechanical stress | Micro-damage triggering supercompensation |
| Radiation hormesis | DNA repair network | Cell × damage/repair sites | Repair capacity heterogeneity | Radiation dose | Low dose upregulating p53/ATR repair |

## Anderson Mapping

### Dose → Disorder

A stressor (pesticide, toxin, caloric deficit, antigen) perturbs the system's
effective disorder W:

```
W_effective(dose) = W_baseline + sensitivity × dose_transform(dose)
```

Where `dose_transform` depends on the system — linear for simple toxins,
logarithmic for dose-response (Hill pharmacology), sigmoidal for receptor-mediated.

### Biphasic Response from Anderson Phase Diagram

The level spacing ratio r(W) defines the system's coordination capacity:
- r > midpoint(GOE, Poisson) → extended regime → signals propagate → coordination
- r < midpoint → localized regime → signals trapped → fragmentation

The hormetic response emerges when:
1. The system's baseline W_0 is near W_c (evolved to operate at criticality)
2. Small dose → W moves slightly → system explores new lattice configurations →
   adaptive pathways activate → net benefit (hormesis)
3. Large dose → W >> W_c → coordination breaks → damage exceeds repair → toxicity

### The Biphasic Curve

```
Response(dose) = coordination_gain(dose) × survival(dose)

coordination_gain(dose) = 1 + amplitude × hill(dose, K_stim, n_stim)
survival(dose) = 1 - hill(dose, K_inh, n_inh)

where K_inh >> K_stim  (inhibition requires much higher dose)
```

This naturally produces:
- dose = 0: response = 1.0 (baseline)
- dose ~ K_stim: response > 1.0 (hormetic peak)
- dose ~ K_inh: response < 1.0 (toxicity begins)
- dose >> K_inh: response → 0 (lethal)

The Anderson connection: K_stim maps to the dose that pushes W to the
near-critical zone; K_inh maps to the dose that pushes W past W_c.

## Cross-Spring Architecture

| Spring | Contribution | Primitives |
|--------|-------------|-----------|
| **wetSpring** | Anderson lattice eigensolver, diversity metrics, Pielou→W mapping, biphasic dose-response model, microbial community dynamics | `anderson_spectral`, `diversity`, `hormesis`, `kinetics`, `ode` |
| **healthSpring** | Tissue binding landscape, PK/PD modeling, colonization resistance, toxicity delocalization | `affinity_landscape`, `pkpd`, `colonization_resistance`, `anderson_gut_lattice` |
| **groundSpring** | Soil transport, pesticide fate, trophic lattice modeling, tillage dimensional collapse | `transport`, `spectral_validation`, `uncertainty` |
| **airSpring** | Atmospheric dispersal, agricultural drift, exposure modeling | `regression`, `hydrology`, `dispersal` |
| **neuralSpring** | ESN regime classification, LSTM time-series prediction of treatment response | `esn`, `lstm`, `regime_classifier` |

## Experimental Program

### Track A: Pesticide Hormesis (wetSpring × groundSpring × airSpring)

Model a trophic lattice (predators, prey, competitors) as an Anderson lattice.
Pesticide dose perturbs W. Predict:
1. Dose at which target pest fitness INCREASES (hormetic zone)
2. Dose at which predator populations collapse first (differential sensitivity)
3. The trophic cascade from predator loss → pest outbreak

Validate against published sub-MIC and sub-lethal dose studies.

### Track B: Immune Calibration (wetSpring × healthSpring)

Model developing immune system as Anderson lattice where antigen exposure sets W.
Predict:
1. The antigen dose/timing that produces optimal W (near W_c) for immune tolerance
2. The hygiene deficit that leaves W too low (allergy-prone)
3. The exposure excess that pushes W too high (chronic inflammation)

Validate against LEAP study data (peanut allergy) and Gonzales AD data.

### Track C: Metabolic Hormesis (healthSpring)

Model metabolic network as Anderson lattice with caloric intake controlling W.
Predict:
1. The caloric restriction that maximizes longevity (near-critical W)
2. The starvation threshold where W exceeds W_c
3. Pathway activation signatures at each regime

Validate against published caloric restriction longevity studies.

### Track D: Low-Affinity Binding (healthSpring × wetSpring)

Already in progress — see healthSpring/work/low_affinity_binding/.
Extends: delocalized binding as the hormetically beneficial regime.

### Track E: Mithridatism / Tolerance (healthSpring)

Model detoxification network as Anderson lattice. Gradual toxin exposure increases
the system's effective W_c (shifts the phase boundary). Predict:
1. The dosing schedule that maximally shifts W_c
2. The cost in metabolic resources (reduced adaptability to novel stressors)
3. The point of diminishing returns

## Connection to Arrow of Causality

The computation generates directional predictions:

1. **Forward**: Given dose → predict Anderson regime → predict biological response
2. **Inverse**: Given observed response → infer Anderson regime → infer effective dose/W
3. **Causal**: If prediction matches observation → the Anderson mechanism is consistent
   with causality. If not → the model is wrong, and the discrepancy points to
   missing physics

The experiment validates or falsifies the computational prediction. Each validated
prediction strengthens the causal claim. Each falsification reveals where the
Anderson model breaks down — which is itself informative.

**Computation finds the arrow. Experiment confirms the direction.**

## Faculty Alignment

| Faculty | Domain | Track |
|---------|--------|-------|
| **Gonzales** | Immunological Anderson, iPSC validation, JAK/IL-31 dose-response | B, D |
| **Lisabeth** | ADDRC HTS, 8K compound screen, full dose-response curves | D |
| **Waters** | QS dose-response, antibiotic sub-MIC effects on biofilm | A, B |
| **Jones** | PFAS analytical chemistry, environmental toxicology | A |
| **Dong** | Agricultural engineering, environmental exposure | A |
| **Liao** | Anaerobic systems, biogas kinetics | A |
| **Anderson (Rika)** | Extremophile genomics, fitness under constraint | A, B |
| **Kachkovskiy** | Spectral theory, Anderson localization mathematics | All |

## Implementation Status

| Component | Status |
|-----------|--------|
| `bio::hormesis` | **Building** — biphasic dose-response, hormetic zone detection, Anderson coupling |
| `bio::binding_landscape` | **Building** — composite binding, colonization occupancy, resistance surface |
| `bio::anderson_spectral` | **Complete** — lattice, sweep, W_c estimation, Pielou→W |
| `bio::diversity` | **Complete** — Shannon, Simpson, Pielou, Bray-Curtis, Chao1 |
| `bio::kinetics` | **Complete** — Monod, Haldane |
| `stats::hill` | **Complete** (barraCuda) — Hill activation/repression |
| Joint experiment | **Planned** — colonization resistance surface (exp_joint_01, exp_joint_02) |
| Validation binaries | **Planned** — hormesis phase diagram, pesticide trophic cascade |
