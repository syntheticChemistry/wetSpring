# Joint Experiment: Anderson Hormesis Across Springs

## Summary

Cross-spring computational experiment unifying hormesis, low-affinity binding,
immune calibration, and trophic cascade modeling under the Anderson localization
framework. Each spring contributes domain-specific lattice models that share
the same spectral eigensolver and diagnostic toolkit.

## Scientific Question

**Is the hormetic zone quantitatively predicted by proximity to the Anderson
critical disorder threshold W_c, across biological scales from molecular
binding to trophic networks?**

If yes: the Anderson framework provides a universal preprocessor for
dose-response experiment design — tell us the lattice, and we'll tell you
where the hormetic zone is before you run the experiment.

If no: the discrepancies reveal which biological systems add physics beyond
the tight-binding Anderson model (nonlinear interactions, feedback loops,
spatial structure, temporal dynamics).

## Experiment Matrix

### wetSpring Experiments

| ID | Title | Model | Validates |
|----|-------|-------|-----------|
| exp_hormesis_01 | Biphasic dose-response via Anderson | Trophic lattice + pesticide perturbation | Hormetic peak at predicted W → W_c |
| exp_hormesis_02 | Trophic cascade Anderson | Multi-species lattice, differential sensitivity | Predator collapse before prey |
| exp_hormesis_03 | QS hormesis | QS community + sub-MIC antibiotic | QS enhancement at sublethal dose |
| exp_joint_01 | Anderson lattice with adhesion-modulated hopping | 1D gut epithelium lattice | Joint with healthSpring exp098 |
| exp_joint_02 | Full 3D colonization resistance surface | (affinity, diversity, disorder) sweep | Phase diagram of resistance |

### healthSpring Experiments (reference)

| ID | Title | Status |
|----|-------|--------|
| exp097 | Affinity landscape & composite targeting | 18/18 pass |
| exp098 | Toxicity landscape & colonization resistance | 22/22 pass |
| exp099 | Hormesis bonus — sub-threshold adaptive response | planned |

### groundSpring Experiments (proposed)

| ID | Title | Model |
|----|-------|-------|
| exp_trophic_01 | Soil trophic lattice under pesticide | Soil food web as Anderson lattice |
| exp_trophic_02 | Tillage hormesis — stress-recovery dynamics | Recovery from dimensional collapse |

### airSpring Experiments (proposed)

| ID | Title | Model |
|----|-------|-------|
| exp_dispersal_01 | Atmospheric pesticide dispersal → exposure landscape | Spatial W gradient from drift |
| exp_dispersal_02 | Multi-source exposure aggregation | Cumulative W from multiple chemical sources |

## Computational Models (wetSpring)

### 1. Biphasic Dose-Response (`bio::hormesis`)

```
response(dose) = (1 + A_stim × hill(dose, K_stim, n_stim)) × (1 - hill(dose, K_inh, n_inh))
```

- `A_stim`: maximum stimulation amplitude (dimensionless, typically 0.1–0.5)
- `K_stim`: dose for half-maximal stimulation
- `K_inh`: dose for half-maximal inhibition (K_inh >> K_stim)
- `n_stim`, `n_inh`: Hill coefficients

### 2. Dose → Anderson Disorder Mapping

```
W(dose) = W_baseline + sensitivity × dose^gamma
```

The exponent gamma captures dose-response shape: gamma=1 (linear),
gamma<1 (saturating, like Monod), gamma>1 (cooperative/threshold).

### 3. Hormetic Zone from Anderson Sweep

Sweep dose → W → anderson_spectral → r(W):
- Identify W values where r is closest to midpoint(GOE, Poisson)
- Map back to dose → this is the predicted hormetic zone
- Compare with experimentally observed NOAEL/LOAEL/hormetic peak

### 4. Colonization Resistance Surface (Joint with healthSpring)

3D parameter sweep:
- Axis 1: Adhesion strength K_base (0.1 to 100 µM)
- Axis 2: Species diversity N_species (1 to 20)
- Axis 3: Epithelial disorder W (0.1 to 5.0)

At each point: Anderson eigensolver → colonization occupancy → resistance score.
The resistance surface manifold reveals the phase boundary between
"diverse-delocalized" (resistant) and "monoculture-localized" (vulnerable).

### 5. Trophic Cascade Lattice

Multi-species Anderson lattice where each species has:
- Population size as lattice occupation
- Interaction strength as hopping parameter
- Species sensitivity to pesticide as on-site energy perturbation

Pesticide dose perturbs on-site energies differentially by species.
Predators (longer generation time, smaller population) have higher
sensitivity → their effective W increases faster → they localize
(population collapses) before prey.

## Shared Primitives

All models use the same barraCuda spectral toolkit:

| Primitive | Source | Used By |
|-----------|--------|---------|
| `anderson_3d` | `barracuda::spectral` | All lattice construction |
| `lanczos` / `lanczos_eigenvalues` | `barracuda::spectral` | Eigenvalue extraction |
| `level_spacing_ratio` | `barracuda::spectral` | Regime diagnostic |
| `hill` / `hill_activation` | `barracuda::stats` | Dose-response curves |
| `shannon` / `pielou_evenness` | `barracuda::stats` | Diversity as W proxy |
| `monod` / `haldane` | `bio::kinetics` | Growth kinetics under stress |
| `rk4_integrate` | `bio::ode` | ODE population dynamics |

## Validation Strategy

### Tier 1: Analytical Known-Values

- Biphasic curve at dose=0 returns baseline (1.0)
- At dose→∞, response→0
- Peak stimulation at predicted dose matches closed-form solution
- Anderson eigensolver matches known W_c for 3D lattice (~16.5)

### Tier 2: Cross-Validation with Existing Experiments

- Gonzales IC50 (Exp280): Hill equation matches → dose-response numerics correct
- Anderson QS (Exp107–156): W_c estimation matches → eigensolver correct
- Haldane inhibition (Exp039): substrate inhibition matches → biphasic shape correct

### Tier 3: Literature Comparison

- Published hormesis databases (Calabrese & Mattson, Pharmacological Research 2023)
- LEAP study peanut allergy dose-response data
- Caloric restriction meta-analysis (Mattison et al., Nature 2017)
- Sub-MIC antibiotic biofilm studies (Andersson & Hughes, Nature Rev Micro 2014)

## wateringHole Handoff

When ready:
```
wateringHole/handoffs/WETSPRING_V130_ANDERSON_HORMESIS_HANDOFF_{DATE}.md
wateringHole/handoffs/WETSPRING_V130_HEALTHSPRING_JOINT_COLONIZATION_HANDOFF_{DATE}.md
```

## The Arrow of Causality

```
Phase 1: Computational prediction
  Anderson model → predicted hormetic zone → predicted phase diagram

Phase 2: Experimental validation
  Plate screen / field trial / clinical data → observed dose-response

Phase 3: Comparison
  prediction vs observation → agreement OR discrepancy

Phase 4: Refinement
  agreement → model captures relevant physics → strengthen causal claim
  discrepancy → model missing something → identify what → refine → repeat
```

Each iteration narrows the gap between model and reality.
The arrow of causality emerges from the convergence.
