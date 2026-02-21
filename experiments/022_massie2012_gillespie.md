# Experiment 022: Massie 2012 Gillespie Stochastic Simulation

**Date**: 2026-02-19
**Status**: COMPLETE — Python baseline validated, Rust Gillespie SSA validated (13/13 checks PASS)
**Track**: 1 (Microbial Ecology, Waters)
**Paper Queue**: #6

---

## Objective

Implement the Gillespie stochastic simulation algorithm (SSA) in pure Rust
for modeling c-di-GMP signal specificity in *Vibrio cholerae*. Massie et al.
2012 shows that cells resolve signal from noise with 60+ enzymes controlling
a single molecule — this requires stochastic, not deterministic, modeling.

Validates against Python baseline and demonstrates the mathematical bridge
from ODE (Exp020) to stochastic regimes needed for single-cell biology.

## Paper Reference

- **Massie et al. 2012** "Quantification of High Specificity Cyclic di-GMP
  Signaling" *PNAS* 109:12746-51
- **Gillespie 1977** "Exact Stochastic Simulation of Coupled Chemical
  Reactions" *J Phys Chem* 81:2340-2361

## Data / Baselines

### Python Baseline

- **Script**: `scripts/gillespie_baseline.py`
- **Output**: `experiments/results/022_gillespie/gillespie_python_baseline.json`
- **Python version**: 3.10.12
- **numpy**: for random number generation

### Ground Truth

Stochastic simulations match deterministic ODE at large molecule counts.
At low counts, stochastic noise creates biologically meaningful variability.
Key validation: mean trajectory converges to ODE solution as N → ∞.

## Design

### Phase 1: Gillespie SSA Module (`bio::gillespie`)

1. Generic Gillespie direct method: propensities → exponential wait → reaction select
2. Seeded PRNG (LCG) for reproducibility
3. State: integer molecule counts
4. Event recording for trajectory analysis

### Phase 2: c-di-GMP Signal Model

1. Simplified 3-reaction system from Massie 2012:
   - DGC synthesis: ∅ → cdGMP (rate = k_dgc)
   - PDE degradation: cdGMP → ∅ (rate = k_pde × [cdGMP])
   - Spontaneous degradation: cdGMP → ∅ (rate = d × [cdGMP])
2. Multiple runs to build ensemble statistics
3. Compare mean ± std against ODE steady state

### Phase 3: Validation Binary (`validate_gillespie`)

1. Mean of N runs converges to analytical steady state (k_dgc / (k_pde + d))
2. Variance matches Poisson expectation for simple birth-death
3. Deterministic with same seed
4. All molecule counts non-negative

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Mean converges to ODE steady state | ±10% at N=1000 runs | Analytical |
| Variance ~ Poisson for birth-death | CV² ~ 1/mean | Theory |
| Deterministic with seed | Bitwise identical | Implementation |
| Non-negative counts | Always | Physical constraint |

## Evolution Path

```
Python (numpy) → Rust CPU (LCG-SSA) → GPU (batch ensemble via WGSL)
```
