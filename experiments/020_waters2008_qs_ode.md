# Experiment 020: Waters 2008 QS/c-di-GMP ODE Model

**Date**: 2026-02-19
**Status**: COMPLETE — Python baseline GREEN (35/35), Rust RK4 validated (16/16 checks PASS)
**Track**: 1 (Microbial Ecology, Waters)
**Paper Queue**: #5

---

## Objective

Implement and validate a Runge–Kutta 4th-order (RK4) ODE integrator in pure
Rust, validated against the Python/scipy `odeint` baseline for the Waters 2008
quorum sensing / c-di-GMP biofilm model. This is the first dynamical systems
primitive for wetSpring and a prerequisite for Massie 2012 (stochastic) and
Fernandez 2020 (bifurcation).

## Paper Reference

- **Waters et al. 2008** "Quorum Sensing Controls Biofilm Formation in
  *V. cholerae* Through Modulation of Cyclic Di-GMP"
  *J Bacteriology* 190:2527-36
- **Massie et al. 2012** PNAS 109:12746-51 (signal specificity)
- **Bridges et al. 2022** PLoS Biol 20:e3001585 (quantitative NspS-MbaA model,
  code: Zenodo 5519935, CC-BY 4.0)

## Data / Baselines

### Python Baseline (already run)

- **Script**: `scripts/waters2008_qs_ode.py`
- **Output**: `experiments/results/qs_ode_baseline/qs_ode_python_baseline.json`
- **Time series**: `experiments/results/qs_ode_baseline/qs_ode_time_series.json`
- **Python version**: 3.10.12
- **scipy.integrate.odeint**: LSODA (adaptive BDF/Adams)
- **Result**: 35/35 checks PASS

### Steady-State Ground Truth (from Python)

| Scenario | N | A | H | C | B |
|----------|---|---|---|---|---|
| Standard Growth | 0.975 | 4.875 | 1.979 | ~0.0 | 0.020 |
| High-Density Inoculum | 0.975 | 4.874 | 1.972 | ~0.0 | 0.104 |
| ΔhapR Mutant | 0.975 | 4.875 | 0.0 | 2.500 | 0.786 |
| DGC Overexpression | 0.975 | 4.875 | 1.979 | 0.662 | 0.452 |

## Design

### Phase 1: RK4 ODE Module (`bio::ode`)

1. Generic RK4 integrator: `rk4_step(f, y, t, dt)` and `rk4_integrate(f, y0, t_span, dt)`
2. Fixed-step with configurable dt (default 0.001 h for this system)
3. State clamp for biological constraints (non-negative concentrations)
4. Return full time series for validation

### Phase 2: Waters 2008 Model (`bio::qs_biofilm`)

1. 5-variable ODE system: N, A, H, C, B
2. Hill activation function
3. 4 scenarios matching Python baseline exactly
4. All parameters from literature (identical to Python script)

### Phase 3: Validation Binary (`validate_qs_ode`)

1. Run all 4 scenarios through Rust RK4
2. Compare steady-state values against Python baseline (from JSON)
3. Tolerance: 1e-3 for steady-state (RK4 vs LSODA differ by method)
4. Biological constraint checks (non-negative, bounded)

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Standard Growth steady state | N=0.975±0.001, B<0.05 | Python baseline |
| ΔhapR biofilm | B>0.7, C>2.0 | Python baseline + Waters 2008 |
| DGC OE elevated c-di-GMP | C>0.5 | Python baseline |
| High-density dispersal | B<0.3 | Python baseline |
| All state variables non-negative | Always | Biological constraint |
| Rust vs Python steady-state | ±1e-3 | Method tolerance (RK4 vs LSODA) |

## Evolution Path

```
Python (scipy odeint) → Rust CPU (RK4) → GPU (batch parameter sweeps via WGSL)
```

GPU promotion: `rk4_batch_f64.wgsl` for parameter space exploration
(ToadStool Phase 10 target: `BatchedRK4F64`).
