# Exp186: Dynamic Anderson W(t) — Community Evolution Under Perturbation

**Date:** February 26, 2026
**Phase:** V55 — Science extensions
**Binary:** `validate_dynamic_anderson` (planned)
**Command:** `cargo run --release --features gpu --bin validate_dynamic_anderson`
**Status:** Protocol defined, implementation pending
**Depends on:** Exp150 (finite-size scaling), Exp170-182 (Track 4 soil QS)

## Purpose

Model how microbial community diversity evolves under perturbation using
time-varying Anderson disorder W(t). This extends the static Anderson
framework (fixed W) to dynamic scenarios like:
- Tillage → no-till transition (W decreases over years)
- Antibiotic treatment → recovery (W spike then decay)
- Seasonal cycles (periodic W(t))

## Physics

The Anderson Hamiltonian with time-varying disorder:

    H(t) = H_hop + W(t) · V_disorder

where W(t) is the disorder parameter that changes with community
diversity over time. As diversity recovers (no-till), W decreases and
the system transitions from localized (QS suppressed) to extended
(QS viable).

Key observable: the level spacing ratio r(t) tracks the
localization → delocalization transition in real time.

## Scenarios

### S1: Tillage → No-Till Transition
- W(0) = 20 (high disorder, tilled soil)
- W(t) = W(0) · exp(-t/τ) + W_∞, τ = 3 years
- W_∞ = 12 (steady-state no-till diversity)
- Track r(t): expect r(0) ≈ POISSON_R → r(∞) ≈ GOE_R

### S2: Antibiotic Perturbation
- W(0) = 14 (healthy gut)
- W(t_ab) = 25 (during antibiotic, t = 0-7 days)
- W(t) = 25 · exp(-(t-7)/τ) + 14, τ = 14 days (recovery)
- Track r(t): expect transient localization, then recovery

### S3: Seasonal Cycle
- W(t) = W_0 + A · sin(2πt/365), W_0 = 16, A = 4
- Tests whether periodic disorder can induce resonant QS transitions
- At W = W_c ± A, system oscillates between extended and localized

## Validation Checks

### S1: Tillage Transition
- [ ] r(0) < midpoint (localized start)
- [ ] r(∞) > midpoint (extended end)
- [ ] Transition time consistent with τ = 3 years
- [ ] W_c crossing detected in r(t) trajectory
- [ ] Results consistent with Brandt farm data (Islam 2014)

### S2: Antibiotic Recovery
- [ ] r drops below midpoint during treatment
- [ ] r recovers to > midpoint within 30 days
- [ ] Recovery trajectory is approximately exponential

### S3: Seasonal
- [ ] r oscillates with period ≈ 365 (within sampling resolution)
- [ ] r range brackets midpoint when W_0 ≈ W_c
- [ ] Time-averaged r depends on W_0 relative to W_c

## Implementation

```rust
fn dynamic_anderson_sweep(
    l: usize,          // lattice size
    w_func: &dyn Fn(f64) -> f64,  // W(t) function
    t_points: &[f64],  // time points
    n_real: usize,     // disorder realizations per point
) -> Vec<(f64, f64, f64)> // (t, r_mean, r_stderr)
```

Each time point constructs a fresh Anderson lattice with W = W(t),
runs Lanczos, and computes r. This is embarrassingly parallel across
time points and realizations.

## Compute Estimate

- Per (L, t, realization): ~0.5s at L=10 on CPU, ~0.05s on GPU
- S1 (100 time points × 8 realizations × 4 L values): ~30 min CPU, ~3 min GPU
- S2 (60 time points × 8 realizations × 4 L values): ~20 min CPU, ~2 min GPU
- S3 (365 time points × 8 realizations × 4 L values): ~100 min CPU, ~10 min GPU

## Provenance

| Item | Value |
|------|-------|
| Theory | Anderson (1958), time-dependent generalization |
| Soil data | Islam et al. (2014), Brandt farm 15-year study |
| Antibiotic | Dethlefsen & Relman, PNAS 108 (2011) |
| Seasonal | Lauber et al., Applied Env. Microbiol. 75 (2009) |
