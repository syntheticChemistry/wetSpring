# Experiment 023 — Fernandez 2020 Bistable Phenotypic Switching

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1 (Microbial Ecology / Signaling)
**Faculty:** Waters (MMG, MSU)

---

## Objective

Reproduce the bistable phenotypic switching model from Fernandez et al. 2020
(PNAS 117:29046-29054), which extends the Waters 2008 QS/c-di-GMP model
with a positive feedback loop that creates bistability and hysteresis.

Cells in *V. cholerae* can occupy either a motile or sessile state depending
on history — this is the hallmark of a bistable dynamical system. The positive
feedback from biofilm state (B) onto DGC (diguanylate cyclase) production
creates a fold bifurcation where both states coexist.

## Paper

Fernandez et al. "V. cholerae adapts to sessile and motile lifestyles by
cyclic di-GMP regulation of cell shape." PNAS 117:29046-29054 (2020).

## Model

Extends the 5-variable Waters 2008 ODE system (N, A, H, C, B) with:
- Positive feedback: DGC_rate += alpha_fb * Hill(B, k_fb, n_fb)
- Steep Hill functions (n_bio = 4, n_fb = 4) for ultrasensitivity
- Reduced HapR repression (k_dgc_rep = 0.3) to allow bistable regime

## Baseline

| Field | Value |
|-------|-------|
| Script | `scripts/fernandez2020_bistable.py` |
| Output | `experiments/results/023_bistable/fernandez2020_python_baseline.json` |
| Python | 3.10.12 |
| Integrator | scipy `odeint` (LSODA adaptive) |
| Date | 2026-02-20 |

## Scenarios

| # | Scenario | alpha_fb | Python B_ss | Python C_ss |
|---|----------|----------|-------------|-------------|
| 1 | Zero feedback | 0.0 | 0.040 | 0.454 |
| 2 | Default (sessile start) | 3.0 | 0.745 | 1.634 |
| 3 | Strong feedback | 8.0 | 0.831 | 3.967 |
| 4 | Bifurcation scan | 0–10 | hysteresis width = 7.2 | — |

## Acceptance Criteria

- Rust steady-state matches Python within 1e-3 (RK4 vs LSODA method tolerance)
- Bifurcation scan detects hysteresis (width > 0)
- Forward sweep stays in low-B attractor (B < 0.1 for all alpha)
- Backward sweep visits high-B attractor (B > 0.5 for some alpha)
- All variables non-negative at all time points
- Bitwise deterministic across reruns

## Validation Binary

```bash
cargo run --bin validate_bistable
```

## Modules Validated

- `bio::bistable` — bistable ODE model, bifurcation scan
- `bio::ode` — RK4 integrator, steady-state analysis
