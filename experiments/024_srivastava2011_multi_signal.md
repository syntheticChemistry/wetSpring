# Experiment 024 — Srivastava 2011 Multi-Input QS Network

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1 (Microbial Ecology / Signaling)
**Faculty:** Waters (MMG, MSU)

---

## Objective

Reproduce the dual-signal quorum sensing model from Srivastava et al. 2011
(J Bacteriology 193:6331-41). V. cholerae integrates two autoinducer signals
(CAI-1 intraspecies, AI-2 interspecies) through separate receptors (CqsS, LuxPQ)
converging on the LuxO phosphorelay to control HapR and downstream biofilm.

## Model

7-variable ODE system: N (cell density), CAI-1, AI-2, LuxO~P, HapR, c-di-GMP, B.
Both signals dephosphorylate LuxO independently; LuxO~P represses HapR via Hill
repression. Downstream c-di-GMP/biofilm follows Waters 2008.

## Scenarios

| # | Scenario | k_cai1 | k_ai2 | Python HapR_ss | Python B_ss |
|---|----------|--------|-------|----------------|-------------|
| 1 | Wild type | 3.0 | 3.0 | 0.543 | 0.413 |
| 2 | CAI-1 only (ΔluxS) | 3.0 | 0.0 | 0.238 | 0.676 |
| 3 | AI-2 only (ΔcqsA) | 0.0 | 3.0 | 0.238 | 0.676 |
| 4 | No QS (ΔluxS ΔcqsA) | 0.0 | 0.0 | 0.031 | 0.777 |
| 5 | Exogenous CAI-1 | 3.0 + 5.0 IC | 3.0 | 0.543 | 0.413 |

## Baseline

| Field | Value |
|-------|-------|
| Script | `scripts/srivastava2011_multi_signal.py` |
| Output | `experiments/results/024_multi_signal/srivastava2011_python_baseline.json` |
| Date | 2026-02-20 |

## Validation Binary

```bash
cargo run --bin validate_multi_signal
```

## Modules Validated

- `bio::multi_signal` — dual-signal QS ODE, 5 scenarios
- `bio::ode` — RK4 integrator, steady-state analysis
