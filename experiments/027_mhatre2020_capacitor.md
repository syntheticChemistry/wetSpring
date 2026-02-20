# Experiment 027 — Mhatre 2020 Phenotypic Capacitor

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1 (Microbial Ecology / Signaling)
**Faculty:** Waters (MMG, MSU)

---

## Objective

Reproduce the phenotypic capacitor model from Mhatre et al. 2020 (PNAS 117:21647-21657).
VpsR acts as a capacitor that integrates c-di-GMP and distributes output to three phenotypic
channels: biofilm, motility, and rugose colony morphology.

## Scenarios

| # | Scenario | VpsR_ss | B_ss | M_ss | R_ss |
|---|----------|---------|------|------|------|
| 1 | Normal | 0.766 | 0.671 | 0.319 | 0.439 |
| 2 | Stress (3×) | 0.769 | 0.672 | 0.316 | 0.441 |
| 3 | Low c-di-GMP | 0.357 | 0.488 | 0.562 | 0.146 |
| 4 | ΔvpsR | 0.000 | 0.000 | 0.667 | 0.000 |

## Validation Binary

```bash
cargo run --bin validate_capacitor
```
