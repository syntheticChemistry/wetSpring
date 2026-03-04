# Exp305: Cross-Spring S93 Evolution Validation + Benchmark

**Date:** 2026-03-04
**Track:** cross-spring
**Status:** DONE — 59/59 checks passed
**Binary:** `validate_cross_spring_s93`
**Handoff:** `WETSPRING_V95_CROSS_SPRING_EVOLUTION_COMPLETE_MAR04_2026.md`

---

## Purpose

Validate the full cross-spring evolution pipeline after rewiring wetSpring V95
to standalone barraCuda v0.3.1. Documents provenance for every primitive and
benchmarks CPU performance in release mode.

## Domains Validated

| Domain | Section | Checks | Key Primitives |
|--------|---------|:------:|----------------|
| CPU Math | D00 | 7 | norm_ppf, gradient_1d, erf, ln_gamma |
| RK45 ODE | D01 | 7 | rk45_integrate, rk4_integrate parity, logistic growth |
| CPU Stats | D02 | 11 | Shannon, jackknife, bootstrap, Kimura, Hargreaves |
| Spectral Theory | D03 | 3 | Anderson eigenvalues, level spacing ratio |
| Linalg | D04 | 3 | graph_laplacian, ridge_regression, NMF |
| Sampling | D05 | 8 | Boltzmann, Sobol, Latin hypercube |
| Numerical | D06 | 4 | trapz, numerical_hessian |
| Bio ODE | D07 | 3 | Lotka-Volterra 2D via RK45 |
| Tolerance Search | D08 | 1 | find_within_ppm CPU |
| KMD | D09 | 2 | CF2 repeat unit, homologue grouping |
| Benchmarks | D10 | 4 | Shannon, gradient, RK45, norm_ppf timing gates |
| Provenance | D11 | 6 | Cross-spring contribution audit |

## Release Benchmarks

| Primitive | Time (µs) |
|-----------|-----------|
| Shannon 10K (100×) | <1 |
| gradient_1d 10K (100×) | 7 |
| RK45 exp decay (10×) | 14 |
| norm_ppf (10K calls) | <0.01 |
| Lotka-Volterra 2D t=50 | 82 |

## Cross-Spring Provenance

See full table in binary header and V95 handoff.
