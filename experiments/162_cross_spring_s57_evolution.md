# Exp162: Cross-Spring S57 Evolution — Rewire + Validate + Benchmark

**Date:** February 24, 2026
**Phase:** 39 → 40 — ToadStool S57 rewire
**Binary:** `validate_cross_spring_s57`
**Command:** `cargo run --release --features gpu --bin validate_cross_spring_s57`
**Result:** 66/66 checks PASS

## Purpose

Validate and benchmark 6 new ToadStool primitives evolved by neuralSpring
in sessions S54 and S56, now wired into wetSpring's bio analysis context.
Demonstrates the cross-spring evolution model at full scale.

## New Primitives Wired (S54-S57 → wetSpring)

| Primitive | Evolved By | Session | wetSpring Use Case | Checks |
|-----------|-----------|---------|-------------------|:------:|
| `graph_laplacian` | neuralSpring baseCamp | S54 | Community interaction network analysis | 11 |
| `effective_rank` | neuralSpring baseCamp | S54 | Diversity matrix spectral diagnostics | 3 |
| `numerical_hessian` | neuralSpring baseCamp | S54 | ML model curvature / convexity analysis | 6 |
| `disordered_laplacian` | neuralSpring | S56 | Anderson disorder on community graphs (QS-disorder coupling) | 32 |
| `belief_propagation_chain` | neuralSpring | S56 | Hierarchical taxonomy classification | 5 |
| `boltzmann_sampling` | neuralSpring | S56 | MCMC parameter optimization for ODE models | 3 |

## Compound Workflows (Multiple Springs in One Pipeline)

- **neuralSpring graph + hotSpring spectral → QS-disorder**: graph_laplacian (S54)
  + disordered_laplacian (S56) + find_all_eigenvalues + level_spacing_ratio
  → detects Anderson localization in biofilm geometry
- **GPU regression**: 5 neuralSpring S31f primitives confirmed working on ToadStool S57
  (Hamming, Jaccard, SpatialPayoff validated; BatchFitness/LocusVar validated at scale in Exp094)

## Cross-Spring Evolution Provenance Map

| Source | Contribution | Beneficiaries |
|--------|-------------|---------------|
| hotSpring | `ShaderTemplate`, `GpuDriverProfile`, `FMR`, `BatchedEigh`, spectral | All Springs |
| wetSpring | 12 bio shaders, GEMM 60×, `math_f64.wgsl`, ODE generic framework | hotSpring HFB, neuralSpring |
| neuralSpring S31f | PairwiseHamming/Jaccard, SpatialPayoff, BatchFitness, LocusVariance | wetSpring bio |
| neuralSpring S54 | graph_laplacian, effective_rank, numerical_hessian + 5 WGSL shaders | wetSpring bio (this exp) |
| neuralSpring S56 | disordered_laplacian, belief_propagation, boltzmann_sampling | wetSpring bio (this exp) |
| airSpring S54 | pow_f64, acos_f64, FMR buffer fixes | All Springs |

## Benchmark Results (Cross-Spring at Scale, ToadStool S57)

| Primitive | Evolved By | Problem | CPU (µs) | GPU (µs) | Speedup |
|-----------|-----------|---------|:--------:|:--------:|:-------:|
| PairwiseHamming | neuralSpring | 500×1000 seqs | 12,244 | 1,023 | 12.0× |
| PairwiseJaccard | neuralSpring | 200×2000 genes | 42,003 | 260 | 161.6× |
| SpatialPayoff | neuralSpring | 256×256 grid | 1,039 | 153 | 6.8× |
| BatchFitness | neuralSpring | 4096×256 genome | 519 | 491 | 1.1× |
| LocusVariance | neuralSpring | 100×10K loci | 1,992 | 366 | 5.4× |
| FMR (Shannon) | hotSpring | 100K f64 | <1 | 2,318 | CPU wins small |
| GEMM 256×256 | wetSpring | 256×256 f64 | 3,572 | 3,497 | 1.0× |

## ODE Lean Benchmark (upstream vs local)

| System | Local CPU (µs) | Upstream CPU (µs) | Speedup | Parity |
|--------|:--------------:|:-----------------:|:-------:|:------:|
| VpsR Capacitor | 2,018 | 1,578 | 1.28× | exact (0.00) |
| Cooperator-Cheater | 1,259 | 1,006 | 1.25× | exact (4.44e-16) |

ToadStool's `integrate_cpu()` is **20-28% faster** than wetSpring's local
implementation — demonstrating the lean advantage.
