# Exp189: Cross-Spring Evolution Benchmark (ToadStool S68)

**Date:** February 26, 2026
**Binary:** `benchmark_cross_spring_s68`
**Features:** `gpu`
**Status:** Protocol ready (requires GPU for §1-2, §5, §7)

## Purpose

Comprehensive cross-spring evolution benchmark validating wetSpring's fully-lean
stack after the V57 rewire to ToadStool S68 (universal precision architecture).
Every delegation chain is validated and benchmarked with provenance tracking.

## Cross-Spring Provenance Map

| Section | Domain | Origin → ToadStool | wetSpring Usage |
|---------|--------|-------------------|-----------------|
| §1 | GPU ODE × 5 | wetSpring V16-V22 → S58 → S68 | `compile_shader_universal(Precision::F64)` |
| §2 | DiversityFusion | wetSpring Write → S63 Absorb | First full Write→Absorb→Lean cycle |
| §3 | CPU diversity | wetSpring → S64 `stats::diversity` | 11 functions delegated |
| §4 | Special functions | hotSpring → S59 `special` | erf, ln_gamma, norm_cdf |
| §5 | Anderson spectral | hotSpring lattice → ToadStool | Track 4 soil pore QS |
| §6 | NMF + ridge | wetSpring → S58 linalg | Drug repurposing, ESN readout |
| §7 | GPU GEMM | wetSpring GemmCached → S62 | Universal precision compile path |
| §8 | CPU stats | airSpring/groundSpring → S64/S66 | regression, hydrology, bootstrap |

## Evolution Timeline (S39 → S68)

- **S39-S44** hotSpring: f64 precision, driver workarounds, Jacobi, RK4/RK45
- **S45-S50** neuralSpring: pairwise ops, graph Laplacian, MCMC, GNN
- **S51-S58** wetSpring: bio ODE × 5, diversity, GEMM, NMF, ridge, Anderson
- **S58** hotSpring: DF64 core (14 shaders), `Fp64Strategy`
- **S60-S62** ToadStool: `SparseGemm`, `TransE`, `TopK`, `PeakDetect`, BGL
- **S63-S64** cross-spring: diversity_fusion, `stats::diversity`, `stats::metrics`
- **S66** cross-spring: regression, hydrology, moving_window, `rawr_mean`
- **S67** ToadStool: universal precision architecture (`compile_shader_universal`)
- **S68** ToadStool: dual-layer universal precision (0 f32-only, 700 shaders)

## Validation Checks

| Check | Expected |
|-------|----------|
| 5 GPU ODE systems finite (128 batches each) | All finite |
| DiversityFusion GPU ≈ CPU (Shannon, Simpson) | `GPU_VS_CPU_F64` |
| CPU diversity local ≡ upstream (Shannon, Simpson, Bray-Curtis) | `EXACT_F64` |
| erf(1.0), ln_gamma(5.0), norm_cdf(1.96) | `ANALYTICAL_F64` |
| Anderson 3D eigenvalues computed, r ∈ (0,1) | Finite, valid range |
| NMF W,H non-negative | ≥ 0.0 |
| Ridge weights finite | Finite |
| GEMM C[0,0] ≈ CPU reference | `GPU_VS_CPU_F64` |
| Cached dispatch faster than first | Speed improvement |
| Pearson anti-correlated ≈ -1 | Within 0.001 |
| dot, l2_norm local ≡ upstream | `EXACT_F64` |

## Compute Estimate

- CPU-only sections: < 1 second
- GPU sections: < 30 seconds (shader compile + dispatch)
- Total: < 60 seconds on RTX 4070
