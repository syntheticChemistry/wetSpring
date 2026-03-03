# Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems

**Date:** 2026-03-02
**Binary:** `validate_cross_spring_evolution_s87`
**Features:** gpu
**ToadStool pin:** S87 (`2dc26792`)
**Result:** 61/61 PASS

## Objective

Comprehensive cross-spring evolution benchmark and validation on ToadStool S87.
Tracks shader provenance through the ecosystem — when and where each primitive was
written, who absorbed it, and which springs now consume it.

## Cross-Spring Shader Provenance

| Spring | Written | Absorbed | Key Primitives | Used By |
|--------|---------|----------|----------------|---------|
| **hotSpring** | v0.4.0 (Feb 14) | S26, S58, S64, S80 | DF64 precision, Anderson spectral, grid/mixing PDE, NVK workarounds, Sovereign compiler | ALL springs |
| **wetSpring** | V6 (Feb 16) | S27, S31, S58, S63, S64 | Bio diversity, 5 ODE systems, DADA2/HMM/alignment, NMF, ridge regression | neuralSpring, groundSpring |
| **neuralSpring** | S-01 (Feb 20) | S27, S54, S56, S64 | GemmF64, graph linalg, pairwise metrics, AlphaFold2, BatchedEncoder | wetSpring (NMF), all springs |
| **airSpring** | V039 (Feb 22) | S40, S70, S81 | Hydrology 6 ET₀ methods, Richards PDE, seasonal pipeline | groundSpring, airSpring |
| **groundSpring** | V54 (Feb 26) | S66, S70, S81 | Bootstrap, jackknife, Wright-Fisher, InterconnectTopology | wetSpring, airSpring |
| **wateringHole** | V69 (Feb 28) | S76, S80, S83 | Boltzmann sampling, Sobol/LHS, BrentGpu, L-BFGS | all springs |

## Key Compositions

Demonstrates how springs compose each other's contributions:

- **wetSpring NMF pipeline** = wetSpring bio (NMF) × neuralSpring (GEMM) × hotSpring (DF64 precision)
- **wetSpring PCoA pipeline** = wetSpring bio (Bray-Curtis) × neuralSpring (BatchedEigh) × hotSpring (precision)
- **All GPU paths** = spring-specific science × hotSpring universal precision layer

## Benchmark Highlights (RTX 4070, DF64 Hybrid)

| Operation | CPU (ms) | GPU (ms) | Speedup | Origin |
|-----------|----------|----------|---------|--------|
| GEMM 256×256 | 28.9 | 4.1 | **7.1×** | neuralSpring→S64 |
| BrayCurtis 50×200 | — | 2.0 | — | wetSpring→S82 |
| GemmCached 256×128×64 | — | 2.5 | — | wetSpring V6 (composed) |
| NMF 100×50 k=5 | 5.2 | — | — | wetSpring→S64 |
| Bootstrap 200×50k | 27.3 | — | — | groundSpring→S70 |
| Boltzmann 5k×2D | 0.26 | — | — | wateringHole→S76 |

## S87-Specific Validation

- Device-lost recovery: `is_lost = false` confirmed (S87 adds `with_device_retry`)
- DF64 roundtrip error: π → 3.55e-15 (near machine epsilon)
- Anderson 4D (S83): 256 sites, builds correctly
- All 6 hydrology ET₀ methods produce valid positive results

## Sections

| § | Domain | Origin | Checks |
|---|--------|--------|--------|
| 0 | GPU Init + Precision | hotSpring | 4 |
| 1 | Bio Diversity Fusion | wetSpring | 4 |
| 2 | GemmF64 GPU scaling | neuralSpring | 6 |
| 3 | GemmCached composed | wetSpring + neural + hot | 6 |
| 4 | Bray-Curtis GPU scaling | wetSpring | 6 |
| 5 | Anderson Spectral 1D→3D→4D | hotSpring | 7 |
| 6 | Hydrology ET₀ (6 methods) | airSpring | 6 |
| 7 | Bootstrap + Evolution | groundSpring | 7 |
| 8 | Sampling + Optimization | wateringHole | 3 |
| 9 | NMF + Graph Theory | wetSpring + neuralSpring | 5 |
| 10 | DF64 Host Protocol | hotSpring + wetSpring | 7 |
| 11 | CPU Throughput Table | cross-spring | 1 |
| 12 | GPU Benchmark Summary | cross-spring | 1 |
| **Total** | | | **61** |
