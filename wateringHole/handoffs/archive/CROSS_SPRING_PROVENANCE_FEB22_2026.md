# Cross-Spring Evolution Provenance — wetSpring

**Date:** February 22, 2026
**Phase:** 22
**Validation:** Exp094 (39/39 PASS), Exp095 (7 benchmarks)

---

## The Biome Model

Each Spring evolves primitives locally, ToadStool absorbs them, and every
Spring benefits. This document traces every shader and system through its
full evolution lifecycle.

## Provenance Map

### hotSpring → ToadStool → wetSpring

| Primitive | Purpose in wetSpring | Absorbed | Speedup |
|-----------|---------------------|----------|---------|
| `FusedMapReduceF64` | Shannon/Simpson diversity, Hill numbers | Session 18 | 10-50× (>1M) |
| `BatchedEighGpu` | PCoA eigendecomposition | Session 25 | blocked (naga) |
| `ShaderTemplate::for_driver_auto` | f64 preamble injection (Ada/NVK) | Session 18 | enables all f64 |
| `math_f64.wgsl` polyfills | exp/log/pow f64 on Ada GPUs | Session 18 | correctness |

### wetSpring → ToadStool → hotSpring

| Primitive | Purpose upstream | Absorbed | Impact |
|-----------|-----------------|----------|--------|
| `GemmCachedF64` | hotSpring HFB nuclear structure | Session 18 | **60× speedup** |
| `math_f64.wgsl` precision fix | All f64 shaders across all Springs | Session 27 | correctness |
| 12 bio WGSL shaders | DNA/protein/ecology GPU ops | Sessions 18–31g | full bio pipeline |
| `bray_curtis_f64.wgsl` | Distance metric (used by neuralSpring) | Session 27 | reusable |
| `kriging_f64.wgsl` | Spatial interpolation | Session 27 | reusable |

### neuralSpring → ToadStool → wetSpring (NEW — Exp094)

| Primitive | Purpose in wetSpring | Absorbed | Speedup |
|-----------|---------------------|----------|---------|
| `PairwiseHammingGpu` | Sequence distance (metagenomics) | Session 31f | **16.4×** |
| `PairwiseJaccardGpu` | Pangenome distance (gene P/A) | Session 31f | **276.7×** |
| `SpatialPayoffGpu` | Game theory fitness (cooperation) | Session 31f | **19.6×** |
| `BatchFitnessGpu` | Population fitness evaluation | Session 31f | **6.5×** |
| `LocusVarianceGpu` | FST decomposition (allele freq) | Session 31f | **19.2×** |

## Cross-Spring Contribution Count

| Spring | Contributed to ToadStool | Consumed from ToadStool |
|--------|------------------------|------------------------|
| hotSpring | ~50 precision/lattice/spectral ops | GEMM 60×, bio metrics |
| wetSpring | 12 bio shaders + GEMM + f64 fix | 24 GPU primitives (lean) |
| neuralSpring | 5 evolutionary/distance ops + matmul router | eigensolvers, f64 precision |

## Total Absorption State

- **24 primitives** in lean mode (19 wetSpring-evolved + 5 neuralSpring-evolved)
- **4 local WGSL shaders** in Write phase (ODE, kmer, unifrac, taxonomy)
- **1 blocker**: ODE `compile_shader()` vs `compile_shader_f64()` in ToadStool

## Benchmark Highlights (Exp095, Release, RTX 4070)

| Primitive | Problem Size | GPU (µs) | Speedup |
|-----------|-------------|----------|---------|
| PairwiseJaccard | 200 genomes × 2K genes | 151 | **276.7×** |
| SpatialPayoff | 256×256 grid | 52 | **19.6×** |
| LocusVariance | 100 pops × 10K loci | 57 | **19.2×** |
| PairwiseHamming | 500 seqs × 1K bp | 978 | **16.4×** |
| BatchFitness | 4096 × 256 genome | 82 | **6.5×** |

---

*Every Spring contributes. Every Spring benefits.*
