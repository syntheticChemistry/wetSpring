# Cross-Spring Evolution: wetSpring Perspective

**Date:** 2026-02-22
**From:** wetSpring (life science & analytical chemistry biome)
**License:** AGPL-3.0-or-later

---

## The Biome Model in Practice

Each Spring writes domain-specific GPU shaders, validates them locally,
and hands off to ToadStool for absorption. No Spring imports from another.
They evolve together through the shared barracuda substrate.

```
hotSpring (physics/precision) ──→ barracuda ←── wetSpring (bio/genomics)
                                      ↑
                                neuralSpring (ML/eigen)
```

wetSpring is the largest consumer of cross-spring evolution effects:
bio computations depend on precision infrastructure (hotSpring), ML
primitives (neuralSpring), and the shared dispatch model (barracuda core).

---

## 1. What wetSpring Gave

### Bio Primitives (12 total absorbed)

| Primitive | Domain | Absorbed | Cross-Spring Value |
|-----------|--------|----------|-------------------|
| SmithWatermanGpu | Sequence alignment | Feb 20 | Basis function alignment (hotSpring) |
| GillespieGpu | Stochastic simulation | Feb 20 | Nuclear decay chains (hotSpring) |
| TreeInferenceGpu | Tree construction | Feb 20 | — |
| FelsensteinGpu | Phylogenetic likelihood | Feb 20 | Model selection patterns |
| HmmBatchForwardF64 | Hidden Markov models | Feb 22 | Sequence models (neuralSpring) |
| AniBatchF64 | Nucleotide identity | Feb 22 | — |
| SnpCallingF64 | Variant detection | Feb 22 | — |
| DnDsBatchF64 | Selection pressure | Feb 22 | — |
| PangenomeClassifyGpu | Gene classification | Feb 22 | — |
| QualityFilterGpu | Read QC | Feb 22 | — |
| Dada2EStepGpu | ASV denoising | Feb 22 | — |
| RfBatchInferenceGpu | Random Forest | Feb 22 | Ensemble methods (neuralSpring), transport prediction (hotSpring) |

### Precision Contributions

| Finding | Impact | Date |
|---------|--------|------|
| `(zero + literal)` pattern for f64 constants | Fixed f32 truncation in all transcendentals | Feb 16 |
| `log_f64` coefficient correction (~1e-3 → ~1e-15) | hotSpring BCS convergence, all Springs using log | Feb 16 |
| RTX 4070 NVVM f64 exp/log failure | `needs_f64_exp_log_workaround()` for Ada Lovelace | Feb 16 |
| SNP binding layout bug | Corrected read-only flag on read-write buffer | Feb 22 |
| `from_existing_simple()` breaks polyfill detection | Real AdapterInfo required for driver workarounds | Feb 22 |

---

## 2. What wetSpring Received

### From hotSpring (Physics/Precision)

| Contribution | How wetSpring Uses It |
|-------------|----------------------|
| `math_f64.wgsl` preamble | All bio shaders use f64 transcendentals through it |
| `ShaderTemplate::for_driver_auto()` | NVK exp/log workaround protects bio shaders |
| `GpuDriverProfile` | Hardware-aware compilation for all GPU modules |
| `compile_shader_f64()` | Central compilation path replacing 8 local shader setups |
| f64 constant precision | All bio f64 computations benefit |

### From neuralSpring (ML/Eigen)

| Contribution | How wetSpring Uses It |
|-------------|----------------------|
| `BatchedEighGpu` | PCoA eigendecomposition, bifurcation analysis |
| `matmul_gpu_evolved` | Taxonomy batch classification via GemmCached |
| `mean_reduce` | Diversity metric aggregation |
| `pairwise_hamming`, `pairwise_jaccard` | Diversity distance computations |

### From barracuda Core

| Contribution | How wetSpring Uses It |
|-------------|----------------------|
| `FusedMapReduceF64` | Shannon, Simpson, observed species, alpha bundle |
| `BrayCurtisF64` | Beta diversity |
| `GemmF64` | Matrix operations across domains |
| `KrigingF64` | Spatial interpolation |
| `VarianceF64`, `CovarianceF64`, `CorrelationF64` | Statistical GPU ops |
| `WeightedDotF64` | Weighted dot products |
| `PrngXoshiro` | GPU random number generation |
| `BufferPool`, `TensorContext` | Memory management |

---

## 3. Concrete Cross-Spring Effects (Observed Feb 22)

### hotSpring → wetSpring: f64 Polyfill Protection
When wetSpring rewired HMM to use `HmmBatchForwardF64`, the shader
compiled through `compile_shader_f64()`, which checks `AdapterInfo`
for Ada Lovelace. On RTX 4070, exp/log polyfills are injected automatically.
This protection was written by hotSpring for its nuclear EOS shaders — it
now protects wetSpring's entire bio shader portfolio.

### neuralSpring → wetSpring: Eigensolver for PCoA
wetSpring's PCoA module delegates eigendecomposition to `BatchedEighGpu`.
This primitive was contributed by neuralSpring for ML Hessian computation.
wetSpring uses it unchanged for ecological distance matrix analysis.

### wetSpring → hotSpring: Precision Fixes
The `log_f64` coefficient fix discovered during DADA2 GPU validation
directly improved hotSpring's BCS bisection solver convergence. The
`(zero + literal)` constant pattern prevents f32 truncation in all
Springs' shaders compiled through `math_f64.wgsl`.

### wetSpring → neuralSpring: HMM and RF Primitives
`HmmBatchForwardF64` is usable for sequence-level neural models.
`RfBatchInferenceGpu` provides GPU-accelerated ensemble inference
that neuralSpring can compose with its training loop.

---

## 4. Absorption Score

| Metric | Before Rewire | After Rewire |
|--------|:------------:|:------------:|
| ToadStool primitives consumed | 15 | **23** |
| Local WGSL shaders | 9 | **1** |
| Cross-spring effects observed | 3 | **6+** |
| Bio primitives in barracuda (from wetSpring) | 4 | **12** |
| Bytes of local shader code | 25,785 | **0** (+ ODE) |

---

## 5. What Remains

| Item | Blocker | Owner |
|------|---------|-------|
| ODE sweep shader | `enable f64;` in ToadStool upstream | ToadStool team |
| CPU math feature | `barracuda::math` feature proposal | ToadStool team |
| `from_existing_simple()` deprecation | Design decision | ToadStool team |
| Auto-dispatch thresholds | Performance profiling needed | Collaboration |

---

*License: AGPL-3.0-or-later. All discoveries, code, and documentation are
sovereign community property.*
