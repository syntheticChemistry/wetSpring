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

## 5. Phase 21 Update: GPU/NPU Readiness + Dispatch Validation

### New Absorption Candidates (Exp081–083)

| Module | Layout | Target | Cross-Spring Value |
|--------|--------|--------|-------------------|
| `kmer` (4^k histogram) | Dense u32 buffer, sorted pairs | GPU | Genomic counting for all Springs |
| `unifrac` (CSR flat tree) | CSR arrays + sample matrix | GPU | Generic tree structure — Felsenstein, NJ, DTL |
| `taxonomy` (int8 quantized) | Affine i8 weights | NPU | FC inference pattern for neuralSpring |

### Dispatch Router Validation (Exp080)

The forge dispatch router was validated across 5 hardware configurations.
Key cross-spring finding: **workload variants with graceful degradation**
are needed. A single ODE workload definition cannot handle both GPU
(ShaderDispatch) and CPU-only environments. Applications must define
GPU-optimal and CPU-fallback variants.

This affects hotSpring's physics ODE workloads identically.

### Flat Serialization Standard (Exp078–079)

The `to_flat()` / `from_flat()` pattern proved bitwise-identical for all 6
ODE modules. This should become a barracuda standard for any parameterized
GPU workload — hotSpring's RK4 physics integrators would benefit directly.

---

## 6. What Remains

| Item | Blocker | Owner |
|------|---------|-------|
| ODE sweep shader | `compile_shader` not `compile_shader_f64` in `batched_ode_rk4.rs:209` | ToadStool team |
| CPU math feature | `barracuda::math` feature proposal | ToadStool team |
| `from_existing_simple()` deprecation | Design decision | ToadStool team |
| Auto-dispatch thresholds | Performance profiling needed | Collaboration |
| `KmerHistogramGpu` shader | Needs WGSL for atomic u32 histogram | ToadStool team |
| `UniFracPairwiseGpu` shader | Needs f64 tree-propagation WGSL | ToadStool team (f64 first) |
| `TaxonomyNpuInference` | AKD1000 FC integration | NPU team |
| Generic `FlatTree` type | Architectural: barracuda::ops::tree | ToadStool team |
| Generic int8 quantization | barracuda::ops::quant | ToadStool team |

---

## 7. Phase 22 Update: Pure GPU Streaming + Full Validation Proof

### Streaming Findings (Cross-Spring Relevance)

Exp090-091 proved that **naive round-trip GPU dispatch is 13-16× slower than CPU**.
ToadStool's `GpuPipelineSession` with pre-warmed pipelines eliminates 92-94% of
that overhead, achieving 441-837× speedup over round-trip at batch scale.

**Cross-spring implication**: hotSpring's physics pipelines (HMC, lattice sweeps)
should adopt the same `execute_to_buffer()` pattern. Any multi-stage GPU pipeline
that reads back to CPU between stages is leaving >90% performance on the table.

### PCIe Direct Transfer (Cross-Spring Relevance)

Exp088 proved GPU→NPU data flow without CPU staging. The buffer layout contracts
(`#[repr(C)]`, flat arrays) are sufficient for cross-substrate transfer. This
validates metalForge's architectural assumption that PCIe peer-to-peer DMA can
be used between heterogeneous accelerators.

## 8. Updated Totals

| Metric | Value |
|--------|-------|
| Experiments | 91 |
| Validation checks | 2,173+ (all PASS) |
| Rust tests | 728 |
| Binaries | 81 |
| ToadStool primitives consumed | 23 |
| Local WGSL shaders | 4 (ODE, kmer, unifrac, taxonomy) |
| Tier A modules | 7 |
| Tier B modules | 2 |
| Handoff version | v8 |

---

*License: AGPL-3.0-or-later. All discoveries, code, and documentation are
sovereign community property.*
