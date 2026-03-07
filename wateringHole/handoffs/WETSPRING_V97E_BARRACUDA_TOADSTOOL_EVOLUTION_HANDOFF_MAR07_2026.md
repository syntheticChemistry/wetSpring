# wetSpring V97e — barraCuda/toadStool Evolution Handoff

**Date:** March 7, 2026
**From:** wetSpring (life science biome)
**To:** barraCuda team, toadStool team, coralReef team
**License:** AGPL-3.0-or-later
**Covers:** Full API consumption inventory, absorption feedback, cross-spring evolution insights, re-export suggestions, next-gen absorption targets

---

## Executive Summary

wetSpring has completed the **full rewire to modern barraCuda APIs** (V97e).
All builder patterns adopted, precision routing wired, provenance API integrated.
This handoff provides:

1. **Complete primitive consumption inventory** — what wetSpring actually uses
2. **Cross-spring evolution insights** — patterns that benefit all springs
3. **Re-export suggestions** — types that need deeper paths than necessary
4. **Absorption feedback** — what works well, what could be improved
5. **Next-gen targets** — what wetSpring needs next from barraCuda/toadStool

---

## §1 Primitive Consumption Inventory

### GPU Ops (`barracuda::ops::*`) — 25+ distinct types

| Primitive | wetSpring Module | Category | Notes |
|-----------|-----------------|----------|-------|
| `FusedMapReduceF64` | diversity, kmd, eic, reconciliation, streaming, spectral_match, merge_pairs, chimera | stats | Most-used single primitive |
| `BrayCurtisF64` | diversity, streaming | diversity | Condensed distance matrices |
| `GemmF64` | gemm_cached, spectral_match | linalg | Taxonomy NB, cosine similarity |
| `BatchedEighGpu` | pcoa | linalg | PCoA eigendecomposition |
| `BatchedOdeRK4F64` | ode_sweep | numerical | 5 ODE systems via trait-generated WGSL |
| `KrigingF64` | kriging | spatial | Spatial interpolation |
| `VarianceF64` | stats_gpu | stats | GPU variance |
| `CorrelationF64` | stats_gpu | stats | GPU Pearson correlation |
| `CovarianceF64` | stats_gpu | stats | GPU covariance |
| `WeightedDotF64` | stats_gpu, eic | linalg | Spectral cosine |
| `PeakDetectF64` | signal_gpu | signal | 1D peak detection |
| `BatchToleranceSearchF64` | tolerance_search_gpu | search | ppm/Da m/z matching |
| `KmdGroupingF64` | kmd_grouping_gpu | chem | Kendrick grouping |
| `PairwiseL2Gpu` | pairwise_l2_gpu | distance | L2 distances |
| `DiversityFusionGpu` | rarefaction | bio | Shannon+Simpson+evenness fused |
| `BatchedMultinomialGpu` | rarefaction | rng | GPU multinomial sampling |
| `ComputeDispatch` | 6 ODE GPU modules, gemm_cached | device | Builder for custom dispatches |

### Bio Ops (`barracuda::ops::bio::*` / top-level re-exports) — 15 types

| Primitive | wetSpring Module | Builder Pattern? |
|-----------|-----------------|:----------------:|
| `HmmBatchForwardF64` | hmm_gpu | Yes (`HmmForwardArgs`) |
| `Dada2EStepGpu` | dada2_gpu | Yes (`Dada2DispatchArgs`) |
| `PangenomeClassifyGpu` | pangenome_gpu | No |
| `DnDsBatchF64` | dnds_gpu | No |
| `AniBatchF64` | ani_gpu | No |
| `SnpCallingF64` | snp_gpu | No |
| `QualityFilterGpu` | quality_gpu | No |
| `RfBatchInferenceGpu` | random_forest_gpu | No |
| `KmerHistogramGpu` | kmer_gpu | No |
| `UniFracPropagateGpu` | unifrac_gpu | No |
| `PairwiseHammingGpu` | hamming_gpu | No |
| `PairwiseJaccardGpu` | jaccard_gpu | No |
| `SpatialPayoffGpu` | spatial_payoff_gpu | No |
| `BatchFitnessGpu` | batch_fitness_gpu | No |
| `LocusVarianceGpu` | locus_variance_gpu | No |

### Validation Binaries — Additional types (in `bin/` only)

| Type | Usage |
|------|-------|
| `GillespieGpu` + `GillespieModel` | 4 validation binaries (builder pattern wired) |
| `SmithWatermanGpu` + `SwConfig` | Alignment validation |
| `TreeInferenceGpu` + `FlatForest` | Decision tree validation |
| `FelsensteinGpu` + `PhyloTree` | Phylogenetic likelihood |
| `HillGateGpu` | Neural-cross-spring bio validation |

### Non-GPU (`barracuda::numerical::`, `stats::`, `special::`, `spectral::`)

| Module | Types | Usage |
|--------|-------|-------|
| `numerical` | 5 ODE types + `BatchedOdeRK4` + `OdeSystem` | CPU derivatives for all 5 ODE systems |
| `stats` | `AlphaDiversity`, `BootstrapMeanGpu`, `KimuraGpu`, `HargreavesBatchGpu`, `JackknifeMeanGpu` | Diversity delegation, extended stats |
| `spectral` | `anderson_2d`, `anderson_3d`, `lanczos`, `level_spacing_ratio` | Anderson localization IPC handler |
| `shaders` | `Precision`, `provenance::*` | Universal precision, cross-spring tracking |
| `esn_v2` | `ESN`, `ESNConfig`, `MultiHeadEsn`, `ExportedWeights` | Reservoir computing bridge |
| `tensor` | `Tensor` | ESN bridge |

### metalForge (forge crate)

| Type | Usage |
|------|-------|
| `WgpuDevice` | Bridge to GPU for cross-substrate dispatch |
| `BandwidthTier` | PCIe/memory tier classification |
| `GPU_DISPATCH_OVERHEAD_US` | Dispatch threshold calibration |
| `stats::mean`, `correlation::std_dev` | Node assembly statistics |

---

## §2 Cross-Spring Evolution Insights

### What wetSpring Learned That Benefits All Springs

1. **Builder patterns dramatically improve crash diagnostics.** When a GPU dispatch
   panics on NVK/NVVM, the `expect()` message now includes which named field
   was being processed, not just "argument 7 of 11."

2. **`PrecisionRoutingAdvice` is essential for NVK.** The `F64NativeNoSharedMem`
   variant correctly routes around shared-memory f64 reduction bugs on NVK.
   Without it, `VarianceF64` and `CorrelationF64` silently return zeros on
   NVK Vulkan. All springs with GPU reductions should check this.

3. **Provenance API enables runtime auditing.** wetSpring's `validate_cross_spring_provenance`
   binary (Exp312) proves that 28 shaders can be queried, filtered by origin/consumer,
   and the cross-spring matrix validated programmatically. This should be
   adopted by all springs for CI.

4. **I/O streaming matters for large genomics.** Deprecating `parse_fastq()` /
   `parse_mzml()` / `parse_ms2()` in favor of streaming iterators reduced peak
   memory from O(file) to O(record). Other springs with file-based I/O should
   consider the same pattern.

5. **`FusedMapReduceF64` is the universal workhorse.** It appears in 8 distinct
   wetSpring GPU modules. Any spring doing GPU statistics should use this
   rather than building custom reduction kernels.

### Cross-Spring Shader Flows (from provenance registry)

| Direction | Count | Key Shaders |
|-----------|:-----:|-------------|
| hotSpring → wetSpring | 5 | DF64 core/transcendentals, stress virial, Verlet neighbor, ESN readout |
| wetSpring → neuralSpring | 3 | Smith-Waterman, Gillespie SSA, HMM forward |
| neuralSpring → wetSpring | 2 | KL divergence, chi-squared |
| airSpring → wetSpring | 3 | Hargreaves ET₀, seasonal pipeline, moving window |
| groundSpring → wetSpring | 2 | Welford mean+variance, chi-squared CDF |
| **Total wetSpring consumed** | **17** | From all 5 springs |

---

## §3 Re-Export Suggestions for barraCuda

These types are consumed by wetSpring but require deep module paths. Springs
would benefit from top-level or shallow re-exports:

| Type | Current Path | Suggested Re-export |
|------|-------------|---------------------|
| `HmmForwardArgs` | `barracuda::ops::bio::hmm::HmmForwardArgs` | `barracuda::HmmForwardArgs` |
| `Dada2DispatchArgs` | `barracuda::ops::bio::dada2::Dada2DispatchArgs` | `barracuda::Dada2DispatchArgs` |
| `Dada2Buffers` | `barracuda::ops::bio::dada2::Dada2Buffers` | (with `Dada2DispatchArgs`) |
| `Dada2Dimensions` | `barracuda::ops::bio::dada2::Dada2Dimensions` | (with `Dada2DispatchArgs`) |
| `GillespieModel` | `barracuda::ops::bio::gillespie::GillespieModel` | `barracuda::GillespieModel` |
| `PrecisionRoutingAdvice` | `barracuda::device::driver_profile::PrecisionRoutingAdvice` | `barracuda::device::PrecisionRoutingAdvice` |
| `Rk45DispatchArgs` | `barracuda::ops::rk45_adaptive::Rk45DispatchArgs` | `barracuda::Rk45DispatchArgs` (for future) |

**Pattern:** barraCuda already re-exports `HmmBatchForwardF64`, `GillespieGpu`, etc.
from `lib.rs`. The builder arg structs should follow the same pattern.

---

## §4 Absorption Feedback

### What Works Well

- **ComputeDispatch builder** — clean, chainable, no boilerplate BGL code
- **Trait-based ODE systems** — `OdeSystem` trait + `BatchedOdeRK4` generates
  WGSL at compile time. Zero local shader maintenance.
- **`compile_shader_universal()`** — single call for any precision variant
- **Buffer pooling** — `TensorContext` + `BufferPool` eliminated manual buffer lifecycle
- **Device-lost detection** — `WgpuDevice::is_lost()` enables graceful CPU fallback

### Areas for Evolution

- **`BatchedOdeRK45F64`** exists but wetSpring hasn't adopted it yet.
  The adaptive step-size controller is ready; wetSpring needs to migrate
  5 fixed-step RK4 ODE systems to adaptive RK45. This is P1 for next session.

- **`mean_variance_to_buffer()`** enables zero-readback chained GPU pipelines
  (mean→normalize→reduce→next kernel). wetSpring's `stats_gpu` module still
  reads mean/variance back to CPU between dispatches. Wiring this would
  eliminate 2 GPU→CPU round-trips per diversity calculation.

- **`GpuCalibration` / `AutoTuner`** — barraCuda has these but wetSpring
  hasn't wired them. For large-scale metagenomics (N>100K samples), auto-tuned
  workgroup sizes could matter.

---

## §5 For toadStool Team

### IPC Namespace

wetSpring's IPC handler (`ipc/handlers/science.rs`) implements:
- `science.diversity` (Shannon, Simpson, Chao1, Bray-Curtis)
- `science.anderson` (2D/3D Anderson localization)
- `science.spectral` (Lanczos eigenvalues, level spacing ratio)

These use `barracuda::spectral::*` and `barracuda::stats::*` directly.
toadStool S130's `science.*` namespace (10 methods) overlaps. Confirm
whether wetSpring should migrate to toadStool's IPC proxy or continue
calling barraCuda directly.

### coralReef Integration

wetSpring has **zero local WGSL shaders** — fully lean. coralReef's
`shader.compile.wgsl` IPC endpoint is relevant when/if wetSpring needs
to compile generated WGSL (e.g. from `BatchedOdeRK4::generate_shader()`)
through coralReef's native compiler for SM70-SM89 targets. Currently
this flows through wgpu's naga backend.

---

## §6 Next-Gen Absorption Targets

| Priority | Target | Impact |
|----------|--------|--------|
| **P1** | `BatchedOdeRK45F64` + `Rk45DispatchArgs` | Adaptive step for all 5 ODE systems |
| **P2** | `mean_variance_to_buffer()` chaining | Zero-readback diversity pipeline |
| **P3** | coralReef shader compilation proxy | Native GPU binary for generated WGSL |
| **P4** | `GpuCalibration` auto-tuning | Large-scale metagenomics optimization |

---

## Verification

```bash
cd wetSpring
cargo fmt -p wetspring-barracuda -p wetspring-forge -- --check
cargo clippy -p wetspring-barracuda --all-targets --features gpu -- -D warnings
cargo clippy -p wetspring-forge --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p wetspring-barracuda -p wetspring-forge --no-deps
cargo test -p wetspring-barracuda -p wetspring-forge
cargo run --features gpu --bin validate_cross_spring_provenance
```

All commands pass as of V97e commit `7f390ff`.
