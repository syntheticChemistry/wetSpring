# wetSpring V105 — barraCuda / toadStool Evolution Handoff

**Date**: 2026-03-10  
**From**: wetSpring V105  
**To**: barraCuda, toadStool, coralReef  
**License**: AGPL-3.0-or-later  
**Covers**: Complete primitive consumption audit, GPU module inventory, API evolution observations, upstream requests, and visualization layer evolution

---

## Executive Summary

- wetSpring V105 consumes **150+ barraCuda primitives** across **47 GPU modules** and 47 CPU modules
- 6 specific upstream requests identified (3 new GPU primitives, 1 CPU primitive, 2 API improvements)
- Visualization layer evolved: 33 scenario builders, LivePipelineSession, Scatter3D, sample-parameterized profiles
- All 1,288 lib + 219 integration tests pass, zero clippy warnings, zero unsafe code
- 3 code-level TODOs track blocked upstream evolution (BipartitionEncodeGpu, CPU Jacobi, merge pairs GPU)

---

## Part 1: Primitive Consumption Inventory

### 1.1 — barracuda::ops (GPU Compute)

| Primitive | Modules Consuming | Count |
|-----------|-------------------|:-----:|
| `FusedMapReduceF64` | merge_pairs, streaming, neighbor_joining, molecular_clock, diversity, chimera, derep, spectral_match, eic, kmd, reconciliation, kmd_grouping | 12 |
| `GemmF64` | gemm_cached, taxonomy, chimera, derep, spectral_match | 5 |
| `BrayCurtisF64` | streaming, diversity | 2 |
| `PeakDetectF64` | signal_gpu | 1 |
| `BatchedEighGpu` | pcoa_gpu | 1 |
| `CorrelationF64` | stats_gpu | 1 |
| `CovarianceF64` | stats_gpu | 1 |
| `VarianceF64` | stats_gpu | 1 |
| `WeightedDotF64` | stats_gpu, eic_gpu | 2 |
| `BatchToleranceSearchF64` | tolerance_search_gpu | 1 |
| `KrigingF64` | kriging | 1 |
| `SparseGemmF64` | drug_repurposing | 1 |
| `TranseScoreF64` | drug_repurposing | 1 |

### 1.2 — barracuda::ops::bio (Domain GPU)

| Primitive | Module |
|-----------|--------|
| `BatchedMultinomialGpu` | rarefaction_gpu |
| `DiversityFusionGpu` | rarefaction_gpu, diversity_fusion_gpu |
| `PairwiseL2Gpu` | pairwise_l2_gpu |
| `QualityFilterGpu` | quality_gpu |
| `UniFracPropagateGpu` | unifrac_gpu |
| `Dada2EStepGpu` + buffers | dada2_gpu |
| `HmmBatchForwardF64` | hmm_gpu |
| `PangenomeClassifyGpu` | pangenome_gpu |
| `KmerHistogramGpu` | kmer_gpu, chimera_gpu, derep_gpu |
| `SpatialPayoffGpu` | spatial_payoff_gpu |
| `BatchFitnessGpu` | batch_fitness_gpu |
| `LocusVarianceGpu` | locus_variance_gpu |
| `PairwiseHammingGpu` | hamming_gpu |
| `PairwiseJaccardGpu` | jaccard_gpu |
| `AniBatchF64` | ani_gpu |
| `SnpCallingF64` | snp_gpu |
| `DnDsBatchF64` | dnds_gpu |
| `RfBatchInferenceGpu` | random_forest_gpu |
| `TaxonomyFcGpu` | taxonomy_gpu |
| `TreeInferenceGpu` | gbm_gpu |
| `SmithWatermanGpu` | alignment binaries |
| `GillespieGpu` | stochastic binaries |
| `FelsensteinGpu` | phylogenetics |

### 1.3 — barracuda::stats (CPU)

| Function | Consumers |
|----------|-----------|
| `diversity::*` (shannon, simpson, chao1, bray_curtis, pielou_evenness, rarefaction_curve, alpha_diversity) | diversity.rs |
| `fit_linear` | calibration.rs, pangenome.rs |
| `mean`, `percentile`, `variance` | rarefaction_gpu, derep |
| `hill` | qs_biofilm |
| `pearson_correlation`, `spearman_correlation`, `correlation_matrix`, `covariance_matrix` | validation binaries |
| `norm_cdf`, `fit_exponential`, `jackknife_mean_variance` | validation binaries |

### 1.4 — barracuda::numerical (ODE/Integration)

| Primitive | Consumers |
|-----------|-----------|
| `CapacitorOde`, `CooperationOde`, `BistableOde`, `MultiSignalOde`, `PhageDefenseOde` | 5 ODE modules + GPU variants |
| `BatchedOdeRK4` + `generate_shader()` | All 5 ODE GPU modules |
| `BatchedOdeRK4F64` | ode_sweep_gpu |
| `trapz` | signal.rs, eic.rs |
| `rk45::rk45_solve` | ode.rs |

### 1.5 — barracuda::spectral

| Primitive | Consumers |
|-----------|-----------|
| `anderson_2d`, `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `find_w_c` | Anderson disorder validation (6+ binaries) |
| `level_spacing_ratio`, `GOE_R`, `POISSON_R` | Spectral analysis binaries |

### 1.6 — barracuda::device

| Primitive | Purpose |
|-----------|---------|
| `WgpuDevice`, `TensorContext` | All GPU modules |
| `BufferPool`, `PooledBuffer` | Memory management |
| `ComputeDispatch` | Shader dispatch |
| `PrecisionRoutingAdvice`, `Fp64Strategy` | DF64/emulation routing |
| `GpuDriverProfile` | Hardware capability query |
| `WgslOpClass` | Latency-aware benchmarking |

---

## Part 2: Upstream Requests

### P1: BipartitionEncodeGpu (NEW)

**Location**: `bio/robinson_foulds_gpu.rs:76–81`  
**Need**: GPU primitive to convert tree bipartition string sets to u32 bit-vectors  
**Why**: `robinson_foulds_gpu` currently delegates to CPU because `PairwiseHammingGpu` operates on bit-vectors but splits are string-encoded. A `BipartitionEncodeGpu` kernel would enable full GPU RF matrix computation.  
**Impact**: Unblocks GPU acceleration for phylogenetic tree comparison at scale.

### P2: CPU Jacobi Eigendecomposition (NEW)

**Location**: `bio/pcoa.rs:20–24`  
**Need**: `barracuda::linalg::jacobi_eigen` or equivalent CPU eigendecomposition  
**Why**: wetSpring has a local `jacobi_eigen` for CPU PCoA. `BatchedEighGpu` exists for GPU but no CPU counterpart in barraCuda's public API. Exposing one would let wetSpring delete its local copy and fully lean.  
**Impact**: Eliminates the last local linear algebra in wetSpring.

### P3: Merge Pairs GPU Primitive (EXISTING REQUEST)

**Location**: `bio/merge_pairs_gpu.rs:6`  
**Need**: GPU merge-pairs primitive for paired-end read merging  
**Why**: Currently CPU-delegated through `FusedMapReduceF64` for overlap scoring only. A dedicated GPU merge kernel would accelerate the full 16S pipeline.

### P4: BatchReconcileGpu (NEW)

**Need**: Wavefront DP primitive for DTL (duplication-transfer-loss) reconciliation  
**Why**: `reconciliation_gpu` currently runs DP on CPU and only uses GPU for cost aggregation. A batch DP primitive would enable full GPU phylogenetic reconciliation.

### P5: OdeResult Helpers (API IMPROVEMENT)

**Observation**: wetSpring ODE modules extract time/state arrays from `RK45Result` manually. A `.time_series()` or `.state_at(t)` helper on the result type would reduce boilerplate across all ODE consumers.

### P6: KmerCounts f64 Convenience (API IMPROVEMENT)

**Observation**: Several wetSpring modules convert `KmerHistogramGpu` u32 output to f64 for downstream stats. A `.as_f64()` or normalized output mode would reduce conversions.

---

## Part 3: API Evolution Observations

### What Works Exceptionally Well

1. **ODE System traits** — `generate_shader()` from `OdeSystem` eliminated all local WGSL (30,424 bytes deleted). The 5 biological ODE systems are fully lean.
2. **FusedMapReduceF64** — Consumed by 12 GPU modules. The most versatile primitive in the library.
3. **PrecisionRoutingAdvice** — Enables wetSpring to route f64 workloads to native/emulation/CPU without knowing the hardware.
4. **BufferPool** — Reduced GPU memory management boilerplate to zero in all 47 GPU modules.
5. **`fit_linear`** — Clean API for calibration curves and pangenome fitting.

### Friction Points (Not Blockers)

1. `PairwiseHammingGpu` requires pre-encoded bit-vectors — string bipartitions can't be used directly (P1 above)
2. No CPU Jacobi in public API despite GPU `BatchedEighGpu` (P2 above)
3. `ComputeDispatch` BGL boilerplate still significant — could benefit from a builder pattern
4. `CsrMatrix` construction is manual; a from-triplets builder would help

---

## Part 4: Visualization Layer Evolution

wetSpring V105 evolved its petalTongue visualization layer significantly:

| Metric | V102 | V105 |
|--------|------|------|
| Scenario builders | 28 | 33 |
| DataChannel types | 8 | 9 (+ Scatter3D) |
| IPC buffer | 4 KB | 64 KB |
| IPC methods | 5 | 9 |
| Capabilities | 16 | 21 |
| Live streaming | StreamSession | StreamSession + LivePipelineSession |
| Scientific ranges | 3 scenarios | All gauge scenarios |

### New Capabilities Relevant to barraCuda

1. **LivePipelineSession** — Streams barraCuda compute results to petalTongue in real time as pipeline stages complete. Each `complete_stage()` pushes data channels.
2. **Scatter3D** — PCoA 3D ordination now visualized; uses barraCuda `BatchedEighGpu` for GPU PCoA → 3-axis scatter.
3. **Sample-parameterized scenarios** — Scientists provide raw data profiles; wetSpring computes all barraCuda math and builds the visualization automatically.
4. **Calibration scenario** — Wraps `barracuda::stats::fit_linear` into a complete calibration report with R² gauge and predicted unknowns.

---

## Part 5: Test and Quality Status

| Metric | Value |
|--------|-------|
| Lib tests | 1,288 (1,231 with json feature) |
| Integration tests | 219 |
| Clippy warnings | 0 |
| `#[allow]` attributes | 0 (all migrated to `#[expect]`) |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]`) |
| Production mocks | 0 |
| Hardcoded paths | 0 (all `data_dir()` / `temp_dir()` / env) |
| Named tolerances | 179 |
| I/O formats | 8 (FASTQ, mzML, MS2, Nanopore, XML, mzXML, JCAMP-DX, BIOM) |

---

## Part 6: What wetSpring Learned for Ecosystem Evolution

1. **`#[expect]` > `#[allow]`** — Migrating 74 attributes to `#[expect]` uncovered 56 stale suppressions. Lint debt accumulates silently with `#[allow]`.

2. **Feature gating visualization** — The `json` feature gate for `serde`/`serde_json` keeps the core crate zero-dependency. Test counts differ by feature — documentation should always specify.

3. **IPC buffer sizing** — 4KB was insufficient for scenarios with 500+ data points. 64KB handles all current scenarios. petalTongue should document its maximum message size.

4. **Scientific ranges are actionable** — Adding normal/warning ranges to gauge channels transforms dashboards from display to decision support. Every gauge should have them.

5. **Sample-parameterized builders** — The most impactful pattern for scientist empowerment. Instead of hardcoded demo data, accept real profiles and compute everything from barraCuda math.

6. **LivePipelineSession pattern** — Wrapping StreamSession with domain-aware stages is reusable across springs. healthSpring does something similar; the pattern should be documented in wateringHole for ecosystem adoption.
