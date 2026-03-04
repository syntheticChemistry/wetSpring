# wetSpring — BarraCuda Requirements

**Last Updated**: March 3, 2026 (Phase 95, standalone barraCuda v0.3.1, 150+ primitives consumed (264 ComputeDispatch ops), fully lean, 1,044 lib tests, full 5-tier GREEN)
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

### Rust CPU Modules (47 modules, 882 barracuda tests, 95.46% line / 93.54% fn / 94.99% branch, V65 S68+ rewire)

| Module Domain | Modules | Status |
|--------------|---------|--------|
| I/O | fastq, mzml, ms2, xml, encoding | Sovereign |
| 16S Pipeline | quality, merge_pairs, derep, dada2, chimera, taxonomy, kmer | Sovereign |
| Diversity | diversity, pcoa, unifrac | Sovereign |
| LC-MS | eic, signal, feature_table, spectral_match, tolerance_search, kmd | Sovereign |
| Spatial | kriging | Sovereign |
| Math Biology | ode, qs_biofilm, gillespie, bistable, multi_signal, cooperation, capacitor, phage_defense | Sovereign |
| Phylogenetics | felsenstein, robinson_foulds, hmm, alignment, bootstrap, placement, neighbor_joining, reconciliation | Sovereign |
| Track 1c | ani, snp, dnds, molecular_clock, pangenome | Sovereign |
| ML | decision_tree, random_forest, gbm | Sovereign |
| Drug repurposing | nmf, transe | Sovereign (NEW — Track 3) |

### GPU Primitives (144 ToadStool primitives + 2 BGL helpers, 0 local WGSL, 1,783 GPU checks)

| ToadStool Primitive | wetSpring Use | Checks | Performance |
|-------------------|---------------|--------|-------------|
| `FusedMapReduceF64` | Shannon, Simpson, alpha diversity, spectral norms | 12/12 | CPU-competitive at small N |
| `BrayCurtisF64` | All-pairs Bray-Curtis distance matrix | 6/6 | 0.40x at 100×100 |
| `BatchedEighGpu` | PCoA ordination via eigendecomposition | 5/5 | Validated at f64 |
| `GemmF64` + `FusedMapReduceF64` | Spectral cosine matching | 8/8 | **926× speedup** |
| `VarianceF64` | Variance / standard deviation | 3/3 | Validated |
| `CorrelationF64` + `CovarianceF64` | Pearson r / covariance | 2/2 | Validated |
| `WeightedDotF64` | Weighted dot product | 2/2 | Validated |
| `SmithWatermanGpu` | Banded wavefront alignment | 3/3 | **Absorbed Feb 20** |
| `TreeInferenceGpu` | Decision tree inference | 6/6 | **Absorbed Feb 20** |
| `GillespieGpu` | Parallel SSA trajectories | skip | NVVM driver issue |
| `FelsensteinGpu` | Phylogenetic pruning likelihood | 15/15 | **Absorbed + composed** |
| `GemmF64::WGSL` | Eliminates fragile include_str! path | — | **Absorbed Feb 20** |

### Local WGSL Shaders (0 — Lean COMPLETE)

Original 12 shaders absorbed by ToadStool (S31d/31g + S39-41). All 5 ODE
shaders deleted — replaced by `BatchedOdeRK4::<S>::generate_shader()`.
Historical record of deleted shaders:

| Shader | Vars | Params | CPU ↔ GPU | Exp |
|--------|------|--------|-----------|-----|
| `phage_defense_ode_rk4_f64.wgsl` | 4 | 11 | Exact parity | 099 |
| `bistable_ode_rk4_f64.wgsl` | 5 | 21 | Exact parity | 100 |
| `multi_signal_ode_rk4_f64.wgsl` | 7 | 24 | Exact parity | 100 |
| `cooperation_ode_rk4_f64.wgsl` | 4 | 13 | Exact parity | 101 |
| `capacitor_ode_rk4_f64.wgsl` | 6 | 16 | Exact parity | 101 |

Absorption target: ToadStool `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`.

### GPU Wrappers (12 — 10 Compose/Lean, 1 Passthrough)

Pure GPU promotion (Exp101) + S62 lean eliminated most Passthrough modules:

| Module | ToadStool Primitive | Strategy |
|--------|-------------------|----------|
| `kmd_gpu` | `FusedMapReduceF64` | Compose |
| `gbm_gpu` | `TreeInferenceGpu` | Compose |
| `merge_pairs_gpu` | `FusedMapReduceF64` | Compose |
| `signal_gpu` | `PeakDetectF64` (S62) | **Lean** (rewired from Passthrough) |
| `feature_table_gpu` | `FMR + WeightedDotF64` | Compose |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Compose |
| `derep_gpu` | `KmerHistogramGpu` | Compose |
| `chimera_gpu` | `GemmCachedF64` | Compose |
| `neighbor_joining_gpu` | `FusedMapReduceF64` | Compose |
| `reconciliation_gpu` | CPU `reconcile_dtl()` per family | **Passthrough** — GPU validated but CPU kernel used; needs `BatchReconcileGpu` |
| `molecular_clock_gpu` | `FusedMapReduceF64` | Compose |

---

## GPU Promotion Status (Feb 22, 2026)

### Resolved (ToadStool absorbed)

| Need | Resolution | Date |
|------|-----------|------|
| ~~ODE solver~~ | `RkIntegrator`, `numerical::rk45` | Feb 19 |
| ~~Gillespie SSA~~ | `GillespieGpu` (NVVM driver skip on RTX 4070) | Feb 20 |
| ~~Smith-Waterman~~ | `SmithWatermanGpu` (3/3 GPU checks) | Feb 20 |
| ~~Felsenstein pruning~~ | `FelsensteinGpu` (15/15 composed) | Feb 20 |
| ~~Decision tree~~ | `TreeInferenceGpu` (6/6 GPU parity) | Feb 20 |
| ~~GemmF64::WGSL~~ | Public const, fragile path eliminated | Feb 20 |

### Resolved (Local WGSL, Exp046-063)

| Need | Resolution | Exp |
|------|-----------|-----|
| ~~Bootstrap resampling~~ | GPU Felsenstein per replicate (15/15) | 046 |
| ~~Phylogenetic placement~~ | GPU Felsenstein per edge (15/15) | 046 |
| ~~HMM forward~~ | Local WGSL `hmm_forward_f64.wgsl` (13/13) | 047 |
| ~~ODE parameter sweeps~~ | Local WGSL `batched_qs_ode_rk4_f64.wgsl` (7/7) | 049 |
| ~~Bifurcation analysis~~ | `BatchedEighGpu` eigenvalues (5/5, bit-exact) | 050 |
| ~~ANI pairwise~~ | Local WGSL `ani_batch_f64.wgsl` (7/7) | 058 |
| ~~SNP calling~~ | Local WGSL `snp_calling_f64.wgsl` (5/5) | 058 |
| ~~dN/dS~~ | Local WGSL `dnds_batch_f64.wgsl` (9/9) | 058 |
| ~~Pangenome classify~~ | Local WGSL `pangenome_classify.wgsl` (6/6) | 058 |
| ~~RF batch inference~~ | Local WGSL `rf_batch_inference.wgsl` (13/13) | 063 |

### Remaining GPU Work

| Operation | Strategy | Priority | Effort |
|-----------|----------|----------|--------|
| ~~K-mer counting GPU~~ | ✅ `kmer_gpu` wraps `KmerHistogramGpu` (Exp099) | Done | — |
| ~~UniFrac GPU~~ | ✅ `unifrac_gpu` wraps `UniFracPropagateGpu` (Exp099) | Done | — |
| ~~Cooperation GPU~~ | ✅ Local WGSL ODE shader (Exp101) | Done | — |
| ~~Capacitor GPU~~ | ✅ Local WGSL ODE shader (Exp101) | Done | — |
| ~~13 Tier B/C modules~~ | ✅ Pure GPU promotion complete (Exp101) | Done | — |
| ~~Taxonomy NPU~~ | ✅ `quantize_affine_i8` (ToadStool S39) | Done | — |
| ~~ODE generic absorption~~ | ✅ All 5 ODE shaders → `generate_shader()` (Lean COMPLETE) | Done | — |
| ~~Signal GPU~~ | ✅ Lean on `PeakDetectF64` (ToadStool S62) | Done | — |
| ~~Track 3 NMF/SpMM/TransE/TopK~~ | ✅ All upstream (ToadStool S58-S60) | Done | — |
| `ComputeDispatch` adoption | Migrate GPU ops from manual BGL to `ComputeDispatch` builder | **P3** | Medium |
| DF64 GEMM adoption | Use `Fp64Strategy::Hybrid` for RTX 4070 GEMM dispatch | **P3** | Low |
| `BandwidthTier` in metalForge | Wire PCIe-aware routing into forge dispatch | **P3** | Low |
| `diversity_fusion` absorption | Hand off to ToadStool (P2-5) | **P2** | Low |

### Track 3 — Drug Repurposing GPU Primitives (ALL DELIVERED)

All Track 3 GPU gaps have been closed by ToadStool S58-S62:

| Operation | ToadStool Primitive | Session | Status |
|-----------|-------------------|---------|--------|
| ~~NMF (f64)~~ | `barracuda::linalg::nmf` (Euclidean + KL) | S58 | ✅ Absorbed — wetSpring uses upstream directly |
| ~~Sparse GEMM~~ | `barracuda::ops::sparse_gemm_f64::SparseGemmF64` | S60 | ✅ Available — CSR × dense |
| ~~Cosine similarity~~ | `barracuda::linalg::nmf::cosine_similarity` | S58 | ✅ Absorbed — pairwise on NMF factors |
| ~~Top-K selection~~ | `barracuda::ops::topk::TopK` | S60 | ✅ Available — 1D bitonic sort |
| TransE scoring | `barracuda::ops::transe_score_f64::TranseScoreF64` | S60 | ✅ Available — GPU KG embedding |

### BarraCuda Evolution Path

```
DONE                                     DONE                              GOAL
──────────────────────────               ────────────────────             ──────────────────
Python baseline (40 scripts)  ────────→  Rust CPU parity (380/380) ────→  ✓ DONE (v1–v8)
GPU diversity (38/38)         ────────→  GPU Parity (29 domains)  ──────→  ✓ DONE (Exp101)
GPU pipeline (88/88)          ────────→  GPU RF inference (13/13) ──────→  NPU for low-power inference
CPU 22.5× faster than Python  ────────→  GPU math PROVEN portable ─────→  Streaming v2: 10 domains (Exp105+106)
12 shaders absorbed (S31d/g + S39-41) ─→  45 GPU modules + 5 local ────→  Full Write→Absorb→Lean cycle
37 MF domains validated       ────────→  metalForge PROVEN (Exp104) ───→  ✓ 25/25 papers three-tier
```

---

## Evolution Readiness: Rust Module → WGSL Shader → Pipeline Stage

Tier classification for GPU shader promotion (Write → Absorb → Lean lifecycle):

- **Tier A (Lean)**: Already uses upstream ToadStool primitives. Ready for sovereign pipeline.
- **Tier B (Compose)**: Combines multiple ToadStool primitives. Ready with minor wiring.
- **Tier C (Write)**: Requires new WGSL shader development or complex adaptation.

### Tier A — Lean (direct upstream primitive, no local WGSL)

| Rust Module | ToadStool Primitive | WGSL Shader (upstream) | Pipeline Stage |
|-------------|--------------------|-----------------------|---------------|
| `diversity_gpu` | `FusedMapReduceF64`, `BrayCurtisF64` | `fused_map_reduce_f64.wgsl`, `bray_curtis_f64.wgsl` | Alpha/beta diversity |
| `ani_gpu` | `AniBatchF64` | `ani_batch_f64.wgsl` | Genome-level ANI |
| `snp_gpu` | `SnpCallingF64` | `snp_calling_f64.wgsl` | Variant calling |
| `dnds_gpu` | `DnDsBatchF64` | `dnds_batch_f64.wgsl` | Selection pressure |
| `pangenome_gpu` | `PangenomeClassifyGpu` | `pangenome_classify.wgsl` | Gene presence/absence |
| `hmm_gpu` | `HmmBatchForwardF64` | `hmm_forward_f64.wgsl` | Sequence classification |
| `quality_gpu` | `QualityFilterGpu` | `quality_filter.wgsl` | Read QC |
| `random_forest_gpu` | `RfBatchInferenceGpu` | `rf_batch_inference.wgsl` | ML inference |
| `signal_gpu` | `PeakDetectF64` | `peak_detect_f64.wgsl` | Chromatographic peaks |
| `pcoa_gpu` | `BatchedEighGpu` | `batched_eigh.wgsl` | Ordination |
| `unifrac_gpu` | `UniFracPropagateGpu` | `unifrac_propagate.wgsl` | Phylogenetic distance |
| `kmer_gpu` | `KmerHistogramGpu` | `kmer_histogram.wgsl` | K-mer counting |
| `dada2_gpu` | `Dada2EStepGpu` | `dada2_e_step.wgsl` | Denoising E-step |
| `felsenstein_gpu` | `FelsensteinGpu` | `felsenstein.wgsl` | Tree likelihood |
| `bistable_gpu` | `BatchedOdeRK4::<Bistable>` | `generate_shader()` | ODE phenotypic switch |
| `batch_fitness_gpu` | `BatchFitnessGpu` | `batch_fitness.wgsl` | Evolutionary fitness |
| `hamming_gpu` | `PairwiseHammingGpu` | `pairwise_hamming.wgsl` | Distance metric |
| `jaccard_gpu` | `JaccardGpu` | `jaccard.wgsl` | Set similarity |
| `spatial_payoff_gpu` | `SpatialPayoffGpu` | `spatial_payoff.wgsl` | Game theory grid |
| `locus_variance_gpu` | `LocusVarianceGpu` | `locus_variance.wgsl` | Population genetics |
| `rarefaction_gpu` | `RarefactionGpu` | `rarefaction.wgsl` | Sampling curves |
| `eic_gpu` | `FusedMapReduceF64` | `fused_map_reduce_f64.wgsl` | Ion chromatograms |

### Tier B — Compose (multiple primitives, minor wiring)

| Rust Module | ToadStool Primitives | Pipeline Stage |
|-------------|---------------------|---------------|
| `spectral_match_gpu` | `GemmF64` + `FusedMapReduceF64` | MS2 library matching (926× speedup) |
| `gemm_cached` | `GemmCached` + `TensorContext` | Shared GEMM pipeline cache |
| `streaming_gpu` | Multiple domain primitives | End-to-end GPU streaming |
| `gbm_gpu` | `TreeInferenceGpu` | Gradient boosted inference |
| `merge_pairs_gpu` | `FusedMapReduceF64` | Paired-end merging |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Tree distance metric |
| `chimera_gpu` | `GemmCachedF64` | Chimera detection |
| `neighbor_joining_gpu` | `FusedMapReduceF64` | Tree construction |
| `reconciliation_gpu` | CPU `reconcile_dtl()` passthrough | Gene/species reconciliation (GPU device validated, CPU kernel) |
| `molecular_clock_gpu` | `FusedMapReduceF64` | Divergence dating |
| `kmd_gpu` | `FusedMapReduceF64` | PFAS homologue detection |

### Tier C — Write (local WGSL extension, pending absorption)

| Rust Module | Local Shader | Absorption Target | Status |
|-------------|-------------|-------------------|--------|
| ~~`diversity_fusion_gpu`~~ | ~~`diversity_fusion_f64.wgsl`~~ | ToadStool P2-5 | ✅ Absorbed by ToadStool S63. Local WGSL deleted. wetSpring leans on `barracuda::ops::bio::diversity_fusion`. |

**Tier C is empty — Full Lean ACHIEVED.** Zero local WGSL shaders remain.

### Blocking Items for Sovereign Pipeline

1. ~~**`diversity_fusion_f64.wgsl` absorption**~~ — ✅ Absorbed by ToadStool S63
2. **`reconciliation_gpu` passthrough** — CPU kernel per family; needs `BatchReconcileGpu` for true GPU promotion
3. **`ComputeDispatch` migration** — P3, medium effort, replaces manual BGL setup
4. **DF64 GEMM adoption** — P3, low effort, `Fp64Strategy::Hybrid` for consumer GPUs
5. **`BandwidthTier` wiring** — P3, low effort, PCIe-aware dispatch in metalForge

---

## ToadStool Handoff Notes

- `log_f64` bug found by wetSpring (coefficients halved) — fixed in ToadStool Feb 16
- Native `log(f64)` crashes NVIDIA NVVM compiler — all transcendentals must use portable implementations
- **NVVM workaround**: force `ShaderTemplate::for_driver_auto(source, true)` for shaders using exp/log
- Spectral cosine achieves 926× GPU speedup — the first "GPU wins decisively" benchmark from any spring
- 47 CPU + 45 GPU Rust modules with 2 runtime dependencies (flate2 + bytemuck) — highest sovereignty ratio in the ecosystem
- **V29 handoff**: cross-spring synthesis, deprecated API removal, dead-code cleanup, evolution handoff
- **12 shaders absorbed + 5 ODE leaned (generate_shader) + 12 composed/lean wrappers (0 Passthrough)** — zero local WGSL remains; 8/9 P0-P3 requests delivered; see `barracuda/EVOLUTION_READINESS.md`
- **Rust edition 2024**, MSRV 1.85 — `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
- **`#![deny(unsafe_code)]`** — edition 2024 makes `std::env::set_var` unsafe; `#[allow]` confined to test env-var calls

---

## Dependency Audit (ecoBin Compliance)

### wgpu C Dependency

The `wgpu` crate (via ToadStool / metalForge) pulls in `renderdoc-sys` (C dependency) on native targets via `wgpu-hal`. This is **acceptable** because:

1. **wgpu is the only practical GPU abstraction for Rust** — cross-platform Vulkan/Metal/DX12/WebGPU without vendor lock-in.
2. **The C dependency is optional** — `renderdoc-sys` is only for RenderDoc debugging integration, not runtime GPU compute.
3. **Actual GPU backends are OS-provided** — Vulkan, Metal, DX12 are system libraries; no C runtime in the hot path.
4. **Pure Rust GPU compute is not yet feasible** — WGSL compilation and GPU dispatch require wgpu; there is no pure-Rust alternative for cross-platform GPU compute.

### Pure Rust Dependencies

All other barracuda dependencies are pure Rust (ecoBin compliant): `flate2`, `bytemuck`, and upstream ToadStool/metalForge crates.

### Mitigation

When wgpu achieves pure-Rust backends (e.g. Lavapipe software renderer without C), we will upgrade. Until then, the single optional C dependency is documented and accepted for GPU pipeline continuity.
