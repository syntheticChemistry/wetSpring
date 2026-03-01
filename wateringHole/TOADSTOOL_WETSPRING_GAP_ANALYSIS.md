# ToadStool barracuda vs wetSpring: Deep Gap Analysis

**Date:** March 1, 2026  
**Scope:** Compare ToadStool barracuda public exports with wetSpring actual usage. Identify unused primitives, adoption opportunities, and maintenance reduction.
**ToadStool Pin:** S70+++ (`1dd7e338`)

**V84 additions:** 7 new CPU domains validated (adapter, placement, PCoA, bootstrap phylo, EIC, KMD, feature table); Python parity proof (Exp253).

---

## Step 1: ToadStool Public Exports

### lib.rs — Top-Level Modules

| Module | Feature | Description |
|--------|---------|-------------|
| `error` | always | Error types |
| `linalg` | always | Linear algebra (nmf, ridge, graph, sparse, solve, eigh, cholesky, svd, qr, lu) |
| `numerical` | always | ODEs, integration, gradients, Hessian, RK45 |
| `special` | always | erf, gamma, bessel, chi_squared, factorial, legendre, laguerre |
| `tolerances` | always | Centralized validation tolerances (LINALG_*, REDUCTION_*, BIO_*, SPECIAL_*) |
| `validation` | always | ValidationHarness, exit_no_gpu, require! macro |
| `math` | always | Re-exports special + erf_batch, erfc_batch |
| `stats` | always | bootstrap, chi2, correlation, diversity, **evolution** (S70), hydrology, **jackknife** (S70), metrics, moving_window_f64, normal, regression, spectral_density |
| `auto_tensor` | gpu | Auto tensor |
| `benchmarks` | gpu | Benchmark suite |
| `compute_graph` | gpu | Compute graph |
| `device` | gpu | WgpuDevice, ComputeDispatch, storage_bgl_entry, uniform_bgl_entry, TensorContext, Fp64Strategy, GpuDriverProfile |
| `dispatch` | gpu | dispatch_for, pairwise_substrate, batch_fitness_substrate, ode_substrate, hmm_substrate, spatial_substrate, DispatchConfig, DispatchTarget |
| `ops` | gpu | 200+ WGSL ops (linalg, bio, fused_map_reduce_f64, bray_curtis_f64, etc.) |
| `shaders` | gpu | Precision, ShaderTemplate |
| `spectral` | gpu | anderson_3d, lanczos, level_spacing_ratio, GOE_R, POISSON_R, etc. |
| `staging` | gpu | GpuRingBuffer, PipelineStats |
| `tensor` | gpu | Tensor |
| `unified_hardware` | gpu | BandwidthTier |
| Bio re-exports | gpu | AniBatchF64, BatchFitnessGpu, Dada2EStepGpu, DnDsBatchF64, FelsensteinGpu, GillespieGpu, HillGateGpu, HmmBatchForwardF64, KmerHistogramGpu, LocusVarianceGpu, MultiObjFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu, PairwiseL2Gpu, PangenomeClassifyGpu, QualityFilterGpu, RfBatchInferenceGpu, SmithWatermanGpu, SnpCallingF64, SpatialPayoffGpu, StencilCooperationGpu, SwarmNnGpu, TaxonomyFcGpu, TreeInferenceGpu, UniFracPropagateGpu, WrightFisherGpu, BatchedMultinomialGpu, FstVariance |

### stats/mod.rs — Public API

- **bootstrap:** bootstrap_ci, bootstrap_mean, bootstrap_median, bootstrap_std, rawr_mean, BootstrapCI
- **chi2:** chi2_decomposed, chi2_decomposed_weighted, Chi2Decomposed
- **correlation:** correlation_matrix, covariance, covariance_matrix, pearson_correlation, spearman_correlation
- **diversity:** alpha_diversity, bray_curtis, bray_curtis_condensed, bray_curtis_matrix, chao1, **chao1_classic** (S70), condensed_index, observed_features, pielou_evenness, rarefaction_curve, shannon, shannon_from_frequencies, simpson, AlphaDiversity
- **evolution** (S70): **kimura_fixation_prob**, **error_threshold**, **detection_power**, **detection_threshold**
- **hydrology:** crop_coefficient, **fao56_et0** (S70), hargreaves_et0, hargreaves_et0_batch, soil_water_balance
- **jackknife** (S70): **jackknife**, **jackknife_mean_variance**, JackknifeResult
- **metrics:** dot, hill, hit_rate, index_of_agreement, l2_norm, mae, mbe, mean, monod, nash_sutcliffe, percentile, r_squared, rmse
- **moving_window_f64:** moving_window_stats_f64, MovingWindowResultF64
- **normal:** norm_cdf, norm_cdf_batch, norm_pdf, norm_pdf_batch, norm_ppf
- **regression:** fit_all, fit_exponential, fit_linear, fit_logarithmic, fit_quadratic, FitResult
- **spectral_density:** empirical_spectral_density, marchenko_pastur_bounds

### ops/bio/mod.rs — Bio GPU Primitives

| Primitive | wetSpring Usage |
|-----------|-----------------|
| AniBatchF64 | ✅ ani_gpu.rs |
| BatchFitnessGpu | ✅ batch_fitness_gpu.rs |
| Dada2EStepGpu | ✅ dada2_gpu.rs |
| DnDsBatchF64 | ✅ dnds_gpu.rs |
| FelsensteinGpu, PhyloTree | ✅ validate_metalforge_v6, validate_streaming_ode_phylo, etc. |
| GillespieGpu, GillespieConfig | ✅ validate_metalforge_full_v2, validate_toadstool_bio |
| HillGateGpu | ✅ cooperation_gpu.rs (V84) |
| HmmBatchForwardF64 | ✅ hmm_gpu.rs |
| KmerHistogramGpu | ✅ validate_local_wgsl_compile |
| LocusVarianceGpu | ✅ locus_variance_gpu.rs |
| MultiObjFitnessGpu | ❌ **UNUSED** |
| PairwiseHammingGpu | ✅ hamming_gpu.rs |
| PairwiseJaccardGpu | ✅ jaccard_gpu.rs |
| PairwiseL2Gpu | ✅ pairwise_l2_gpu.rs (V75) |
| PangenomeClassifyGpu | ✅ pangenome_gpu.rs |
| QualityFilterGpu | ✅ quality_gpu.rs |
| RfBatchInferenceGpu | ✅ random_forest_gpu.rs |
| SmithWatermanGpu, SwConfig | ✅ validate_metalforge_full_v2 |
| SnpCallingF64 | ✅ snp_gpu.rs |
| SpatialPayoffGpu | ✅ spatial_payoff_gpu.rs |
| StencilCooperationGpu | ✅ cooperation_gpu.rs (V84) |
| SwarmNnGpu | ❌ **UNUSED** |
| TaxonomyFcGpu | ✅ validate_local_wgsl_compile |
| TreeInferenceGpu, FlatForest | ✅ decision_tree, gbm, reconciliation |
| UniFracPropagateGpu | ✅ unifrac_gpu.rs |
| WrightFisherGpu | ❌ **UNUSED** |
| BatchedMultinomialGpu | ✅ rarefaction_gpu.rs (V75) |
| FstVariance, fst_variance_decomposition | ✅ fst_variance.rs (V75) |

---

## Step 2: wetSpring barracuda Imports Catalog

### Direct `use barracuda::` Imports (wetSpring)

| Import | Files |
|--------|-------|
| `barracuda::special::erf` | validate_barracuda_cpu_v13/v14, validate_paper_math_control_v1, validate_soil_*, validate_notill_*, benchmark_python_vs_rust_v2 |
| `barracuda::stats::norm_cdf` | validate_barracuda_cpu_v13/v14, validate_paper_math_control_v1, validate_soil_*, validate_notill_*, benchmark_python_vs_rust_v2, validate_pure_gpu_pipeline |
| `barracuda::stats::pearson_correlation` | benchmark_cross_spring_modern, validate_pure_gpu_pipeline |
| `barracuda::stats::AlphaDiversity` | bio/diversity.rs (re-export) |
| `barracuda::stats::{correlation::variance, pearson_correlation}` | validate_pure_gpu_pipeline |
| `barracuda::stats::spearman_correlation` | validate_cold_seep_pipeline |
| `barracuda::special::{erf, ln_gamma, regularized_gamma_p}` | benchmark_cross_spring_modern |
| `barracuda::numerical::trapz` | benchmark_cross_spring_modern |
| `barracuda::numerical::{BistableOde, OdeSystem}` | bio/bistable.rs |
| `barracuda::numerical::{CapacitorOde, CooperationOde, MultiSignalOde, PhageDefenseOde}` | bio/*.rs |
| `barracuda::numerical::ode_generic::{BatchedOdeRK4, OdeSystem}` | bio/*_gpu.rs (5 ODE modules) |
| `barracuda::numerical::CapacitorOde` | bio/capacitor_gpu.rs |
| `barracuda::linalg::{graph_laplacian, ridge_regression}` | benchmark_cross_spring_modern |
| `barracuda::linalg::nmf::{self, NmfConfig, NmfObjective}` | validate_cross_spring_s62, validate_metalforge_drug_repurposing, validate_repodb_nmf, validate_matrix_pharmacophenomics |
| `barracuda::linalg::sparse::CsrMatrix` | validate_gpu_drug_repurposing |
| `barracuda::device::{WgpuDevice, storage_bgl_entry, uniform_bgl_entry}` | bio/*_gpu.rs, gpu.rs, gemm_cached.rs |
| `barracuda::device::{GpuDriverProfile, TensorContext, Fp64Strategy, latency::WgslOpClass}` | gpu.rs |
| `barracuda::shaders::Precision` | validate_pure_gpu_streaming_v4, validate_cross_spring_evolution_v71, validate_barracuda_gpu_v6, bio/*_gpu.rs, gemm_cached.rs |
| `barracuda::spectral::*` | handlers.rs, validate_cold_seep_pipeline, validate_barracuda_gpu_v4, validate_spectral_cross_spring, validate_dynamic_anderson, validate_df64_anderson, validate_soil_*, validate_anderson_*, validate_*, validate_real_ncbi_pipeline |
| `barracuda::unified_hardware::BandwidthTier` | metalForge bridge.rs, dispatch.rs |
| `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64` | reconciliation_gpu, derep_gpu, chimera_gpu, rarefaction_gpu, kmd_gpu, validate_* |
| `barracuda::ops::linalg::gemm_f64::GemmF64` | derep_gpu, chimera_gpu, gemm_cached.rs |
| `barracuda::ops::kriging_f64::{KrigingF64, KrigingResult, VariogramModel}` | bio/kriging.rs |
| `barracuda::ops::peak_detect_f64::PeakDetectF64` | bio/signal_gpu.rs |
| `barracuda::ops::transe_score_f64::TranseScoreF64` | validate_cross_spring_s62, validate_gpu_drug_repurposing |
| `barracuda::ops::linalg::BatchedEighGpu` | validate_gpu_ode_sweep |
| `barracuda::ops::sparse_gemm_f64::SparseGemmF64` | validate_gpu_drug_repurposing |
| `barracuda::ops::bio::diversity_fusion::*` | bio/diversity_fusion_gpu.rs (re-export) |
| Bio re-exports (FelsensteinGpu, PhyloTree, etc.) | validate_metalforge_*, validate_toadstool_bio, bio/*_gpu.rs |

**Note:** wetSpring uses `wetspring_barracuda::validation::{Validator, data_dir, exit_skipped}` — **not** `barracuda::validation::ValidationHarness`.

---

## Step 3: Gap Analysis & Recommendations

### 1. ComputeDispatch Builder

| Aspect | Status |
|--------|--------|
| **ToadStool** | `ComputeDispatch::new(device, "Op").shader(src, "main").storage_read(0, &buf).storage_rw(1, &out).uniform(2, &params).dispatch_1d(n).submit()` |
| **wetSpring** | Manual BGL via `storage_bgl_entry`, `uniform_bgl_entry` in 6 ODE modules + GemmCached |
| **Relevance** | High — bio, chemistry, environmental GPU modules |
| **Replace local?** | Yes — bistable_gpu, capacitor_gpu, cooperation_gpu, multi_signal_gpu, phage_defense_gpu, gemm_cached |
| **Maintenance** | Reduces ~80 lines of boilerplate per module (BGL + pipeline + pass setup) |

**Recommendation:** **Adopted V75.** Refactored the 6 ODE GPU modules and GemmCached to use `ComputeDispatch` instead of manual BGL. ~400 lines removed, single source of truth for bind-group layout.

---

### 2. ValidationHarness vs wetSpring Validator

| Aspect | Status |
|--------|--------|
| **ToadStool** | `ValidationHarness::new("name").check_abs(label, obs, exp, tol).check_rel(...).check_upper(...).check_lower(...).check_bool(...).require!(h, result, label).finish()` |
| **wetSpring** | `Validator::new("name").check(...).check_count(...).check_pass(...).section(...).finish()` + `data_dir()`, `exit_skipped()` |
| **Relevance** | High — 158+ validation binaries |
| **Replace local?** | Partial — API differs (check_abs vs check with tolerance inline) |
| **Maintenance** | Marginal — would require rewiring 158 binaries; wetSpring's Validator has `data_dir`, `exit_skipped` which ValidationHarness lacks |

**Recommendation:** **Keep local Validator.** Documented in ABSORPTION_MANIFEST: "ValidationHarness available but not consumed — wetSpring keeps local Validator with simpler API suited to Python-baseline pattern." Adoption would require 158-binary rewire for marginal benefit. Consider adding `check_rel`, `check_upper`, `check_lower` to wetSpring's Validator if needed.

---

### 3. Stats Functions Not Consumed

| Function | Domain Relevance | Could Replace Local? | Recommendation |
|----------|------------------|----------------------|----------------|
| `bootstrap_ci`, `bootstrap_mean`, `rawr_mean` | Phylogenetic bootstrap (Exp031) | Yes — bootstrap workload | **Adopted V84** — bootstrap_ci, rawr_mean consumed. |
| `chi2_decomposed`, `chi2_decomposed_weighted` | Goodness-of-fit (PFAS, spectral) | Maybe | **Evaluate** — if any validation does χ² by hand, delegate. |
| `correlation_matrix`, `covariance_matrix` | PCoA, spectral | Maybe | **Evaluate** — PCoA uses BatchedEighGpu; correlation_matrix may simplify some paths. |
| `hargreaves_et0`, `crop_coefficient`, `soil_water_balance` | airSpring hydrology | No — wetSpring is bio/PFAS | **Skip** — not in wetSpring domain. |
| `fit_exponential`, `fit_logarithmic`, `fit_quadratic`, `fit_all` | Heaps law, growth curves | Yes — pangenome uses fit_linear | **Adopted V84** — fit_exponential, fit_quadratic, fit_logarithmic, fit_all consumed. |
| `moving_window_stats_f64` | airSpring IoT | No | **Skip** — not in wetSpring domain. |
| `index_of_agreement`, `nash_sutcliffe`, `hit_rate` | Model validation | Maybe | **Evaluate** — if any validation compares model vs observed, adopt. |
| `empirical_spectral_density`, `marchenko_pastur_bounds` | Random matrix theory | Maybe | **Evaluate** — spectral module uses anderson_3d, lanczos; these could complement. |

---

### 4. Bio Ops Unused by wetSpring

| Primitive | Domain Relevance | Could Replace Local? | Recommendation |
|-----------|------------------|----------------------|----------------|
| **WrightFisherGpu** | Population genetics, allele frequency | Yes — locus_variance, Fst | **Adopt** — if wetSpring adds population genetics (Fst, allele trajectories), use. |
| **StencilCooperationGpu** | Spatial game theory, biofilm | Yes — cooperation_gpu uses BatchedOdeRK4 | **Evaluate** — cooperation_gpu is ODE-based; StencilCooperationGpu is stencil-based. Different models. |
| **BatchedMultinomialGpu** | Rarefaction, bootstrap resampling | Yes — rarefaction_gpu, metalForge bootstrap | **Adopt** — rarefaction uses FusedMapReduceF64; BatchedMultinomialGpu is purpose-built for multinomial sampling. metalForge workload references it. |
| **FstVariance** | Fst decomposition, population structure | Yes — locus_variance_gpu | **Adopt** — LocusVarianceGpu is used; FstVariance provides decomposition. Add if Fst analysis is needed. |
| **HillGateGpu** | Two-input Hill AND (QS logic) | Maybe | **Evaluate** — wetSpring has Hill kinetics in ODEs; HillGateGpu is discrete logic. Different use case. |
| **MultiObjFitnessGpu** | Evolutionary algorithms | Maybe | **Evaluate** — batch_fitness_gpu uses BatchFitnessGpu; MultiObjFitness is for Pareto fronts. |
| **PairwiseL2Gpu** | Euclidean distance matrix | Yes — PCoA, diversity | **Adopt** — PairwiseJaccardGpu, PairwiseHammingGpu used; PairwiseL2Gpu would complete the set for continuous traits. |
| **SwarmNnGpu** | Swarm neural network | Maybe | **Skip** — niche; no current wetSpring workload. |

---

### 5. math_f64 / ShaderTemplate

| Aspect | Status |
|--------|--------|
| **ToadStool** | `ShaderTemplate::with_math_f64_auto(shader)`, `ShaderTemplate::math_f64_subset(&["sqrt_f64"])`, `ShaderTemplate::for_driver_auto(shader, true)` |
| **wetSpring** | Uses `device.compile_shader_universal(&wgsl, Precision::F64, Some("label"))` — which internally uses ShaderTemplate/precision pipeline |
| **Direct ShaderTemplate use?** | No — wetSpring does not call ShaderTemplate directly |
| **compile_op_shader?** | wetSpring does not use abstract op compilation; uses concrete ops (GemmF64::WGSL, BatchedOdeRK4::generate_shader()) |

**Recommendation:** **No change.** wetSpring correctly uses `compile_shader_universal`; ShaderTemplate is an internal implementation detail. No adoption needed.

---

### 6. Precision Enum (F64 vs Df64)

| Aspect | Status |
|--------|--------|
| **wetSpring** | All GPU modules use `Precision::F64` |
| **ToadStool** | Supports F16, F32, F64, Df64 (double-single for ~10× throughput on FP32 cores) |
| **Correct usage?** | Yes — wetSpring uses `Precision::F64` for scientific accuracy |

**Recommendation:** **Consider Df64 for throughput-critical paths.** GemmCached, ODE sweeps could optionally use `Precision::Df64` on consumer GPUs for ~10× speedup where precision is acceptable. Document as optional optimization.

---

### 7. Dispatch System (dispatch_for, substrates)

| Aspect | Status |
|--------|--------|
| **ToadStool** | `dispatch_for("matmul", 1000)` → CPU or GPU; `ode_substrate(n_systems, n_steps)`, `hmm_substrate(n_states, n_obs)`, etc. |
| **wetSpring** | Does **not** use dispatch_for or substrates — always uses GPU when available |
| **Relevance** | Medium — could avoid GPU dispatch overhead for small workloads |

**Recommendation:** **Evaluate.** wetSpring validation binaries typically assume GPU or skip. For production pipelines with variable batch sizes, `dispatch_for` or `ode_substrate` could route small batches to CPU. Low priority.

---

### 8. barracuda::tolerances

| Aspect | Status |
|--------|--------|
| **ToadStool** | `tolerances::LINALG_MATMUL`, `BIO_HMM`, `SPECIAL_ERF`, etc. with `check(computed, expected, &tol)` |
| **wetSpring** | Own `tolerances` module (bio, instrument, gpu, spectral) — 97 named constants |
| **Overlap** | Both define similar categories (linalg, bio, special) |

**Recommendation:** **Keep local.** wetSpring's tolerances are domain-specific (bio, instrument, spectral) and already centralized. Could optionally import ToadStool's `tolerances::check()` helper if it simplifies validation code.

---

## Summary: Priority Recommendations

| Priority | Primitive | Action |
|----------|-----------|--------|
| ~~P0~~ | ~~ComputeDispatch~~ | **DONE V75** — 6 modules refactored |
| ~~P1~~ | ~~BatchedMultinomialGpu~~ | **DONE V75** — rarefaction_gpu adopted |
| ~~P1~~ | ~~PairwiseL2Gpu~~ | **DONE V75** — pairwise_l2_gpu.rs |
| ~~P2~~ | ~~FstVariance~~ | **DONE V75** — fst_variance.rs |
| ~~P2~~ | ~~bootstrap_ci, rawr_mean~~ | **DONE V84** — consumed |
| **P2** | Precision::Df64 | Evaluate — throughput optimization for ODE/GEMM |
| **P3** | dispatch_for, substrates | Evaluate — CPU fallback for small batches |
| ~~P3~~ | ~~fit_exponential, fit_quadratic~~ | **DONE V84** — consumed |
| **Skip** | ValidationHarness | Keep local Validator |
| **Skip** | hydrology, moving_window | Not in wetSpring domain |
| **Skip** | ShaderTemplate direct use | No change — use compile_shader_universal |

---

## Appendix: ToadStool lib.rs pub use/pub mod Quick Reference

```text
pub mod: error, linalg, numerical, special, tolerances, validation
pub mod math: special re-exports
pub mod stats: bootstrap, chi2, correlation, diversity, hydrology, metrics, moving_window_f64, normal, regression, spectral_density
[gpu] pub mod: auto_tensor, benchmarks, compute_graph, device, dispatch, ops, shaders, spectral, staging, tensor, unified_hardware, ...
[gpu] pub use ops::bio::{AniBatchF64, BatchFitnessGpu, Dada2EStepGpu, ...}
prelude: BarracudaError, Result, ComputeGraph, Device, Tensor, ...
```
