# Cross-Spring Shader Evolution

> **V149 note (Apr 20, 2026):** This document preserves the V126 shader
> provenance map as a fossil record. Current state: barraCuda v0.3.12,
> **800+** WGSL shaders, **150+** primitives consumed, **zero** local WGSL
> (fully lean). wetSpring guideStone Level 4+ — NUCLEUS validated (38/38
> pass, 4 skip, exit 0), calling barraCuda over IPC via `primalspring::composition`
> API (48 consumed methods, v0.9.17 manifest). **10** open primal gaps (7 resolved).
> See `docs/PRIMAL_GAPS.md` for composition status.

**Last updated**: Mar 16, 2026 — V126
**Validated by**: V98+ cross-spring evolution (Exp319-320), upstream rewire (zero API breakage), Exp313-318 (173/173), standalone barraCuda v0.3.5, 1,443+ tests

---

## Overview

barraCuda is the shared GPU/NPU compute substrate for the ecoPrimals
ecosystem. Its 784+ WGSL shaders across 38 categories evolved through cross-spring contributions:
each spring (hotSpring, wetSpring, neuralSpring, airSpring) contributed
domain-specific shaders that were absorbed into ToadStool, refined, and made
available to all springs. This document tracks that evolution.

---

## Shader Census by Origin

| Origin | Shader Count | Op Count | Domains |
|--------|-------------|----------|---------|
| hotSpring | ~46 | ~35 | Nuclear HFB, lattice QCD (SU(3), Wilson, HMC), MD, ESN, precision, DF64 core-streaming |
| wetSpring | ~35 | ~25 | Metagenomics, DADA2, ANI, dN/dS, PFAS, SSA, NMF, ODE bio (5 → trait), TransE, taxonomy |
| neuralSpring | ~14 | ~12 | ML, pairwise metrics, evolutionary, spectral IPR, graph linalg, GNN |
| airSpring | ~5 | ~8 | IoT, precision agriculture, Richards PDE, Kriging, van Genuchten |
| ToadStool-native | 508+ | 300+ | math(103), activation(37), linalg(33), loss(31), norm(27), reduce(24), tensor(43), etc. |
| **Total** | **608** | **350+** | 38 shader categories |

---

## Evolution Timeline

| Date | Event | Shaders | Direction |
|------|-------|---------|-----------|
| Feb 14 | hotSpring MD handoff | ~10 MD + cell list | hotSpring → ToadStool |
| Feb 15 | hotSpring GPU sovereignty Phase 1 | Vulkan f64 bypass | hotSpring → ToadStool |
| Feb 16 | wetSpring handoff v1 | 3 bio shaders | wetSpring → ToadStool |
| Feb 17 | Three-springs unified wateringHole | — | Architecture |
| Feb 19 | wetSpring v2-v3 | Gillespie, SmithWaterman | wetSpring → ToadStool |
| Feb 20 | wetSpring v4 | Felsenstein, TreeInference, GemmF64 | wetSpring → ToadStool |
| Feb 20 | neuralSpring S-01/S-11 | TensorSession ML ops | neuralSpring → ToadStool |
| Feb 22 | ToadStool S39 | 18 Spring shaders absorbed (7 bio + 11 HFB) | Consolidation |
| Feb 22 | ToadStool S39 | sparse_eigh + quantize_affine_i8 | ToadStool native |
| Feb 22 | ToadStool S40 | Richards PDE + moving_window | airSpring → ToadStool |
| Feb 22 | ToadStool S41 | 6 f64 shader compile fixes | Cross-spring fix |
| Feb 22 | ToadStool S42 | 19 new WGSL (612 total) | Consolidation |
| Feb 23 | ToadStool HEAD | loop_unroller + Jacobi eigh fix | Cross-spring fix |
| Feb 23 | wetSpring Phase 31 | PCoA naga + spectral cross-spring | wetSpring validates |
| Feb 23 | wetSpring Phase 32 | NCBI-scale GPU (6 exps) | wetSpring validates |
| Feb 23 | wetSpring Phase 33 | NPU reservoir (6 exps) | wetSpring validates |
| Feb 23 | wetSpring Phase 34 | Rewire + cross-spring benchmark | wetSpring validates |
| Feb 24 | ToadStool S54 | graph_laplacian, effective_rank, numerical_hessian, spectral_density, Marchenko-Pastur | neuralSpring → ToadStool |
| Feb 24 | ToadStool S56 | belief_propagation, boltzmann_sampling, disordered_laplacian | neuralSpring → ToadStool |
| Feb 24 | ToadStool S58 | NMF (Euclidean+KL), 5 bio ODE systems, Fp64Strategy, df64_core.wgsl | wetSpring + hotSpring → ToadStool |
| Feb 24 | ToadStool S59 | ridge_regression, anderson_3d_correlated, ValidationHarness, NMF re-exports | wetSpring + neuralSpring → ToadStool |
| Feb 24 | wetSpring V30 | S59 lean: NMF, ridge, ODE, Anderson rewired upstream (~1,312 lines removed) | wetSpring leans on ToadStool |
| Feb 24 | ToadStool S60 | TranseScoreF64, SparseGemmF64, BandwidthTier | neuralSpring + hotSpring → ToadStool |
| Feb 24 | ToadStool S62 | PeakDetectF64, cpu-math feature gate | wetSpring + neuralSpring → ToadStool |
| Feb 25 | wetSpring V33 | CPU-math lean: barracuda always-on, ~177 lines dual-path removed. PeakDetect + TransE GPU. | wetSpring full lean on ToadStool |
| Feb 25 | ToadStool DF64 | DF64 core-streaming: HMC gauge force, Wilson plaquette, action, kinetic energy — FP32 cores on consumer GPUs (~10× throughput) | hotSpring → ToadStool |
| Feb 25 | ToadStool DF64 | gemm_df64.wgsl, lennard_jones_df64.wgsl — DF64 tiled GEMM + pairwise MD forces | hotSpring → ToadStool |
| Feb 25 | ToadStool DF64 | ComputeDispatch builder, storage_bgl_entry/uniform_bgl_entry, gpu_ctx(), unified_hardware refactor | ToadStool architecture |
| Feb 25 | wetSpring V35 | DF64 lean: BGL helpers adopted (6 files, ~258 lines removed), compile_shader_f64, PeakDetect bug reported | wetSpring leans on ToadStool |
| Feb 25 | wetSpring Exp166 | Modern systems benchmark: 5 GPU ODE systems, GEMM cached, cross-spring provenance (19/19 PASS) | wetSpring validates |
| Feb 25 | wetSpring V40 | ToadStool S39-S62+DF64 catch-up: 44→49 primitives, 7/9 P0-P3 delivered, 0 Passthrough, SparseGemmF64 + TopK wired | wetSpring leans on ToadStool |
| Feb 25 | wetSpring Exp168 | Cross-spring evolution validation: hotSpring precision → wetSpring bio → neuralSpring pop-gen → Track 3 GPU (~25 PASS) | wetSpring validates |
| Feb 25 | wetSpring V43 | ToadStool catch-up review: ABSORPTION_TRACKER 46/46 DONE, `normal_cdf` → upstream (50th primitive), 8/9 P0-P3, V40-V42 items documented for ToadStool tracker | wetSpring leans on ToadStool |
| Feb 25 | wetSpring V44 | Complete rewire: `find_w_c` (4 files), `anderson_sweep_averaged` (1 file), `pearson_correlation` CPU delegation, 53 primitives total | wetSpring leans on ToadStool |
| Feb 25 | wetSpring Exp169 | Modern cross-spring evolution benchmark: 4-spring provenance map, CPU math+stats+spectral validated (12/12 PASS) | wetSpring validates |
| Feb 26 | wetSpring V50 | **ODE derivative rewire**: 5 local RHS functions replaced with `barracuda::numerical::ode_bio::*Ode::cpu_derivative`. ~200 lines duplicate derivative math eliminated. c-di-GMP guard preserved as thin wrapper. `hill()` + `qs_rhs()` exposed as public API. Zero local derivative math remains. | wetSpring leans on ToadStool |
| Feb 26 | wetSpring V50 | Revalidation: QS ODE 16/16, Bistable 14/14, Cooperation 20/20, Capacitor 18/18, Multi-Signal 19/19, Phage 12/12, CPU-full 50/50, cross-spring 9/9+12/12. 823 lib tests. | wetSpring validates |

---

## S58-S59 Cross-Spring Evolution: Where Things Evolved to Be Helpful

The S58-S59 absorption cycle demonstrates the full power of cross-spring evolution.
Primitives that originated in one biome became shared infrastructure benefiting all.

### hotSpring precision shaders → used everywhere

hotSpring's `df64_core.wgsl` and `Fp64Strategy` (S58) originated from lattice QCD
where f64 precision is mandatory for gauge field computations. ToadStool absorbed
this as a universal precision strategy: `Native` on hardware with f64, `Hybrid`
(double-single emulation) on f32-only GPUs. This benefits:

- **wetSpring**: ODE bio shaders now generate f64-correct code on all GPUs
- **neuralSpring**: Spectral methods (IPR, Lanczos) maintain f64 across GPU tiers
- **airSpring**: Kriging spatial interpolation benefits from f64 precision on mobile GPUs

### wetSpring bio shaders → used by neuralSpring and beyond

wetSpring's biological ODE systems (Capacitor, Cooperation, MultiSignal, Bistable,
PhageDefense) were absorbed in S58 as `barracuda::numerical::ode_bio`. The ODE
`OdeSystem` trait pattern and `BatchedOdeRK4` generic template that evolved from
wetSpring's Write phase now powers:

- **neuralSpring**: Evolutionary dynamics ODE (Wright-Fisher, Lotka-Volterra)
- **hotSpring**: Nuclear structure ODE (HFB iterations use same batched pattern)
- **ToadStool**: Any future ODE system gets GPU shaders for free via `generate_shader()`

wetSpring NMF (S58) became `barracuda::linalg::nmf`, available to:

- **neuralSpring**: Latent factor discovery in neural activity
- **airSpring**: Sensor network decomposition

### neuralSpring primitives → used by wetSpring

neuralSpring's graph theory primitives (S54: `graph_laplacian`, `effective_rank`,
`numerical_hessian`) are used by wetSpring for:

- Community network analysis (species interaction graphs)
- Hessian-based sensitivity analysis of ODE models
- Effective rank of gene expression matrices

neuralSpring's `ValidationHarness` (S59) provides structured tolerance-aware
validation with `check_abs`/`check_rel`/`require!` macros.

### V50: ODE Derivative Lean — Completing the Full Circle

The V50 rewire completes the ODE absorption cycle at the CPU derivative level.
wetSpring originally wrote 5 biological ODE systems with local RHS functions
(~200 lines of hill functions, Monod kinetics, multi-variable derivatives).
ToadStool absorbed these as `barracuda::numerical::ode_bio::*Ode` (S58), adding
both GPU WGSL generation and CPU `cpu_derivative` methods.

V50 closes the loop: wetSpring now delegates derivative *computation* back to
barracuda, keeping only the integration framework (trajectory storage, clamping,
steady-state analysis) and a thin c-di-GMP convergence guard. The result:

- **Zero local derivative math** — all ODE equations live in barracuda
- **Single source of truth** — change a derivative in barracuda, all springs get it
- **GPU/CPU parity** — same `cpu_derivative` logic mirrors the WGSL `deriv()` function
- **Cross-spring benefit** — neuralSpring's evolutionary ODE, hotSpring's nuclear ODE,
  and any future spring ODE all use the same `OdeSystem` trait infrastructure

This demonstrates the full Write → Absorb → Lean → **Rewire** cycle:
```
wetSpring writes (v24-v25) → ToadStool absorbs (S58) → wetSpring leans (V30) → rewires CPU (V50)
```

### S60-S62: Full CPU-math lean + new GPU primitives

ToadStool S60-S62 introduced the `cpu-math` feature gate, making `barracuda::special`,
`barracuda::linalg`, and `barracuda::numerical` available **without GPU dependencies**.
wetSpring leveraged this architectural change to eliminate all `#[cfg(not(feature = "gpu"))]`
fallback code:

| What was removed | Lines | Now delegates to |
|-----------------|:-----:|------------------|
| `special.rs` local erf (A&S 7.1.26) | ~15 | `barracuda::special::erf` |
| `special.rs` local ln_gamma (Lanczos) | ~30 | `barracuda::special::ln_gamma` |
| `special.rs` local regularized_gamma (series) | ~30 | `barracuda::special::regularized_gamma_p` |
| `esn.rs` local Cholesky ridge solver | ~95 | `barracuda::linalg::ridge_regression` |
| `eic.rs` local trapezoidal loop | ~7 | `barracuda::numerical::trapz` |
| **Total** | **~177** | **5 upstream functions** |

New primitives wired in S60-S62:

| Primitive | Origin | Used by wetSpring | Also benefits |
|-----------|--------|-------------------|---------------|
| `PeakDetectF64` (S62) | Shared need across springs | `bio/signal_gpu.rs` — LC-MS peak detection | neuralSpring (spike detection), airSpring (anomaly detection) |
| `TranseScoreF64` (S60) | neuralSpring KG embeddings | `validate_knowledge_graph_embedding.rs` — drug repurposing | All springs with knowledge graphs |
| `SparseGemmF64` (S60) | hotSpring sparse lattice ops | Available for drug-disease NMF (~5% fill) | neuralSpring (sparse attention), wetSpring (future) |

### Measured benefits (ODE lean benchmark, S62)

| System | Local CPU µs | Upstream µs | Speedup |
|--------|-------------|-------------|---------|
| Capacitor | 1,165 | 774 | **1.51×** |
| Cooperation | 837 | 623 | **1.34×** |
| MultiSignal | 1,589 | 1,200 | **1.32×** |
| Bistable | 1,715 | 1,415 | **1.21×** |
| PhageDefense | 85 | 61 | **1.39×** |

Upstream integrators are 21-51% faster because ToadStool optimizes the shared
`integrate_cpu()` across all springs' usage patterns. Performance improved since
S59 (was 10-43%) due to ToadStool's continuous optimization across all consumers.

### S62+DF64: Core-streaming + architectural cleanup

ToadStool's DF64 expansion (post-S62) introduced a major performance architecture:
routing f64 workloads through FP32 cores on consumer GPUs via double-float (f32-pair)
arithmetic. On RTX 3090, staple multiplications (40% of HMC) now run on 10,496 FP32
cores instead of 164 FP64 units — ~10x throughput for the compute-dominant inner loop.

| New shader/infrastructure | Origin | Benefit |
|--------------------------|--------|---------|
| `su3_df64.wgsl` | hotSpring lattice QCD | SU(3) matrix algebra on FP32 cores |
| `su3_hmc_force_df64.wgsl` | hotSpring | Hybrid gauge force (DF64 compute, f64 projection) |
| `wilson_plaquette_df64.wgsl` | hotSpring | Hybrid plaquette measurement |
| `wilson_action_df64.wgsl` | hotSpring | Hybrid per-site action |
| `kinetic_energy_df64.wgsl` | hotSpring | Hybrid kinetic energy |
| `gemm_df64.wgsl` | shared | Tiled GEMM with DF64 accumulation (benefits all springs) |
| `lennard_jones_df64.wgsl` | hotSpring MD | O(N²) pairwise forces with DF64 |
| `ComputeDispatch` builder | ToadStool | Eliminates ~80-line boilerplate per GPU op |
| `storage_bgl_entry/uniform_bgl_entry` | ToadStool | BGL helper functions (adopted by wetSpring: 6 files, ~258 lines saved) |
| `BarracudaError::gpu_ctx()` | ToadStool | Compact error wrapping for GPU ops |
| `unified_hardware` refactor | ToadStool | 1012-line monolith → 6 focused modules |

**DF64 cross-spring impact**: the `gemm_df64.wgsl` tiled GEMM benefits any spring
doing matrix math on consumer GPUs. wetSpring's drug repurposing NMF (200×150 matrices),
neuralSpring's neural network weight updates, and airSpring's kriging systems all
benefit from ~10x consumer GPU throughput. Once ToadStool exposes
`GemmF64::wgsl_shader_for_device()` publicly, downstream springs can auto-select
DF64 in their cached GEMM pipelines.

**Known issue**: `PeakDetectF64` WGSL shader has an f32→f64 type mismatch bug
(`prominence[idx] = 0.0;` should be `0.0lf`). Reported in wetSpring V35 handoff.

---

## Cross-Spring Benefit Map

### hotSpring contributions used by wetSpring

| Shader/Op | Contributed | Used in wetSpring |
|-----------|------------|-------------------|
| `batched_eigh_*.wgsl` | Feb 15 | PCoA eigendecomposition (Exp101-102, 106) |
| `batched_qs_ode_rk4_f64.wgsl` | Feb 16 | QS biofilm ODE integration (10+ exps) |
| `fused_map_reduce_f64.wgsl` | Feb 14 | Shannon entropy, Simpson, convergence norms |
| `weighted_dot_f64.wgsl` | Feb 14 | EIC integration, spectral matching |
| `esn_reservoir_update.wgsl` | Feb 22 | NPU reservoir deployment (Exp114-119) |
| `esn_readout.wgsl` | Feb 22 | NPU readout quantization (Exp114-119) |
| `batched_bisection_f64.wgsl` | Feb 22 | Eigenvalue refinement in PCoA |
| `broyden_f64.wgsl` | Feb 22 | HFB density mixing pattern |
| Spectral module (Lanczos, Anderson) | Feb 22 | QS-disorder analysis (Exp113, 119) |

### neuralSpring contributions used by wetSpring

| Shader/Op | Contributed | Used in wetSpring |
|-----------|------------|-------------------|
| `pairwise_hamming.wgsl` | Feb 20 | Metagenomic comparison (Exp037-042) |
| `pairwise_jaccard.wgsl` | Feb 20 | Community similarity (Exp037-042) |
| `batch_fitness_eval.wgsl` | Feb 20 | Evolutionary dynamics (Exp037-042) |
| `locus_variance.wgsl` | Feb 20 | Population genetics (Exp037-042) |
| `spatial_payoff.wgsl` | Feb 20 | Spatial cooperation games (Exp037-042) |
| `batch_ipr.wgsl` | Feb 22 | Spectral cross-spring (Exp101) |
| `pairwise_l2.wgsl` | Feb 20 | metalForge distance metrics |

### airSpring contributions used by wetSpring

| Shader/Op | Contributed | Used in wetSpring |
|-----------|------------|-------------------|
| `moving_window.wgsl` | Feb 22 | Bloom time-series (future integration) |
| `kriging_f64.wgsl` | Feb 22 | Spatial diversity mapping |

### wetSpring contributions used by other springs

| Shader/Op | Absorbed | Used by |
|-----------|----------|---------|
| `bray_curtis_f64.wgsl` | Feb 16 | airSpring (sensor similarity), neuralSpring |
| `felsenstein_f64.wgsl` | Feb 20 | neuralSpring (phylogenetic priors) |
| `gillespie_ssa_f64.wgsl` | Feb 19 | hotSpring (decay), neuralSpring (evolution) |
| `smith_waterman_banded_f64.wgsl` | Feb 19 | neuralSpring (sequence alignment) |
| `kmer_histogram.wgsl` | Feb 22 | ToadStool-wide (k-mer ops) |
| `taxonomy_fc.wgsl` | Feb 22 | ToadStool-wide (taxonomy classification) |
| `unifrac_propagate.wgsl` | Feb 22 | ToadStool-wide (phylogenetic distance) |
| `dada2_e_step.wgsl` | Feb 22 | ToadStool-wide (amplicon denoising) |
| `quality_filter.wgsl` | Feb 22 | ToadStool-wide (read quality control) |
| `snp_calling_f64.wgsl` | Feb 22 | ToadStool-wide (variant calling) |
| `hmm_forward_f64.wgsl` | Feb 22 | ToadStool-wide (HMM inference) |
| `pangenome_classify.wgsl` | Feb 22 | ToadStool-wide (gene family assignment) |
| `dnds_batch_f64.wgsl` | Feb 22 | ToadStool-wide (selection analysis) |
| `ani_batch_f64.wgsl` | Feb 22 | ToadStool-wide (nucleotide identity) |
| `rf_batch_inference.wgsl` | Feb 22 | neuralSpring (forest inference) |

### Multi-spring shaders (evolved through joint feedback)

| Shader | Contributing Springs | Key Insight |
|--------|---------------------|-------------|
| `batched_elementwise_f64.wgsl` | airSpring, wetSpring, hotSpring | Universal elementwise ops |
| `fused_map_reduce_f64.wgsl` | wetSpring, airSpring, hotSpring | Map-reduce across all domains |
| `moving_window.wgsl` | airSpring, wetSpring | Time-series windowing |
| `kriging_f64.wgsl` | airSpring, wetSpring | Spatial interpolation |
| `bray_curtis_f64.wgsl` | wetSpring, airSpring | Dissimilarity metrics |

---

## Import Modernization (Phase 34)

16 wetSpring files migrated from deep import paths to crate-root re-exports:

| Before | After |
|--------|-------|
| `barracuda::ops::bio::batch_fitness::BatchFitnessGpu` | `barracuda::BatchFitnessGpu` |
| `barracuda::ops::bio::locus_variance::LocusVarianceGpu` | `barracuda::LocusVarianceGpu` |
| `barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu` | `barracuda::PairwiseHammingGpu` |
| `barracuda::ops::bio::pairwise_jaccard::PairwiseJaccardGpu` | `barracuda::PairwiseJaccardGpu` |
| `barracuda::ops::bio::spatial_payoff::SpatialPayoffGpu` | `barracuda::SpatialPayoffGpu` |
| `barracuda::ops::bio::ani::AniBatchF64` | `barracuda::AniBatchF64` |
| `barracuda::ops::bio::dnds::DnDsBatchF64` | `barracuda::DnDsBatchF64` |
| `barracuda::ops::bio::dada2::Dada2EStepGpu` | `barracuda::Dada2EStepGpu` |
| `barracuda::ops::bio::hmm::HmmBatchForwardF64` | `barracuda::HmmBatchForwardF64` |
| `barracuda::ops::bio::pangenome::PangenomeClassifyGpu` | `barracuda::PangenomeClassifyGpu` |
| `barracuda::ops::bio::quality_filter::QualityFilterGpu` | `barracuda::QualityFilterGpu` |
| `barracuda::ops::bio::rf_inference::RfBatchInferenceGpu` | `barracuda::RfBatchInferenceGpu` |
| `barracuda::ops::bio::snp::SnpCallingF64` | `barracuda::SnpCallingF64` |
| `barracuda::ops::bio::unifrac_propagate::UniFracPropagateGpu` | `barracuda::UniFracPropagateGpu` |

Two config types (`QualityConfig`, `UniFracConfig`) remain on deep paths — ToadStool
S42 does not re-export them at the crate root.

---

## Validation

| Suite | Result | Count |
|-------|--------|-------|
| Library tests | PASS | 823/823 |
| Integration tests | PASS | 21/21 |
| Doc tests | PASS | 19/19 |
| GPU feature compile | PASS | All bins + lib |
| Exp114 NPU QS Classifier | PASS | 13/13 |
| Exp115 NPU Phylo Placement | PASS | 9/9 |
| Exp116 NPU Genome Binning | PASS | 9/9 |
| Exp117 NPU Spectral Screen | PASS | 8/8 |
| Exp118 NPU Bloom Sentinel | PASS | 11/11 |
| Exp119 NPU Disorder Classifier | PASS | 9/9 |
| **Exp120 Cross-Spring Evolution** | **PASS** | **9/9** |
| **Exp168 Cross-Spring S62+DF64** | **PASS** | **~25** |
| **Exp169 Cross-Spring Modern Benchmark** | **PASS** | **12/12** |
| **Exp170-178 Track 4 Soil QS (CPU)** | **PASS** | **183** |
| **Exp179-182 Track 4 Three-Tier** | **PASS** | **138** |
| Exp151 Anderson Correlated Disorder (S59 lean) | PASS | 9/9 |
| Exp158 MATRIX Pharmacophenomics (S59 lean) | PASS | 9/9 |
| Exp159 NMF Drug Repurposing (S59 lean) | PASS | 7/7 |
| Exp160 repoDB NMF Reproduction (S59 lean) | PASS | 9/9 |
| **ODE Lean Benchmark (S59)** | **PASS** | **11/11** |
| Clippy (pedantic+nursery, all features) | PASS | 0 warnings |
| Format check | PASS | 0 diffs |
