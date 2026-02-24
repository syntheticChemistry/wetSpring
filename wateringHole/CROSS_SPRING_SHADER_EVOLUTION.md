# Cross-Spring Shader Evolution

**Last updated**: Feb 24, 2026 — ToadStool S59 (650+ WGSL shaders)
**Validated by**: wetSpring Exp120 `benchmark_cross_spring_evolution`, V30 lean validation

---

## Overview

ToadStool BarraCuda is the shared GPU/NPU compute substrate for the ecoPrimals
ecosystem. Its 650+ WGSL shaders evolved through cross-spring contributions:
each spring (hotSpring, wetSpring, neuralSpring, airSpring) contributed
domain-specific shaders that were absorbed into ToadStool, refined, and made
available to all springs. This document tracks that evolution.

---

## Shader Census by Origin

| Origin | Shader Count | Op Count | Domains |
|--------|-------------|----------|---------|
| hotSpring | ~35 | ~25 | Nuclear HFB, lattice QCD, MD, ESN, precision, df64 emulation |
| wetSpring | ~22 | ~18 | Metagenomics, DADA2, ANI, dN/dS, PFAS, SSA, NMF, ODE bio |
| neuralSpring | ~14 | ~12 | ML, pairwise metrics, evolutionary, spectral IPR, graph linalg |
| airSpring | ~5 (shared) | ~8 | IoT, precision agriculture, Richards, Kriging |
| ToadStool-native | 100+ | 200+ | Math, linalg, NN, FHE, attention |
| **Total** | **650+** | **300+** | |

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

### Measured benefits (ODE lean benchmark)

| System | Local CPU µs | Upstream µs | Speedup |
|--------|-------------|-------------|---------|
| Capacitor | 3,176 | 1,813 | **1.75×** |
| Cooperation | 921 | 836 | **1.10×** |
| MultiSignal | 1,568 | 1,229 | **1.28×** |
| Bistable | 1,686 | 1,382 | **1.22×** |
| PhageDefense | 83 | 64 | **1.30×** |

Upstream integrators are 10-43% faster because ToadStool optimizes the shared
`integrate_cpu()` across all springs' usage patterns.

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
| Library tests | PASS | 759/759 |
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
| Exp151 Anderson Correlated Disorder (S59 lean) | PASS | 9/9 |
| Exp158 MATRIX Pharmacophenomics (S59 lean) | PASS | 9/9 |
| Exp159 NMF Drug Repurposing (S59 lean) | PASS | 7/7 |
| Exp160 repoDB NMF Reproduction (S59 lean) | PASS | 9/9 |
| **ODE Lean Benchmark (S59)** | **PASS** | **11/11** |
| Clippy (pedantic+nursery, all features) | PASS | 0 warnings |
| Format check | PASS | 0 diffs |
