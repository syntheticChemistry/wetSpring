# wetSpring v18 — Cross-Spring Rewire & Evolution Handoff

**Date:** February 23, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-or-later
**Previous:** V17 — NPU Reservoir (Exp107-119, Feb 23)
**Context:** Phase 34 completes the full import modernization to ToadStool S42,
documents cross-spring shader provenance for the entire 612-shader corpus,
and benchmarks the cross-spring synergy. This handoff captures the current
state of wetSpring's BarraCuda consumption, remaining absorption targets,
and recommendations for ToadStool's evolution.

---

## Part 1: Executive Summary

| Metric | Value |
|--------|-------|
| Experiments | 120 |
| Validation checks | 2,673+ |
| Rust tests | 750 (676 lib + 60 integration + 14 doc) |
| CPU bio modules | 41 |
| GPU bio modules | 42 (27 lean + 5 write + 7 compose + 3 passthrough) |
| NPU module | 1 (ESN → int8 → Akida) |
| ToadStool primitives consumed | 31 |
| Local WGSL shaders | 5 (ODE domains) |
| Import paths modernized | 16 (Phase 34) |
| BarraCuda session alignment | S42 (612 WGSL, BarraCuda rename) |

### Key Phase 34 achievements

1. **16 bio imports modernized** — all `barracuda::ops::bio::module::Type` paths
   migrated to crate-root re-exports (`barracuda::Type`). Two config types
   (`QualityConfig`, `UniFracConfig`) remain deep — recommend root re-export.
2. **Cross-spring shader provenance documented** — 612 WGSL shaders traced to
   origin springs (35 hotSpring, 22 wetSpring, 14 neuralSpring, 5 airSpring, 100+ native).
3. **Exp120 cross-spring evolution benchmark** — exercises diversity (wetSpring),
   QS ODE (hotSpring precision), ESN reservoir (hotSpring/neuralSpring), documents
   evolution timeline and synergy map. 9/9 checks pass.
4. **Full BarraCUDA→BarraCuda rename** — 56 replacements across 14 active docs,
   aligned with ToadStool S42 crate rename.

---

## Part 2: How wetSpring Uses BarraCuda

### 2.1 Consumption Pattern

wetSpring's 42 GPU modules fall into four categories by how they consume
BarraCuda primitives:

| Category | Count | Pattern | Example |
|----------|-------|---------|---------|
| **Lean** | 27 | Direct delegation to ToadStool op | `diversity_gpu` → `FusedMapReduceF64` |
| **Write** | 5 | Local WGSL shader, local dispatch | `bistable_gpu` → `bistable_ode_rk4_f64.wgsl` |
| **Compose** | 7 | Wire multiple ToadStool primitives | `kmd_gpu` → `FusedMapReduceF64` + custom math |
| **Passthrough** | 3 | Accept GPU buffers, CPU kernel | `gbm_gpu` → CPU ensemble, GPU I/O |

### 2.2 ToadStool Primitives Consumed (31)

**Root re-exports (used directly):**
- `FelsensteinGpu`, `PhyloTree`, `FelsensteinResult`
- `FlatForest`, `TreeInferenceGpu`
- `GillespieConfig`, `GillespieGpu`
- `SmithWatermanGpu`, `SwConfig`
- `KmerHistogramGpu`, `TaxonomyFcGpu`, `UniFracPropagateGpu`
- `BatchFitnessGpu`, `LocusVarianceGpu`
- `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `SpatialPayoffGpu`
- `AniBatchF64`, `DnDsBatchF64`, `Dada2EStepGpu`
- `HmmBatchForwardF64`, `PangenomeClassifyGpu`
- `QualityFilterGpu`, `RfBatchInferenceGpu`, `SnpCallingF64`

**Deep paths (not root-exported):**
- `barracuda::ops::bio::unifrac_propagate::UniFracConfig`
- `barracuda::ops::bio::quality_filter::QualityConfig`

**Non-bio ops:**
- `FusedMapReduceF64`, `BrayCurtisF64`, `GemmF64`
- `CorrelationF64`, `CovarianceF64`, `VarianceF64`, `WeightedDotF64`
- `BatchedEighGpu`, `BatchedOdeRK4F64`, `BatchedRk4Config`
- `KrigingF64`, `KrigingResult`, `VariogramModel`

**Device layer:**
- `WgpuDevice`, `TensorContext`, `GpuDriverProfile`, `WgslOpClass`
- `BufferPool`, `PooledBuffer`

**Spectral module:**
- `anderson_hamiltonian`, `find_all_eigenvalues`, `lanczos`, `lanczos_eigenvalues`
- `level_spacing_ratio`, `lyapunov_exponent`, `anderson_2d`, `anderson_3d`
- `almost_mathieu_hamiltonian`, `GOE_R`, `POISSON_R`

**Special functions:**
- `barracuda::special::{erf, ln_gamma, regularized_gamma_p}`

### 2.3 Local WGSL Shaders (5 — Write Phase)

These are the remaining local shaders pending ToadStool absorption:

| Shader | Vars | Params | Absorption Target |
|--------|------|--------|-------------------|
| `phage_defense_ode_rk4_f64.wgsl` | 4 | 11 | `BatchedOdeRK4Generic<4,11>` |
| `bistable_ode_rk4_f64.wgsl` | 5 | 21 | `BatchedOdeRK4Generic<5,21>` |
| `multi_signal_ode_rk4_f64.wgsl` | 7 | 24 | `BatchedOdeRK4Generic<7,24>` |
| `cooperation_ode_rk4_f64.wgsl` | 4 | 13 | `BatchedOdeRK4Generic<4,13>` |
| `capacitor_ode_rk4_f64.wgsl` | 6 | 16 | `BatchedOdeRK4Generic<6,16>` |

All use `compile_shader_f64()` with `fmax`/`fclamp`/`fpow` polyfills.
CPU ↔ GPU parity proven for all 5 (Exp099-101).

---

## Part 3: Cross-Spring Shader Evolution

### 3.1 Provenance Census (ToadStool S42: 612 WGSL)

| Origin | Shaders | Ops | Key Domains |
|--------|---------|-----|-------------|
| hotSpring | ~35 | ~25 | Nuclear HFB, lattice QCD, MD, ESN, precision |
| wetSpring | ~22 | ~18 | Bio, genomics, PFAS, Gillespie, Felsenstein |
| neuralSpring | ~14 | ~12 | ML, pairwise, evolutionary, IPR |
| airSpring | ~5 | ~8 | IoT, agriculture, Richards, Kriging |
| ToadStool-native | 100+ | 200+ | Math, linalg, NN, FHE, attention |

### 3.2 Cross-Spring Benefit to wetSpring

| From | Primitive | Used in wetSpring |
|------|-----------|-------------------|
| hotSpring | `batched_eigh_*.wgsl` | PCoA eigendecomposition (Exp101-102, 106) |
| hotSpring | RK4/RK45 f64 precision | QS biofilm ODE (10+ experiments) |
| hotSpring | ESN reservoir (`esn_v2`) | NPU deployment (Exp114-119) |
| hotSpring | Broyden mixing | HFB pattern → potential iterative convergence |
| hotSpring | Spectral module | QS-disorder analysis (Exp113, 119) |
| neuralSpring | `pairwise_hamming/jaccard` | Metagenomic comparison (Exp037-042) |
| neuralSpring | `batch_fitness`, `locus_variance` | Population genetics (Exp037-042) |
| neuralSpring | `batch_ipr` | Spectral cross-spring (Exp107) |
| airSpring | `moving_window` | Bloom time-series (future integration) |
| airSpring | `kriging_f64` | Spatial diversity mapping |

### 3.3 wetSpring Contributions to Other Springs

| Shader/Op | Absorbed | Available to |
|-----------|----------|-------------|
| `bray_curtis_f64.wgsl` | S39 | airSpring (sensor similarity) |
| `felsenstein_f64.wgsl` | S39 | neuralSpring (phylogenetic priors) |
| `gillespie_ssa_f64.wgsl` | S39 | hotSpring (decay), neuralSpring (evolution) |
| `smith_waterman_banded_f64.wgsl` | S39 | neuralSpring (sequence alignment) |
| `kmer_histogram.wgsl` | S39 | All springs (k-mer ops) |
| `taxonomy_fc.wgsl` | S39 | All springs (classification) |
| `unifrac_propagate.wgsl` | S39 | All springs (phylogenetic distance) |
| `dada2_e_step.wgsl` | S39 | All springs (amplicon denoising) |
| `quality_filter.wgsl` | S39 | All springs (read QC) |
| `snp_calling_f64.wgsl` | S39 | All springs (variant calling) |
| `hmm_forward_f64.wgsl` | S39 | neuralSpring (f32 log variant) |
| `pangenome_classify.wgsl` | S39 | All springs (gene family assignment) |
| `dnds_batch_f64.wgsl` | S39 | All springs (selection analysis) |
| `ani_batch_f64.wgsl` | S39 | All springs (nucleotide identity) |
| `rf_batch_inference.wgsl` | S39 | neuralSpring (forest inference) |

---

## Part 4: Absorption Targets for ToadStool

### P1 — High Priority

| Target | Description | Impact |
|--------|-------------|--------|
| **Root re-export `QualityConfig`** | Currently deep path only | 1 file simplified |
| **Root re-export `UniFracConfig`** | Currently deep path only | 1 file simplified |
| **`BatchedOdeRK4Generic<N,P>`** | Generic ODE solver replacing 5 local shaders | Eliminates 5 WGSL shaders from wetSpring (and future hotSpring ODE) |
| **`esn_v2::ESN::to_npu_weights()`** | ESN readout quantization to int8 | NPU deployment pattern for all springs |

### P2 — Medium Priority

| Target | Description | Impact |
|--------|-------------|--------|
| **`quantize_affine_i8` for `Vec<f64>`** | Currently only supports `Vec<f32>` | wetSpring ESN readout uses f64 |
| **`spectral::lyapunov_exponent` docs** | Missing doc comments | Cross-spring usability |
| **`FlatTree` constructors from `PhyloTree`** | Currently manual CSR construction | Simplify UniFrac GPU path |
| **`BatchedOdeRK45F64`** | Adaptive step-size ODE | Higher-order biology models |

### P3 — Low Priority / Future

| Target | Description | Impact |
|--------|-------------|--------|
| **`moving_window_stats` for bloom** | airSpring IoT → wetSpring bloom monitoring | Cross-spring synergy |
| **`chi_squared_f64`** | Statistical test primitive | Enrichment testing, pangenomics |
| **GPU ESN reservoir update** | `esn_reservoir_update.wgsl` + `esn_readout.wgsl` | GPU-accelerated ESN training |

---

## Part 5: Validation Summary

### 5.1 Hardware Tiers

| Tier | Description | Experiments | Checks |
|------|-------------|:-----------:|:------:|
| BarraCuda CPU | Rust math matches Python | Exp035,043,057,070,079,085,102 | 380/380 |
| BarraCuda GPU | GPU matches CPU reference | Exp064,071,087,092,101 | 702+ |
| metalForge | Substrate-independent (CPU/GPU/NPU) | Exp060,065,080,084,086,088,093,103,104 | 234+ |
| Streaming | Pure GPU, zero CPU round-trips | Exp072,073,075,089,090,091,105,106 | 252+ |
| Cross-spring | neuralSpring + spectral primitives | Exp094,095,107 | 71 |
| NCBI-scale | Real-scale GPU extensions | Exp108-113 | 78 |
| NPU reservoir | ESN → int8 → Akida | Exp114-119 | 59 |
| Cross-spring evolution | Shader provenance + rewire | Exp120 | 9 |

### 5.2 Paper Controls

29 papers reproduced or proxied across four tracks. All use open data
(NCBI SRA, Zenodo, MassBank, EPA, published ODE parameters, or synthetic).
No proprietary data dependencies. Each paper has:

- **BarraCuda CPU** control (Rust math matches Python/paper baseline)
- **BarraCuda GPU** control (GPU math matches CPU reference truth)
- **metalForge** control (substrate-independent: CPU ↔ GPU ↔ NPU same answer)

| Track | Papers | Faculty | Key Methods |
|-------|:------:|---------|-------------|
| Track 1 | 8 | Waters | QS ODE, Gillespie, bistable, cooperation, phage defense |
| Track 1b | 6 | Liu | HMM, Smith-Waterman, Felsenstein, NJ, DTL, bootstrap |
| Track 1c | 6 | R. Anderson | ANI, SNP, dN/dS, pangenome, molecular clock |
| Track 2 | 4 | Jones | LC-MS, EIC, spectral matching, KMD, PFAS |
| Proxy | 2 | Cahill, Smallwood | Algae time-series, bloom surveillance |
| Cross-spring | 1 | Kachkovskiy | Anderson localization, QS-disorder bridge |

### 5.3 Full Suite (Feb 23, 2026)

```
cargo test --lib              → 676 passed, 1 ignored
cargo test --tests            → 21 passed
cargo check --features gpu    → clean (0 errors)
benchmark_cross_spring_evolution → 9/9 PASS
validate_npu_qs_classifier       → 13/13 PASS
validate_npu_phylo_placement     → 9/9 PASS
validate_npu_genome_binning      → 9/9 PASS
validate_npu_spectral_screen     → 8/8 PASS
validate_npu_bloom_sentinel      → 11/11 PASS
validate_npu_disorder_classifier → 9/9 PASS
```

---

## Part 6: ToadStool S39-S42 Integration Review

| Session | Feature | wetSpring Impact |
|---------|---------|-----------------|
| S39 | 18 Spring shader absorption (7 bio + 11 HFB) | All 8 wetSpring bio modules now delegate to upstream |
| S39 | `sparse_eigh`, `quantize_affine_i8` | `BatchedEighGpu` used for PCoA; `quantize_affine_i8` f32-only |
| S40 | Richards PDE + `moving_window_stats` | airSpring → potential wetSpring bloom integration |
| S41 | 6 f64 shader compile fixes | Critical: fixed `batch_pair_reduce`, `bray_curtis`, `cosine_similarity`, `fused_map_reduce`, `kmd_grouping`, `batch_tolerance_search` |
| S42 | 19 new WGSL (612 total), BarraCuda rename | wetSpring aligned: 16 imports modernized, all docs renamed |
| HEAD | `loop_unroller` u32 fix, Jacobi eigh fix | Both confirmed working in wetSpring validation |

### Discrepancies Noted

1. **`QualityConfig` / `UniFracConfig`** — not root-exported despite main types being exported
2. **`quantize_affine_i8`** — only accepts `Vec<f32>`, wetSpring ESN uses `f64` readout weights
3. **`esn_v2::ESN`** — no `to_npu_weights()` method; wetSpring implemented locally in `bio::esn`

---

## Part 7: Evolution Timeline

```
Feb 14  hotSpring MD → ToadStool MD primitives
Feb 15  hotSpring GPU sovereignty Phase 1 → f64 Vulkan bypass
Feb 16  wetSpring handoff v1 → initial bio shaders
Feb 17  Three-springs unified wateringHole
Feb 19  wetSpring v2-v3 → Gillespie, SmithWaterman
Feb 20  wetSpring v4 → Felsenstein, TreeInference, GemmF64
Feb 20  neuralSpring S-01/S-11 → TensorSession ML ops
Feb 22  ToadStool S39 → absorb 18 Spring shaders (7 bio + 11 HFB)
Feb 22  ToadStool S40 → Richards PDE + moving window stats
Feb 22  ToadStool S41 → 6 f64 shader compile fixes (critical)
Feb 22  ToadStool S42 → 19 new WGSL (612 total), BarraCuda rename
Feb 23  ToadStool HEAD → loop_unroller fix, Jacobi eigh fix
Feb 23  wetSpring Phase 31 → PCoA naga bug + spectral cross-spring
Feb 23  wetSpring Phase 32 → NCBI-scale GPU (Exp108-113)
Feb 23  wetSpring Phase 33 → NPU reservoir deployment (Exp114-119)
Feb 23  wetSpring Phase 34 → Cross-spring rewire + evolution benchmark (Exp120)
```

---

## Part 8: Recommendations

1. **Generic ODE solver** — `BatchedOdeRK4Generic<N_VARS, N_PARAMS>` would eliminate
   all 5 local wetSpring WGSL shaders and serve hotSpring's growing ODE needs.
   Each spring would supply only the derivative function; ToadStool provides
   the RK4 integration, dispatch, and f64 polyfill infrastructure.

2. **Root re-export config types** — `QualityConfig` and `UniFracConfig` are the
   only two types still requiring deep import paths. Adding them to the crate
   root would complete the re-export surface for all bio ops.

3. **f64 quantization path** — `quantize_affine_i8` should accept `Vec<f64>` in
   addition to `Vec<f32>`. wetSpring's ESN readout weights are f64; the current
   workaround is a local `to_npu_weights()` implementation in `bio::esn.rs`.

4. **ESN NPU weight export** — Adding `to_npu_weights()` to `esn_v2::ESN` would
   give all springs the NPU deployment pattern that wetSpring validated in
   Exp114-119.

5. **Cross-spring provenance tracking** — Consider maintaining a machine-readable
   provenance tag per shader (e.g., `// @origin: wetSpring`, `// @absorbed: S39`).
   The current comment-based tracking required manual audit to reconstruct.

6. **Shader evolution documentation** — The cross-spring shader evolution document
   (`wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md`) should be maintained as a
   living document. As springs contribute new shaders, the provenance table
   should be updated.

---

*This handoff follows the unidirectional pattern: wetSpring proposes, ToadStool absorbs.*
*License: AGPL-3.0-or-later*
