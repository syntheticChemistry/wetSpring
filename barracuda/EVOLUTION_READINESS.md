# wetSpring Evolution Readiness

**Date:** March 16, 2026 (V126 — barraCuda v0.3.5 + toadStool S155 + coralReef Phase 10)
**Pattern:** Write → Absorb → Lean (inherited from hotSpring)
**Status:** 44 GPU modules + CPU modules + IPC + vault + provenance + visualization (all lean, 0 local WGSL, 0 local derivative/regression math), 150+ primitives consumed (standalone barraCuda v0.3.5, wgpu 28). 1,443+ tests (0 failures), 376 experiments, 5,707+ checks, 214 named tolerances (zero inline literals), 354 binaries. `cargo clippy -D warnings -W pedantic -W nursery` CLEAN, **0 silent fallbacks**, **0 `#[allow()]` in entire codebase**. `#![forbid(unsafe_code)]` on all crate roots. All primal names via `primal_names::*` constants. **V125:** Structured `IpcError` enum (28 sites, healthSpring/biomeOS pattern), dual-format `extract_capabilities()` (groundSpring/ludoSpring), generic `socket_env_var()`/`discover_primal()` (sweetGrass pattern), 18 binary OrExit import fixes — all 354 binaries compile clean. **V126:** `DispatchOutcome<T>` protocol vs application error separation, `health.liveness`/`health.readiness` probes, `IpcError` query helpers (`is_retriable()`/`is_timeout_likely()`/`is_method_not_found()`/`is_connection_error()`). **V124:** Workspace `deny.toml`, typed `compute.dispatch` IPC client, structured `tracing`. **V123:** Zero-panic `OrExit`, dual-format discovery, `extract_rpc_error()`. **V122:** `#[expect(reason)]` across 276+ binaries. **See also:** `wateringHole/handoffs/WETSPRING_V125_IPCERROR_CAPABILITIES_DISCOVERY_HANDOFF_MAR16_2026.md`.

### Full Lean Phase

All local WGSL shaders have been absorbed by barraCuda (via ToadStool absorption) and deleted.
The 5 ODE systems completed Write → Absorb → Lean (S58).
`diversity_fusion_f64.wgsl` completed Write → Absorb → Lean (S63).
`bio::diversity` delegated to `barracuda::stats::diversity` (S64).
`special::{dot, l2_norm}` delegated to `barracuda::stats::{dot, l2_norm}` (S64).

**V50 ODE derivative lean:** 5 local RHS functions (`capacitor_rhs`, `coop_rhs`,
`multi_rhs`, `bistable_rhs`, `defense_rhs`) replaced with
`barracuda::numerical::ode_bio::*Ode::cpu_derivative`. Local helper functions
(`hill()`, `hill_repress()`, `monod()`) removed. ~200 lines of duplicate
derivative math eliminated. c-di-GMP convergence guard preserved as thin
wrapper for bistable and multi-signal (fixed-step RK4 stability refinement).

wetSpring is now **fully lean** — zero local math, zero local WGSL, zero local
ODE derivative code.

`barracuda` is now an **always-on** dependency (`default-features = false` for CPU
builds, `barracuda/gpu` for GPU builds). This eliminated all `#[cfg(not(feature = "gpu"))]`
dual-path fallback code for `erf`, `ln_gamma`, `regularized_gamma`, `ridge_regression`,
and `trapz`. Zero duplicate math remains in the codebase.

See `ABSORPTION_MANIFEST.md` for the full ledger.

### S65 Evolution (Phase 43 — V40 catch-up)

ToadStool S39-S62+DF64 (55+ commits since V39) delivered massive infrastructure:

**Absorbed and leaned:**
- `PeakDetectF64` (S62) — `signal_gpu` rewired from Passthrough to Lean
- `storage_bgl_entry`/`uniform_bgl_entry` from `barracuda::device::compute_pipeline`
  (6 files: 5 ODE GPU + `gemm_cached.rs`, ~258 lines BGL boilerplate removed)
- `compile_shader_f64` directly on `GemmF64::WGSL` (replaces `ShaderTemplate::for_driver_auto`)
- `GemmF64::wgsl_shader_for_device()` now public — DF64 GEMM auto-selection unblocked

**Available for future wiring:**
- `compile_shader_universal()` — single call for any precision (F16/F32/F64/DF64) (S67) — **IN USE** by 6 GPU modules
- `Precision::Df64` — DF64 variant for enhanced precision on FP32 cores (S67) — available via `GpuF64::optimal_precision()`
- `compile_template()` — template-based `{{SCALAR}}` shader compilation (S67)
- `Precision::op_preamble()` — abstract precision ops layer (S68)
- `downcast_f64_to_f16()` — F16 downcast with sentinel protection (S68)
- `WgpuDevice::is_lost()` — device-lost detection (S68+) — **WIRED** in `GpuF64::is_lost()` (V63)
- `WgpuDevice::acquire_dispatch()` — concurrency permit (S68+) — transparent via `submit_and_poll`
- `WgpuDevice::max_concurrent_dispatches()` — concurrency budget query (S68+)
- `submit_and_poll()` — resilient submit with catch_unwind (S68+) — **WIRED** in 5 ODE GPU + 1 GEMM module (V64)
- `ComputeDispatch` builder — eliminates 80-line bind-group/pipeline boilerplate
- `Fp64Strategy` auto-detect — Native/Hybrid selection per GPU era — **WIRED** in `GpuF64::fp64_strategy()` (V64)
- DF64 core-streaming — routes f64 through FP32 cores on consumer GPUs (RTX 4070: 5888 FP32 cores vs 92 FP64 units)
- `BandwidthTier` — PCIe-aware routing for metalForge dispatch
- `SparseGemmF64` — CSR × dense GEMM for drug repurposing sparse matrices
- `TranseScoreF64` — GPU TransE KG embedding scoring
- `TopK` — GPU bitonic sort for drug-disease ranking
- `unified_hardware` refactored to 6 focused modules (types, traits, scheduler, discovery, cpu_executor, transfer)
- `from_existing_simple()` deprecated (S68+) — wetSpring already uses `from_existing()` with real `AdapterInfo`

**Upstream requests** (V40 status — 7/9 delivered):
1. ~~Make `GemmF64::wgsl_shader_for_device()` public~~ → **DELIVERED** (S65) — with `Fp64Strategy` auto-detect (Native/Hybrid), DF64 GEMM for FP32 cores
2. ~~Fix `PeakDetectF64` WGSL shader~~ → **DELIVERED** (S62) — full f64 op + `peak_detect_f64.wgsl`; `signal_gpu` already leaned
3. ~~`ComputeDispatch` with cached-pipeline variant~~ → **DELIVERED** (S65) — `ComputeDispatch` builder in `device::compute_pipeline`
4. ~~`barracuda::math::{dot, l2_norm}`~~ → **DELIVERED** (S60) as GPU ops: `NormReduceF64::l2()`, `FusedMapReduceF64::dot()`, `WeightedDotF64::dot()`. CPU `special::{dot, l2_norm}` remain as thin local helpers for validation math.
5. Absorb `diversity_fusion_f64.wgsl` → **DONE** (absorbed S63)
6. ~~`BatchedOdeRK4Generic<N, P>`~~ → **DELIVERED** (S58) via `OdeSystem` trait + `generate_shader()`; all 5 ODE systems leaned
7. ~~GPU Top-K selection~~ → **DELIVERED** (S60) — `ops::topk::TopK` (1D indices, WGSL bitonic sort)
8. ~~NPU int8 quantization helpers~~ → **DELIVERED** (S39) — `quantize_affine_i8`
9. Tolerance module pattern for ToadStool validation → **DELIVERED** (S52) — `barracuda::tolerances` module with `Tolerance` struct + `check()` + 12 named constants. wetSpring keeps its own flat `tolerances.rs` (77 domain-specific constants) which is complementary.

### Next Lean Phase: Absorption Candidates

Following hotSpring's pattern of writing validated extensions as proposals for
ToadStool absorption, these wetSpring modules are candidates for new Lean phase:

| Module | Location | Status | Absorption benefit |
|--------|----------|--------|-------------------|
| `diversity_fusion_f64.wgsl` | barracuda::ops::bio | DONE (absorbed S63) | Fused Shannon + Simpson + evenness |
| `forge::bridge` | `metalForge/forge/src/bridge.rs` | 47 tests | Multi-substrate dispatch |
| `forge::dispatch` | `metalForge/forge/src/dispatch.rs` | 47 tests | Universal workload routing |

All GPU bio modules are now either Lean (upstream primitive) or Compose
(wire upstream primitives). No GPU modules remain in Passthrough.

### Code Quality (Phase 15+)

All modules pass `clippy::pedantic` + `clippy::nursery` (0 warnings, `-D` enforced
in CI). Zero `#[allow()]` in production code — all lint suppressions use `#[expect(reason)]`.
`cargo fmt` (0 diffs), `cargo doc` (0 warnings with and without `--all-features`).
All tolerances centralized in `tolerances/` (214 named constants across 10 submodules —
includes bio, gpu, spectral, instrument domains; zero inline literals remain).
V121: 14 new tolerance constants, ~50 inline literal replacements, crate-level
`#[allow()]` → `#[expect(reason)]`, all hardcoded primal names → `primal_names::*`.
`bio::spectral_match` uses `special::{dot, l2_norm}` instead of inline computation.
`#![deny(unsafe_code)]` enforced crate-wide — **zero unsafe blocks** in library or
test code as of Feb 24, 2026. Test env-var manipulation refactored to pure-function
`resolve_data_dir()` pattern, eliminating all `unsafe { set_var/remove_var }` calls.
All 158 binaries carry `# Provenance` headers. Data paths use `validation::data_dir()`
for capability-based discovery. NCBI API key resolution evolved to capability-based
cascade (env var → `WETSPRING_DATA_ROOT` → XDG config → legacy paths).
`flate2` uses `rust_backend` — zero C dependencies (ecoBin compliant). All 42
Python baselines carry SPDX-License-Identifier + Date headers. DADA2 algorithmic
constants fully documented with provenance (Callahan et al. 2016, R package defaults).
Integration tests use streaming APIs exclusively (`Ms2Iter`, `MzmlIter`, `FastqIter`)
— deprecated buffering APIs (`parse_ms2`, `parse_mzml`, `parse_fastq`) are no longer
exercised outside their own deprecation-gated unit tests. V114: Last 4 validation
binaries migrated from deprecated batch APIs to streaming (FastqIter, Ms2Iter).
GPU buffer limits and dispatch thresholds fully documented with hardware provenance.
CI enforces fmt, clippy (pedantic+nursery), test, doc, and json feature check on
every push/PR. **Rust edition 2024**, MSRV 1.87.

See also: `ABSORPTION_MANIFEST.md` for the full absorption ledger.

---

## Absorption Tiers

| Tier | Meaning | Action |
|------|---------|--------|
| **✅ Absorbed** | ToadStool has the primitive; wetSpring consumes upstream | Lean on upstream |
| **A** | Local code ready for handoff — GPU-friendly, validated, WGSL written | Write handoff doc |
| **B** | CPU-validated, needs GPU-friendly refactoring | Refactor for absorption |
| **C** | CPU-only, no GPU path planned | Keep local |

---

## CPU Modules (40)

| Module | Domain | GPU Tier | ToadStool Primitive | Notes |
|--------|--------|----------|-------------------|-------|
| `alignment` | Smith-Waterman | ✅ Absorbed | `SmithWatermanGpu` | Exp044 |
| `ani` | Average Nucleotide Identity | ✅ Absorbed | `AniBatchF64` | Rewired Feb 22, 2026 |
| `bistable` | ODE toggle switch | ✅ Absorbed | `BatchedOdeRK4::<BistableOde>` | S58 lean, `generate_shader()` |
| `bootstrap` | Phylo resampling | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `capacitor` | Phenotypic capacitor | ✅ Absorbed | `BatchedOdeRK4::<CapacitorOde>` | S58 lean, `generate_shader()` |
| `chimera` | Chimera detection | ✅ Absorbed | `GemmCachedF64` | GEMM-based sketch scoring (pure GPU promotion) |
| `cooperation` | Game theory QS | ✅ Absorbed | `BatchedOdeRK4::<CooperationOde>` | S58 lean, `generate_shader()` |
| `dada2` | Error model | ✅ Absorbed | `Dada2EStepGpu` | Rewired Feb 22, 2026 |
| `decision_tree` | PFAS ML | ✅ Absorbed | `TreeInferenceGpu` | Exp044 |
| `derep` | Dereplication | ✅ Absorbed | `KmerHistogramGpu` | Parallel hashing (pure GPU promotion) |
| `diversity` | α/β diversity | ✅ Absorbed | `BrayCurtisF64`, `FMR` | Exp004/016 |
| `dnds` | Nei-Gojobori dN/dS | ✅ Absorbed | `DnDsBatchF64` | Rewired Feb 22, 2026 |
| `eic` | EIC/XIC extraction | ✅ Absorbed | `FMR` | Lean |
| `feature_table` | Feature extraction | ✅ Absorbed | `FMR` + `WeightedDotF64` | Chains eic_gpu + signal_gpu (pure GPU promotion) |
| `felsenstein` | Pruning likelihood | ✅ Absorbed | `FelsensteinGpu` | Exp046 |
| `gbm` | GBM inference | ✅ Absorbed | `TreeInferenceGpu` | Batch inference (pure GPU promotion) |
| `gillespie` | Stochastic SSA | ✅ Absorbed | `GillespieGpu` | Exp044 |
| `hmm` | Hidden Markov Model | ✅ Absorbed | `HmmBatchForwardF64` | Rewired Feb 22, 2026 |
| `kmd` | Kendrick mass defect | ✅ Absorbed | `FusedMapReduceF64` | Element-wise KMD (pure GPU promotion) |
| `kmer` | K-mer counting | ✅ Absorbed | `KmerHistogramF64` | ToadStool S40 (Exp081) |
| `merge_pairs` | Read merging | ✅ Absorbed | `FusedMapReduceF64` | Batch overlap scoring (pure GPU promotion) |
| `molecular_clock` | Strict/relaxed clock | ✅ Absorbed | `FusedMapReduceF64` | Relaxed rates element-wise (pure GPU promotion) |
| `multi_signal` | Multi-signal QS | ✅ Absorbed | `BatchedOdeRK4::<MultiSignalOde>` | S58 lean, `generate_shader()` |
| `neighbor_joining` | NJ tree construction | ✅ Absorbed | `FusedMapReduceF64` | GPU distance matrix + CPU NJ loop (pure GPU promotion) |
| `ode` | RK4 integrator | ✅ Absorbed | `BatchedOdeRK4F64` | ToadStool S41 (Exp049) |
| `pangenome` | Gene clustering | ✅ Absorbed | `PangenomeClassifyGpu` | Rewired Feb 22, 2026 |
| `pcoa` | PCoA ordination | ✅ Absorbed | `BatchedEighGpu` | Exp016 |
| `phage_defense` | Phage-bacteria defense | ✅ Absorbed | `BatchedOdeRK4::<PhageDefenseOde>` | S58 lean, `generate_shader()` |
| `phred` | Quality scoring | C | — | Per-base lookup |
| `placement` | Phylo placement | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `qs_biofilm` | QS/c-di-GMP ODE | ✅ Absorbed | `BatchedOdeRK4F64` | ToadStool S41 (Exp049) |
| `quality` | Read quality | ✅ Absorbed | `QualityFilterGpu` | Rewired Feb 22, 2026. Adapter logic extracted to `adapter.rs` |
| `random_forest` | RF ensemble | ✅ Absorbed | `RfBatchInferenceGpu` | Rewired Feb 22, 2026 |
| `reconciliation` | DTL reconciliation | ✅ Absorbed | `BatchReconcileGpu` | Workgroup-per-family (pure GPU promotion) |
| `robinson_foulds` | Tree distance | ✅ Absorbed | `PairwiseHammingGpu` | Bipartition bit-vectors (pure GPU promotion) |
| `signal` | Signal processing | ✅ Absorbed | `FusedMapReduceF64` | Batch peak detection (pure GPU promotion) |
| `snp` | SNP calling | ✅ Absorbed | `SnpCallingF64` | Rewired Feb 22, 2026 |
| `spectral_match` | Spectral cosine | ✅ Absorbed | `FMR` spectral cosine | Exp016 |
| `taxonomy/` | Naive Bayes classify | ✅ Absorbed / NPU | `TaxonomyFcF64` | ToadStool S40; `types`, `kmers`, `classifier` submodules (Exp083) |
| `tolerance_search` | Tolerance search | ✅ Absorbed | `BatchTolSearchF64` | Exp016 |
| `unifrac/` | UniFrac distance | ✅ Absorbed | `UniFracPropagateF64` | ToadStool S40; `tree`, `flat_tree`, `distance` submodules (Exp082) |

---

## GPU Modules (42)

| Module | Wraps | ToadStool Primitive | Status |
|--------|-------|-------------------|--------|
| `batch_fitness_gpu` | EA batch fitness | `BatchFitnessGpu` | Lean (neuralSpring) |
| `hamming_gpu` | Pairwise Hamming | `PairwiseHammingGpu` | Lean (neuralSpring) |
| `jaccard_gpu` | Pairwise Jaccard | `PairwiseJaccardGpu` | Lean (neuralSpring) |
| `locus_variance_gpu` | FST per-locus | `LocusVarianceGpu` | Lean (neuralSpring) |
| `spatial_payoff_gpu` | Spatial PD payoff | `SpatialPayoffGpu` | Lean (neuralSpring) |
| `ani_gpu` | ANI pairwise | `AniBatchF64` | ✅ Lean (Feb 22) |
| `chimera_gpu` | Chimera GEMM scoring | `GemmCachedF64` | ✅ Promoted (pure GPU) |
| `dada2_gpu` | DADA2 E-step | `Dada2EStepGpu` | ✅ Lean (Feb 22) |
| `diversity_gpu` | α/β diversity | `BrayCurtisF64`, `FMR` | Lean |
| `dnds_gpu` | dN/dS GPU | `DnDsBatchF64` | ✅ Lean (Feb 22) |
| `eic_gpu` | EIC extraction | `FMR` | Lean |
| `gemm_cached` | Matrix multiply | `GemmCachedF64` | Lean |
| `hmm_gpu` | HMM forward | `HmmBatchForwardF64` | ✅ Lean (Feb 22) |
| `kriging` | Spatial interpolation | `KrigingF64` | Lean |
| `ode_sweep_gpu` | ODE parameter sweep | `BatchedOdeRK4F64` | ✅ Lean (ToadStool S41 fixed compile_shader_f64) |
| `pangenome_gpu` | Pangenome classify | `PangenomeClassifyGpu` | ✅ Lean (Feb 22) |
| `pcoa_gpu` | PCoA eigenvalues | `BatchedEighGpu` | Lean |
| `quality_gpu` | Quality filtering | `QualityFilterGpu` | ✅ Lean (Feb 22) |
| `rarefaction_gpu` | Rarefaction curves | `PrngXoshiro` | Lean |
| `random_forest_gpu` | RF batch inference | `RfBatchInferenceGpu` | ✅ Lean (Feb 22) |
| `snp_gpu` | SNP calling | `SnpCallingF64` | ✅ Lean (Feb 22); gracefully skips on wgpu binding mismatch (catch_unwind) |
| `spectral_match_gpu` | Spectral cosine | `FMR` | Lean |
| `stats_gpu` | Variance/correlation | `FMR` | Lean |
| `streaming_gpu` | Streaming pipeline | Multiple | Lean |
| `taxonomy_gpu` | Taxonomy scoring | `FMR` | Lean |
| `kmer_gpu` | K-mer histogram | `KmerHistogramGpu` | ✅ Lean (Exp099) |
| `unifrac_gpu` | UniFrac propagation | `UniFracPropagateGpu` | ✅ Lean (Exp099) |
| `bistable_gpu` | Bistable QS ODE | `BatchedOdeRK4::<BistableOde>` | ✅ Lean (S58 → `generate_shader()`) |
| `multi_signal_gpu` | Dual-signal QS ODE | `BatchedOdeRK4::<MultiSignalOde>` | ✅ Lean (S58 → `generate_shader()`) |
| `phage_defense_gpu` | Phage defense ODE | `BatchedOdeRK4::<PhageDefenseOde>` | ✅ Lean (S58 → `generate_shader()`) |
| `cooperation_gpu` | Cooperation ODE | `BatchedOdeRK4::<CooperationOde>` | ✅ Lean (S58 → `generate_shader()`) |
| `capacitor_gpu` | Capacitor ODE | `BatchedOdeRK4::<CapacitorOde>` | ✅ Lean (S58 → `generate_shader()`) |
| `kmd_gpu` | Kendrick mass defect | `FusedMapReduceF64` | ✅ Promoted (pure GPU) |
| `gbm_gpu` | GBM batch inference | `TreeInferenceGpu` | ✅ Promoted (pure GPU) |
| `merge_pairs_gpu` | Read merging | `FusedMapReduceF64` | ✅ Promoted (pure GPU) |
| `signal_gpu` | Peak detection | `PeakDetectF64` (S62) | ✅ Lean (S62 → upstream GPU peaks) |
| `feature_table_gpu` | Feature extraction | `FMR` + `WeightedDotF64` | ✅ Promoted (pure GPU) |
| `robinson_foulds_gpu` | Tree distance | `PairwiseHammingGpu` | ✅ Promoted (pure GPU) |
| `derep_gpu` | Dereplication | `KmerHistogramGpu` | ✅ Promoted (pure GPU) |
| `neighbor_joining_gpu` | NJ tree | `FusedMapReduceF64` | ✅ Promoted (pure GPU) |
| `reconciliation_gpu` | DTL reconciliation | `BatchReconcileGpu` | ✅ Promoted (pure GPU) |
| `molecular_clock_gpu` | Molecular clock | `FusedMapReduceF64` | ✅ Promoted (pure GPU) |

---

## Local WGSL Shader Inventory (0 — Lean COMPLETE)

Original 12 shaders absorbed by ToadStool (S31d/31g + S39-41). All 5 ODE
shaders **deleted** — replaced by `BatchedOdeRK4::<S>::generate_shader()` via
`OdeSystem` trait implementations in `bio::ode_systems`. Per-variable clamping
handled by derivative-level guards (`fmax_d` in WGSL, `.max(0.0)` in CPU).

| Shader (deleted) | Vars | Params | Domain | Status |
|-------------------|------|--------|--------|--------|
| ~~`phage_defense_ode_rk4_f64.wgsl`~~ | 4 | 11 | Monod phage-bacteria defense | **Lean COMPLETE** |
| ~~`bistable_ode_rk4_f64.wgsl`~~ | 5 | 21 | QS + cooperative feedback | **Lean COMPLETE** |
| ~~`multi_signal_ode_rk4_f64.wgsl`~~ | 7 | 24 | V. cholerae dual-signal | **Lean COMPLETE** |
| ~~`cooperation_ode_rk4_f64.wgsl`~~ | 4 | 13 | Cooperative QS game theory | **Lean COMPLETE** |
| ~~`capacitor_ode_rk4_f64.wgsl`~~ | 6 | 16 | Phenotypic capacitor | **Lean COMPLETE** |

### Shader Generation Notes

All GPU ODE modules now use `BatchedOdeRK4::<S>::generate_shader()` which:
1. Generates WGSL from `OdeSystem::wgsl_derivative()` at runtime
2. Uses `WgpuDevice::compile_shader_universal(source, Precision::F64)` (V57 rewire)
3. Workgroup dispatch at 64 threads
- Function names avoid `_f64` suffix to prevent `ShaderTemplate` rewriting
- No `// @unroll_hint` in comments (optimizer matches via `contains()`)

---

## ToadStool Primitives Consumed (144 — barracuda always-on)

### Original 15 (pre-Feb 22)

| Primitive | Module(s) | Exp |
|-----------|----------|-----|
| `BrayCurtisF64` | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (Shannon) | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (Simpson) | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (spectral cosine) | spectral_match_gpu | 016 |
| `GemmCachedF64` | gemm_cached | 016 |
| `BatchedEighGpu` | pcoa_gpu, validate_gpu_ode_sweep | 016/050 |
| `BatchTolSearchF64` | tolerance_search | 016 |
| `PrngXoshiro` | rarefaction_gpu | 016 |
| `SmithWatermanGpu` | alignment (via barracuda) | 044 |
| `GillespieGpu` | gillespie (via barracuda) | 044 |
| `TreeInferenceGpu` | decision_tree (via barracuda) | 044 |
| `FelsensteinGpu` | felsenstein, bootstrap, placement | 046 |
| `ShaderTemplate::for_driver_auto` | ode_sweep_gpu, gemm_cached | 047+ |
| `LogsumexpWgsl` | (available, not yet wired) | — |
| `BatchedOdeRK4F64` | ode_sweep_gpu, qs_biofilm, bistable, multi_signal, phage_defense | S41 (Exp049) |

### 8 Bio Primitives (absorbed Feb 22, cross-spring evolution)

| Primitive | wetSpring Module | Origin | ToadStool Session |
|-----------|-----------------|--------|------------------|
| `HmmBatchForwardF64` | hmm_gpu | wetSpring Exp047 | 31d |
| `AniBatchF64` | ani_gpu | wetSpring Exp058 | 31d |
| `SnpCallingF64` | snp_gpu | wetSpring Exp058 | 31d |
| `DnDsBatchF64` | dnds_gpu | wetSpring Exp058 | 31d |
| `PangenomeClassifyGpu` | pangenome_gpu | wetSpring Exp058 | 31d |
| `QualityFilterGpu` | quality_gpu | wetSpring Exp016 | 31d |
| `Dada2EStepGpu` | dada2_gpu | wetSpring Exp016 | 31d |
| `RfBatchInferenceGpu` | random_forest_gpu | wetSpring Exp063 | 31g |

These primitives live in `barracuda::ops::bio::*` and are available to all
springs. neuralSpring's metalForge pipeline can use wetSpring's bio shaders;
hotSpring's precision f64 polyfills improve wetSpring's numerical accuracy.

### 5 neuralSpring Primitives (Exp094, consumed by wetSpring)

| Primitive | wetSpring Use | Exp |
|-----------|---------------|-----|
| `PairwiseHammingGpu` | SNP-based strain distance matrices | 094 |
| `PairwiseJaccardGpu` | Gene presence/absence similarity | 094 |
| `SpatialPayoffGpu` | Spatial PD payoff for cooperation models | 094 |
| `BatchFitnessGpu` | EA batch fitness for evolutionary simulations | 094 |
| `LocusVarianceGpu` | FST per-locus AF variance for population genetics | 094 |

### 4 ToadStool S54-S57 Primitives (Cross-spring evolution)

| Primitive | wetSpring Use | ToadStool Session |
|-----------|---------------|------------------|
| `graph_laplacian` | Community network analysis | S54 |
| `effective_rank` | Gene expression matrix rank | S54 |
| `numerical_hessian` | ODE model sensitivity | S54 |
| `boltzmann_sampling` | Parameter sweep MCMC | S56 |

### 7 ToadStool S58-S59 Primitives (wetSpring lean)

| Primitive | wetSpring Use | ToadStool Session |
|-----------|---------------|------------------|
| NMF (Euclidean + KL) | Drug repurposing, pharmacophenomics | S58 |
| 5× ODE bio systems | Bistable, Capacitor, Cooperation, MultiSignal, PhageDefense | S58 |
| `ridge_regression` | ESN reservoir readout | S59 |
| `anderson_3d_correlated` | Correlated disorder validation | S59 |
| `trapz` | EIC peak integration | S59 |
| `erf`, `ln_gamma`, `regularized_gamma_p` | Special functions (always-on CPU math) | S59 |
| `norm_cdf` | Normal CDF (delegates from `special::normal_cdf`) | S59 |

### 8 ToadStool S60-S65 Primitives (V40 catch-up)

| Primitive | wetSpring Use | ToadStool Session | Status |
|-----------|---------------|------------------|--------|
| `TranseScoreF64` | GPU knowledge graph scoring (TransE) | S60 | Available — Track 3 |
| `SparseGemmF64` | Sparse NMF for drug repurposing | S60 | Available — Track 3 |
| `TopK` | Drug-disease pair ranking | S60 | Available — Track 3 |
| `PeakDetectF64` | GPU LC-MS peak detection | S62 | ✅ Lean — `signal_gpu` rewired |
| `BandwidthTier` | PCIe-aware routing for metalForge | S62 | Available |
| `ComputeDispatch` | Eliminates BGL/pipeline boilerplate | S65 | Available |
| `Fp64Strategy` | Native/Hybrid f64 auto-selection | DF64 | Available |
| DF64 GEMM (`gemm_df64.wgsl`) | ~10× throughput on FP32 cores (RTX 4070) | DF64 | Available |

---

## Absorption Queue (handoff to ToadStool)

### All Lean — nothing queued for Write

All local WGSL shaders absorbed and deleted. All CPU-math dual-paths eliminated.
barracuda is always-on. The next evolution cycle will focus on wiring remaining
unused primitives (SparseGemmF64 for NMF, BandwidthTier for metalForge dispatch).

### Pure GPU Promotion Complete

All formerly Tier B/C modules now have GPU wrappers. No modules remain in
Tier B or Tier C. The remaining CPU-only domain is `phred` (per-base lookup,
no parallelism benefit) and `fastq_parsing` (I/O-bound).

---

## Validation Coverage by Tier

| Tier | CPU Modules | GPU Modules | CPU Checks | GPU Checks |
|------|:-----------:|:-----------:|:----------:|:----------:|
| ✅ Absorbed (Lean) | 41 | 42 (all lean on upstream) | 1,476+ | 1,578+ |
| Compose | — | 7 (wire ToadStool primitives) | — | — |
| Passthrough | — | 3 (GPU buffers, CPU kernel) | — | — |
| Write (local WGSL) | 0 | 0 | — | — |
| Dispatch routing | — | — | 80 | — |
| Streaming/transfer | — | — | 57 | 82 |
| **Total** | **41** | **42** | **1,476+** | **1,578+** |

---

## Write → Absorb → Lean History

| Date | Event |
|------|-------|
| Feb 16 | Handoff v1: diversity shaders, log_f64 bug, BrayCurtis pattern |
| Feb 17 | Handoff v2: bio primitives requested (SW, Gillespie, Felsenstein, DT) |
| Feb 19 | Handoff v3: primitive verification, fragile GEMM path eliminated |
| Feb 20 | ToadStool absorbs 4 bio primitives (commit cce8fe7c) |
| Feb 20 | Exp046: FelsensteinGpu composed for bootstrap + placement |
| Feb 20 | Exp047: Local HMM shader written + validated (absorption candidate) |
| Feb 20 | Exp049: Local ODE shader written (upstream fix candidate) |
| Feb 20 | Exp050: BatchedEighGpu validated for bifurcation (bit-exact) |
| Feb 20 | Track 1c: 5 new modules (ani, dnds, molecular_clock, pangenome, snp) |
| Feb 20 | Exp051-056: R. Anderson deep-sea metagenomics (133 new checks) |
| Feb 20 | Exp057: BarraCuda CPU v4 — 23 domains, 128/128 parity checks |
| Feb 20 | Exp058: GPU Track 1c — 4 new WGSL shaders, 27/27 GPU checks |
| Feb 20 | Exp059: 25-domain benchmark — 22.5× Rust over Python |
| Feb 20 | Exp060: metalForge cross-substrate — 20/20 CPU↔GPU parity |
| Feb 20 | Exp061/062: RF + GBM inference — 29/29 CPU checks (domains 24-25) |
| Feb 20 | Exp063: GPU RF batch inference — 13/13 GPU checks (SoA WGSL shader) |
| Feb 21 | Phase 15: Code quality hardening — pedantic clippy, tolerance centralization, provenance headers |
| Feb 21 | 97% bio+io, bench/mod 97.5% coverage (56% overall), 650 tests (587 lib + 50 integration + 13 doc), 0 clippy warnings |
| Feb 21 | All inline tolerance literals → named constants in `tolerances.rs` (now 43, including 4 Jacobi constants) |
| Feb 21 | All data paths → `validation::data_dir()` for capability-based discovery |
| Feb 21 | Phase 17: metalForge absorption engineering — shaped all modules for ToadStool readiness |
| Feb 21 | `bio::special` consolidated (erf, ln_gamma, regularized_gamma) for extraction to `barracuda::math` |
| Feb 21 | PRIMITIVE_MAP updated with absorption readiness gaps and shared math extraction plan |
| Feb 21 | `bio::special` extraction documented in EVOLUTION_READINESS.md |
| Feb 21 | Exp064: GPU Parity v1 — 26/26 checks across 8 consolidated GPU domains |
| Feb 21 | Exp065: metalForge Full Cross-System — 35/35 substrate-independence proof |
| Feb 21 | Exp066: CPU vs GPU scaling benchmark — crossover characterization all domains |
| Feb 21 | Exp067: Dispatch overhead profiling — measured fixed GPU dispatch cost per domain |
| Feb 21 | Exp068: Pipeline caching — 38% dispatch overhead reduction (6 local WGSL modules) |
| Feb 21 | Exp069: Python → Rust CPU → GPU three-tier — full value chain proven |
| Feb 21 | PCIe topology documented: RTX 4070 + Titan V + AKD1000, P2P DMA paths |
| Feb 21 | Exp072: GPU streaming pipeline — pre-warmed FMR, 1.27x speedup over individual dispatch |
| Feb 21 | Exp073: Dispatch overhead quantified — streaming beats individual at all batch sizes |
| Feb 21 | Exp074: Substrate router — GPU↔NPU↔CPU routing, PCIe topology-aware, fallback proven |
| Feb 21 | Exp075: Pure GPU 5-stage pipeline — diversity→Bray-Curtis→PCoA→stats→spectral, 0.1% overhead |
| Feb 21 | Exp076: Cross-substrate pipeline — GPU→NPU→CPU data flow, 12 samples, latency profiled |
| Feb 22 | **Rewire: 8 bio WGSL shaders absorbed by ToadStool 31d/31g, wetSpring rewired to `barracuda::ops::bio::*`** |
| Feb 22 | Deleted 8 local shaders (25 KB): hmm, ani, snp, dnds, pangenome, quality, dada2, rf |
| Feb 22 | Fixed ToadStool SNP binding layout (is_variant read_only→read_write, removed phantom binding 6) |
| Feb 22 | Fixed wetSpring `GpuF64::new()` — pass real `AdapterInfo` to `WgpuDevice::from_existing()` for correct f64 polyfill detection |
| Feb 22 | Re-validated: 633 lib tests, 451 GPU checks, 0 clippy warnings, clean docs |
| Feb 22 | Exp078: ODE GPU sweep readiness — flat param APIs for 5 ODE modules (10 new tests) |
| Feb 22 | Tier promotions: multi_signal B→A, phage_defense B→A, cooperation C→B |
| Feb 22 | Benchmarked: CPU vs GPU scaling, dispatch overhead, phylo+HMM, streaming pipeline (all 23 ToadStool primitives) |
| Feb 22 | Exp079: BarraCuda CPU v6 — 48 checks proving flat API preserves bitwise ODE math (6 modules) |
| Feb 22 | Exp080: metalForge dispatch routing — 35 checks across 5 substrate configs (forge crate) |
| Feb 22 | Exp081: kmer GPU histogram (4^k flat buffer + sorted pairs) — promoted B→A |
| Feb 22 | Exp082: unifrac CSR flat tree + sample matrix — promoted B→A |
| Feb 22 | Exp083: taxonomy int8 quantization for NPU FC dispatch — promoted B→A/NPU |
| Feb 22 | Tier B → 2 remaining (cooperation B, no others). 7 modules now Tier A. |
| Feb 22 | Exp085: Tier A layout fidelity — kmer histogram/sorted-pairs, unifrac CSR, taxonomy int8 round-trips |
| Feb 22 | Exp086: metalForge pipeline proof — 5-stage dispatch + parity across 4 hardware configs |
| Feb 22 | Exp087: GPU Extended Domains — EIC, PCoA, Kriging, Rarefaction (--features gpu) |
| Feb 22 | Exp088: metalForge PCIe Direct Transfer — 32 checks, 6 paths + buffer contracts |
| Feb 22 | Exp089: ToadStool Streaming Dispatch — 25 checks, 5 patterns + determinism |
| Feb 22 | Exp090: Pure GPU Streaming Pipeline — 80 checks, 4 modes (RT/stream/parity/scaling) |
| Feb 22 | Exp091: Streaming vs Round-Trip Benchmark — quantifies 441-837× streaming advantage |
| Feb 22 | Exp092: CPU vs GPU All 16 Domains Head-to-Head — 48/48 checks across 16 domains |
| Feb 22 | Exp093: metalForge Full Cross-Substrate v3 — 28/28 checks, 16 domains substrate-independent |
| Feb 22 | WGSL Write phase: kmer_histogram_f64, unifrac_propagate_f64, taxonomy_fc_f64 shaders |
| Feb 22 | ABSORPTION_MANIFEST.md created — tracking Write → Absorb → Lean cycle |
| Feb 22 | metalForge forge crate v0.2.0 — streaming dispatch module, CpuCompute capability |
| Feb 22 | ToadStool review (session 39): 5 new bio primitives available (LocusVariance, PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness) |
| Feb 22 | ODE blocker updated: `enable f64;` cleared but `compile_shader` vs `compile_shader_f64` bug in upstream `batched_ode_rk4.rs:209` |
| Feb 22 | Revalidated: 654 lib tests, clippy clean, GPU feature compiles against upstream HEAD (d45fdfb3) |
| Feb 22 | **Exp094: Cross-Spring Evolution Validation** — 39/39 checks validating 5 neuralSpring primitives (PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance) wired and consumed by wetSpring |
| Feb 22 | **Exp095: Cross-Spring Scaling Benchmark** — 7 benchmarks across 5 neuralSpring primitives, 6.5×–277× GPU speedup at realistic bio problem sizes |
| Feb 22 | **Exp096: Local WGSL Compile + Dispatch** — 10/10 checks validating local WGSL compile and dispatch path |
| Feb 22 | SNP GPU gracefully skips on upstream wgpu binding mismatch (ToadStool snp shader); wrapped with catch_unwind |
| Feb 22 | 5 new bio module GPU wrappers: hamming_gpu, jaccard_gpu, spatial_payoff_gpu, batch_fitness_gpu, locus_variance_gpu |
| Feb 22 | Absorbed count: 24 primitives (19 wetSpring + 5 neuralSpring) |
| Feb 22 | **Phase 15+ structural audit**: `taxonomy.rs` → `taxonomy/` (types, kmers, classifier), `unifrac.rs` → `unifrac/` (tree, flat_tree, distance), adapter logic → `adapter.rs` |
| Feb 22 | PCoA Jacobi constants centralized in `tolerances.rs` (4 new: convergence, element_skip, tau_overflow, sweep_multiplier — Golub & Van Loan provenance) |
| Feb 22 | DADA2 constants documented with provenance (Callahan et al. 2016, R package defaults) |
| Feb 22 | All 40 Python baselines now carry SPDX-License-Identifier: AGPL-3.0-or-later |
| Feb 22 | Doc link fixes (`kmer.rs` `to_histogram`/`to_sorted_pairs` → `Self::*`), `cargo doc` clean |
| Feb 22 | Revalidated: 664 lib tests (10 new from adapter module), clippy pedantic clean, fmt clean, doc clean |
| Feb 22 | Taxonomy classifier: removed `#[allow(dead_code)]` — exposed `taxon_priors()` and `n_kmers_total()` as public accessors |
| Feb 22 | CI hardened: `RUSTDOCFLAGS="-D warnings"`, clippy `-D pedantic -D nursery`, json feature build check |
| Feb 22 | Cargo.lock updated to latest compatible transitive deps (syn, wasm-bindgen, bumpalo, js-sys) |
| Feb 22 | Fixed 6 `clippy::nursery` lints: needless_collect → `.count()`/`.len()`, single-item `into_iter` → explicit `HashMap::insert` |
| Feb 22 | Full audit: all files under 1000 lines, all clone() justified, all `#[allow(dead_code)]` in bins justified, 0 unnecessary deps |
| Feb 22 | **Edition 2024 upgrade**: `edition = "2024"`, MSRV 1.85. Import reordering, `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions |
| Feb 22 | `#![forbid(unsafe_code)]` → `#![deny(unsafe_code)]` for edition 2024 `env::set_var`/`remove_var` safety (test-only `unsafe`) |
| Feb 22 | Taxonomy classifier: `taxon_priors()` and `n_kmers_total()` accessors replace `#[allow(dead_code)]` |
| Feb 22 | CI: `RUSTDOCFLAGS="-D warnings"`, clippy `-D pedantic -D nursery`, json feature build check |
| Feb 22 | Lockfile updated (25 transitive deps), 6 clippy::nursery fixes, 3 midpoint + 2 const fn lint fixes |
| Feb 22 | All 40 Python baselines: SPDX + Date provenance headers (34 newly stamped from git creation dates) |
| Feb 22 | 4 fuzz targets verified (FASTQ, MS2, mzML, XML), error module reviewed (9-variant sovereign type), API surface audited |
| Feb 22 | **Phase 27: Write phase** — 3 new local WGSL ODE shaders created (phage_defense, bistable, multi_signal) |
| Feb 22 | **Exp099: CPU vs GPU Expanded** — kmer_gpu, unifrac_gpu, phage_defense_gpu wrappers + metalForge GPU→CPU→GPU pipeline |
| Feb 22 | **Exp100: metalForge Cross-Substrate v4** — 28/28 checks, 3 ODE domains exact parity, NPU routing, GPU→GPU→CPU pipeline |
| Feb 22 | 5 new GPU wrappers: `kmer_gpu`, `unifrac_gpu`, `bistable_gpu`, `multi_signal_gpu`, `phage_defense_gpu` |
| Feb 22 | GPU module count: 25 → 30. Local WGSL: 0 → 3. Absorbed ToadStool primitives: 32 → 30 (recounted after tier reclassification) |
| Mar 12 | **V114 Deep Audit:** 15 binaries received `required-features` gates fixing default build. 52 clippy pedantic/nursery warnings fixed. 4 validation binaries migrated from deprecated batch parsers to streaming APIs. Inline tolerances replaced with `tolerances::*`. VRAM estimate evolved from hardcoded 12 GB to capability-based `max_buffer_size` derivation. |

---

## Comparison with hotSpring Evolution

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Computational physics | Life science & analytical chemistry |
| CPU modules | 50+ (physics, lattice, MD, spectral) | 41 (bio, signal, ML) |
| GPU modules | 34 WGSL shaders | 42 modules (Lean phase) |
| Absorbed | complex64, SU(3), plaquette, HMC, CellList | SW, Gillespie, DT, Felsenstein, GEMM, HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF + 5 neuralSpring (PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance) |
| WGSL pattern | `pub const WGSL: &str` inline | `include_str!("../shaders/...")` |
| metalForge | GPU + NPU hardware characterization | GPU + NPU + cross-substrate validation |
| Handoffs | `../wateringHole/handoffs/` (36+ docs) | `ecoPrimals/archive/wetspring-early-handoffs-feb2026/` (v1-v9 fossil) |
| Tests | 454 | 1,288 lib + 219 integration |
| Validation | 418 checks | 8,300+ checks |
| Experiments | 31 suites | 260 experiments |
| Line coverage | — | 97% bio+io (55% overall) |
| Pipeline caching | Upstream (ToadStool native) | Local (Exp068, 38% overhead reduction) |
| Three-tier proof | CPU→GPU→NPU | Python→CPU→GPU→NPU (Exp069) |
| PCIe topology | Documented | Documented + P2P routing (PCIE_TOPOLOGY.md) |
| Streaming dispatch | — | Pre-warmed pipeline, 1.27× speedup (Exp072) |
| Dispatch overhead | — | Quantified at 4 batch sizes (Exp073) |
| Substrate routing | — | GPU↔NPU↔CPU router validated (Exp074) |
| Pure GPU pipeline | — | 5-stage, 0.1% overhead, 31 checks (Exp075) |
| Cross-substrate E2E | — | GPU→NPU→CPU, latency profiled (Exp076) |
| Pure GPU streaming | — | Zero CPU round-trips, 441-837× over round-trip (Exp090) |
| PCIe direct transfer | — | GPU→NPU without CPU staging (Exp088) |

| WGSL pattern | `pub const WGSL` inline | upstream for Lean |
| Write cycle | Active (physics ODEs) | Lean phase (all bio ODEs absorbed) |

Both Springs follow the same pipeline: **Python → Rust CPU → GPU → ToadStool absorption**.
hotSpring's `pub const WGSL` inline approach and wetSpring's upstream Lean
approach both work for absorption. wetSpring is in Lean phase (all absorbed);
hotSpring actively in Write phases for new domain-specific ODE shaders.
Convergent handoff patterns via `wateringHole/`.

---

## Unwired Upstream Primitives — Evolution Plan

Available ToadStool/BarraCuda primitives not yet consumed by wetSpring.
Each has a concrete use case; wiring is deferred until the domain requires it.

| Primitive | Session | Domain | wetSpring Use Case | Priority |
|-----------|---------|--------|--------------------|----------|
| `LogsumexpWgsl` | Early | GPU log-sum-exp | HMM forward pass stability (currently stable without it) | Low |
| `SparseGemmF64` | S60 | Sparse matrix | Track 3 drug repurposing NMF (CSR × dense) | Medium |
| `TranseScoreF64` | S60 | KG embedding | Track 3 knowledge graph drug-disease scoring | Medium |
| `TopK` | S60 | GPU sorting | Track 3 drug-disease pair ranking | Medium |
| `BandwidthTier` | S62 | PCIe routing | metalForge PCIe-aware cross-substrate dispatch | Low |
| `ComputeDispatch` | S65 | Pipeline builder | Eliminate remaining BGL/pipeline boilerplate in GPU modules | Medium |
| `ValidationHarness` | S59 | Test framework | Richer check API (check_abs/check_rel/check_upper) — local `Validator` preferred for now | Low |

**Track 3 blocking:** `SparseGemmF64`, `TranseScoreF64`, and `TopK` are the next
wiring targets when the drug repurposing sub-thesis advances to GPU promotion.

---

## External Dependency Evolution — Pure Rust Audit

All production dependencies are pure Rust or have pure Rust backends.

| Dependency | Version | Pure Rust? | Notes |
|------------|---------|:----------:|-------|
| `barracuda` | path (v0.3.5) | **Yes** | Standalone math primal, zero FFI |
| `flate2` | 1.0 | **Yes** | `rust_backend` feature → miniz_oxide (no C zlib) |
| `bytemuck` | 1 | **Yes** | Zero-copy GPU buffer casting |
| `serde` | 1 (optional) | **Yes** | Derive only |
| `serde_json` | 1 (optional) | **Yes** | JSON for viz/config |
| `wgpu` | 28 (optional, gpu) | **Mostly** | Pure Rust WebGPU; `renderdoc-sys` C dep pulled on native via `wgpu-hal` (debug instrumentation only, not required) |
| `tokio` | 1 (optional, gpu) | **Yes** | Async runtime for GPU device init |
| `chacha20poly1305` | 0.10 (optional, vault) | **Yes** | RustCrypto AEAD |
| `ed25519-dalek` | 2.2 (optional, vault) | **Yes** | Dalek Ed25519 signing |
| `blake3` | 1.8 (optional, vault) | **Yes** | BLAKE3 hashing (pure Rust, default-features=false + features=["pure"]) |
| `akida-driver` | path (optional, npu) | **Yes** | BrainChip AKD1000 via ioctl |
| `bingocube-nautilus` | path (optional, nautilus) | **Yes** | Evolutionary reservoir computing |

### Sovereign I/O (zero external parsers)

| Format | Implementation | External dep? |
|--------|---------------|:-------------:|
| XML (mzML) | `io::xml` sovereign pull parser | **No** |
| Base64 | `encoding.rs` sovereign decoder | **No** |
| FASTQ | `io::fastq` streaming parser | **No** |
| MS2 | `io::ms2` streaming parser | **No** |
| Newick | `bio::newick` parser | **No** |
| JSON (minimal) | `ncbi_data` bracket-depth splitter | **No** (full JSON via optional `serde_json`) |

### Evolution target: `renderdoc-sys`

The only transitive C dependency is `renderdoc-sys` via `wgpu-hal`. This provides
RenderDoc GPU debugging instrumentation. It does not affect runtime correctness
and is only linked on native targets. Evolution path:
1. **Current:** Accept as debug-only transitive dep (does not affect ecoBin compliance for CPU builds)
2. **Future:** When wgpu makes renderdoc optional, disable it via feature flags
3. **Sovereign:** coralReef sovereign GPU compiler (Level 3+) eliminates wgpu entirely

### ecoBin compliance

CPU builds (`default-features = false`) have **zero C dependencies**: pure Rust
throughout. GPU builds carry `renderdoc-sys` only; all computation is pure Rust/WGSL.
