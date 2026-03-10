# Absorption Manifest: wetSpring → ToadStool/BarraCuda

**Date:** March 9, 2026 (V105)
**Pattern:** Write → Absorb → Lean (adopted from hotSpring)
**barraCuda:** standalone v0.3.3 `a898dee`
**Status:** 150+ primitives consumed (264 ComputeDispatch ops) via `compile_shader_universal`, 0 local WGSL (fully lean), 0 local ODE derivative math, 0 local regression math, 5 GPU ODE via trait-generated WGSL, 47 GPU modules (all lean), 0 Tier B/C, 0 Passthrough, 1,288 lib + 219 integration tests, standalone barraCuda v0.3.3, 334 experiments, 9,060+ checks, 316 binaries, 179 named tolerances, clippy pedantic CLEAN (`--all-features`). **V105:** petalTongue visualization evolution — 9 DataChannel types, 33 scenario builders, StreamSession, Songbird capabilities, IPC science→viz wiring, Exp333-334 (78/78 PASS). V100: 173/173 PASS.

---

## Methodology: Write → Absorb → Lean

wetSpring follows hotSpring's proven absorption cycle:

```
Write → Validate → Hand off → Absorb → Lean
─────────────────────────────────────────────
Implement     Test against    Document in    ToadStool adds    Rewire to
on CPU +      Python +        wateringHole/  shaders/ops       upstream,
WGSL          known physics   handoffs/                        delete local
```

### Principles

1. **Biome model** — Springs don't import each other. wetSpring, hotSpring,
   neuralSpring each lean on ToadStool/BarraCuda independently. ToadStool
   absorbs what works; all Springs benefit.

2. **Shaders in `.wgsl` files** — `include_str!` for large, `pub const` for small.
   No opaque blobs. Every shader has documented binding layouts and dispatch geometry.

3. **CPU reference first** — Every GPU/NPU path has a validated CPU baseline
   (Python → Rust CPU → Rust GPU → streaming).

4. **Handoffs via wateringHole** — Structured documents with code locations,
   binding indices, workgroup sizes, test results, and tolerance rationale.

5. **Backward-compatible lean** — Re-exports and type aliases (not breaking changes)
   when switching from local to upstream.

6. **Named tolerances** — Central constants in `tolerances.rs`, not magic numbers.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Write | Local WGSL ODE shaders | **5 shaders deleted** — GPU modules use `generate_shader()` from `OdeSystem` trait impls (`bio/ode_systems.rs`) |
| Compose | GPU wrappers wiring ToadStool primitives | **7 modules** (kmd, merge_pairs, RF, derep, NJ, reconciliation, molecular_clock) |
| Passthrough | Accept GPU buffers, CPU kernel | **0 modules** — all 3 former Passthrough promoted (V40) |
| Validate | CPU ↔ GPU parity for all shaders | All 5 ODE: exact parity (Exp099/100/101) |
| Hand off | wateringHole/handoffs/ documents | **V84** active (supersedes V83), V7-V82 archived |
| Absorb | ToadStool integrates as `ops::bio::*` | **150+ primitives** consumed (ToadStool S66: all DONE, +46 cross-spring total) |
| Lean | Rewire to upstream, delete local code | 150+ primitives consumed (S66), 5 `OdeSystem` trait rewires, BGL boilerplate removed, 0 Passthrough |

---

## Absorbed (Lean Phase — 24 primitives)

These modules consume upstream ToadStool/BarraCuda primitives.
Local WGSL deleted; wetSpring imports from `barracuda::*`.

| wetSpring Module / Primitive | Upstream Primitive | Absorbed Date | Exp |
|------------------|-------------------|---------------|-----|
| `alignment` | `SmithWatermanGpu` | Feb 20 | 044 |
| `ani_gpu` | `AniBatchF64` | Feb 22 | 058 |
| `dada2_gpu` | `Dada2EStepGpu` | Feb 22 | — |
| `decision_tree` | `TreeInferenceGpu` | Feb 20 | 044 |
| `diversity_gpu` | `FusedMapReduceF64`, `BrayCurtisF64` | Feb 22 | 016 |
| `dnds_gpu` | `DnDsBatchF64` | Feb 22 | 058 |
| `eic_gpu` | `FusedMapReduceF64`, `WeightedDotF64` | Feb 22 | 087 |
| `felsenstein` | `FelsensteinGpu` | Feb 20 | 046 |
| `gemm_cached` | `GemmF64::WGSL` | Feb 20 | 016 |
| `gillespie` | `GillespieGpu` | Feb 20 | 044 |
| `hmm_gpu` | `HmmBatchForwardF64` | Feb 22 | 047 |
| `kriging` | `KrigingF64` (via GEMM) | Feb 22 | 087 |
| `pangenome_gpu` | `PangenomeClassifyGpu` | Feb 22 | 058 |
| `pcoa_gpu` | `BatchedEighGpu` | Feb 22 | 087 |
| `quality_gpu` | `QualityFilterGpu` | Feb 22 | — |
| `random_forest_gpu` | `RfBatchInferenceGpu` | Feb 22 | 063 |
| `rarefaction_gpu` | `FusedMapReduceF64`, `PrngXoshiro` | Feb 22 | 087 |
| `snp_gpu` | `SnpCallingF64` | Feb 22 | 058 |
| `spectral_match_gpu` | `FusedMapReduceF64`, `GemmF64` | Feb 22 | 016 |
| (cross-spring) | `PairwiseHammingGpu` | Feb 22 | 094 |
| (cross-spring) | `PairwiseJaccardGpu` | Feb 22 | 094 |
| (cross-spring) | `SpatialPayoffGpu` | Feb 22 | 094 |
| (cross-spring) | `BatchFitnessGpu` | Feb 22 | 094 |
| (cross-spring) | `LocusVarianceGpu` | Feb 22 | 094 |

---

## Local WGSL Shaders (Write Phase — LEAN COMPLETE)

All 5 ODE shaders have been **deleted** (30,424 bytes). GPU modules now use
`BatchedOdeRK4::<S>::generate_shader()` at runtime, producing WGSL from the
`OdeSystem` trait implementations in `bio/ode_systems.rs`.

| System | Struct | Vars | Params | WGSL Lines | CPU Parity |
|--------|--------|:----:|:------:|:----------:|:----------:|
| Phage Defense | `PhageDefenseOde` | 4 | 11 | 142 | Derivative-level exact |
| Bistable | `BistableOde` | 5 | 21 | 169 | Exact (0.00) |
| Multi-Signal | `MultiSignalOde` | 7 | 24 | 199 | Exact (4.44e-16) |
| Cooperation | `CooperationOde` | 4 | 13 | 148 | Exact (4.44e-16) |
| Capacitor | `CapacitorOde` | 6 | 16 | 170 | Exact (0.00) |

### GPU Module Pattern (post-lean)

```rust
let wgsl = BatchedOdeRK4::<XxxOde>::generate_shader();
let module = device.compile_shader_f64(&wgsl, Some("Xxx ODE"));
// Pipeline setup unchanged, dispatch_workgroups(n.div_ceil(64), 1, 1)
```

---

## Compose Phase (7 GPU Wrappers)

GPU wrappers that wire existing ToadStool primitives for GPU-accelerated
workflows. No local WGSL needed — these compose upstream ops.

| Module | ToadStool Primitive | Strategy | Exp |
|--------|-------------------|----------|-----|
| `kmd_gpu` | `KmerHistogramGpu` | Kendrick mass defect via k-mer histogram | 101 |
| `merge_pairs_gpu` | `PairwiseHammingGpu` | Overlap scoring via Hamming distance | 101 |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Bipartition distance via Hamming | 101 |
| `derep_gpu` | `KmerHistogramGpu` | Sequence hashing via k-mer histogram | 101 |
| `neighbor_joining_gpu` | `GemmCachedF64` | Distance matrix operations | 101 |
| `reconciliation_gpu` | `FusedMapReduceF64` | DTL cost aggregation (CPU DP core; blocked on `BatchReconcileGpu`) | 101 |
| `molecular_clock_gpu` | `GemmCachedF64` | Rate matrix operations | 101 |

---

## Passthrough Phase — ALL PROMOTED (V40)

All 3 former Passthrough modules have been promoted. Zero Passthrough remains.

| Module | Was | Now | How |
|--------|-----|-----|-----|
| `gbm_gpu` | Passthrough (sequential boosting) | ✅ Compose (`TreeInferenceGpu`) | Pure GPU batch inference (promoted Exp101) |
| `feature_table_gpu` | Passthrough (feature extraction) | ✅ Compose (`FMR` + `WeightedDotF64`) | Chains eic_gpu + signal_gpu (promoted Exp101) |
| `signal_gpu` | Passthrough (CPU peaks) | ✅ Lean (`PeakDetectF64` S62) | Rewired to upstream GPU peak detection |

---

## Tier B/C — All Promoted (Phase 28)

All 13 former Tier B/C modules have been promoted to GPU-capable:

| Former Tier | Module | Promoted To | Notes |
|-------------|--------|-------------|-------|
| B | `cooperation` | Write (local WGSL 4v/13p) | ODE RK4 f64 shader |
| C | `capacitor` | Write (local WGSL 6v/16p) | ODE RK4 f64 shader |
| C | `kmd` | Compose (`KmerHistogramGpu`) | Kendrick mass defect |
| C | `gbm` | Passthrough | Sequential boosting |
| C | `merge_pairs` | Compose (`PairwiseHammingGpu`) | Overlap merging |
| C | `signal` | Passthrough | Peak detection |
| C | `feature_table` | Passthrough | Feature extraction |
| C | `robinson_foulds` | Compose (`PairwiseHammingGpu`) | Tree distance |
| C | `derep` | Compose (`KmerHistogramGpu`) | Dereplication |
| C | `chimera` | Compose (`GemmCachedF64`) | Chimera scoring (upgraded) |
| C | `neighbor_joining` | Compose (`GemmCachedF64`) | NJ tree construction |
| C | `reconciliation` | Compose (`TreeInferenceGpu`) | DTL reconciliation |
| C | `molecular_clock` | Compose (`GemmCachedF64`) | Molecular clock |

---

## CPU Math Extraction — COMPLETE

All local math has been delegated to barracuda upstream:

| Local Function | File | Upstream Target | Status |
|----------------|------|-----------------|--------|
| `erf()` | `special.rs` | `barracuda::special::erf` | ✅ Delegated |
| `ln_gamma()` | `special.rs` | `barracuda::special::ln_gamma` | ✅ Delegated |
| `regularized_gamma_lower()` | `special.rs` | `barracuda::special::regularized_gamma_p` | ✅ Delegated |
| `normal_cdf()` | `special.rs` | `barracuda::stats::norm_cdf` | ✅ Delegated (V43) |
| `dot()`, `l2_norm()` | `special.rs` | `barracuda::stats::{dot, l2_norm}` | ✅ Delegated (S64) |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` | ✅ Delegated |
| `solve_ridge()` | `bio/esn.rs` | `barracuda::linalg::ridge_regression` | ✅ Delegated |

### ODE Derivative Delegation (V50)

5 ODE system RHS functions replaced with `barracuda::numerical::ode_bio::*Ode::cpu_derivative`:

| System | wetSpring File | barracuda Primitive | Guard |
|--------|---------------|-------------------|-------|
| Capacitor | `bio/capacitor.rs` | `CapacitorOde::cpu_derivative` | None |
| Cooperation | `bio/cooperation.rs` | `CooperationOde::cpu_derivative` | None |
| Multi-Signal | `bio/multi_signal.rs` | `MultiSignalOde::cpu_derivative` | c-di-GMP convergence |
| Bistable | `bio/bistable.rs` | `BistableOde::cpu_derivative` | c-di-GMP convergence |
| Phage Defense | `bio/phage_defense.rs` | `PhageDefenseOde::cpu_derivative` | None |

Local helpers (`hill()`, `hill_repress()`, `monod()`) and full RHS functions
removed (~200 lines). wetSpring retains:
- `rk4_integrate` (trajectory storage + clamping — not in barracuda's batched API)
- `OdeResult` / `steady_state_mean` (trajectory analysis)
- c-di-GMP convergence guard (thin wrapper for fixed-step RK4 stability)
- `QsBiofilm` base model (not absorbed — monostable variant stays local)
- All param structs (ergonomic named fields, `Default` impls, domain docs)

**Zero duplicate derivative math remains in the codebase.**

---

## metalForge Forge Crate

The `metalForge/forge/` crate (`wetspring-forge` v0.3.0, 113 tests) provides:

| Module | Purpose | Absorption Path |
|--------|---------|-----------------|
| `probe` | GPU (wgpu) + CPU (/proc) + NPU (/dev) discovery | `barracuda::device::discovery` |
| `inventory` | Unified substrate inventory | `barracuda::device::inventory` |
| `dispatch` | Capability-based workload routing | `barracuda::dispatch` |
| `bridge` | forge ↔ barracuda device bridge | Integration seam |

**Bridge docstring:** "When ToadStool absorbs forge, this bridge becomes the
integration point — substrate discovery feeds directly into device creation."

---

## New Upstream Primitives Available (ToadStool S54-S57)

### S54 Primitives (neuralSpring baseCamp → ToadStool)

| Primitive | Module | Tests | Potential wetSpring Use |
|-----------|--------|:-----:|------------------------|
| `graph_laplacian` | `barracuda::linalg` | 3 | Community network analysis |
| `effective_rank` | `barracuda::linalg` | 3 | Spectral diagnostics |
| `numerical_hessian` | `barracuda::numerical` | 3 | Optimization landscape |

5 new WGSL shaders: `symmetrize.wgsl`, `laplacian.wgsl` (linalg),
`hessian_column.wgsl` (numerical), `histogram.wgsl` (stats),
`metropolis.wgsl` (sample).

GPU fixes from airSpring also landed: `pow_f64` fractional exponent,
`acos_simple` → `acos_f64`, `FusedMapReduceF64` buffer conflict resolution.

### S56 Primitives (neuralSpring → ToadStool)

| Primitive | Module | Tests | Potential wetSpring Use |
|-----------|--------|:-----:|------------------------|
| `belief_propagation_chain` | `barracuda::linalg::graph` | 3 | Chain PGM forward pass |
| `boltzmann_sampling` | `barracuda::sample::metropolis` | 3 | CPU MCMC sampling |
| `disordered_laplacian` | `barracuda::linalg::graph` | 3 | Anderson disorder — **directly relevant to QS-disorder coupling** |

### S57 Status

4,224 core tests, 650+ WGSL shaders, 46 cross-spring absorption items
complete. All root docs synced. 222 lines commented-out code removed.

### wetSpring Consumption of S54-S56 Primitives (Exp162)

| Primitive | Module | Session | wetSpring Use | Exp |
|-----------|--------|---------|---------------|-----|
| `graph_laplacian` | `barracuda::linalg` | S54 | Community network spectral analysis | 162 |
| `effective_rank` | `barracuda::linalg` | S54 | Diversity matrix diagnostics | 162 |
| `numerical_hessian` | `barracuda::numerical` | S54 | ML model curvature analysis | 162 |
| `disordered_laplacian` | `barracuda::linalg` | S56 | QS-disorder coupling on community graphs | 162 |
| `belief_propagation_chain` | `barracuda::linalg` | S56 | Hierarchical taxonomy classification | 162 |
| `boltzmann_sampling` | `barracuda::sample` | S56 | MCMC parameter optimization | 162 |

### S60-S62+DF64 Primitives (V40 Catch-Up)

ToadStool S60-S62+DF64 delivered major infrastructure and closed all
wetSpring P0-P3 evolution requests except diversity_fusion absorption.

| Primitive | Module | Session | wetSpring Status | Notes |
|-----------|--------|---------|------------------|-------|
| `PeakDetectF64` | `barracuda::ops::peak_detect_f64` | S62 | ✅ Lean — `signal_gpu` rewired | Closes P1-2 (f32→f64 fix) |
| `ComputeDispatch` builder | `barracuda::device::compute_pipeline` | S62+DF64 | Available — adoption candidate | Closes P1-3 (cached-pipeline) |
| `SparseGemmF64` | `barracuda::ops::sparse_gemm_f64` | S60 | Available — Track 3 drug repurposing | CSR × dense f64 GEMM |
| `TranseScoreF64` | `barracuda::ops::transe_score_f64` | S60 | Available — Track 3 KG scoring | GPU TransE embedding |
| `TopK` | `barracuda::ops::topk` | S60 | Available — Track 3 drug ranking | Closes P3-7 (GPU Top-K) |
| `BandwidthTier` | `barracuda::unified_hardware::types` | S62 | Available — metalForge PCIe routing | PCIe-aware dispatch |
| `Fp64Strategy` | `barracuda::device::driver_profile` | DF64 | Available — DF64 auto-selection | Native/Hybrid per GPU |
| DF64 GEMM | `barracuda::shaders::linalg::gemm_df64` | DF64 | Available — RTX 4070 FP32-core GEMM | ~10× throughput for compute-dominant loops |

All P0-P3 request resolution documented in V40 handoff.

---

## Cross-Spring Contributions

Patterns from hotSpring and neuralSpring that wetSpring leans on:

| Contribution | Source | wetSpring Usage |
|-------------|--------|----------------|
| `ShaderTemplate::for_driver_auto` | hotSpring NVK workaround | All f64 WGSL compilation |
| `GpuDriverProfile` | hotSpring device capabilities | Driver-specific polyfill selection |
| `ReduceScalarPipeline` | hotSpring pipeline feedback | FMR dispatch optimization |
| `BatchedEighGpu` (NAK-optimized) | hotSpring eigensolve | PCoA ordination |
| `(zero + literal)` f64 pattern | neuralSpring naga fix | All f64 shader constants |
| `GemmCached` 60× speedup | wetSpring → absorbed | hotSpring HFB uses this |

---

## Validation Summary

| Category | Checks | Status |
|----------|:------:|--------|
| CPU parity (Python → Rust) | 1,476 | ALL PASS |
| GPU parity (CPU → GPU) | 1,578+ | ALL PASS |
| BarraCuda CPU parity (v1-v8) | 380/380 | ALL PASS |
| Streaming dispatch | 80 | ALL PASS |
| Layout fidelity | 35 | ALL PASS |
| Transfer/streaming | 57 | ALL PASS |
| Finite-size scaling (Exp150) | 14 | ALL PASS |
| Correlated disorder (Exp151) | 8 | ALL PASS |
| ODE lean benchmark | 11 | ALL PASS |
| Paper queue extensions (Exp152-156) | 42 | ALL PASS |
| Drug repurposing track (Exp157-161) | 40 | ALL PASS |
| **Total** | **3,132+** | **ALL PASS** |

**Phase 39 additions (Feb 24, 2026):**
- Exp150: W_c = 16.26 (disorder-averaged, L=6–12, 8 realizations)
- Exp151: Correlated disorder pushes W_c > 28 (biofilm clustering facilitates QS)
- Exp152: Physical communication pathways — 8 modes, 6/8 Anderson-susceptible (9 checks)
- Exp153: Nitrifying QS — R:P = 2.3:1 matches eavesdropper prediction (12 checks)
- Exp154: Marine interkingdom QS — 10/10 Anderson predictions correct (6 checks)
- Exp155: Myxococcus — critical density + NP geometry bootstrap (7 checks)
- Exp156: Dictyostelium — non-Hermitian relay defeats localization (8 checks)
- Exp157-161: Drug repurposing track — NMF, pathway scoring, KG embedding (40 checks)
  - New local module: `bio::nmf` (Lee & Seung multiplicative updates, cosine sim, top-K)
  - ToadStool absorption targets: NMF update shader, sparse GEMM, weighted NMF mask
- ODE lean benchmark: 5 systems × generate_shader(), upstream 20–33% faster

**V37 revalidation (Feb 25, 2026):**
- Revalidated against ToadStool `02207c4a` (S62+DF64 expansion)
- 806 tests pass (759 barracuda + 47 forge), 0 failures
- `cargo clippy --all-targets` clean (pedantic + nursery)
- 95.75% library coverage (llvm-cov)
- Deep debt cleanup: ncbi.rs sovereignty (capability-based HTTP), I/O parser lean
  (deprecated buffering removed), 56 binaries modernized (NaN-safe sorts, descriptive expects),
  tolerance provenance (5 constants with commit hashes), CI expanded (coverage gate + forge jobs)

**V40 ToadStool catch-up (Feb 25, 2026):**
- Reviewed ToadStool S39-S62+DF64 commit evolution (55+ commits since last handoff)
- Confirmed: 8/9 P0-P3 evolution requests delivered by ToadStool (V43: tolerance module confirmed)
  - P0-1: `GemmF64::wgsl_shader_for_device()` public + DF64 auto-select ✅
  - P1-2: `PeakDetectF64` f64 end-to-end (S62) ✅ — `signal_gpu` already leaned
  - P1-3: `ComputeDispatch` builder (S62+DF64) ✅
  - P1-4: `dot`/`l2_norm` as GPU ops (`NormReduceF64`, `FusedMapReduceF64`) ✅
  - P2-6: `BatchedOdeRK4` via `OdeSystem` trait + `generate_shader()` (S58) ✅
  - P3-7: `TopK` GPU primitive (S60) ✅
  - P3-8: `quantize_affine_i8` (S39) ✅
- P2-5 (diversity_fusion absorption) DONE (absorbed S63). P2-9 (tolerance pattern) confirmed DELIVERED (S52).
- Track 3 GPU fully unblocked: NMF, SpMM, TransE, cosine, Top-K all upstream
- 3 former Passthrough modules promoted (V40): signal_gpu → Lean, gbm_gpu → Compose, feature_table_gpu → Compose (0 Passthrough remaining)
- 5 new upstream primitives available: SparseGemmF64, TranseScoreF64, BandwidthTier, Fp64Strategy, DF64 GEMM
- All tests pass (806), clippy clean, docs clean against ToadStool HEAD (02207c4a)

**V44 complete cross-spring rewire (Feb 25, 2026):**
- Reviewed ToadStool ABSORPTION_TRACKER: all 46 wetSpring V16-V22 items DONE
- ToadStool S42-S62+DF64: massive absorption cycle — 650+ shaders, 4,500+ tests
- `normal_cdf` rewired: `special::normal_cdf` → `barracuda::stats::norm_cdf` (50th primitive)
- `ValidationHarness` (S59) available but not consumed — wetSpring keeps local `Validator`
  with simpler API (check/check_count/check_pass/section) suited to Python-baseline pattern.
  ToadStool's `ValidationHarness` (check_abs/check_rel/check_upper/check_lower) is
  richer but would require rewiring 158 binaries for marginal benefit.
- `barracuda::tolerances` module (S52) confirmed delivered — closes P2-9.
  wetSpring's flat `tolerances.rs` (77 domain constants) is complementary.
- All items absorbed. 9/9 evolution requests DONE.
- 823 lib tests pass, 0 clippy warnings (pedantic+nursery), fmt clean

**V50 ODE derivative rewire (Feb 26, 2026):**
- 5 ODE RHS functions rewired to `barracuda::numerical::ode_bio::*Ode::cpu_derivative`
- ~200 lines local derivative math eliminated (hill, hill_repress, monod helpers + RHS bodies)
- c-di-GMP convergence guard preserved as thin wrapper for bistable + multi_signal
- `which_exists()` rewritten as pure Rust PATH scan (no subprocess)
- `interpret_output()` takes ownership (eliminates stdout clone)
- 4 new `try_load_json_array` error-path tests added
- 823 lib tests pass, 0 clippy warnings (pedantic+nursery), fmt clean, docs clean

**V83 Extended cross-spring rewire (Mar 1, 2026):**
- Exp248: BarraCuda CPU v18 — bootstrap_ci, rawr_mean, fit_exponential, fit_quadratic,
  fit_logarithmic, fit_all, cross-spring stats (36/36 checks)
- Exp249: Cross-Spring Evolution Benchmark S70+++ with provenance map (34/34 checks)
- Exp250: GPU v10 — StencilCooperationGpu, HillGateGpu dispatched; WrightFisher/Symmetrize/Laplacian
  findings for ToadStool S71 (12/12 checks)
- Fp64Strategy::Concurrent wired in gpu.rs
- 8 new primitives consumed: stats (bootstrap_ci, rawr_mean, fit_exponential, fit_quadratic,
  fit_logarithmic, fit_all), GPU bio (StencilCooperationGpu, HillGateGpu)
- 2 upstream findings filed for ToadStool S71
- Total: 144 ToadStool primitives consumed (85 prior + 8 V83)

**V82 ToadStool S70+++ rewire (Mar 1, 2026):**
- ToadStool pin advanced: S68+ (`e96576ee`) → S70+++ (`1dd7e338`)
- 13 commits, 324 files changed, 9,440 insertions in upstream barracuda crate
- No breaking changes — clean compile, 1,223 tests pass, 0 clippy warnings
- 3 new primitives consumed: `stats::evolution` (kimura_fixation_prob, error_threshold,
  detection_power, detection_threshold), `stats::jackknife` (jackknife_mean_variance,
  generalized jackknife), `stats::diversity::chao1_classic` (integer-count Chao 1984)
- Exp247: ToadStool S70+++ Rewire Validation — 42/42 checks PASS
- Available but not consumed: staging::pipeline::PipelineBuilder, Fp64Strategy::Concurrent,
  SymmetrizeGpu, LaplacianGpu, 6 new WGSL shaders (batched_elementwise, seasonal_pipeline,
  anderson_coupling, lanczos_iteration, linear_regression, matrix_correlation)
- Total: 85 ToadStool primitives consumed (82 prior + 3 S70)
