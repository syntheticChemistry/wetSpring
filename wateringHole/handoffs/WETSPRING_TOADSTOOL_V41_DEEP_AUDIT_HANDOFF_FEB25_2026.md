# wetSpring → ToadStool Handoff V41: Deep Audit + Evolution Handoff

**Date:** February 25, 2026
**From:** wetSpring (Phase 46, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**Supersedes:** V40 (ToadStool S39-S62+DF64 catch-up)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring completed a comprehensive deep audit of the entire codebase,
documentation, and validation surface. This handoff summarizes the audit
findings, documents our full barracuda usage map, and provides actionable
evolution recommendations for the ToadStool/BarraCuda team.

**Post-V41 numbers:** 918 tests (871 barracuda + 47 forge), 96.48%
library coverage, **49** ToadStool primitives + 2 BGL helpers consumed,
0 Passthrough modules, 70 named tolerance constants, 0 clippy warnings,
`#![deny(missing_docs)]` enforced.

---

## Part 1: Audit Findings — What's Clean

wetSpring passes all quality gates with zero issues:

| Gate | Status |
|------|--------|
| `cargo fmt --check` | 0 diffs |
| `cargo clippy --all-targets` | 0 warnings (pedantic + nursery) |
| `cargo test` | 871 passed (barracuda), 0 failed, 1 ignored |
| `cargo doc --no-deps` | 0 warnings, 87 API pages |
| `cargo llvm-cov --lib` | 96.48% line coverage |
| `#![deny(unsafe_code)]` | Enforced crate-wide |
| `#![deny(clippy::expect_used, unwrap_used)]` | Enforced crate-wide |
| `#![deny(missing_docs)]` | Enforced crate-wide (escalated from `warn` in V41) |
| SPDX headers | All `.rs` and `.py` files |
| File size limit | All under 1000 LOC (max: 924) |
| Inline tolerance literals | 0 (all use `tolerances::` constants) |
| Production `unwrap()`/`expect()` | 0 in library code |
| Production mocks | 0 |
| TODO/FIXME markers | 0 |
| External C dependencies | 0 (`flate2` uses `rust_backend`) |

---

## Part 2: Complete Barracuda Usage Map

### CPU Primitives Used (always-on, `default-features = false`)

| Domain | Primitive | wetSpring Consumer | Notes |
|--------|-----------|-------------------|-------|
| Special | `barracuda::special::erf` | `special.rs` (delegates) | Error function |
| Special | `barracuda::special::ln_gamma` | `special.rs` (delegates) | Log-gamma |
| Special | `barracuda::special::regularized_gamma_p` | `special.rs` (delegates) | Incomplete gamma |
| Numerical | `barracuda::numerical::trapz` | `bio/eic.rs` | Peak integration |
| Numerical | `barracuda::numerical::numerical_hessian` | Cross-spring bins | Hessian for optimization |
| Numerical | `barracuda::numerical::ode_generic::BatchedOdeRK4` | ODE GPU modules | Batched RK4 ODE solver |
| Numerical | Bio ODE systems (5) | ODE GPU modules | Bistable, MultiSignal, PhageDefense, Cooperation, Capacitor |
| LinAlg | `barracuda::linalg::ridge_regression` | `bio/esn.rs` | ESN readout training |
| LinAlg | `barracuda::linalg::nmf` | Drug repurposing bins | NMF factorization |
| LinAlg | `barracuda::linalg::sparse::CsrMatrix` | GPU drug bins | Sparse NMF |
| LinAlg | `barracuda::linalg::graph_laplacian` | Cross-spring bins | Graph spectral |
| LinAlg | `barracuda::linalg::effective_rank` | Cross-spring bins | Diversity diagnostics |
| LinAlg | `barracuda::linalg::disordered_laplacian` | Cross-spring bins | Anderson disorder |
| LinAlg | `barracuda::linalg::belief_propagation_chain` | Cross-spring bins | Taxonomy |
| Spectral | Anderson Hamiltonians, Lanczos, level statistics | Cross-spring bins | Spectral theory |
| Sampling | `barracuda::sample::boltzmann_sampling` | Cross-spring bins | MCMC optimization |

### GPU Primitives Used (49 + 2 BGL helpers)

**Math Ops (12):**
`FusedMapReduceF64`, `BrayCurtisF64`, `GemmF64`, `BatchedEighGpu`,
`PeakDetectF64`, `SparseGemmF64`, `TranseScoreF64`, `KrigingF64`,
`WeightedDotF64`, `VarianceF64`, `CorrelationF64`, `CovarianceF64`

**Bio Ops (15):**
`UniFracPropagateGpu`, `RfBatchInferenceGpu`, `HmmBatchForwardF64`,
`Dada2EStepGpu`, `AniBatchF64`, `DnDsBatchF64`, `SnpCallingF64`,
`QualityFilterGpu`, `PangenomeClassifyGpu`, `KmerHistogramGpu`,
`PairwiseJaccardGpu`, `PairwiseHammingGpu`, `SpatialPayoffGpu`,
`LocusVarianceGpu`, `BatchFitnessGpu`

**Phylogenetic (6):**
`FelsensteinGpu`, `PhyloTree`, `TreeInferenceGpu`, `FlatForest`,
`GillespieGpu`, `SmithWatermanGpu`

**ODE (6):**
`BatchedOdeRK4<Bistable>`, `BatchedOdeRK4<MultiSignal>`,
`BatchedOdeRK4<PhageDefense>`, `BatchedOdeRK4<Cooperation>`,
`BatchedOdeRK4<Capacitor>` + `generate_shader()`

**Infrastructure (8+2):**
`WgpuDevice`, `TensorContext`, `GpuDriverProfile`, `WgslOpClass`,
`RarefactionGpu`, `TaxonomyFcGpu`, `GemmCached`, `NmfConfig` +
`storage_bgl_entry`, `uniform_bgl_entry`

### Local CPU Implementations (no barracuda equivalent used)

| Local Module | Functions | Duplication Risk |
|-------------|-----------|-----------------|
| `special::dot` | Dot product | **Low** — GPU has `FMR::dot()`, CPU helper is 1 line |
| `special::l2_norm` | L2 norm | **Low** — GPU has `NormReduceF64::l2()`, CPU helper is 1 line |
| `bio/diversity` | Shannon, Simpson, Chao1, etc. | **None** — CPU reference for GPU validation |
| `bio/pcoa` | Jacobi eigendecomposition | **None** — CPU reference for `BatchedEighGpu` |
| `bio/signal` | `find_peaks` | **None** — CPU reference for `PeakDetectF64` |
| All other `bio/*` | CPU implementations | **None** — intentional CPU/GPU split for validation |

**Verdict: Zero duplication.** All CPU modules exist as validation references
for their GPU counterparts. The `special::dot`/`l2_norm` helpers are thin
1-line functions used only in validation binaries.

---

## Part 3: Evolution Recommendations for ToadStool

### P0 — Remaining Requests (from V40)

| # | Request | Effort | Impact |
|---|---------|--------|--------|
| 1 | `diversity_fusion_f64.wgsl` absorption | Medium | Eliminates sole remaining local WGSL shader |
| 2 | `ComputeDispatch` migration guide | Low | Document migration path from manual BGL setup |

### P1 — New Recommendations from Deep Audit

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 1 | **`barracuda::special::{dot, l2_norm}` CPU exports** | wetSpring (and likely other springs) use thin local `dot`/`l2_norm` helpers. Exporting these from barracuda CPU math would eliminate the only local math helpers. |
| 2 | **`PhyloTree` unification** | wetSpring has `bio::unifrac::PhyloTree` (CPU) and barracuda has `PhyloTree` (GPU). Validation binaries convert between them. A shared `barracuda::PhyloTree` that both CPU and GPU code uses would simplify the interface. |
| 3 | **`Fp64Strategy` documentation** | Consumer GPU users (RTX 4070) need clear guidance on when `Hybrid` vs `Native` DF64 is selected. A doc page on `Fp64Strategy` auto-detection would help springs choose correctly. |

### P2 — Long-Term Evolution Observations

| # | Observation | Detail |
|---|------------|--------|
| 1 | **Per-domain GPU polyfill status** | wetSpring's `needs_f64_exp_log_workaround()` drives log/exp polyfill on Ada Lovelace. As ToadStool adds DF64 transcendentals, the polyfill should fade. Track which domains still need it. |
| 2 | **Batch ODE generic capacity** | 5 wetSpring ODE systems all lean on `BatchedOdeRK4<S>::generate_shader()`. This pattern works extremely well. Consider documenting it as the standard for new springs adding ODE domains. |
| 3 | **metalForge integration path** | wetSpring's `metalForge/forge/` crate does substrate discovery and routing. When ToadStool absorbs forge, the `bridge.rs` module becomes the integration point. Keep the bridge stable. |
| 4 | **ESN/NPU pipeline** | wetSpring has a working ESN → int8 → NPU pipeline for reservoir computing. If ToadStool adds NPU primitives, the wetSpring ESN module is a good test case. |

---

## Part 4: Dependency Health Report

wetSpring's dependency surface is minimal and sovereignty-compliant:

| Dependency | Version | Type | Notes |
|-----------|---------|------|-------|
| `barracuda` | 0.2.0 (path) | Always-on | CPU math, GPU ops |
| `bytemuck` | 1.25.0 | Direct | Pure Rust, zero-copy GPU buffers |
| `flate2` | 1.1.9 | Direct | Pure Rust (`rust_backend` = miniz_oxide) |
| `wgpu` | 22 | Optional (`gpu`) | WebGPU compute |
| `tokio` | 1 | Optional (`gpu`) | Async GPU init only |
| `serde_json` | 1 | Optional (`json`) | 2 binaries only |
| `tempfile` | 3.25.0 | Dev-only | Test temp dirs |

**Zero C dependencies.** No `openssl`, no `ring`, no HTTP/TLS crates.
HTTP transport uses capability-discovered system `curl`/`wget` — zero
compile-time coupling.

---

## Part 5: Coverage and Test Breakdown

### Library Coverage by Module Family

| Family | Coverage | Tests | Notes |
|--------|----------|-------|-------|
| bio/* (47 modules) | ~97% | ~550 | CPU reference implementations |
| io/* (FASTQ, mzML, MS2, XML) | 87-97% | ~80 | Streaming parsers |
| special, tolerances, validation | 99%+ | ~40 | Thin wrappers + constants |
| bench/power, bench/hardware | 71-81% | ~25 | Hardware-dependent code untestable |
| ncbi | 80% | ~30 | Network-dependent code untestable |
| encoding, error, phred | 99%+ | ~20 | Sovereign utilities |
| **Total** | **96.48%** | **871** | |

### Validation Binary Coverage

| Category | Binaries | Checks |
|----------|----------|--------|
| CPU validation | 92 | ~1,476 |
| GPU validation | 42 | ~702 |
| metalForge cross-substrate | 12 | ~300 |
| Benchmark | 11 | ~200 |
| Cross-spring | 8 | ~150 |
| NCBI-scale hypothesis | 6 | ~146 |
| NPU reservoir | 6 | ~59 |
| **Total** | **158** | **3,300+** |

---

## Part 6: Paper Queue Control Matrix

All 43 completed reproductions use publicly accessible data or published
model parameters. The three-tier validation (CPU + GPU + metalForge) is
complete for all 30 actionable papers:

| Track | Papers | CPU | GPU | metalForge | Data Source |
|-------|:------:|:---:|:---:|:----------:|-------------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | NCBI SRA, published params |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | PhyNetPy/SATé, algorithms |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | NCBI SRA, MBL, MG-RAST |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | MassBank, EPA, Zenodo |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | repoDB, PMC, ROBOKOP |
| Phase 37-39 extensions | 9 | 9/9 | — | — | Published params, NCBI |
| Cross-spring (spectral) | 1 | 1/1 | 1/1 | — | Algorithmic |
| **Total** | **40** | **40/40** | **31/31** | **30/30** | All public |

Zero proprietary data dependencies. 9 more papers queued (Track 4: No-Till
Soil QS) — awaiting implementation.

---

## Part 7: Evolution Readiness — Rust → WGSL → Pipeline Stage

### Tier A — Lean (22 modules, direct upstream primitive)

| Rust Module | ToadStool Primitive | Pipeline Stage |
|-------------|--------------------|----|
| `diversity_gpu` | `FusedMapReduceF64`, `BrayCurtisF64` | Alpha/beta diversity |
| `ani_gpu` | `AniBatchF64` | Genome-level ANI |
| `snp_gpu` | `SnpCallingF64` | Variant calling |
| `dnds_gpu` | `DnDsBatchF64` | Selection pressure |
| `pangenome_gpu` | `PangenomeClassifyGpu` | Gene presence/absence |
| `hmm_gpu` | `HmmBatchForwardF64` | Sequence classification |
| `quality_gpu` | `QualityFilterGpu` | Read QC |
| `random_forest_gpu` | `RfBatchInferenceGpu` | ML inference |
| `signal_gpu` | `PeakDetectF64` | Chromatographic peaks |
| `pcoa_gpu` | `BatchedEighGpu` | Ordination |
| `unifrac_gpu` | `UniFracPropagateGpu` | Phylogenetic distance |
| `kmer_gpu` | `KmerHistogramGpu` | K-mer counting |
| `dada2_gpu` | `Dada2EStepGpu` | Denoising E-step |
| `felsenstein_gpu` | `FelsensteinGpu` | Tree likelihood |
| `bistable_gpu` | `BatchedOdeRK4<Bistable>` | ODE phenotypic switch |
| `batch_fitness_gpu` | `BatchFitnessGpu` | Evolutionary fitness |
| `hamming_gpu` | `PairwiseHammingGpu` | Distance metric |
| `jaccard_gpu` | `JaccardGpu` | Set similarity |
| `spatial_payoff_gpu` | `SpatialPayoffGpu` | Game theory grid |
| `locus_variance_gpu` | `LocusVarianceGpu` | Population genetics |
| `rarefaction_gpu` | `RarefactionGpu` | Sampling curves |
| `eic_gpu` | `FusedMapReduceF64` | Ion chromatograms |

### Tier B — Compose (11 modules, multiple primitives)

| Rust Module | ToadStool Primitives | Pipeline Stage |
|-------------|---------------------|---|
| `spectral_match_gpu` | `GemmF64` + `FusedMapReduceF64` | MS2 library matching |
| `gemm_cached` | `GemmCached` + `TensorContext` | Shared GEMM cache |
| `streaming_gpu` | Multiple domain primitives | End-to-end GPU streaming |
| `gbm_gpu` | `TreeInferenceGpu` | Gradient boosted inference |
| `merge_pairs_gpu` | `FusedMapReduceF64` | Paired-end merging |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Tree distance metric |
| `chimera_gpu` | `GemmCachedF64` | Chimera detection |
| `neighbor_joining_gpu` | `FusedMapReduceF64` | Tree construction |
| `reconciliation_gpu` | `TreeInferenceGpu` | Gene/species reconciliation |
| `molecular_clock_gpu` | `FusedMapReduceF64` | Divergence dating |
| `kmd_gpu` | `FusedMapReduceF64` | PFAS homologue detection |

### Tier C — Write (1 module, pending absorption)

| Rust Module | Local Shader | Absorption Target |
|-------------|-------------|-------------------|
| `diversity_fusion_gpu` | `diversity_fusion_f64.wgsl` | ToadStool P2-5 |

### Blocking Items for Full Lean

1. **`diversity_fusion_f64.wgsl` absorption** — the sole remaining local WGSL shader
2. **`ComputeDispatch` migration** — P3, medium effort, replaces manual BGL setup
3. **DF64 GEMM adoption** — P3, low effort, `Fp64Strategy::Hybrid` for consumer GPUs
4. **`BandwidthTier` wiring** — P3, low effort, PCIe-aware dispatch in metalForge

---

## Part 8: Lessons Learned (for ToadStool evolution)

### What worked well

1. **`BatchedOdeRK4<S>::generate_shader()`** — trait-based WGSL generation for ODE
   systems is the ideal pattern. 5 biological ODE systems lean on it with zero local
   WGSL. The struct+trait approach (implement `OdeSystem`, get WGSL for free) should
   be the standard for all parametric compute kernels.

2. **`FusedMapReduceF64` versatility** — used by 10+ wetSpring modules for Shannon,
   Simpson, dot products, norms, weighted sums. The fused map+reduce pattern is the
   right abstraction for element-wise → scalar reductions.

3. **Lean absorption cycle** — the Write → Absorb → Lean pattern works exactly as
   designed. Phase 20 absorbed 8 shaders, deleted 25 KB local code. Springs write
   proposals, ToadStool absorbs as shared primitives, springs lean. No coupling between
   springs.

4. **`needs_f64_exp_log_workaround()`** — capability-based polyfill detection is much
   better than hardcoded workarounds. Consumer GPUs (RTX 40-series) get the polyfill
   automatically, compute-class GPUs don't.

5. **Cross-spring evolution** — neuralSpring graph primitives (S54-S56) used directly
   in wetSpring Anderson-QS analysis proves the convergence hub model works. Springs
   don't import each other; they both lean on ToadStool.

### What could be improved

1. **`PhyloTree` dual types** — validation binaries need conversion between
   `wetspring::bio::unifrac::PhyloTree` and `barracuda::PhyloTree`. A single shared
   type would simplify the interface.

2. **CPU `dot`/`l2_norm` gap** — these trivial helpers shouldn't need local
   implementations. Adding them to `barracuda::special` or `barracuda::math` would
   close the last local math gap.

3. **`ComputeDispatch` adoption** — the builder pattern is ready upstream but springs
   still use manual BGL setup in some modules. A migration guide with before/after
   examples would accelerate adoption.

---

## Part 9: Handoff Artifacts

| Artifact | Location |
|----------|----------|
| This handoff | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V41_DEEP_AUDIT_HANDOFF_FEB25_2026.md` |
| Previous handoff | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V40_CATCH_UP_FEB25_2026.md` |
| Evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| Absorption manifest | `barracuda/ABSORPTION_MANIFEST.md` |
| Specs | `specs/BARRACUDA_REQUIREMENTS.md` |
| Cross-spring shader map | `wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` |
| Archive | `wateringHole/handoffs/archive/` (35 previous handoffs) |

---

*wetSpring Phase 46 — all 3,300+ checks pass, 918 tests, 96.48% coverage,
49 ToadStool primitives + 2 BGL helpers consumed, 0 local WGSL (1 extension),
70 named tolerances, 0 clippy warnings.*
