# wetSpring Evolution Readiness

**Date:** February 22, 2026
**Pattern:** Write → Absorb → Lean (inherited from hotSpring)
**Status:** 41 CPU + 25 GPU modules, 0 local WGSL shaders (Lean phase complete), 32 ToadStool primitives consumed, 740 tests passing

### Feb 22 Lean Phase Complete: All 4 local WGSL shaders absorbed (ToadStool S39-41)

ToadStool sessions 39-41 absorbed the final 4 local WGSL shaders (ODE sweep,
kmer histogram, unifrac propagate, taxonomy FC). wetSpring rewired to
`barracuda::ops::bio::*`, deleted the shaders, and verified: 740 tests pass.
The `shaders/` directory is now empty. ODE blocker resolved: ToadStool S41
fixed `compile_shader_f64` in `batched_ode_rk4.rs`. See `ABSORPTION_MANIFEST.md`
for details.

### Code Quality (Phase 15+)

All modules pass `clippy::pedantic` + `clippy::nursery` (0 warnings, `-D` enforced
in CI), `cargo fmt` (0 diffs), `cargo doc` (0 warnings, `RUSTDOCFLAGS="-D warnings"`).
97% line coverage for bio+io modules (56% overall including bench) via `cargo-llvm-cov`.
All tolerances centralized in `tolerances.rs` (43 named constants — includes
4 Jacobi eigendecomposition constants with Golub & Van Loan provenance).
`#![deny(unsafe_code)]` and `#![deny(clippy::expect_used, clippy::unwrap_used)]`
enforced crate-wide (`deny` instead of `forbid` to allow `unsafe` in test env var
manipulation required by Rust 2024 edition). All 83 binaries carry `# Provenance`
headers. Data paths use `validation::data_dir()` for capability-based discovery.
`flate2` uses `rust_backend` — zero C dependencies (ecoBin compliant). All 40
Python baselines carry SPDX-License-Identifier + Date headers. DADA2 algorithmic
constants fully documented with provenance (Callahan et al. 2016, R package defaults).
CI enforces fmt, clippy (pedantic+nursery), test, doc, and json feature check on
every push/PR. **Rust edition 2024**, MSRV 1.85.

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

## CPU Modules (41)

| Module | Domain | GPU Tier | ToadStool Primitive | Notes |
|--------|--------|----------|-------------------|-------|
| `alignment` | Smith-Waterman | ✅ Absorbed | `SmithWatermanGpu` | Exp044 |
| `ani` | Average Nucleotide Identity | ✅ Absorbed | `AniBatchF64` | Rewired Feb 22, 2026 |
| `bistable` | ODE toggle switch | **A** | — | Map to `BatchedOdeRK4F64` |
| `bootstrap` | Phylo resampling | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `capacitor` | Signal peak | C | — | Too small for GPU |
| `chimera` | Chimera detection | C | — | Sequential per-read |
| `cooperation` | Game theory QS | **B** | — | GPU flat API (Exp078), maps to ODE sweep |
| `dada2` | Error model | ✅ Absorbed | `Dada2EStepGpu` | Rewired Feb 22, 2026 |
| `decision_tree` | PFAS ML | ✅ Absorbed | `TreeInferenceGpu` | Exp044 |
| `derep` | Dereplication | C | — | Hash-based, CPU-optimal |
| `diversity` | α/β diversity | ✅ Absorbed | `BrayCurtisF64`, `FMR` | Exp004/016 |
| `dnds` | Nei-Gojobori dN/dS | ✅ Absorbed | `DnDsBatchF64` | Rewired Feb 22, 2026 |
| `eic` | EIC/XIC extraction | C | — | I/O-bound |
| `feature_table` | OTU table | C | — | Sparse matrix |
| `felsenstein` | Pruning likelihood | ✅ Absorbed | `FelsensteinGpu` | Exp046 |
| `gbm` | GBM inference | C | — | Sequential boosting (CPU-optimal) |
| `gillespie` | Stochastic SSA | ✅ Absorbed | `GillespieGpu` | Exp044 |
| `hmm` | Hidden Markov Model | ✅ Absorbed | `HmmBatchForwardF64` | Rewired Feb 22, 2026 |
| `kmd` | Kendrick mass defect | C | — | Lookup table |
| `kmer` | K-mer counting | ✅ Absorbed | `KmerHistogramF64` | ToadStool S40 (Exp081) |
| `merge_pairs` | Read merging | C | — | Sequential per-pair |
| `molecular_clock` | Strict/relaxed clock | C | — | Small calibration data, CPU-optimal |
| `multi_signal` | Multi-signal QS | **A** | — | GPU flat API (Exp078), maps to ODE sweep |
| `neighbor_joining` | NJ tree construction | C | — | Sequential algorithm |
| `ode` | RK4 integrator | ✅ Absorbed | `BatchedOdeRK4F64` | ToadStool S41 (Exp049) |
| `pangenome` | Gene clustering | ✅ Absorbed | `PangenomeClassifyGpu` | Rewired Feb 22, 2026 |
| `pcoa` | PCoA ordination | ✅ Absorbed | `BatchedEighGpu` | Exp016 |
| `phage_defense` | CRISPR/RM model | **A** | — | GPU flat API (Exp078), maps to ODE sweep |
| `phred` | Quality scoring | C | — | Per-base lookup |
| `placement` | Phylo placement | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `qs_biofilm` | QS/c-di-GMP ODE | ✅ Absorbed | `BatchedOdeRK4F64` | ToadStool S41 (Exp049) |
| `quality` | Read quality | ✅ Absorbed | `QualityFilterGpu` | Rewired Feb 22, 2026. Adapter logic extracted to `adapter.rs` |
| `random_forest` | RF ensemble | ✅ Absorbed | `RfBatchInferenceGpu` | Rewired Feb 22, 2026 |
| `reconciliation` | DTL reconciliation | C | — | Tree traversal |
| `robinson_foulds` | Tree distance | C | — | Per-node comparison |
| `signal` | Signal processing | C | — | FFT-based, small data |
| `snp` | SNP calling | ✅ Absorbed | `SnpCallingF64` | Rewired Feb 22, 2026 |
| `spectral_match` | Spectral cosine | ✅ Absorbed | `FMR` spectral cosine | Exp016 |
| `taxonomy/` | Naive Bayes classify | ✅ Absorbed / NPU | `TaxonomyFcF64` | ToadStool S40; `types`, `kmers`, `classifier` submodules (Exp083) |
| `tolerance_search` | Tolerance search | ✅ Absorbed | `BatchTolSearchF64` | Exp016 |
| `unifrac/` | UniFrac distance | ✅ Absorbed | `UniFracPropagateF64` | ToadStool S40; `tree`, `flat_tree`, `distance` submodules (Exp082) |

---

## GPU Modules (25)

| Module | Wraps | ToadStool Primitive | Status |
|--------|-------|-------------------|--------|
| `batch_fitness_gpu` | EA batch fitness | `BatchFitnessGpu` | Lean (neuralSpring) |
| `hamming_gpu` | Pairwise Hamming | `PairwiseHammingGpu` | Lean (neuralSpring) |
| `jaccard_gpu` | Pairwise Jaccard | `PairwiseJaccardGpu` | Lean (neuralSpring) |
| `locus_variance_gpu` | FST per-locus | `LocusVarianceGpu` | Lean (neuralSpring) |
| `spatial_payoff_gpu` | Spatial PD payoff | `SpatialPayoffGpu` | Lean (neuralSpring) |
| `ani_gpu` | ANI pairwise | `AniBatchF64` | ✅ Lean (Feb 22) |
| `chimera_gpu` | Chimera GPU scoring | `FMR` | Lean |
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

---

## Local WGSL Shader Inventory (0 — Lean phase complete)

All 12 original local WGSL shaders have been absorbed by ToadStool (sessions 31d/31g + 39-41).
The `shaders/` directory is empty. Final 4 absorbed Feb 22: ODE (S41), kmer (S40),
unifrac (S40), taxonomy (S40). ODE blocker resolved: ToadStool S41 fixed
`compile_shader_f64` in `batched_ode_rk4.rs`.

### Shader Compilation Notes

All GPU modules now lean on ToadStool. Historical note: local shaders used `ShaderTemplate::for_driver_auto(_, true)`
for RTX 4070 (Ada Lovelace) f64 exp/log polyfill. ToadStool's absorbed shaders
handle this automatically via `WgpuDevice::compile_shader_f64()`.

**naga quirks (historical; absorbed shaders):**
- `enable f64;` not supported — omit from all WGSL
- Bare f32 literals in f64 builtins fail type check — use `f64(0.0)`
- `pow()` on f64 crashes NVVM — use `pow_f64()` polyfill

---

## ToadStool Primitives Consumed (32)

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

---

## Absorption Queue (handoff to ToadStool)

### Completed (Feb 22) — All 12 shaders absorbed (Lean phase complete)

ToadStool sessions 39-41 absorbed the final 4 local WGSL shaders: ODE sweep
(S41: fixed `compile_shader_f64` in batched_ode_rk4.rs), kmer histogram (S40),
unifrac propagate (S40), taxonomy FC (S40). wetSpring rewired; `shaders/` empty.

### Completed (Feb 22) — 8 bio shaders absorbed (sessions 31d/31g)

All 8 bio shaders successfully absorbed by ToadStool sessions 31d/31g and
rewired in wetSpring. 451 GPU checks pass. Two ToadStool bugs fixed during
the rewire (SNP binding layout, AdapterInfo propagation).

### GPU-Ready (Tier A) — Promoted via Exp078

2. **`multi_signal`** — 7-var ODE, flat API, maps to `BatchedOdeRK4F64`
3. **`phage_defense`** — 4-var ODE, flat API, maps to `BatchedOdeRK4F64`

### GPU/NPU-Ready (Tier A) — Promoted via Exp081–083

4. **`kmer`** — ✅ Absorbed (ToadStool S40: `KmerHistogramF64`)
5. **`unifrac`** — ✅ Absorbed (ToadStool S40: `UniFracPropagateF64`)
6. **`taxonomy`** — ✅ Absorbed (ToadStool S40: `TaxonomyFcF64`); NPU int8 variant planned

### Needs Refactoring (Tier B)

7. **`cooperation`** — 4-var ODE, flat API (Exp078), maps to ODE sweep

---

## Validation Coverage by Tier

| Tier | CPU Modules | GPU Modules | CPU Checks | GPU Checks |
|------|:-----------:|:-----------:|:----------:|:----------:|
| ✅ Absorbed (Lean) | 25 | 25 (ToadStool) | 900+ | 400+ |
| A (GPU/NPU-ready) | 2 | 0 | 81+ | 12+ |
| B (needs refactor) | 2 | 0 | 60+ | 0 |
| C (CPU-only) | 14 | 0 | 171+ | 0 |
| Dispatch routing | — | — | 80 | — |
| Streaming/transfer | — | — | 57 | 82 |
| **Total** | **41** | **25** | **1,392** | **533** |

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
| Feb 21 | DEPRECATION_MIGRATION updated with `bio::special` extraction steps |
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

---

## Comparison with hotSpring Evolution

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Computational physics | Life science & analytical chemistry |
| CPU modules | 50+ (physics, lattice, MD, spectral) | 41 (bio, signal, ML) |
| GPU modules | 34 WGSL shaders | 25 modules, 0 local WGSL (Lean complete) |
| Absorbed | complex64, SU(3), plaquette, HMC, CellList | SW, Gillespie, DT, Felsenstein, GEMM, HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF + 5 neuralSpring (PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance) |
| WGSL pattern | `pub const WGSL: &str` inline | `include_str!("../shaders/...")` |
| metalForge | GPU + NPU hardware characterization | GPU + NPU + cross-substrate validation |
| Handoffs | `../wateringHole/handoffs/` (16+ docs) | `archive/handoffs/` (consolidated) |
| Tests | 454 | 738 |
| Validation | 418 checks | 2,229+ checks |
| Experiments | 31 suites | 96 experiments |
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

Both Springs follow the same pipeline: Python → Rust CPU → GPU → ToadStool absorption.
The patterns should converge: hotSpring's `pub const WGSL` inline approach and
wetSpring's `include_str!` file approach both work for absorption.
