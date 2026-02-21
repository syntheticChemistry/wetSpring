# wetSpring Evolution Readiness

**Date:** February 21, 2026
**Pattern:** Write → Absorb → Lean (inherited from hotSpring)
**Status:** 41 CPU + 20 GPU modules, 9 local WGSL shaders, 15 ToadStool primitives consumed

### Code Quality (Phase 15)

All modules pass `clippy::pedantic` + `clippy::nursery` (0 warnings), `cargo fmt`
(0 diffs), `cargo doc` (0 warnings). 93.5% line coverage via `cargo-llvm-cov`.
All tolerances centralized in `tolerances.rs` (22 named constants). Zero `unsafe`
and zero `.unwrap()` in production code. All 61 binaries carry `# Provenance`
headers. Data paths use `validation::data_dir()` for capability-based discovery.
`flate2` uses `rust_backend` — zero C dependencies (ecoBin compliant).

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
| `ani` | Average Nucleotide Identity | **A** | Local WGSL | `ani_batch_f64.wgsl` (Exp058) |
| `bistable` | ODE toggle switch | **A** | — | Map to `BatchedOdeRK4F64` |
| `bootstrap` | Phylo resampling | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `capacitor` | Signal peak | C | — | Too small for GPU |
| `chimera` | Chimera detection | C | — | Sequential per-read |
| `cooperation` | Game theory QS | C | — | CPU-only model |
| `dada2` | Error model | **A** | Local WGSL | `dada2_e_step.wgsl` |
| `decision_tree` | PFAS ML | ✅ Absorbed | `TreeInferenceGpu` | Exp044 |
| `derep` | Dereplication | C | — | Hash-based, CPU-optimal |
| `diversity` | α/β diversity | ✅ Absorbed | `BrayCurtisF64`, `FMR` | Exp004/016 |
| `dnds` | Nei-Gojobori dN/dS | **A** | Local WGSL | `dnds_batch_f64.wgsl` (Exp058) |
| `eic` | EIC/XIC extraction | C | — | I/O-bound |
| `feature_table` | OTU table | C | — | Sparse matrix |
| `felsenstein` | Pruning likelihood | ✅ Absorbed | `FelsensteinGpu` | Exp046 |
| `gbm` | GBM inference | C | — | Sequential boosting (CPU-optimal) |
| `gillespie` | Stochastic SSA | ✅ Absorbed | `GillespieGpu` | Exp044 |
| `hmm` | Hidden Markov Model | **A** | Local WGSL | `hmm_forward_f64.wgsl` (Exp047) |
| `kmd` | Kendrick mass defect | C | — | Lookup table |
| `kmer` | K-mer counting | **B** | — | Needs lock-free hash GPU primitive |
| `merge_pairs` | Read merging | C | — | Sequential per-pair |
| `molecular_clock` | Strict/relaxed clock | C | — | Small calibration data, CPU-optimal |
| `multi_signal` | Multi-signal QS | **B** | — | Maps to ODE sweep |
| `neighbor_joining` | NJ tree construction | C | — | Sequential algorithm |
| `ode` | RK4 integrator | **A** | Local WGSL | `batched_qs_ode_rk4_f64.wgsl` (Exp049) |
| `pangenome` | Gene clustering | **A** | Local WGSL | `pangenome_classify.wgsl` (Exp058) |
| `pcoa` | PCoA ordination | ✅ Absorbed | `BatchedEighGpu` | Exp016 |
| `phage_defense` | CRISPR/RM model | **B** | — | Maps to ODE sweep |
| `phred` | Quality scoring | C | — | Per-base lookup |
| `placement` | Phylo placement | ✅ Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `qs_biofilm` | QS/c-di-GMP ODE | **A** | Local WGSL | `batched_qs_ode_rk4_f64.wgsl` (Exp049) |
| `quality` | Read quality | **A** | Local WGSL | `quality_filter.wgsl` |
| `random_forest` | RF ensemble | **A** | Local WGSL | `rf_batch_inference.wgsl` (Exp063) |
| `reconciliation` | DTL reconciliation | C | — | Tree traversal |
| `robinson_foulds` | Tree distance | C | — | Per-node comparison |
| `signal` | Signal processing | C | — | FFT-based, small data |
| `snp` | SNP calling | **A** | Local WGSL | `snp_calling_f64.wgsl` (Exp058) |
| `spectral_match` | Spectral cosine | ✅ Absorbed | `FMR` spectral cosine | Exp016 |
| `taxonomy` | Naive Bayes classify | **B** / NPU | — | NPU candidate (FC model) |
| `tolerance_search` | Tolerance search | ✅ Absorbed | `BatchTolSearchF64` | Exp016 |
| `unifrac` | UniFrac distance | **B** | — | Needs tree traversal GPU primitive |

---

## GPU Modules (20)

| Module | Wraps | ToadStool Primitive | Status |
|--------|-------|-------------------|--------|
| `ani_gpu` | ANI pairwise | Local WGSL | **A** — handoff candidate |
| `chimera_gpu` | Chimera GPU scoring | `FMR` | Lean |
| `dada2_gpu` | DADA2 E-step | Local WGSL | **A** — handoff candidate |
| `diversity_gpu` | α/β diversity | `BrayCurtisF64`, `FMR` | Lean |
| `dnds_gpu` | dN/dS GPU | Local WGSL | **A** — handoff candidate |
| `eic_gpu` | EIC extraction | `FMR` | Lean |
| `gemm_cached` | Matrix multiply | `GemmCachedF64` | Lean |
| `hmm_gpu` | HMM forward | Local WGSL | **A** — handoff candidate |
| `kriging` | Spatial interpolation | `KrigingF64` | Lean |
| `ode_sweep_gpu` | ODE parameter sweep | Local WGSL | **A** — handoff candidate |
| `pangenome_gpu` | Pangenome classify | Local WGSL | **A** — handoff candidate |
| `pcoa_gpu` | PCoA eigenvalues | `BatchedEighGpu` | Lean |
| `quality_gpu` | Quality filtering | Local WGSL | **A** — handoff candidate |
| `rarefaction_gpu` | Rarefaction curves | `PrngXoshiro` | Lean |
| `random_forest_gpu` | RF batch inference | Local WGSL | **A** — handoff candidate |
| `snp_gpu` | SNP calling | Local WGSL | **A** — handoff candidate |
| `spectral_match_gpu` | Spectral cosine | `FMR` | Lean |
| `stats_gpu` | Variance/correlation | `FMR` | Lean |
| `streaming_gpu` | Streaming pipeline | Multiple | Lean |
| `taxonomy_gpu` | Taxonomy scoring | `FMR` | Lean |

---

## Local WGSL Shader Inventory

| Shader | Domain | GPU Checks | Status | Absorption Target |
|--------|--------|:----------:|--------|------------------|
| `quality_filter.wgsl` | Read quality | 88 (pipeline) | **A** | New `ParallelFilter<T>` |
| `dada2_e_step.wgsl` | DADA2 error model | 88 (pipeline) | **A** | New `BatchPairReduce<f64>` |
| `hmm_forward_f64.wgsl` | HMM batch forward | 13 (Exp047) | **A** | New `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | QS ODE sweep | 7 (Exp049) | **A** | Fix upstream `BatchedOdeRK4F64` |
| `ani_batch_f64.wgsl` | ANI pairwise identity | 7 (Exp058) | **A** | New `AniBatchF64` |
| `snp_calling_f64.wgsl` | SNP calling | 5 (Exp058) | **A** | New `SnpCallingF64` |
| `dnds_batch_f64.wgsl` | dN/dS (Nei-Gojobori) | 9 (Exp058) | **A** | New `DnDsBatchF64` |
| `pangenome_classify.wgsl` | Pangenome classify | 6 (Exp058) | **A** | New `PangenomeClassifyGpu` |
| `rf_batch_inference.wgsl` | RF batch inference | 13 (Exp063) | **A** | New `RfBatchInferenceGpu` |

### Shader Compilation Notes

All 9 local shaders except `quality_filter.wgsl` require
`ShaderTemplate::for_driver_auto(source, ...)` on RTX 4070 (Ada Lovelace).
The `dN/dS` shader requires forced polyfill (`true`) for `log()`.

**naga quirks:**
- `enable f64;` not supported — omit from all WGSL
- Bare f32 literals in f64 builtins fail type check — use `f64(0.0)`
- `pow()` on f64 crashes NVVM — use `pow_f64()` polyfill

---

## ToadStool Primitives Consumed (15)

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
| `ShaderTemplate::for_driver_auto` | hmm_gpu, ode_sweep_gpu, Track 1c, RF | 047+ |
| `LogsumexpWgsl` | (available, not yet wired) | — |
| `BatchedOdeRK4F64` | (blocked: `enable f64;` + `pow` crash) | — |

---

## Absorption Queue (handoff to ToadStool)

### Ready Now (Tier A) — 9 shaders

1. **`hmm_forward_f64.wgsl`** — 13/13 GPU checks (Exp047)
2. **`batched_qs_ode_rk4_f64.wgsl`** — 7/7 GPU checks (Exp049)
3. **`ani_batch_f64.wgsl`** — 7/7 GPU checks (Exp058)
4. **`snp_calling_f64.wgsl`** — 5/5 GPU checks (Exp058)
5. **`dnds_batch_f64.wgsl`** — 9/9 GPU checks (Exp058)
6. **`pangenome_classify.wgsl`** — 6/6 GPU checks (Exp058)
7. **`rf_batch_inference.wgsl`** — 13/13 GPU checks (Exp063)
8. **`dada2_e_step.wgsl`** — 88 pipeline checks (Exp016)
9. **`quality_filter.wgsl`** — 88 pipeline checks (Exp016)

Plus **NVVM driver profile fix** — Ada Lovelace RTX 40-series
`needs_f64_exp_log_workaround()` should return `true`.

### Needs Refactoring (Tier B)

10. **`kmer`** — Needs lock-free hash table GPU primitive (P3)
11. **`unifrac`** — Needs tree traversal GPU primitive (P3)
12. **`taxonomy`** — NPU candidate (Naive Bayes → FC model → int8)
13. **`multi_signal`** / **`phage_defense`** — Map to ODE sweep pattern

---

## Validation Coverage by Tier

| Tier | CPU Modules | GPU Modules | CPU Checks | GPU Checks |
|------|:-----------:|:-----------:|:----------:|:----------:|
| ✅ Absorbed | 10 | 11 (lean) | 500+ | 200+ |
| A (handoff ready) | 10 | 9 (local WGSL) | 400+ | 60+ |
| B (needs refactor) | 6 | 0 | 150+ | 0 |
| C (CPU-only) | 15 | 0 | 191+ | 0 |
| **Total** | **41** | **20** | **1,241** | **260** |

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
| Feb 20 | Exp057: BarraCUDA CPU v4 — 23 domains, 128/128 parity checks |
| Feb 20 | Exp058: GPU Track 1c — 4 new WGSL shaders, 27/27 GPU checks |
| Feb 20 | Exp059: 25-domain benchmark — 22.5× Rust over Python |
| Feb 20 | Exp060: metalForge cross-substrate — 20/20 CPU↔GPU parity |
| Feb 20 | Exp061/062: RF + GBM inference — 29/29 CPU checks (domains 24-25) |
| Feb 20 | Exp063: GPU RF batch inference — 13/13 GPU checks (SoA WGSL shader) |
| Feb 21 | Phase 15: Code quality hardening — pedantic clippy, tolerance centralization, provenance headers |
| Feb 21 | 93.5% line coverage, 552 tests (539 lib + 13 doc), 0 clippy warnings |
| Feb 21 | All inline tolerance literals → 22 named constants in `tolerances.rs` |
| Feb 21 | All data paths → `validation::data_dir()` for capability-based discovery |

---

## Comparison with hotSpring Evolution

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Computational physics | Life science & analytical chemistry |
| CPU modules | 50+ (physics, lattice, MD, spectral) | 41 (bio, signal, ML) |
| GPU modules | 34 WGSL shaders | 20 modules + 9 local WGSL shaders |
| Absorbed | complex64, SU(3), plaquette, HMC, CellList | SW, Gillespie, DT, Felsenstein, GEMM |
| WGSL pattern | `pub const WGSL: &str` inline | `include_str!("../shaders/...")` |
| metalForge | GPU + NPU hardware characterization | GPU + NPU + cross-substrate validation |
| Handoffs | `wateringHole/handoffs/` (16+ docs) | `archive/handoffs/` (consolidated) |
| Tests | 454 | 552 |
| Validation | 418 checks | 1,501 checks |
| Experiments | 31 suites | 63 experiments |
| Line coverage | — | 93.5% |

Both Springs follow the same pipeline: Python → Rust CPU → GPU → ToadStool absorption.
The patterns should converge: hotSpring's `pub const WGSL` inline approach and
wetSpring's `include_str!` file approach both work for absorption.
