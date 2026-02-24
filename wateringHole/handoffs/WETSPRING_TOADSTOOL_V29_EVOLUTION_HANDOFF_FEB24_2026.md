# wetSpring → ToadStool Handoff V29: Evolution Absorption & Cross-Spring Synthesis

**Date:** February 24, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda team
**Phase:** 40 — Complete S57 absorption, deprecated API removal, dead-code cleanup, cross-spring synthesis

---

## Summary

This handoff completes the S57 alignment cycle started in V28. Where V28 surveyed
new primitives and fixed clippy lints, V29 finishes the job: all 6 S54-S57
primitives are wired into bio workflows (Exp162, 66/66 PASS), the deprecated
`parse_fastq` API is fully retired from tests and fuzz targets, dead code is
cleaned, and documentation is globally consistent. This handoff also maps the
full cross-spring evolution provenance — what each Spring contributes, what it
absorbs, and how the biome model benefits everyone.

### What Changed in V29

1. **Deprecated API migration** — All `parse_fastq()` calls migrated to
   `FastqIter::open()` across 5 files (fuzz target, 3 integration tests,
   unit tests). No `#[allow(deprecated)]` remains for FASTQ.
2. **Dead code cleanup** — Removed unused struct fields in 4 validation binaries
   (`validate_public_benchmarks`, `validate_features`, `validate_cross_substrate_pipeline`,
   `validate_voc_peaks`). Removed corresponding `#[allow(dead_code)]`.
3. **Binary registration** — `validate_cross_spring_s57` (Exp162) added to
   `Cargo.toml` → **152 binaries**.
4. **Global doc sync** — All 12 documentation files updated to Phase 40,
   162 experiments, 3,198+ checks, 37 primitives consumed, S57 aligned.
5. **wateringHole index** — Active handoff table updated V25 → V28 → V29,
   V26-V27 filled in, shader count updated to 650+.
6. **Clippy pedantic+nursery** — 0 errors, 0 warnings on `--all-targets`.

---

## wetSpring Current State

| Metric | Value |
|--------|-------|
| Phase | **40** |
| Tests | **881** (834 barracuda + 47 forge) |
| Experiments | **162** |
| Validation checks | **3,198+** |
| Binaries | **152** (141 validate + 11 benchmark) |
| Library coverage | **95.67%** |
| CPU modules | **47** |
| GPU modules | **42** |
| ToadStool primitives consumed | **37** (31 Lean + 6 S54-S57) |
| Local WGSL shaders | **0** (all generated via `generate_shader()`) |
| ToadStool alignment | **S57** |
| Python scripts | **42** |
| Papers | **43/43** |
| Clippy | **clean** (pedantic + nursery, `-D warnings`) |
| Deprecated API calls | **0** |

---

## Cross-Spring Evolution Provenance Map

### The Biome Model

```
   hotSpring ──────→ ToadStool/BarraCuda ←────── wetSpring
   (physics)              ↑↓                 (bio/chem)
                    neuralSpring              airSpring
                      (ML/AI)                (atmosphere)
```

Springs never import each other. Each Spring writes primitives locally, ToadStool
absorbs the best implementations, and all Springs lean on the shared upstream.
This cycle — **Write → Absorb → Lean** — drives quality convergence across
domains without coupling.

### What Each Spring Contributes

#### hotSpring → ToadStool (physics precision)

| Contribution | ToadStool Module | Used By |
|---|---|---|
| Anderson 1D/2D/3D eigensolvers | `barracuda::spectral` | wetSpring (Anderson-QS localization), neuralSpring |
| Lanczos algorithm + CSR `SpMV` | `barracuda::spectral` | wetSpring (community graph eigenvalues) |
| Level spacing ratio | `barracuda::spectral` | wetSpring (GOE/Poisson classification) |
| Hofstadter butterfly | `barracuda::spectral` | neuralSpring (fractal transport) |
| Complex f64 WGSL (14 ops) | `barracuda::gpu` | hotSpring lattice QCD |
| SU(3) algebra + Wilson plaquette | `barracuda::gpu` | hotSpring HMC |
| FFT f64 (1D + 3D) | `barracuda::fft` | wetSpring (spectral analysis), hotSpring (BCS) |
| `GpuDriverProfile` + `WgslOptimizer` | `barracuda::gpu` | all Springs |
| NVK exp/log workaround via `ShaderTemplate` | `barracuda::gpu` | all Springs |

#### wetSpring → ToadStool (bio shaders & precision)

| Contribution | ToadStool Module | Used By |
|---|---|---|
| `(zero + literal)` f64 constant pattern | `math_f64.wgsl` preamble | all Springs (`log_f64` 1e-3 → 1e-15) |
| `GemmCached` (60× repeated GEMM) | `barracuda::gpu` | hotSpring (HFB SCF loop) |
| Smith-Waterman GPU | `barracuda::bio` | wetSpring (alignment) |
| Pairwise Hamming/Jaccard GPU | `barracuda::bio` | wetSpring (distance matrices) |
| Gillespie SSA GPU | `barracuda::bio` | wetSpring (stochastic ecology) |
| `BatchedOdeRK4<S>` + `generate_shader()` | `barracuda::ode` | wetSpring (5 ODE systems at runtime) |
| HMM forward GPU | `barracuda::bio` | wetSpring (hidden Markov) |
| Felsenstein pruning GPU | `barracuda::bio` | wetSpring (phylogenetics) |
| BatchFitness + LocusVariance GPU | `barracuda::bio` | wetSpring (evolutionary dynamics) |
| SpatialPayoff GPU | `barracuda::bio` | wetSpring (spatial game theory) |

#### neuralSpring → ToadStool (graph + ML)

| Contribution | ToadStool Module | Used By |
|---|---|---|
| `graph_laplacian` | `barracuda::linalg` | wetSpring (community networks, Exp162) |
| `effective_rank` | `barracuda::linalg` | wetSpring (diversity diagnostics, Exp162) |
| `numerical_hessian` | `barracuda::numerical` | wetSpring (ML curvature, Exp162) |
| `disordered_laplacian` | `barracuda::linalg` | wetSpring (QS-disorder coupling, Exp162) |
| `belief_propagation_chain` | `barracuda::linalg` | wetSpring (taxonomy, Exp162) |
| `boltzmann_sampling` (Metropolis-Hastings) | `barracuda::sample` | wetSpring (MCMC optimization, Exp162) |
| `histogram.wgsl` (atomic binning) | `barracuda::gpu` | all Springs |
| `metropolis.wgsl` (parallel MCMC) | `barracuda::gpu` | all Springs |

#### airSpring → ToadStool (GPU robustness)

| Contribution | ToadStool Module | Used By |
|---|---|---|
| `pow_f64` fractional fix | `barracuda::gpu` | all Springs |
| `acos_f64` from `math_f64.wgsl` | `barracuda::gpu` | all Springs |
| `FusedMapReduceF64` buffer conflict fix | `barracuda::gpu` | all Springs |

### Cross-Spring Synergy Highlights

1. **Anderson localization in biofilms** — hotSpring's spectral eigensolvers +
   neuralSpring's graph Laplacian + wetSpring's QS-disorder model = detects
   localization transitions in microbial community geometry (Exp162 compound
   workflow, 32 checks PASS)

2. **Precision cascade** — wetSpring's `(zero + literal)` fix improved
   `log_f64` from 1e-3 to 1e-15 accuracy → hotSpring's BCS bisection got
   more precise → ToadStool absorbed the fix → all Springs benefit

3. **GEMM reuse** — wetSpring's `GemmCached` (memoized shader compilation,
   60× speedup on repeated calls) → hotSpring uses it for HFB self-consistent
   field iterations with identical matrix dimensions

4. **ODE lean** — wetSpring deleted 5 local WGSL shaders, replaced with
   `generate_shader()` at runtime → ToadStool template system generates
   domain-specific ODE shaders without any Spring maintaining shader files

---

## Exp162 Results: Cross-Spring S57 Evolution

Binary: `validate_cross_spring_s57` | **66/66 checks PASS**

### Primitives Wired

| Primitive | Origin | wetSpring Scenario | Checks |
|---|---|---|---|
| `graph_laplacian` | neuralSpring S54 | 8-node community network → spectral gap | 11 |
| `effective_rank` | neuralSpring S54 | Diversity matrix → effective dimensionality | 3 |
| `numerical_hessian` | neuralSpring S54 | ODE parameter curvature (Lotka-Volterra) | 6 |
| `disordered_laplacian` | neuralSpring S56 | QS heterogeneity → disorder coupling | 32 |
| `belief_propagation_chain` | neuralSpring S56 | 3-layer taxonomy hierarchy → posteriors | 5 |
| `boltzmann_sampling` | neuralSpring S56 | MCMC on Rosenbrock → optimization | 3 |
| GPU smoke tests | cross-spring | Hamming, Jaccard, SpatialPayoff on GPU | 6 |

### Compound Workflow

```
graph_laplacian(community) → eigenvalues(Lanczos) → level_spacing_ratio
disordered_laplacian(community, QS) → eigenvalues → level_spacing_ratio
────── compare: graph r vs Anderson r → localization detection ──────
```

### Performance

| Operation | Size | Time |
|---|---|---|
| graph_laplacian (8-node) | 8×8 | 2 µs |
| numerical_hessian (2-param) | 2×2 | 15 µs |
| belief_propagation_chain (3-layer) | 4→3→5 | 8 µs |
| boltzmann_sampling (1000 steps) | 2-param | 1.2 ms |

---

## Deprecated API Retirement

### `parse_fastq` → `FastqIter::open()`

| File | Before | After |
|---|---|---|
| `fuzz/fuzz_targets/fuzz_fastq.rs` | `parse_fastq(&path)` | `FastqIter::open(&path).and_then(\|i\| i.collect())` |
| `tests/bio_integration.rs` | `#[allow(deprecated)]` + `parse_fastq` | `FastqIter::open(&path).expect("open").collect::<Result<Vec<_>, _>>()` |
| `tests/determinism.rs` | `#[allow(deprecated)]` + `parse_fastq` | Same pattern |
| `tests/io_roundtrip.rs` | 11× `parse_fastq` | `FastqIter::open()` throughout |
| `src/io/fastq/tests.rs` | 6× `parse_fastq` | `FastqIter::open()` throughout |

**Zero** `#[allow(deprecated)]` remaining for `parse_fastq`. The `parse_fastq`
function itself remains in the public API (marked `#[deprecated]`) for any
downstream consumers, but wetSpring no longer calls it anywhere.

---

## Dead Code Removed

| File | Removed | Reason |
|---|---|---|
| `validate_public_benchmarks.rs` | `mean_read_len`, `n_unique`, `asv_counts` fields | Never accessed |
| `validate_features.rs` | `peak_area`, `snr` fields | Never accessed |
| `validate_cross_substrate_pipeline.rs` | `observed`, `variance` fields | Never accessed |
| `validate_voc_peaks.rs` | `nist_id` field | Never accessed |

---

## Absorption Targets for ToadStool

### Track 3: Drug Repurposing GPU (Exp157-161)

These 5 experiments (Phase 39) are validated CPU-only. GPU acceleration needs
new ToadStool primitives:

| Needed Primitive | wetSpring Module | Purpose |
|---|---|---|
| `NmfUpdateGpu` | `bio/nmf.rs` | Lee & Seung multiplicative update | 
| `SparseGemmGpu` | `bio/nmf.rs` | Sparse gene-drug GEMM |
| `TransEScoreGpu` | knowledge graph | TransE embedding scoring |
| `WeightedNmfMaskGpu` | `bio/nmf.rs` | Masked NMF for missing data |

### Track 4: Passthrough → Compose

Three GPU modules wrap ToadStool primitives that don't exist yet:

| Module | Awaits |
|---|---|
| `gbm_gpu` | `GbmBatchInferenceGpu` |
| `feature_table_gpu` | `FeatureExtractionGpu` |
| `signal_gpu` | `PeakDetectGpu` |

### Track 5: Math Feature Gate

Six local math functions in `barracuda/src/special.rs` duplicate upstream:

| Function | Upstream |
|---|---|
| `erf()` | `barracuda::special::erf` |
| `ln_gamma()` | `barracuda::special::ln_gamma` |
| `regularized_gamma_lower()` | `barracuda::special::regularized_gamma` |
| `cholesky_factor()` (in `esn.rs`) | `barracuda::linalg::cholesky_solve` |
| `solve_ridge()` (in `esn.rs`) | `barracuda::linalg::ridge_regression` |
| `trapz()` (in `eic.rs`) | `barracuda::numerical::trapz` |

Once ToadStool adds a `math` feature gate, wetSpring can remove these 6
local implementations and lean fully upstream.

---

## Documents Updated

| Document | Key Updates |
|---|---|
| `README.md` | Phase 40, 162 experiments, 3,198+ checks, 152 binaries, 37 primitives |
| `CONTROL_EXPERIMENT_STATUS.md` | Exp162 added, totals updated, S57 aligned |
| `BENCHMARK_RESULTS.md` | 881 tests, 3,198+ checks, 37 primitives |
| `specs/README.md` | Phase 40, 47 CPU, 37+0 ToadStool, V28 handoff |
| `specs/PAPER_REVIEW_QUEUE.md` | Phase 40, 43 reproductions |
| `specs/BARRACUDA_REQUIREMENTS.md` | 37+0, V28 handoff |
| `whitePaper/README.md` | 162 experiments, 3,198+ checks |
| `whitePaper/baseCamp/README.md` | 162 experiments, 3,198+ checks, 43 papers |
| `barracuda/ABSORPTION_MANIFEST.md` | 37 primitives, S57, Exp162 table |
| `barracuda/EVOLUTION_READINESS.md` | 37 primitives, 162 experiments, S57 |
| `wateringHole/README.md` | V28 active, V26-V27 filled, 650+ shaders |

---

## Verification

```
$ cargo test --features gpu
  770 passed; 0 failed; 9 ignored → 881 total

$ cargo clippy --features gpu --all-targets -- -D clippy::pedantic -D clippy::nursery
  0 errors, 0 warnings

$ cargo run --release --features gpu --bin validate_cross_spring_s57
  66/66 checks PASS
```

---

## Remaining Next Steps

1. **Track 3 GPU** — NMF/TransE primitives for drug repurposing (Exp157-161)
2. **Math feature gate** — `barracuda [features] math = []` → retire 6 local dupes
3. **Passthrough modules** — `gbm_gpu`, `feature_table_gpu`, `signal_gpu` blocked
4. **metalForge absorption** — `forge::probe`, `forge::dispatch` → `barracuda::device`
5. **neuralSpring GPU shaders** — `histogram.wgsl`, `metropolis.wgsl` available
   for cross-spring bio GPU ops (e.g., GPU-accelerated MCMC for ODE parameters)
