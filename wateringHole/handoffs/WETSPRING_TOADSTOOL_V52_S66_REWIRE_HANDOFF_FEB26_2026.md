# wetSpring → ToadStool/BarraCuda V52 S66 Rewire Handoff

**Date:** February 26, 2026
**Phase:** 52 (V52 — S66 rewire + full GPU validation on local hardware)
**ToadStool pin:** `045103a7` (S66 Wave 5)
**Previous pin:** `17932267` (S65)
**wetSpring status:** Fully lean — 0 local WGSL, 0 local derivative math, 0 local regression, delegated hill/percentile/mean

---

## What Changed in ToadStool Since V50 (S65 → S66 Wave 5)

### 3 commits: `bd62d939`, `95eaad92`, `045103a7`

| Category | Addition | Source |
|----------|----------|--------|
| `stats::regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `fit_all` → `FitResult` | airSpring absorption |
| `stats::metrics` | `hill`, `monod`, `mae`, `dot`, `l2_norm` | wetSpring/neuralSpring |
| `stats::bootstrap` | `bootstrap_ci`, `bootstrap_mean`, `bootstrap_median`, `bootstrap_std`, `rawr_mean` | hotSpring absorption |
| `stats::hydrology` | `hargreaves_et0`, `crop_coefficient`, `soil_water_balance` | groundSpring absorption |
| `stats::moving_window_f64` | `moving_window_stats_f64` → `MovingWindowResultF64` | airSpring absorption |
| `stats::diversity` | `shannon_from_frequencies` (pre-normalized) | internal |
| `device` | `compile_shader_df64` — DF64 WGSL compilation with ILP optimizer | internal |
| `ops::df64_shaders` | `WGSL_ELEMENTWISE_{ADD,SUB,MUL,FMA}_DF64`, `WGSL_{SUM,MEAN}_REDUCE_DF64` | internal |
| `ops::lattice` | `NeighborMode::precompute` — 4D periodic neighbor table for HMC | hotSpring absorption |
| `ops::rk_stage` | Re-export `BatchedOdeRK4F64`, `BatchedRk4Config` | doc cleanup |
| Smart refactoring | 14 test files extracted, `compute_graph.rs` reduced, all files < 1000 LOC | deep debt |

### Quality gates: 2,541 barracuda tests, 0 warnings, 0 unsafe (production), 694 WGSL shaders

---

## wetSpring V52 Rewires

### 1. `hill()` → `barracuda::stats::hill` (S66)

`bio/qs_biofilm.rs:167` — local `powf(n)` implementation replaced with upstream
delegation. Retains `x <= 0 → 0` physical guard (upstream has no guard).
**Lines eliminated:** 3 (math body replaced with 1-line delegation).

### 2. `fit_heaps_law` → `barracuda::stats::fit_linear` (S66)

`bio/pangenome.rs:160-182` — manual 18-line log-log linear regression
(sums, products, denominator check) replaced with `fit_linear(&ln_g, &ln_n).map(|r| r.params[0])`.
**Lines eliminated:** 15.

### 3. `compute_ci` → `barracuda::stats::{mean, percentile}` (S66)

`bio/rarefaction_gpu.rs:231-259` — local mean/sort/percentile computation
replaced with `barracuda::stats::mean` + `barracuda::stats::percentile` calls.
**Lines eliminated:** 8.

### 4. `shannon_from_frequencies` re-export (S66)

`bio/diversity.rs` — new re-export of `barracuda::stats::shannon_from_frequencies`
for pre-normalized frequency data (GPU output path optimization).

---

## V51 GPU Validation (same session)

Prior to the S66 rewire, V51 completed full GPU validation on local hardware:

| Item | Result |
|------|--------|
| **GPU validators** | 70 validators, 1,578 checks, ALL PASS |
| **Hardware** | RTX 4070 (Ada) + Titan V |
| **Fixes** | BatchFitness/LocusVariance f32→f64 (upstream f64 shader evolution) |
| **Tolerance corrections** | 8 validators: `GPU_VS_CPU_TRANSCENDENTAL` (1e-10) → `GPU_LOG_POLYFILL` (1e-7) for chained Shannon/cosine |
| **Library tests (GPU)** | 830 pass |
| **Total validation checks** | 4,494+ |

---

## Primitives Consumed (updated)

| Category | Count | Source |
|----------|-------|--------|
| Core ops (reduce, linalg, gemm, matmul) | 18 | ToadStool core |
| Bio ops (ODE, diversity, NMF, peak detect) | 21 | wetSpring → absorbed |
| Stats (hill, monod, percentile, mean, fit_linear, shannon_from_frequencies) | 6 NEW | S66 |
| ML (random forest, HMM, TransE, ESN) | 8 | ToadStool ML |
| Distance (Hamming, Jaccard, SpatialPayoff, BatchFitness, LocusVariance) | 5 | neuralSpring → absorbed |
| Staging (unidirectional streaming, ring buffer) | 2 | ToadStool core |
| BGL helpers | 2 | ToadStool core |
| **Total primitives consumed** | **79** (+6 from S66) |

---

## Absorption Candidates for ToadStool

### Ready to absorb (wetSpring → ToadStool)

| Item | Location | Effort |
|------|----------|--------|
| Monostable `QsBiofilmOde` | `bio/qs_biofilm.rs` | Low — only missing ODE system |
| `rk4_trajectory` helper | `bio/ode_solvers.rs` | Low — convenience wrapper |
| `ConvergenceGuard` pattern | `bio/multi_signal_gpu.rs` | Medium — c-di-GMP guard |
| Streaming FASTQ/mzML/MS2 parsers | `bio/io/` | Medium — zero-copy |

### NOT candidates (domain-specific, stays local)

- Pangenome analysis pipeline (`bio/pangenome.rs`)
- Taxonomy classifier (`bio/taxonomy/`)
- Phylogenetic bootstrap (`bio/bootstrap.rs`)

---

## What wetSpring Does NOT Use (S66 features available but not needed)

| Feature | Why not |
|---------|---------|
| `compile_shader_df64` | ODE shaders use native f64; DF64 path is for FP32-only GPUs |
| `stats::hydrology` | No hydrological modeling in wetSpring |
| `stats::moving_window_f64` | No time-series sliding window analysis |
| `ops::lattice::NeighborMode` | Lattice QCD is hotSpring domain |
| `ops::df64_shaders` | DF64 elementwise ops for FP32-only path |
| `bootstrap_ci` (full resampling) | Rarefaction uses multinomial subsampling, not bootstrap resampling |

---

## Verification

```
cargo test --lib                  → 823 pass, 0 fail
cargo clippy --features gpu       → 0 warnings (pedantic + nursery)
cargo fmt                         → 0 diffs
GPU validators (70)               → 1,578 checks PASS (RTX 4070)
CPU validators (57)               → 917 checks PASS
```
