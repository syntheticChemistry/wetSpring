# wetSpring → ToadStool Handoff V30: S59 Lean — Upstream Convergence

**Date:** February 24, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda team
**Phase:** 41 — S59 lean: NMF, ridge regression, ODE systems, correlated Anderson rewired to upstream

---

## Summary

This handoff completes the Lean cycle for ToadStool S58-S59 absorptions. Four
categories of local code that ToadStool absorbed from wetSpring are now rewired
back to upstream, closing the Write-Absorb-Lean loop:

1. **NMF** — `bio/nmf.rs` deleted; 3 binaries rewired to `barracuda::linalg::nmf`
2. **Ridge regression** — `solve_ridge`/`cholesky_factor` in `esn.rs` replaced
   by `barracuda::linalg::ridge_regression` (GPU-gated; no-GPU fallback retained)
3. **ODE bio systems** — `ode_systems.rs` deleted; 5 GPU wrappers + 1 benchmark
   rewired to `barracuda::numerical::ode_bio::*Ode`
4. **Anderson correlated disorder** — `build_correlated_anderson_3d` deleted;
   `validate_correlated_disorder.rs` rewired to `barracuda::spectral::anderson_3d_correlated`

### What Changed in V30

1. **`bio/nmf.rs` removed** (482 lines) — NMF (Euclidean + KL divergence), cosine
   similarity, reconstruction error, and `top_k_predictions` now consumed from
   `barracuda::linalg::nmf`. `top_k_cosine` (wetSpring-only) inlined into the
   single binary that uses it.
2. **`bio/ode_systems.rs` removed** (715 lines) — 5 `OdeSystem` trait impls
   (`BistableOde`, `CapacitorOde`, `CooperationOde`, `MultiSignalOde`,
   `PhageDefenseOde`) now consumed from `barracuda::numerical::ode_bio`.
   CPU scenario functions (`run_*`, `scenario_*`, `bifurcation_scan`) remain
   local as they are wetSpring-specific validation logic.
3. **`esn.rs` ridge solve rewired** (~100 lines behind `#[cfg(feature = "gpu")]`) —
   GPU builds delegate to `barracuda::linalg::ridge_regression`; no-GPU builds
   retain the local Cholesky-based solver for standalone operation.
4. **`validate_correlated_disorder.rs` local builder removed** (~115 lines) —
   Uses `barracuda::spectral::anderson_3d_correlated` directly.
5. **`eic.rs` trapezoidal integration** — `integrate_peak()` now delegates to
   `barracuda::numerical::trapz` when `gpu` feature is active; sovereign
   trapezoidal fallback retained for no-GPU builds.
6. **Tolerance cleanup** — 4 NMF-specific constants removed from `tolerances.rs`
   (`NMF_CONVERGENCE`, `NMF_CONVERGENCE_TIGHT`, `NMF_CONVERGENCE_EXACT`,
   `NMF_INIT_FLOOR`), now 56 constants remain.
6. **Cargo.toml** — 3 NMF binaries now `required-features = ["gpu"]` since they
   import from ToadStool barracuda.
7. **Clippy pedantic + nursery** — 0 errors, 0 warnings on `--all-targets --all-features`.

---

## wetSpring Current State

| Metric | Value |
|--------|-------|
| Phase | **41** |
| Tests | **806** (759 barracuda + 47 forge) |
| Experiments | **162** |
| Validation checks | **3,198+** |
| Binaries | **152** |
| CPU modules | **46** (was 47; `nmf` removed) |
| GPU modules | **42** |
| ToadStool primitives consumed | **42** (37 prior + NMF, ridge, ODE bio, correlated Anderson, trapz) |
| Local WGSL shaders | **0** (all generated via `generate_shader()`) |
| ToadStool alignment | **S59** |
| Tolerance constants | **56** (was 60; 4 NMF constants removed) |
| Clippy | **clean** (pedantic + nursery, `-D warnings`) |
| Lines removed | **~1,312** (nmf 482 + ode_systems 715 + anderson ~115) |

---

## Lean Details

### NMF Lean

| Before (wetSpring local) | After (ToadStool upstream) |
|---|---|
| `wetspring_barracuda::bio::nmf::nmf()` → panics on bad input | `barracuda::linalg::nmf::nmf()` → `Result<NmfResult, BarracudaError>` |
| `wetspring_barracuda::bio::nmf::NmfConfig` default uses local `tolerances::NMF_CONVERGENCE` | `barracuda::linalg::nmf::NmfConfig` default uses `1e-4` (same value) |
| `wetspring_barracuda::bio::nmf::top_k_cosine` | Inlined into `validate_matrix_pharmacophenomics.rs` |
| Local dense matmul, Frobenius error, KL divergence | ToadStool uses identical algorithm |

**Affected binaries:** `validate_repodb_nmf`, `validate_nmf_drug_repurposing`,
`validate_matrix_pharmacophenomics` — all now require `--features gpu`.

### Ridge Regression Lean

| Before | After |
|---|---|
| `esn.rs` local `cholesky_factor()` + `solve_ridge()` (~100 lines) | `#[cfg(feature = "gpu")]`: `barracuda::linalg::ridge_regression()` |
| | `#[cfg(not(feature = "gpu"))]`: local fallback retained |

No change to ESN public API (`Esn::train`, `train_stateful`, `train_stateless`).

### ODE Systems Lean

| Before | After |
|---|---|
| `ode_systems.rs`: 5 local `OdeSystem` impls | `barracuda::numerical::ode_bio::{BistableOde, CapacitorOde, ...}` |
| `super::ode_systems::BistableOde` in GPU wrappers | `barracuda::numerical::BistableOde` (re-export) |
| Local `cpu_derivative()` + `wgsl_derivative()` | Identical implementation in ToadStool (absorbed from wetSpring) |

**Stays local:** `bio/bistable.rs`, `bio/capacitor.rs`, `bio/cooperation.rs`,
`bio/multi_signal.rs`, `bio/phage_defense.rs` — these contain scenario functions
and CPU integrators used by validation binaries.

### Anderson Correlated Disorder Lean

| Before | After |
|---|---|
| `validate_correlated_disorder.rs` local `LcgRng` + `build_correlated_anderson_3d()` (~115 lines) | `barracuda::spectral::anderson_3d_correlated(l, disorder, xi_corr, seed)` |

Same algorithm (exponential kernel smoothing + variance normalisation).

---

## Absorption Targets for ToadStool (Unchanged from V29)

### Track 3: Drug Repurposing GPU (Exp157-161)

| Needed Primitive | Purpose |
|---|---|
| `NmfUpdateGpu` | Lee & Seung multiplicative update shader |
| `SparseGemmGpu` | Sparse gene-drug GEMM |
| `TransEScoreGpu` | TransE embedding scoring |
| `WeightedNmfMaskGpu` | Masked NMF for missing data |

### Track 4: Passthrough → Compose

| Module | Awaits |
|---|---|
| `gbm_gpu` | `GbmBatchInferenceGpu` |
| `feature_table_gpu` | `FeatureExtractionGpu` |
| `signal_gpu` | `PeakDetectGpu` |

### Track 5: Math Feature Gate

5 local math functions in `special.rs` duplicate upstream (was 6; ridge now lean):

| Function | Upstream |
|---|---|
| `erf()` | `barracuda::special::erf` |
| `ln_gamma()` | `barracuda::special::ln_gamma` |
| `regularized_gamma_lower()` | `barracuda::special::regularized_gamma` |
| `trapz()` (in `eic.rs`) | `barracuda::numerical::trapz` |

Once ToadStool adds a `math` feature gate, wetSpring can remove these local
implementations and lean fully upstream.

---

## Verification

```
$ cargo fmt -- --check
  0 diffs

$ cargo clippy --all-targets --all-features -- -D clippy::pedantic -D clippy::nursery
  0 errors, 0 warnings

$ cargo test --features gpu
  759 passed; 0 failed; 9 ignored

$ cargo test --tests
  (no-gpu) 0 failures

$ cargo doc --no-deps --all-features
  0 warnings
```

---

## Cross-Spring Evolution: S58-S59 Absorption Cycle Complete

```
wetSpring (Write)          ToadStool (Absorb)           wetSpring (Lean)
─────────────────          ──────────────────           ─────────────────
bio/nmf.rs           ──→   linalg/nmf.rs (S58)    ──→   barracuda::linalg::nmf (V30)
bio/ode_systems.rs   ──→   numerical/ode_bio/ (S58) ──→  barracuda::numerical::*Ode (V30)
bio/esn.rs ridge     ──→   linalg/ridge.rs (S59)  ──→   barracuda::linalg::ridge (V30)
correlated anderson  ──→   spectral/anderson (S59) ──→   barracuda::spectral::* (V30)
```

This Lean pass removes ~1,312 lines of code that now exist upstream in ToadStool,
completing the full Write-Absorb-Lean cycle for S58-S59.
