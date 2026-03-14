# wetSpring V109 — Upstream Rewire + NUCLEUS Atomics: barraCuda/toadStool Evolution Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date**: 2026-03-10
**From**: wetSpring V109 (Eastgate)
**To**: barraCuda / toadStool / coralReef
**License**: AGPL-3.0-or-later
**Covers**: Upstream API migration (SpringDomain, sync GPU, DADA2), mixed hardware validation (NPU→GPU PCIe bypass), NUCLEUS atomic coordination (Tower/Node/Nest via biomeOS graph), evolution feedback, absorption opportunities

---

## Executive Summary

- wetSpring V109 validates 6 new experiments (Exp347-352) with **145/145 PASS** proving upstream rewire correctness, mixed hardware dispatch, and NUCLEUS atomic deployment
- **Upstream API breaks handled**: SpringDomain→SCREAMING_SNAKE_CASE, `shannon_gpu` async→sync, `ln_gamma` returns `Result<f64>`, `variance` not exported (use `covariance(x,x)`), `plasma_dispersion`/`spectral::stats` now require `gpu` feature
- **Full 8-tier chain validated**: Paper Math → CPU → Python Parity → GPU → ToadStool Dispatch → Streaming → metalForge (mixed HW) → NUCLEUS (biomeOS graph)
- **NUCLEUS Tower/Node/Nest all READY**: biomeOS graph execution coordinates cross-track QS analysis
- IPC dispatch ~117µs/call, bit-exact vs direct function calls
- Test suite: 1,151/1,154 pass (3 known pre-existing GPU f32 failures)
- `cargo clippy --features gpu,ipc` — **ZERO warnings**

---

## Part 1: What Changed (V109)

| Change | Impact |
|--------|--------|
| `SpringDomain` → SCREAMING_SNAKE_CASE | 5 files migrated (`provenance.rs`, 2 cross-spring validators, GPU v14, provenance validator) |
| `shannon_gpu` async→sync | Returns `Result<f64>` directly; `rt.block_on()` removed from callers |
| `GpuF64::new()` remains async | Still needs `rt.block_on(GpuF64::new())` for GPU init |
| `GPU_CPU_PARITY` → `GPU_VS_CPU_F64` | Tolerance constant renamed; updated in all GPU validators |
| `ln_gamma` returns `Result<f64>` | Callers need `.expect()` or `?` |
| `variance` not publicly exported | Use `covariance(x, x)` for sample variance |
| `plasma_dispersion` requires `gpu` | Non-GPU builds must exclude spectral/plasma modules |
| DADA2 duplicate import fix | Removed `init_error_model` from `pub(crate)` re-export |
| NPU→GPU PCIe bypass validated | metalForge v19: direct transfer without CPU roundtrip |
| NUCLEUS v4 validated | Tower/Node/Nest atomics via biomeOS IPC graph execution |

---

## Part 2: Upstream API Evolution Detail

### 2.1 — SpringDomain Migration

barraCuda changed `SpringDomain` from an enum with PascalCase variants to a struct with `SCREAMING_SNAKE_CASE` associated constants:

```rust
// OLD (broken):
SpringDomain::WetSpring
SpringDomain::HotSpring

// NEW (current):
SpringDomain::WET_SPRING
SpringDomain::HOT_SPRING
SpringDomain::NEURAL_SPRING
```

**toadStool action:** Ensure all springs consuming `shaders::provenance` migrate to the new API. wetSpring has completed this migration.

### 2.2 — GPU Diversity Sync API

`shannon_gpu` and `simpson_gpu` are now synchronous functions returning `Result<f64>`:

```rust
// OLD:
let h = rt.block_on(shannon_gpu(&gpu, &counts)).expect("GPU");

// NEW:
let h = shannon_gpu(&gpu, &counts).expect("GPU");
```

`GpuF64::new()` remains async — device creation still needs tokio runtime.

**toadStool action:** Update any dispatch code that wraps `shannon_gpu`/`simpson_gpu` in `block_on`.

### 2.3 — Tolerance Constants

```rust
// OLD:
tolerances::GPU_CPU_PARITY  // removed

// NEW:
tolerances::GPU_VS_CPU_F64           // 1e-6 (standard)
tolerances::GPU_VS_CPU_TRANSCENDENTAL // 1e-10 (erf, ln)
tolerances::GPU_VS_CPU_BRAY_CURTIS   // 1e-10 (BC pairs)
tolerances::GPU_VS_CPU_ENSEMBLE      // 1e-4 (stochastic)
tolerances::GPU_VS_CPU_HMM_BATCH     // 1e-3 (HMM forward)
```

### 2.4 — Feature Gate Changes

`plasma_dispersion` and `spectral::stats` now depend on `ops` and `linalg::eigh` respectively, which are `#[cfg(feature = "gpu")]`. Non-GPU builds that use spectral stats will fail to compile.

**toadStool action:** Consider whether `spectral::stats::level_spacing_ratio` should have a CPU-only fallback path. Currently it requires the `eigh` eigendecomposition which is GPU-gated.

---

## Part 3: Primitive Consumption Summary (V109)

wetSpring consumes **150+ barraCuda primitives** across these categories:

| Category | Examples | V109 Status |
|----------|----------|-------------|
| Stats | `mean`, `bootstrap_ci`, `jackknife_mean_variance`, `pearson_correlation`, `spearman_correlation`, `fit_linear`, `fit_exponential`, `covariance`, `norm_cdf` | All validated (Exp347 D65) |
| Linalg | `graph_laplacian`, `effective_rank`, `ridge_regression` | All validated (Exp347 D66) |
| Special | `erf`, `ln_gamma` (now `Result`) | All validated (Exp347 D67) |
| Numerical | `trapz` (was `trapezoidal`) | Validated (Exp349 S10) |
| GPU diversity | `FusedMapReduceF64`, `BrayCurtisF64` (via `shannon_gpu`, `simpson_gpu`) | Sync API validated (Exp348 D43) |
| Provenance | `shaders::provenance::SpringDomain::WET_SPRING`, `shaders_from`, `shaders_consumed_by`, `cross_spring_matrix`, `report::evolution_report`, `report::shader_count` | SCREAMING_SNAKE_CASE validated (Exp347 D68) |

---

## Part 4: NUCLEUS Atomic Deployment Findings

### 4.1 — Tower/Node/Nest Readiness

| Atomic | Components | Status |
|--------|-----------|--------|
| Tower | BearDog + Songbird | READY |
| Node | Tower + ToadStool | READY |
| Nest | Tower + NestGate + Squirrel | READY |
| Full | All primals | READY |

### 4.2 — IPC Dispatch Performance

| Metric | Value |
|--------|-------|
| Direct dispatch (Shannon, 200 taxa) | ~0.01µs/call |
| IPC dispatch (JSON-RPC, same) | ~117µs/call |
| Overhead ratio | ~10,000× (micro-benchmark; real workloads much lower) |
| Accuracy | Bit-exact (`EXACT_F64` tolerance) |

**toadStool action:** For sub-microsecond operations, IPC overhead dominates. Consider batch dispatch (send N operations per IPC call) or shared-memory IPC for latency-sensitive paths.

### 4.3 — biomeOS Graph Execution

Cross-track QS analysis (T6 anaerobic + T4 soil + T1 algae) validated through biomeOS graph coordination. Bray-Curtis distance matrix computed across all 3 tracks via single graph invocation.

---

## Part 5: Absorption Opportunities

| Item | Source | Recommendation |
|------|--------|---------------|
| Biogas kinetics (Gompertz, first-order, Monod, Haldane) | wetSpring `validate_barracuda_cpu_v26/v27` | Absorb into `barracuda::bio::kinetics` module |
| Anderson W mapping | wetSpring diversity → disorder → P(QS) pipeline | Absorb into `barracuda::bio::anderson` module |
| Batch dispatch | NUCLEUS IPC overhead analysis | Add `dispatch_batch("science.diversity", [params1, params2, ...])` |
| Spectral stats CPU fallback | `level_spacing_ratio` requires GPU `eigh` | Add CPU eigendecomposition path |
| Variance export | `barracuda::stats::variance` exists but not re-exported | Add to `pub use correlation::{...}` block |

---

## Part 6: Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Absorb Gompertz/Monod/Haldane into `barracuda::bio::kinetics` | toadStool | Medium |
| 2 | Re-export `variance` from `barracuda::stats` | barraCuda | Low |
| 3 | Consider CPU fallback for `spectral::stats` | barraCuda | Medium |
| 4 | Batch IPC dispatch for micro-operations | biomeOS | Low |
| 5 | Verify all springs migrated to SCREAMING_SNAKE_CASE `SpringDomain` | toadStool | High |
| 6 | Update `shannon_gpu`/`simpson_gpu` dispatch wrappers for sync API | toadStool | High |

---

## Appendix: V109 Validation Results

| Exp | Name | Checks | Status |
|-----|------|--------|--------|
| 347 | BarraCuda CPU v27 — upstream rewire + Track 6 | 39/39 | PASS |
| 348 | CPU vs GPU v11 — sync diversity API + GPU parity | 19/19 | PASS |
| 349 | ToadStool Dispatch v4 — compute dispatch (6 sections) | 32/32 | PASS |
| 350 | Pure GPU Streaming v13 — 7-stage pipeline | 17/17 | PASS |
| 351 | metalForge v19 — NPU→GPU PCIe + CPU fallback | 22/22 | PASS |
| 352 | NUCLEUS v4 — Tower/Node/Nest + biomeOS graph | 16/16 | PASS |
