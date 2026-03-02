SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring → ToadStool/BarraCUDA Handoff V92D — Deep Debt Evolution + Pedantic Hardening

**Date**: March 2, 2026
**From**: wetSpring (V92D)
**To**: ToadStool/BarraCUDA team
**ToadStool pin**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later
**Supersedes**: V92C (Deep Audit & GPU Test Evolution)

---

## Executive Summary

- **Zero panics in library code**: ESN bridge `block_on()` converted from `panic!` to
  `Result<O, BarracudaError>` with `??` chaining at all 6 call sites. This was the last
  `panic!` path in non-test library code.
- **IPC handler refactor**: 8-argument `insert_metric_if_requested` → `MetricCtx` struct
  with `dispatch_metric` helper. Follows clippy pedantic `too_many_arguments` resolution.
  Pattern recommended for ToadStool GPU dispatch wrappers with similar argument counts.
- **Modern idiomatic Rust throughout**: `mul_add` for fused multiply-add, `f64::from` for
  lossless casts, `f64::midpoint` for precision-correct midpoints, inlined format args,
  `if let` over single-arm `match`, `RangeInclusive::contains` over manual comparisons.
- **Full `--all-features` pedantic clean**: `cargo clippy --workspace --all-targets
  --all-features -- -D warnings -W clippy::pedantic` passes with zero warnings. Previous
  V92C was clean per-target; V92D is clean for the full feature matrix.
- **1,309 tests pass** (up from 1,276 at V92C). No failures, 1 hardware-dependent ignore.

---

## Part 1: Panic Elimination (Library Code)

### Before (V92C)

```rust
fn block_on<F, O>(f: F) -> O
where F: Future<Output = O> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map(|rt| rt.block_on(f))
        .unwrap_or_else(|e| panic!("tokio runtime: {e}"))
}
```

### After (V92D)

```rust
fn block_on<F, O>(f: F) -> Result<O, BarracudaError>
where F: Future<Output = O> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map(|rt| rt.block_on(f))
        .map_err(|e| BarracudaError::Gpu(format!("tokio runtime: {e}")))
}
```

All 6 call sites use `block_on(async_fn())?? ` — the outer `?` handles runtime
creation, the inner `?` handles the async operation's own `Result`.

### Relevance to ToadStool

ToadStool's `barracuda` crate has similar patterns in GPU initialization paths.
Any `panic!` in library code makes error recovery impossible for downstream consumers.
The `block_on → Result` pattern should be applied to any ToadStool runtime init.

---

## Part 2: MetricCtx Pattern (Handler Refactor)

### Before (V92C — in science.rs)

```rust
fn insert_metric_if_requested(
    metrics: &[&str],
    out: &mut Map<String, Value>,
    name: &str,
    counts: &[f64],
    cpu_fn: fn(&[f64]) -> f64,
    gpu: &Option<GpuF64>,
    gpu_method: &str,
    threshold: usize,
) { ... }
```

### After (V92D)

```rust
struct MetricCtx<'a> {
    metrics: &'a [&'a str],
    out: &'a mut Map<String, Value>,
    counts: &'a [f64],
    gpu: &'a Option<GpuF64>,
    threshold: usize,
}

impl MetricCtx<'_> {
    fn insert(&mut self, name: &str, compute: impl FnOnce() -> f64) { ... }
}

fn dispatch_metric(
    counts: &[f64],
    cpu_fn: fn(&[f64]) -> f64,
    gpu_result: Option<f64>,
) -> f64 {
    gpu_result.unwrap_or_else(|| cpu_fn(counts))
}
```

### Relevance to ToadStool

ToadStool's `ComputeDispatch` builder already avoids the many-argument pattern for
GPU dispatch. The `MetricCtx` pattern is complementary for IPC/service layers where
dispatch context (GPU device, threshold, output map) is shared across multiple metric
insertions. Consider a generic `DispatchCtx<D>` in barracuda if Springs share this pattern.

---

## Part 3: Modern Rust Idioms Applied

| Idiom | Before | After | Files |
|-------|--------|-------|:-----:|
| `mul_add` | `1.0 + f64::from(i) * 0.5` | `f64::from(i).mul_add(0.5, 1.0)` | 1 |
| `f64::from` | `i as f64` (for `i: i32`) | `f64::from(i)` | 4 |
| `f64::midpoint` | `(a + b) / 2.0` | `f64::midpoint(a, b)` | 2 |
| Inlined format | `println!("{}", x)` | `println!("{x}")` | 6 |
| `if let` | `match (Ok(a), Ok(b)) { ... }` | `if let (Ok(a), Ok(b)) = ...` | 1 |
| `contains` | `x >= 0.0 && x <= 1.0` | `(0.0..=1.0).contains(&x)` | 2 |
| Doc backticks | `barracuda::bio` in docs | `` `barracuda::bio` `` | 50+ |

### Relevance to ToadStool

These are Rust 2024 edition best practices enforced by `clippy::pedantic`. ToadStool
should adopt the same `--all-features -W clippy::pedantic` gate. Key wins:

- `mul_add` is not just style — it uses hardware FMA when available, giving both
  better precision (single rounding) and better performance
- `f64::midpoint` avoids overflow for large values (`(f64::MAX + f64::MAX) / 2.0`
  overflows; `f64::midpoint` does not)
- `f64::from` over `as` prevents silent truncation on wider integer types

---

## Part 4: Shared Helpers

### `validation::bench<T>()`

```rust
pub fn bench<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    (result, ms)
}
```

Used by validation binaries for consistent timing. Returns both the computation
result and elapsed milliseconds. Consider absorbing into ToadStool's validation
harness if other Springs need the same pattern.

---

## Part 5: Barracuda Usage Census (V92D)

### Consumed ToadStool Primitives: 93

| Category | Primitives | Count |
|----------|-----------|:-----:|
| Stats | mean, variance, shannon, simpson, bray_curtis, chao1, pielou, norm_cdf, hill, fit_linear, percentile | 11 |
| Linalg | gemm, nmf, ridge, svd | 4 |
| Sparse | sparse_gemm_f64 | 1 |
| Spectral | anderson_1d/2d/3d, lanczos, level_spacing_ratio, spectral_bandwidth, spectral_condition_number, classify_spectral_phase | 8 |
| Numerical | ode_bio (5 systems), rk4, rk45 | 7 |
| Bio | diversity (shannon/simpson/bray_curtis/chao1/pielou → upstream), DADA2, chimera, taxonomy, unifrac | 9+ |
| Graph | graph_laplacian, effective_rank, disordered_laplacian, belief_propagation_chain, boltzmann_sampling | 5 |
| ML | transe_score_f64, peak_detect_f64 | 2 |
| ESN | MultiHeadEsn, esn_v2 (bridge) | 2 |
| GPU infra | compile_shader_universal, ComputeDispatch, storage_bgl_entry, uniform_bgl_entry, GpuF64 | 5 |
| Other | Various cross-spring primitives (hotSpring precision, neuralSpring ML, airSpring hydro, groundSpring bootstrap) | 39 |

### Zero Local Math

- 0 local WGSL shaders
- 0 local ODE derivative functions
- 0 local regression implementations
- 0 local diversity metric implementations
- 0 unsafe code blocks
- 0 C dependencies (CPU build); 1 transitive C dep (renderdoc-sys via wgpu, GPU build)

---

## Part 6: API Pain Points (Unchanged from V92C, Still Relevant)

1. **`FitResult.params: Vec<f64>`** — positional access is error-prone. Named fields
   (slope, intercept, r_squared) would prevent indexing bugs.
2. **`SpectralAnalysis::from_eigenvalues(eigenvalues, gamma)`** — the `gamma` parameter
   is always 1.0 for bio/ecology use. A `from_eigenvalues_default` or builder would help.
3. **GPU test mock** — `MockGpuF64` returning CPU results would enable CI testing without
   hardware. Currently 32+ wetSpring GPU tests require `#[ignore]` on CI.
4. **`MultiHeadEsn::from_exported_weights()`** — constructor for importing pre-trained
   weights (requested by hotSpring too). Currently only `new()` is available.

---

## Part 7: Absorption Opportunities for ToadStool

### From This Session

| Pattern | Where | Benefit |
|---------|-------|---------|
| `MetricCtx` struct for dispatch | `ipc/handlers/science.rs` | Reduces argument counts in dispatch wrappers |
| `block_on → Result` | `bio/esn/toadstool_bridge.rs` | Eliminates panics in runtime init |
| `bench<T>()` helper | `validation.rs` | Consistent timing in validation harnesses |
| Provenance classification headers | All 249 validators | Standardized categorization for CI |

### From Full wetSpring History

| What | Lines | Tests | Notes |
|------|:-----:|:-----:|-------|
| 103 named tolerance constants | ~250 | 103 (hierarchy test) | Bio-specific, but naming convention is general |
| `BioBrain` attention state machine | ~250 | 6 | Domain-agnostic brain adapter |
| IPC discover module | ~100 | 4 | Capability-based primal discovery |
| Provenance headers (standard) | — | — | `//! Validation class:` + `//! Provenance:` |

---

## Part 8: What wetSpring Does NOT Need from ToadStool

| Category | Reason |
|----------|--------|
| New GPU primitives | Fully lean on 93; no Tier C (Write) items |
| Local WGSL support | Zero local shaders; all generated via traits |
| DF64 dual-layer | Not currently needed; f64 precision sufficient |
| `BandwidthTier` | metalForge dispatch handles bandwidth routing locally |

---

## Part 9: Quality Gate

| Gate | Status |
|------|--------|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic` | PASS (zero warnings) |
| `cargo test --workspace` | 1,309 passed, 0 failed, 1 ignored |
| `cargo doc --workspace --no-deps` | PASS |
| Zero `unsafe` code | PASS |
| Zero `panic!` in library code | PASS (V92D: last panic eliminated) |
| Zero `todo!()`/`unimplemented!()` | PASS |
| AGPL-3.0-or-later headers | PASS |

---

## Part 10: Recommended ToadStool Evolution Priorities (wetSpring Perspective)

1. **`MultiHeadEsn::from_exported_weights()`** — unblocks both hotSpring and wetSpring
   ESN GPU migration. Highest-impact single API addition.
2. **`FitResult` named fields** — prevents indexing bugs across all Springs.
3. **`MockGpuF64`** — enables CI testing of GPU dispatch paths without hardware.
4. **Pedantic clippy gate** — adopt `--all-features -W clippy::pedantic` in ToadStool CI.
   Modern idioms (`mul_add`, `f64::from`, `f64::midpoint`) improve both precision and safety.
5. **`SpectralAnalysis` default gamma** — minor API improvement, benefits wetSpring + neuralSpring.

---

*wetSpring V92D — 1,309 tests, 272 experiments, 7,220+ checks, 255 binaries,
93 ToadStool primitives, 0 local WGSL, 0 unsafe, 0 panics. Fully lean.*
