# SPDX-License-Identifier: AGPL-3.0-only

# wetSpring V97 → barraCuda v0.3.3 + wgpu 28 Rewire Handoff

**Date:** 2026-03-05
**From:** wetSpring V97 (1,047 lib tests + 200 forge tests, zero clippy warnings)
**To:** barraCuda team (math primitives), toadStool team (hardware dispatch)
**License:** AGPL-3.0-only

---

## Executive Summary

wetSpring has completed a full rewire to barraCuda v0.3.3 and wgpu 28. This
is the second major version bump since the standalone extraction (v0.3.1 →
v0.3.3). All 1,247 tests pass. Zero clippy warnings (pedantic+nursery). Zero
code duplication with upstream. This handoff documents what changed, what we
found, and what we recommend.

---

## 1. What Changed

### 1.1 wgpu 22 → 28 Migration

| Pattern | Files | Change |
|---------|-------|--------|
| `wgpu::Maintain::Wait` → `wgpu::PollType::Wait { submission_index: None, timeout: None }` | 20 | wgpu 28 `PollType` is struct variant; `poll()` returns `Result` |
| `Instance::new(desc)` → `Instance::new(&desc)` | 2 | wgpu 28 takes reference |
| `request_adapter().await` | 1 | Returns `Result` (was `Option`) |
| `DeviceDescriptor` fields | 1 | Added `experimental_features`, `trace` |
| `request_device(desc, trace)` → `request_device(desc)` | 1 | `trace` absorbed into descriptor |
| `Arc::new(device)` / `Arc::new(queue)` | 1 | wgpu 28 Device/Queue are internally Arc'd |
| `enumerate_adapters()` | 1 | Now async; wrapped in `pollster::block_on()` |

### 1.2 Dependency Updates

| Crate | Old | New | Notes |
|-------|-----|-----|-------|
| `wgpu` (wetspring-barracuda) | 22 | 28 | Matches upstream barraCuda |
| `wgpu` (wetspring-forge) | 22 | 28 | Same |
| `pollster` (wetspring-forge) | — | 0.4 | New dep for async `enumerate_adapters` |

### 1.3 Cargo.toml Fixes

- `validate_emp_anderson_atlas` bin now has `required-features = ["ipc"]`
  (was compiling unconditionally but using gated `ipc` module)
- Barracuda version comments updated from v0.3.1 to v0.3.3

---

## 2. Upstream Bug Found and Fixed

**File:** `barraCuda/crates/barracuda/src/special/chi_squared.rs`
**Issue:** `use crate::device::capabilities::WORKGROUP_SIZE_1D;` at module scope
without `#[cfg(feature = "gpu")]` guard. Broke CPU-only builds (`default-features = false`).
**Fix:** Added `#[cfg(feature = "gpu")]` to the import. The struct `ChiSquaredBatchGpu`
and its `impl` were already properly gated.

This was committed directly in the local barraCuda checkout. The barraCuda team
should adopt this fix upstream.

---

## 3. New Primitives Available (Not Yet Consumed)

barraCuda v0.3.3 (unreleased head) added these primitives that wetSpring could
adopt in future phases:

| Primitive | Benefit |
|-----------|---------|
| `VarianceF64::mean_variance()` | Fused single-pass Welford (saves a dispatch vs separate mean + variance) |
| `CorrelationF64` fused shader | 5-accumulator Pearson in one dispatch |
| `ChiSquaredBatchGpu` | Batch chi-squared PDF/CDF on GPU (for enrichment/pangenomics) |
| DF64 fused shaders | `mean_variance_df64.wgsl`, `correlation_full_df64.wgsl` |
| TensorContext fast path | 15+ ops with pooled buffers and pipeline cache |
| `Fp64Strategy` dispatch | Automatic Native/DF64 selection per GPU hardware |

wetSpring already uses `VarianceF64`, `CorrelationF64`, `CovarianceF64`, and
`WeightedDotF64` via thin wrappers in `stats_gpu.rs`. These wrappers will
automatically benefit from the upstream TensorContext migration and DF64
support without code changes.

---

## 4. Zero Duplication Confirmed

wetSpring has zero reimplementation overlap with barraCuda:

- All 32 `ops/bio/*` modules in barraCuda were absorbed from wetSpring (handoff v4–v8)
- wetSpring's GPU modules are thin wrappers calling barraCuda primitives
- No local WGSL shaders (full lean since V50)
- No local derivative/regression math

---

## 5. Quality Gates

| Gate | Status |
|------|--------|
| `cargo check` (CPU) | PASS |
| `cargo check --features gpu` | PASS |
| `cargo check --all-features` | PASS |
| `cargo test -p wetspring-barracuda` | 1,047 passed, 0 failed |
| `cargo test -p wetspring-forge` | 200 passed, 0 failed |
| `cargo clippy --features gpu -- -W clippy::pedantic -W clippy::nursery` | ZERO warnings |
| `cargo clippy -p wetspring-forge -- -W clippy::pedantic -W clippy::nursery` | ZERO warnings |
| `cargo fmt --check` | PASS |
| `cargo doc --all-features --no-deps` | PASS (273 files) |

---

## 6. Recommendations for barraCuda Team

1. **Adopt `chi_squared.rs` GPU gate fix** — the `WORKGROUP_SIZE_1D` import needs
   `#[cfg(feature = "gpu")]` for CPU-only consumers.
2. **Consider CHANGELOG version bump** — the unreleased changes (fused shaders,
   TensorContext migration, Fp64Strategy, DF64 naga rewriter fix) represent
   significant evolution beyond the v0.3.3 tag.
3. **wgpu 28 breaking change doc** — the existing `BREAKING_CHANGES.md` is excellent;
   `PollType::Wait` struct variant caught us (the error message is helpful but the
   migration path wasn't in the doc yet).

---

*This handoff is unidirectional: wetSpring → barraCuda/toadStool. No response expected.*
