# wetSpring → ToadStool/BarraCUDA Handoff: V57 ToadStool S68 Catch-Up

**Date:** February 26, 2026
**From:** wetSpring V57
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`) — advanced from S66 (`045103a7`)
**wetSpring:** 961 tests, 882 barracuda lib, 96.67% llvm-cov, 82 named tolerances, 189 experiments

---

## Executive Summary

V57 catches wetSpring up to ToadStool S68 (19 commits since S66 pin).
ToadStool's S67/S68 introduced universal precision architecture and evolved
all 291 f32-only shaders to f64 canonical. wetSpring revalidated cleanly
against S68 HEAD after fixing a CPU-build regression in ToadStool's
`numerical/mod.rs` and `stats/mod.rs` (feature-gate fix contributed upstream).
All 882 lib tests pass, zero clippy warnings, no API breakage.

---

## Part 1: ToadStool S67/S68 Changes Reviewed

### S67 — Universal Precision Architecture (Feb 24, 2026)

| Feature | Module | Status |
|---------|--------|--------|
| `compile_shader_universal(source, precision)` | `shaders::precision` | Available |
| `Precision::Df64` variant | `shaders::precision` | Available |
| `compile_template(template, precision)` | `shaders::precision` | Available |
| `downcast_f64_to_f32()` | `shaders::precision` | Available |
| `downcast_f64_to_f32_with_transcendentals()` | `shaders::precision` | Available |
| 12 universal shader templates | `shaders/templates/` | Available |

### S68 — Dual-Layer Universal Precision (Feb 26, 2026)

| Feature | Module | Status |
|---------|--------|--------|
| `Precision::op_preamble()` | `shaders::precision` | Available |
| `compile_op_shader()` | `shaders::precision` | Available |
| `downcast_f64_to_f16()` | `shaders::precision` | Available |
| `sovereign/df64_rewrite.rs` (naga-guided DF64) | `sovereign` | Available |
| 291 f32-only shaders → f64 canonical | All modules | Universal |
| 5 near-duplicate pairs consolidated | elementwise, sum/mean/std_dim | Cleaner |
| 122 shader tests (unit + e2e + chaos + fault) | tests | Validated |
| Zero f32-only shaders remaining | All | Milestone |

### Metrics at S68

| Metric | Value |
|--------|-------|
| WGSL shaders | 700 (0 orphans, 0 f32-only) |
| Barracuda tests | 2,546+ |
| Workspace tests | 21,599 |
| Clippy warnings | 0 |
| `unsafe` blocks | 2 (documented) |

---

## Part 2: Fix Contributed — CPU Feature-Gate Regression

### Problem

S68 added `crate::shaders::precision::downcast_f64_to_f32()` calls in
`numerical/mod.rs` (`wgsl_hessian_column()`) and `stats/mod.rs`
(`WGSL_HISTOGRAM`, `WGSL_BOOTSTRAP_MEAN_F64`). These modules are declared
as CPU-only in `lib.rs` (lines 80-86), but `crate::shaders` is gated behind
`#[cfg(feature = "gpu")]`. This breaks all `default-features = false`
consumers (wetSpring, and potentially other Springs).

### Fix

Added `#[cfg(feature = "gpu")]` to the three affected items:

```rust
// numerical/mod.rs
#[cfg(feature = "gpu")]
pub fn wgsl_hessian_column() -> &'static str { ... }

// stats/mod.rs
#[cfg(feature = "gpu")]
pub const WGSL_BOOTSTRAP_MEAN_F64: &str = ...;
#[cfg(feature = "gpu")]
pub static WGSL_HISTOGRAM: ... = ...;
```

### Verification

- wetSpring: 882 lib tests pass, 0 clippy warnings (pedantic)
- ToadStool: 2,568 tests pass (24 failures are pre-existing GPU-dependent tests on CPU-only machine)

---

## Part 3: wetSpring Consumption of S67/S68

### Already Working (unchanged APIs)

All 79 ToadStool primitives consumed by wetSpring continue to work without
modification. The universal precision architecture is backward-compatible:

- `compile_shader_f64()` still works (now delegates internally to universal precision)
- `BatchedOdeRK4::<S>::generate_shader()` still works
- All `barracuda::stats::*`, `barracuda::special::*`, `barracuda::linalg::*`,
  `barracuda::numerical::*`, `barracuda::spectral::*` unchanged
- GPU ops (`FusedMapReduceF64`, `BrayCurtisF64`, etc.) unchanged

### Available for Future Wiring

| Primitive | Benefit for wetSpring | Priority |
|-----------|----------------------|----------|
| `compile_shader_universal()` | Single compile call for any precision | Low — current `compile_shader_f64` works |
| `Precision::Df64` | DF64 precision for Anderson Lanczos at L ≥ 24 | Medium — Exp187 |
| `compile_template()` | Template-based shader compilation | Low |
| `downcast_f64_to_f16()` | F16 downcast for NPU quantization | Low — NPU uses int8 |
| `op_preamble()` | Abstract precision ops | Low |

### Cleaned

- `gpu.rs` doc comment updated: removed stale "3 local WGSL shaders" reference,
  replaced with accurate "zero local shaders, all generated via `BatchedOdeRK4`"

---

## Part 4: Recommendations for ToadStool

### Immediate (fix needed)

1. **Merge the feature-gate fix** for `numerical/mod.rs` and `stats/mod.rs`.
   Any `default-features = false` consumer is broken without it.

### Medium-term

1. **Audit other CPU-only modules** for similar `crate::shaders::` references
   that may have leaked during the S68 f32→f64 evolution waves.
   Pattern to check: any code in `error`, `linalg`, `numerical`, `special`,
   `tolerances`, `validation`, or `stats` modules that references `crate::shaders`.

2. **GPU Lanczos kernel** — wetSpring Exp184b validates Anderson L=14-20 on
   CPU with 16 disorder realizations. GPU acceleration would enable L=24+
   and 100+ realizations (Exp187 protocol ready).

3. **DF64 Lanczos** — combine `Precision::Df64` with Lanczos eigensolve for
   enhanced precision at large lattice sizes. The DF64 pipeline from S68 is
   perfectly suited for this.

---

## Part 5: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/wetSpring/barracuda

# Full test suite (882 tests)
cargo test --lib

# Clippy pedantic (0 warnings)
cargo clippy --lib --bins -- -W clippy::pedantic

# Build with GPU feature
cargo build --features gpu

# Specific ToadStool-consuming tests
cargo test --lib -- bio::diversity
cargo test --lib -- special::
cargo test --lib -- ncbi::
```

---

## Part 6: File Changes in V57

| File | Change |
|------|--------|
| `barracuda/src/gpu.rs` | Updated stale doc comment (local WGSL → zero local) |
| `(ToadStool) numerical/mod.rs` | `#[cfg(feature = "gpu")]` on `wgsl_hessian_column()` |
| `(ToadStool) stats/mod.rs` | `#[cfg(feature = "gpu")]` on `WGSL_HISTOGRAM`, `WGSL_BOOTSTRAP_MEAN_F64` |

---

## Part 7: Pin History

| Version | ToadStool Pin | Session | Key Changes |
|---------|--------------|---------|-------------|
| V57 | `f0feb226` | S68 | Universal precision catch-up, feature-gate fix |
| V56 | `045103a7` | S66 | Science pipeline, NCBI, NestGate, biomeOS |
| V55 | `045103a7` | S66 | Deep debt, idiomatic Rust |
| V54 | `045103a7` | S66 | Codebase audit, supply-chain |
| V53 | `045103a7` | S66 | Cross-spring evolution benchmarks |
| V44 | `02207c4a` | S62+DF64 | Complete cross-spring rewire |
| V40 | `02207c4a` | S62+DF64 | 55-commit catch-up |
