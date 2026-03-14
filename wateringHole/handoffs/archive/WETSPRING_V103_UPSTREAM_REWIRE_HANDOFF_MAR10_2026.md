# wetSpring V103 — Upstream Rewire & Modern Rust Evolution

**Date:** March 10, 2026
**From:** wetSpring V103
**To:** barraCuda team, toadStool team, coralReef team
**License:** AGPL-3.0-or-later
**Synced against:** barraCuda `a898dee`, toadStool S130+ (`bfe7977b`), coralReef Phase 10 (`d29a734`)

---

## Executive Summary

wetSpring V103 is an **upstream rewire** that catches up to the latest
barraCuda (v0.3.3, deep debt, typed errors, named constants), toadStool
S130+ (clippy pedantic, `#[expect]` evolution, spring sync), and coralReef
Phase 10 (AMD E2E verified, sovereign shader compilation).

**Key changes:**

1. **`#[allow(clippy::...)]` → `#[expect(clippy::...)]` evolution** across the
   entire workspace (~209 files). This is the ToadStool S131 pattern — stale
   suppressions now produce compile errors instead of silently hiding.

2. **37 stale suppressions discovered and removed** — the `#[expect]` evolution
   immediately caught 37 suppressions that were no longer needed. These included:
   - `cast_precision_loss` on functions that no longer cast
   - `too_many_lines` on functions that were refactored below threshold
   - `similar_names` where bindings were renamed
   - `needless_range_loop` where iterators were adopted
   - `unnecessary_wraps` where return types changed

3. **Hardcoded `/tmp/` paths → `std::env::temp_dir()`** for platform portability
   (6 paths evolved: 3 validation binaries, 1 production discovery path in
   `PetalTonguePushClient::discover()`, 1 test helper, 1 integration test).

4. **`#![deny(unsafe_code)]` → `#![forbid(unsafe_code)]`** in `barracuda/src/lib.rs`.
   `forbid` cannot be overridden by inner `#[allow]` — strongest possible guarantee.
   metalForge already had `#![forbid(unsafe_code)]`.

5. **Zero API breakage** against barraCuda HEAD — all 1,513 tests pass.

6. **Full deep debt audit completed:**
   - External dependencies: All pure Rust (`flate2` uses `rust_backend`). Only
     `wgpu` has native deps (unavoidable — WebGPU requires Vulkan/Metal/DX12).
   - File sizes: All under 1000 lines (wateringHole limit). Largest library
     file: `streaming_gpu.rs` at 686 lines.
   - Unsafe code: Zero. Both crates `forbid(unsafe_code)`.
   - Mocks: All isolated to `#[cfg(test)]`. Zero production mocks.
   - `unwrap()`/`expect()`: Zero in production library code. All in tests.
   - Hardcoding: All URLs use env var override with default fallback. All socket
     paths use capability-based discovery (env var → XDG → temp_dir cascade).
     Remaining `/tmp/` references are only in test assertions and doc comments.

---

## Changes Applied

### `#[expect]` Evolution (209 files)

Every `#[allow(clippy::...)]` and `#![allow(clippy::...)]` across both workspace
crates (`wetspring-barracuda`, `wetspring-forge`) has been evolved to
`#[expect(clippy::...)]` / `#![expect(clippy::...)]`.

Benefits:
- **Self-cleaning**: Any future barraCuda or Rust evolution that eliminates a lint
  will surface as a compile error ("unfulfilled lint expectation") instead of
  becoming invisible dead code.
- **Audit trail**: Every remaining suppression is provably needed.
- **Ecosystem alignment**: Follows ToadStool S131 pattern, groundSpring V98.

### Stale Suppression Removal (37 instances)

| File | Removed Lint | Reason |
|------|-------------|--------|
| `bio/alignment.rs` | `cast_possible_wrap`, `many_single_char_names` | Bindings reduced; re-added `many_single_char_names` with SW notation justification |
| `bio/chimera.rs` | `cast_precision_loss` | Cast no longer present |
| `bio/dada2/core.rs` | 3× `needless_range_loop`, `cast_precision_loss` | Loops restructured |
| `bio/derep.rs` | `cast_precision_loss` | Cast eliminated |
| `bio/esn/reservoir.rs` | `needless_range_loop` (module-level) | Iterators adopted |
| `bio/gillespie.rs` | `cast_possible_truncation` | Cast eliminated |
| `bio/ncbi_data/campy.rs` | `cast_precision_loss`, `cast_possible_truncation` | Casts eliminated |
| `bio/ncbi_data/vibrio.rs` | `cast_precision_loss`, `cast_possible_truncation` | Casts eliminated |
| `bio/pangenome.rs` | `cast_precision_loss` (module-level) | Now uses local `#[expect]` |
| `bio/qs_biofilm.rs` | `many_single_char_names` | Bindings renamed |
| `bio/quality/mod.rs` | `cast_possible_truncation` | Cast no longer present |
| `bio/snp.rs` | `cast_precision_loss` | Cast eliminated |
| `bio/unifrac/distance.rs` | 2× `cast_precision_loss` | Casts eliminated |
| `bio/taxonomy/classifier.rs` | `cast_precision_loss`, `cast_possible_truncation` | Tightened to specific functions |
| `ipc/handlers/science.rs` | `unnecessary_wraps` | Return type changed |
| `ipc/handlers/mod.rs` | `unnecessary_wraps` (module-level) | No longer needed |
| `visualization/scenarios/pipeline.rs` | `cast_precision_loss` | Cast eliminated |
| `bench/power.rs` | `cast_precision_loss` in multi-lint | Kept `significant_drop_tightening` |
| `bio/esn/npu.rs` | `cast_precision_loss` in multi-lint | Kept `cast_possible_truncation` |
| `metalForge/forge/src/node/assembly.rs` | `cast_possible_truncation` | `usize` → `u64` on 64-bit is no-op |
| 15 validation binaries | Various `too_many_lines`, `similar_names`, `cast_precision_loss` | Functions refactored or casts eliminated |

### Hardcoded Path Evolution (3 files)

| File | Old | New |
|------|-----|-----|
| `bin/dump_wetspring_scenarios.rs` | `/tmp/nonexistent-stream-demo.sock` | `std::env::temp_dir().join(...)` |
| `bin/validate_visualization_v2.rs` | `/tmp/nonexistent-viz-v2.sock` | `std::env::temp_dir().join(...)` (V101) |
| `metalForge/.../validate_viz_evolution_v1.rs` | `/tmp/nonexistent-exp333.sock` | `std::env::temp_dir().join(...)` |

---

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace -- -D warnings -W clippy::pedantic` | PASS (0 warnings) |
| `cargo test --workspace` | 1,513 tests PASS, 0 failures, 1 ignored |
| `cargo doc --workspace --no-deps` | 182+ pages generated |
| Zero `#[allow(clippy::...)]` remaining | Confirmed — all evolved to `#[expect]` |

---

## Upstream Sync Status

| Dependency | Commit | Status |
|-----------|--------|--------|
| barraCuda | `a898dee` (v0.3.3) | Compiles clean, zero API breakage |
| toadStool | `bfe7977b` (S130+) | Pattern-aligned (`#[expect]` evolution) |
| coralReef | `d29a734` (Phase 10) | No direct dependency; sovereign path ready |

---

## What Remains from barraCuda v0.3.3 Handoff

| Item | Status |
|------|--------|
| `SumReduceF64`/`VarianceReduceF64` Fp64Strategy fix | Available upstream; wetSpring consumes via `barracuda::` |
| `barracuda::math::{dot, l2_norm}` re-export | `dot`/`l2_norm` at `barracuda::stats::` (used); `barracuda::math::` does not include them yet |
| `Rk45DispatchArgs` for `BatchedOdeRK45F64` | Available at `barracuda::ops::rk45_adaptive::`; not yet consumed |
| `HmmForwardArgs` builder | Available; current usage still works |
| `Dada2DispatchArgs` builder | Available; current usage still works |
| `PrecisionRoutingAdvice` | Available at `barracuda::device::driver_profile::`; not yet consumed |
| Ada Lovelace `F64NativeNoSharedMem` | Available upstream; benefits Hybrid GPU users |
| `shared_mem_f64` runtime probe | Available upstream |
| `fused_ops_healthy()` canary | Available in `test_prelude` |
| `Fp64Strategy::Sovereign` | Future — when coralReef backend lands in barraCuda |

---

## Recommended Next Steps

1. **Adopt `BatchedOdeRK45F64`** — Replace fixed-step RK4 with adaptive Dormand-Prince
   for regulatory network ODE systems (phage defense, bistable, cooperation).
2. **Adopt `PrecisionRoutingAdvice`** — Query GPU driver profile for dispatch decisions
   in GPU validation binaries.
3. **Adopt builder types** — `HmmForwardArgs`, `Dada2DispatchArgs`, `Rk45DispatchArgs`
   for improved readability (positional args → named fields).
4. **`Fp64Strategy::Sovereign`** — When barraCuda adds `ComputeDispatch::CoralReef`,
   adopt sovereign compilation path.
