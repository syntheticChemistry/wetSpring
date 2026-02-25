# wetSpring → ToadStool Handoff V35: DF64 Evolution Lean + BGL Helpers

**Date:** February 25, 2026
**From:** wetSpring (Phase 42, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**ToadStool:** S62+DF64 (post-S62 DF64 expansion commits)

---

## Summary

wetSpring reviewed ToadStool's post-S62 evolution (3 commits: DF64 core-streaming
for HMC, GEMM DF64, Lennard-Jones DF64, ComputeDispatch builder, BGL helpers,
`gpu_ctx()`, and `unified_hardware` refactor). We adopted the applicable pieces:

1. **`storage_bgl_entry` / `uniform_bgl_entry`** — adopted in 6 files (5 ODE GPU +
   `gemm_cached.rs`), replacing ~258 lines of manual BGL entry boilerplate
2. **`GemmF64::WGSL` via `compile_shader_f64`** — simplified from
   `ShaderTemplate::for_driver_auto(GEMM_WGSL, ...)` to direct
   `device.compile_shader_f64(GemmF64::WGSL, ...)` (1 file, 3 lines)
3. **Identified DF64 GEMM adoption blocker** — `wgsl_shader_for_device()` and
   DF64 shader sources are private; wetSpring cannot auto-select DF64 yet

## What We Adopted

| File | What changed | Lines saved |
|------|-------------|:-----------:|
| `src/bio/gemm_cached.rs` | `storage_bgl_entry`/`uniform_bgl_entry`, `compile_shader_f64` | ~48 |
| `src/bio/bistable_gpu.rs` | `storage_bgl_entry`/`uniform_bgl_entry` in `bgl_entries()` | ~42 |
| `src/bio/capacitor_gpu.rs` | Same | ~42 |
| `src/bio/cooperation_gpu.rs` | Same | ~42 |
| `src/bio/multi_signal_gpu.rs` | Same | ~42 |
| `src/bio/phage_defense_gpu.rs` | Same | ~42 |
| **Total** | **6 files** | **~258** |

## What We Didn't Adopt (and Why)

### ComputeDispatch builder
Our ODE GPU modules cache the pipeline at `new()` and reuse it across `integrate()`
calls. `ComputeDispatch` is a one-shot builder (shader → pipeline → dispatch → submit
in one chain). Our pattern needs the pipeline to outlive the dispatch call. The BGL
helpers give us the 80% cleanup without requiring architectural change.

### `BarracudaError::gpu_ctx()`
wetSpring uses `crate::error::Error::Gpu(format!(...))`, not `BarracudaError`. Adopting
`gpu_ctx()` would require either switching to `BarracudaError` across 27 files (~95
call sites) or adding a parallel helper on our `Error` type. Not worth the churn for
a formatting helper.

### DF64 GEMM auto-selection
`GemmF64::wgsl_shader_for_device()` is private. The DF64 GEMM WGSL source
(`gemm_df64.wgsl`) is only accessible via private `const` inside `gemm_f64.rs`. We
can't replicate the auto-selection in our local `GemmCached` until ToadStool either:
1. Makes `wgsl_shader_for_device()` public, or
2. Exposes `WGSL_DF64` as a public constant like `GemmF64::WGSL`

**Recommended**: Make `GemmF64::wgsl_shader_for_device()` public — all downstream
consumers that cache GEMM pipelines need this.

### `unified_hardware` refactor
wetSpring doesn't use `unified_hardware` directly. The refactor from 1012-line
monolith to 6 focused modules is internal to ToadStool.

## Bug Report: PeakDetectF64 WGSL Shader

**File:** `crates/barracuda/src/shaders/signal/peak_detect_f64.wgsl`, line 49
**Bug:** `prominence[idx] = 0.0;` assigns f32 literal to f64 array
**Error:** `naga: The type of [48] doesn't match the type stored in [47]`
**Impact:** `PeakDetectF64::execute()` panics on shader validation
**Fix:** Change `0.0` to `0.0lf` (WGSL f64 literal) or `f64(0.0)`

This bug was present in S62 and remains unfixed in the latest commit. wetSpring's
`validate_gpu_drug_repurposing.rs` (Exp164) gracefully handles this with
`catch_unwind` and falls back to CPU peak detection.

## Cross-Spring Evolution Notes

### DF64 core-streaming is significant
The DF64 expansion (HMC, GEMM, Lennard-Jones) demonstrates that FP32-core routing
is viable for f64 workloads. For wetSpring's drug repurposing pipeline (200×150 NMF
matrices), DF64 GEMM would give ~10x throughput on consumer GPUs. This directly
benefits Track 3 validation.

### What wetSpring needs from ToadStool next
1. **Public DF64 GEMM shader access** (or public `wgsl_shader_for_device()`)
2. **PeakDetect f64 literal fix** (trivial one-line fix)
3. **`ComputeDispatch` with cached pipeline variant** — a builder that returns the
   pipeline + BGL for reuse across calls (not just one-shot)

## Validation

```
cargo fmt    — clean
cargo clippy — 0 warnings (pedantic + nursery, --features gpu)
cargo test   — 752 passed (CPU-only)
validate_metalforge_drug_repurposing — 9/9 PASS
benchmark_ode_lean_crossspring       — 11/11 PASS
validate_gpu_drug_repurposing        — 8/8 PASS
```

## Status

| Metric | Value |
|--------|-------|
| ToadStool alignment | **S62+DF64** |
| Primitives consumed | **44** (+ 2 BGL helpers) |
| Local WGSL shaders | **0** |
| BGL boilerplate removed | **~258 lines** |
| Tests | 752 CPU / 759 GPU |
| Experiments | 165 |
| Validation checks | 3,242+ |
