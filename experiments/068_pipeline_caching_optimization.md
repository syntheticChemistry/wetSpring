# Exp068: Pipeline Caching Optimization — Local WGSL Modules

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross/GPU
**Binary:** `benchmark_dispatch_overhead` (re-run before/after)

---

## Objective

Refactor all 6 local WGSL GPU modules to cache compiled shader pipelines
at initialization, eliminating per-call shader compilation overhead.
This follows the same pattern used by `QualityFilterCached` and `GemmCached`,
making the modules more absorption-ready for ToadStool.

---

## Modules Refactored

| Module | Shader | Polyfill | Before | After | Improvement |
|--------|--------|----------|-------:|------:|:-----------:|
| `ani_gpu` | `ani_batch_f64.wgsl` | No | 5,855µs | 5,398µs | -8% |
| `snp_gpu` | `snp_calling_f64.wgsl` | No | 10,169µs | 6,384µs | -37% |
| `dnds_gpu` | `dnds_batch_f64.wgsl` | Yes (log) | 9,900µs | 4,703µs | -52% |
| `pangenome_gpu` | `pangenome_classify.wgsl` | No | 5,788µs | 3,043µs | -47% |
| `random_forest_gpu` | `rf_batch_inference.wgsl` | No | 2,493µs | 1,600µs | -36% |
| `hmm_gpu` | `hmm_forward_f64.wgsl` | Yes (exp/log) | 5,086µs | 3,229µs | -37% |
| **Average** | | | **5,078µs** | **3,137µs** | **-38%** |

ToadStool primitives (Shannon FMR, Bray-Curtis) already had this optimization:
Shannon 994→546µs (-45%), Bray-Curtis 335→193µs (-42%).

---

## Pattern

**Before**: each `batch_xxx()` call compiled the WGSL shader, created the
compute pipeline, and extracted the bind group layout. This added ~3-5ms
of shader compilation overhead per call.

**After**: `new()` compiles the shader and caches `pipeline` + `bgl` in
the struct. Subsequent calls only create buffers, bind group, and dispatch.

```rust
// Cached pipeline — compiled once in new()
pub struct XxxGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
```

This matches the `QualityFilterCached` / `GemmCached` / `FusedMapReduceF64`
pattern already used by the streaming pipeline (`GpuPipelineSession`).

---

## Remaining Overhead (~3ms)

After pipeline caching, the remaining dispatch overhead is:
- Buffer creation (`create_buffer_init`): ~1-2ms
- Bind group creation: ~0.1ms
- Command encode + submit: ~0.1ms
- GPU poll/sync + readback: ~1-2ms

This is per-dispatch and cannot be cached (buffer sizes vary with input).
ToadStool's streaming dispatch amortizes this by chaining stages.

---

## Correctness

All existing validation binaries pass after the refactoring:
- Exp064: 26/26 PASS
- Exp065: 35/35 PASS

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline tool | Exp067 dispatch overhead (before) |
| Exact command | `cargo run --features gpu --release --bin benchmark_dispatch_overhead` |
| Data | Minimal test vectors (measures overhead, not throughput) |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!_OS 22.04 |
