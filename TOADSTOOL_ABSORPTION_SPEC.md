# ToadStool Absorption Spec — wetSpring Local Extensions

**Date:** 2026-02-19
**From:** wetSpring (ecoPrimals)
**To:** ToadStool team
**License:** AGPL-3.0-or-later

## Summary

wetSpring built local extensions to ToadStool primitives during GPU pipeline
development. These extensions are working in production (68/68 parity checks,
73.8% buffer reuse). This doc describes each extension and the recommended
absorption path into ToadStool.

## Extension 1: GemmCached — Pre-compiled GEMM Pipeline

**File:** `wetSpring/barracuda/src/bio/gemm_cached.rs`
**Problem:** `GemmF64::execute()` recreates shader module, bind group layout,
and compute pipeline on every call. In streaming workloads (GEMM per sample),
this adds ~0.5ms per dispatch.

**Solution:** `GemmCached::new(device, ctx)` compiles once, stores the pipeline.
`execute()` reuses the cached pipeline — only creates buffers + bind group.

**Measured impact:**
- Small workload (5 queries): 6.1ms → 3.9ms (**36% faster**)
- First-sample penalty eliminated (36ms → 9.8ms)
- Average per-sample: 14.3ms → 11.0ms (23% improvement)

**Absorption into ToadStool:**

```rust
// Current GemmF64 API:
let c = GemmF64::execute(device, &a, &b, m, k, n, batch_size)?;

// Proposed evolution:
let gemm = GemmF64::new(device.clone());  // compile pipeline once
let c = gemm.execute(&a, &b, m, k, n, batch_size)?;  // reuse cached pipeline
let buf = gemm.execute_to_buffer(&a, &b, m, k, n, batch_size)?;  // no readback
```

**Key code to absorb:**
1. Hoist `compile_shader_f64()` + `create_compute_pipeline()` to `new()`
2. Store pipeline + bind group layout as struct fields
3. `execute()` creates only data buffers + bind group per call
4. Add `execute_to_buffer()` returning `wgpu::Buffer` for chaining

## Extension 2: BufferPool Integration for GEMM

**File:** `wetSpring/barracuda/src/bio/gemm_cached.rs` (in `acquire_and_upload`)
**Problem:** Every GEMM dispatch allocates 3 new GPU buffers (A, B, C). In
streaming workloads, these are similar-sized across calls.

**Solution:** Use `TensorContext::buffer_pool().acquire_pooled()` for A, B, C
buffers. Power-of-2 bucketing enables cross-call reuse.

**Measured impact:**
- 16 allocations, 45 reuses = **73.8% reuse rate**
- Eliminates ~30 buffer allocations across 10-sample pipeline

**Absorption into ToadStool:**
Modify `GemmF64::execute()` to accept an optional `&TensorContext` and use
its buffer pool. Or add `GemmF64::execute_pooled(&self, ctx, ...)`.

## Extension 3: execute_to_buffer() — Chaining Primitive

**File:** `wetSpring/barracuda/src/bio/gemm_cached.rs`
**Problem:** GEMM always reads results back to CPU. For chaining (GEMM → argmax),
the intermediate data should stay on GPU.

**Solution:** `execute_to_buffer()` dispatches GEMM and returns the output
`wgpu::Buffer` without readback. Caller decides when to read back.

**Absorption into ToadStool:**
Add `execute_to_buffer()` as an alternative to `execute()` in `GemmF64`.
This is the foundation for `PipelineBuilder` integration where GEMM output
feeds directly into subsequent GPU stages.

## Extension 4 (Future): GPU Argmax Kernel

**Not yet implemented.** Currently, GEMM scores are read back to CPU for
argmax + bootstrap confidence. A GPU argmax kernel would:
- Reduce readback from ~3.5MB to ~1KB per taxonomy dispatch
- Enable GEMM → argmax chaining via `execute_to_buffer()`
- This would be a new ToadStool primitive: `ArgmaxF64`

## Infrastructure Already Working

These ToadStool systems are wired and actively used by wetSpring:

| System | Usage | Stats |
|--------|-------|-------|
| `TensorContext` | Session-level GPU context | Wired in `GpuPipelineSession` |
| `BufferPool` | Buffer reuse for GEMM | 73.8% reuse rate |
| `FusedMapReduceF64` | Shannon, Simpson, sum | Pre-compiled, reused |
| `GemmF64` (via GemmCached) | Taxonomy classification | Cached pipeline |
| `BrayCurtisF64` | Beta diversity | Wired in `diversity_gpu.rs` |

## How to Test

```bash
cd wetSpring/barracuda
cargo run --features gpu --release --bin validate_16s_pipeline_gpu
```

Expects: 68/68 checks PASS, BufferPool stats > 0% reuse, scaling
benchmark shows GPU advantage at all workload sizes.
