# ToadStool Absorption Spec — wetSpring Local Extensions

**Date:** 2026-02-19
**From:** wetSpring (ecoPrimals)
**To:** ToadStool team
**License:** AGPL-3.0-or-later

## Summary

wetSpring built local extensions to ToadStool primitives during GPU pipeline
development. These extensions are working in production (88/88 parity checks,
93.0% buffer reuse, 2.45× GPU speedup). This doc describes each extension and
the recommended absorption path into ToadStool.

## Extension 1: QualityFilterCached — Per-Read Parallel Filtering

**Files:**
- `wetSpring/barracuda/src/shaders/quality_filter.wgsl` — WGSL shader
- `wetSpring/barracuda/src/bio/quality_gpu.rs` — Rust wrapper

**Problem:** Quality filtering scans each read sequentially (leading trim,
trailing trim, sliding window). With 80K-1.2M reads, this is embarrassingly
parallel across reads but sequential within each read.

**Solution:** Custom WGSL shader with one GPU thread per read. All integer
arithmetic (u32), no f64. Quality bytes packed 4-per-u32 for efficient transfer.
Pre-compiled pipeline + BufferPool integration.

**Measured impact:**
- Math parity: 100% (all read counts match CPU exactly)
- Speed: ~0.85× vs CPU (QF is memory-bound, not compute-bound)
- Proves GPU CAN handle per-read operations even if CPU is faster here

**ToadStool absorption → `ParallelFilter<T>`:**
A per-element parallel scan-and-filter primitive where each thread processes
one element independently. Shader template parameterized by element type
and filter logic.

## Extension 2: Dada2Gpu — Batch Pair-Wise Reduction

**Files:**
- `wetSpring/barracuda/src/shaders/dada2_e_step.wgsl` — WGSL shader
- `wetSpring/barracuda/src/bio/dada2_gpu.rs` — Rust wrapper + EM loop

**Problem:** DADA2's E-step computes `log_p_error(seq, center)` for all
(sequence, center) pairs — O(seqs × centers × seq_length). This is the
pipeline's largest bottleneck (326ms avg on CPU).

**Solution:** GPU batch dispatch: one thread per (seq, center) pair. Each
thread sums precomputed `ln(err[from][to][qual])` lookup values over all
positions. **Key insight: no GPU transcendentals needed.** The ln() values
are precomputed on CPU and uploaded as a flat f64 table (672 values = 5KB).

The full EM loop stays on CPU (argmax, error model update, Poisson test,
convergence check). Only the E-step dispatch is GPU. Data (bases, quals,
lengths) uploaded once; only center_indices and log_err updated per iteration.

**Measured impact:**
- **24.4× average speedup** (326ms CPU → 13ms GPU)
- Identical ASV counts and total reads across all 10 samples
- Reduced pipeline total from 1.21× to 2.45× GPU speedup
- BufferPool reuse: 93% (bases/quals buffers reused across iterations)

**ToadStool absorption → `BatchPairReduce<f64>`:**

```rust
// Proposed ToadStool primitive:
let reducer = BatchPairReduce::<f64>::new(device, reduce_shader);
// Computes f(element_a[i], element_b[j]) reduced over shared dimension
let scores = reducer.execute(
    &data_a,      // [N × L] elements
    &data_b,      // [M × L] elements  
    &lookup_table, // precomputed per-element values
    n, m, l,
)?;
// Returns [N × M] matrix of reduced values
```

## Extension 3: GemmCached — Pre-compiled GEMM Pipeline

**File:** `wetSpring/barracuda/src/bio/gemm_cached.rs`

**Problem:** `GemmF64::execute()` recreates shader, BGL, and compute pipeline
on every call (~0.5ms overhead per dispatch).

**Solution:** `GemmCached::new(device, ctx)` compiles once at init.
`execute()` reuses cached pipeline — only buffer management per call.

**Measured impact:**
- First-sample penalty eliminated (36ms → 9.8ms)
- Average per-sample: 14.3ms → 11.0ms (23% improvement)
- Small workload (5 queries): 16.5× GPU speedup

**ToadStool absorption:**

```rust
// Current: GemmF64::execute(device, &a, &b, m, k, n, batch)?;
// Proposed:
let gemm = GemmF64::new(device);  // compile once
let c = gemm.execute(&a, &b, m, k, n, batch)?;  // reuse cached
let buf = gemm.execute_to_buffer(&a, &b, m, k, n, batch)?;  // no readback
```

## Extension 4: BufferPool Integration

**Files:** All GPU modules (`gemm_cached.rs`, `quality_gpu.rs`, `dada2_gpu.rs`)

**Problem:** GPU buffer allocation is expensive. Streaming workloads dispatch
similar-sized operations repeatedly.

**Solution:** All modules use `TensorContext::buffer_pool().acquire_pooled()`
with power-of-2 bucketing. Buffers auto-returned on drop.

**Measured impact:** 29 allocations, 385 reuses = **93.0% reuse rate**.

## Extension 5 (Future): GPU Argmax Kernel

**Not yet implemented.** GEMM scores are read back for CPU argmax + bootstrap.
A GPU argmax kernel would reduce readback from ~3.5MB to ~1KB per dispatch.

## Infrastructure Actively Used

| System | Usage | Stats |
|--------|-------|-------|
| `TensorContext` | Session-level GPU context | Wired in `GpuPipelineSession` |
| `BufferPool` | Buffer reuse for all GPU modules | 93.0% reuse rate |
| `FusedMapReduceF64` | Shannon, Simpson, sum | Pre-compiled, reused |
| `GemmF64` (via GemmCached) | Taxonomy classification | Cached pipeline |
| `BrayCurtisF64` | Beta diversity | Wired in `diversity_gpu.rs` |
| `ShaderTemplate` | f64 driver workaround | Used by DADA2 + GEMM shaders |

## How to Test

```bash
cd wetSpring/barracuda
cargo run --features gpu --release --bin validate_16s_pipeline_gpu
```

Expects: 88/88 checks PASS, BufferPool 93%+ reuse, DADA2 GPU 20×+ speedup,
overall pipeline 2×+ GPU speedup.
