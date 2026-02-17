# Handoff: wetSpring → ToadStool/BarraCUDA

> **Note:** This is a historical document from the initial ToadStool integration
> (Phase 3, 17/17 GPU checks). wetSpring has since expanded significantly —
> see [`EVOLUTION_READINESS.md`](EVOLUTION_READINESS.md) for current status
> (Phase 4, 94/94 total checks, 293 tests, 24 modules, zero custom WGSL shaders).

**Date:** February 16, 2026
**From:** wetSpring (Life science + analytical chemistry validation study)
**To:** ToadStool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## STATED GOAL: GPU Diversity Metrics at f64 — ACHIEVED

wetSpring's first GPU validation confirms: **Shannon entropy, Simpson index,
and Bray-Curtis distance matrices compute correctly on consumer GPU at f64
precision**, validated against CPU baselines.

| Metric | GPU vs CPU Error | Tolerance | Checks |
|--------|:----------------:|:---------:|:------:|
| Shannon entropy | ≤ 1e-10 | `GPU_VS_CPU_TRANSCENDENTAL` | 3/3 PASS |
| Simpson index | ≤ 1e-6 | `GPU_VS_CPU_F64` | 3/3 PASS |
| Bray-Curtis distances | ≤ 1e-10 | `GPU_VS_CPU_BRAY_CURTIS` | 4/4 PASS |

Hardware: RTX 4070 12 GB, `SHADER_F64` confirmed, Vulkan backend via wgpu 0.19.

---

## Critical Discovery: `log_f64` Bug in ToadStool's `math_f64.wgsl`

### The Problem

Native `log(f64)` is **not in the WGSL spec** and is **rejected by NVIDIA's
NVVM compiler** with `NVVM compilation failed: 1`. This blocks any shader
that uses `log()` on f64 values.

ToadStool's `math_f64.wgsl` provides a software `log_f64()` implementation
using atanh-polynomial range reduction. However, **the polynomial coefficients
in ToadStool's version are approximately 2x too large**, producing ~1e-3
precision instead of ~1e-15.

### Root Cause

The atanh series expansion of `log(y)` is:

```
log(y) = 2 * atanh((y-1)/(y+1)) = 2 * s * (1 + s²/3 + s⁴/5 + s⁶/7 + ...)
```

ToadStool's coefficients appear to include the leading factor of 2 inside the
polynomial (i.e., `c1 ≈ 0.6666...` instead of `0.3333...`). Since the outer
multiplication by 2 is also present, the polynomial terms are doubled.

### The Fix (Applied in wetSpring)

The corrected `log_f64` in `barracuda/src/shaders/shannon_map_f64.wgsl` uses:

```wgsl
let c1 = zero + 0.3333333333333367565;   // ≈ 1/3
let c2 = zero + 0.1999999999970470954;   // ≈ 1/5
let c3 = zero + 0.1428571437183119575;   // ≈ 1/7
let c4 = zero + 0.1111109921607489198;   // ≈ 1/9
let c5 = zero + 0.0909178608080902506;   // ≈ 1/11
let c6 = zero + 0.0765691884960468666;   // ≈ 1/13
let c7 = zero + 0.0739909930255829295;   // ≈ 1/15 (minimax-optimized)
```

These are the standard atanh series coefficients `≈ 1/(2k+1)` with minimax
optimization. The outer `2 * s * (1 + s² * P(s²))` provides the factor of 2.

### Action Required

**Push the corrected coefficients back to ToadStool's `math_f64.wgsl`.**
The file is at `phase1/toadstool/crates/barracuda/shaders/math/math_f64.wgsl`.
Halve the existing polynomial coefficients (or replace with wetSpring's values)
and verify with the Shannon entropy test case:

```
counts = [10, 20, 30, 40]  →  Shannon = 1.27985422...
CPU Shannon - GPU Shannon should be < 1e-10
```

### f64 Constant Construction in WGSL

A second precision issue: `f64(0.333...)` in WGSL truncates through f32,
losing ~7 digits of precision. The workaround is:

```wgsl
let zero = x - x;           // guaranteed f64 zero
let c1 = zero + 0.333...;   // AbstractFloat literal → full f64
```

This `zero + literal` pattern preserves all 15-16 significant digits. It
should be adopted throughout ToadStool's WGSL shaders wherever f64 constants
appear.

---

## What wetSpring Wrote (Absorb or Reference)

### Three WGSL f64 Shaders

All shaders are at `wetSpring/barracuda/src/shaders/`:

#### 1. `shannon_map_f64.wgsl` (107 lines)

**Operation:** Per-element Shannon contribution: `-p * ln(p)` where `p = count / total`.

- Includes the corrected `log_f64()` function (74 lines)
- Bindings: `counts[]` (read), `output[]` (read_write), `total_buf[]` (read), `ShannonParams` (uniform)
- Workgroup size: 256
- Dispatch: `ceil(n / 256)`
- Output: map result per element; CPU or `SumReduceF64` performs the final summation

**Absorption recommendation:** Extract `log_f64()` into ToadStool's shared
`math_f64.wgsl` (with corrected coefficients). The Shannon map kernel itself
could live in a new `shaders/statistics/` directory or be absorbed into a
general map-reduce framework.

#### 2. `simpson_map_f64.wgsl` (51 lines)

**Operation:** Per-element Simpson contribution: `p²` where `p = count / total`.

- Pure arithmetic — no transcendentals
- Same binding layout as Shannon
- Trivially correct at f64 precision

**Absorption recommendation:** This is a thin wrapper around `p * p`. Could
be a specialization of a general elementwise-map shader.

#### 3. `bray_curtis_pairs_f64.wgsl` (95 lines)

**Operation:** All-pairs Bray-Curtis condensed distance matrix.

- One thread per pair: `BC(a,b) = Σ|a_k - b_k| / Σ(a_k + b_k)`
- `pair_to_ij()` function converts condensed index to (i,j) via integer sqrt
- Bindings: `samples[N*D]` (read, flattened), `output[N*(N-1)/2]` (read_write), `BcParams` (uniform: n_samples, n_features, n_pairs)
- Workgroup size: 256
- Dispatch: `ceil(n_pairs / 256)`
- O(N² × D) where N = samples, D = features

**Absorption recommendation:** The `pair_to_ij()` helper is reusable for any
pairwise distance kernel. Consider a general `pairwise_distance_f64.wgsl` that
accepts a distance function selector (Bray-Curtis, Euclidean, cosine, etc.).

### Rust GPU Bridge: `GpuF64`

Located at `wetSpring/barracuda/src/gpu.rs` (272 lines):

- Wraps `wgpu::Device` + `wgpu::Queue` + `barracuda::device::{WgpuDevice, TensorContext}`
- Initializes with `SHADER_F64` + `SHADER_F16` features
- Provides: `create_pipeline()`, `create_f64_buffer()`, `create_uniform_buffer()`, `dispatch_and_read()`
- Respects `science_limits()`: 512 MiB storage, 1 GiB `max_buffer_size`, 16 storage buffers
- Environment variable: `WETSPRING_WGPU_BACKEND` for backend selection

This follows the same pattern as hotSpring's GPU bridge. The `GpuF64` struct
could potentially be generalized into a shared `BarraCUDA` utility.

### Dispatch Functions: `diversity_gpu.rs`

Located at `wetSpring/barracuda/src/bio/diversity_gpu.rs` (234 lines):

- `shannon_gpu(gpu, counts) → f64` — map via shader, CPU sum
- `simpson_gpu(gpu, counts) → f64` — map via shader, CPU sum
- `bray_curtis_condensed_gpu(gpu, samples) → Vec<f64>` — one dispatch, all pairs
- All gate on `require_f64()` and return `Error::Gpu` on failure

### Validation Tolerances

Located at `wetSpring/barracuda/src/tolerances.rs` (85 lines):

| Constant | Value | Use |
|----------|-------|-----|
| `EXACT` | 0.0 | Integer/deterministic checks |
| `ANALYTICAL_F64` | 1e-12 | CPU f64 analytical formulas |
| `GPU_VS_CPU_F64` | 1e-6 | GPU vs CPU (pure arithmetic) |
| `GPU_VS_CPU_TRANSCENDENTAL` | 1e-10 | GPU vs CPU (log_f64, exp_f64) |
| `GPU_VS_CPU_BRAY_CURTIS` | 1e-10 | GPU vs CPU (BC distance) |

---

## What wetSpring Uses from ToadStool/BarraCUDA

### Currently Used

| Primitive | Location | wetSpring Use |
|-----------|----------|---------------|
| `WgpuDevice` | `device/wgpu_device.rs` | GPU initialization, feature detection |
| `TensorContext` | `device/tensor_context.rs` | Buffer pool, solver buffers |
| `science_limits()` | `device/wgpu_device.rs` | Buffer size limits |
| `SHADER_F64` feature | wgpu | f64 in WGSL shaders |

### Not Yet Used (Ready to Wire)

| Primitive | Location | wetSpring Future Use |
|-----------|----------|---------------------|
| `BatchedEighGpu` | `ops/linalg/` | PCoA eigensolve on BC distance matrix |
| `SumReduceF64` | `ops/reduce/` | Replace CPU sum after map shaders (fused map-reduce) |
| `prng_xoshiro.wgsl` | `shaders/numerical/` | GPU rarefaction (random subsampling) |
| `cumsum_f64.wgsl` | `shaders/reduce/` | Rarefaction accumulation |
| `PipelineBuilder` | `pipeline/mod.rs` | Multi-kernel chaining (parse→diversity→ordination) |
| `sparse_matvec_f64.wgsl` | `shaders/linalg/` | Sparse feature tables (LC-MS) |
| `batched_bisection_f64.wgsl` | `shaders/optimizer/` | m/z tolerance search |
| `Fft1DF64` | `ops/fft/` | Peak detection assist (LC-MS) |
| `cosine_similarity.wgsl` | `shaders/math/` (f32) | MS2 spectral matching — **needs f64 lift** |

### What wetSpring Wants ToadStool to Build

#### Priority 1: Fused Map-Reduce

wetSpring's current Shannon/Simpson shaders map per-element, then sum on CPU.
A fused `map_reduce_f64` primitive that combines the map kernel with
`SumReduceF64` in a single pipeline would:
- Eliminate the CPU readback + sum step
- Reduce dispatch count from 1 + 1 to 1
- Enable scaling to millions of elements without CPU bottleneck

**Interface sketch:**
```rust
gpu.map_reduce_f64(
    shader_source: &str,    // the map kernel
    data: &[f64],
    params: &[u8],
    reduce_op: ReduceOp::Sum,
) -> f64
```

#### Priority 2: `cosine_similarity_f64.wgsl`

The existing `cosine_similarity.wgsl` is f32 only. wetSpring needs f64 for
MS2 spectral matching (Track 2). This is a straightforward lift — dot product
+ norm — but needs to be in the shared library.

#### Priority 3: `exp_f64()` in `math_f64.wgsl`

Once `log_f64()` is corrected, the next transcendental needed is `exp_f64()`
for future BCS-like kernels and softmax operations. The same atanh/polynomial
approach (but for exp) with the `zero + literal` constant pattern.

---

## Evolution Lessons from wetSpring

### 1. WGSL f64 Constant Precision

**Always** use `zero + literal` for f64 constants in WGSL. The pattern:
```wgsl
let zero = x - x;
let constant = zero + 0.6931471805599453;  // full f64 precision
```
Never use `f64(0.693...)` — it truncates through f32.

### 2. Native f64 Builtins Are Unreliable

As of wgpu 0.19 / Vulkan / NVIDIA:
- `sqrt(f64)` — WORKS (confirmed by hotSpring)
- `abs(f64)` — WORKS
- `min/max(f64)` — WORKS
- `log(f64)` — **REJECTED by NVVM** (not in WGSL spec)
- `exp(f64)` — **REJECTED by NVVM** (not in WGSL spec)
- `pow(f64)` — **REJECTED by NVVM** (not in WGSL spec)

Any shader using transcendentals on f64 must use software implementations.
ToadStool's `math_f64.wgsl` is the right place for these, once the `log_f64`
coefficients are corrected.

### 3. Dispatch Pattern for Ecology/Chemistry

Life science and analytical chemistry have a different GPU profile than physics:

| Physics (hotSpring) | Ecology/Chemistry (wetSpring) |
|---------------------|-------------------------------|
| Small matrices, many iterations | Large vectors, few iterations |
| Eigensolve dominates | Reduce/distance dominates |
| Matrix dimension ~4-50 | Vector length ~10,000-1,000,000 |
| Per-iteration sync needed | Single dispatch often sufficient |
| GPU-resident pipeline critical | Map-reduce pipeline critical |

wetSpring workloads are "embarrassingly parallel" — each element or pair is
independent. This means:
- Single-dispatch kernels (no iteration loops in shader)
- High GPU utilization from thread count alone
- Bottleneck is data transfer, not dispatch overhead

### 4. Validation Binary Pattern

wetSpring follows hotSpring's validation pattern but adapted for two-phase
comparison (CPU baseline → GPU result):

```rust
// CPU compute
let cpu_shannon = diversity::shannon(&counts);

// GPU compute
let gpu_shannon = diversity_gpu::shannon_gpu(&gpu, &counts)?;

// Compare
let ok = validation::check("Shannon GPU", gpu_shannon, cpu_shannon, GPU_VS_CPU_TRANSCENDENTAL);
```

Exit codes: 0 (all pass), 1 (any fail), 2 (skip — e.g., no SHADER_F64).

---

## What wetSpring Has Validated (Cumulative)

### Phase 2 (CPU): 36/36 checks pass

| Binary | Modules | Checks |
|--------|---------|:------:|
| `validate_fastq` | `io::fastq` | 9 |
| `validate_diversity` | `bio::diversity` + `bio::kmer` | 14 |
| `validate_mzml` | `io::mzml` + `io::xml` | 7 |
| `validate_pfas` | `io::ms2` + `bio::tolerance_search` | 6 |

### Phase 3 (GPU): 12/12 checks pass

| Binary | Shader | Checks |
|--------|--------|:------:|
| `validate_diversity_gpu` | `shannon_map_f64.wgsl` | 3 |
| `validate_diversity_gpu` | `simpson_map_f64.wgsl` | 3 |
| `validate_diversity_gpu` | `bray_curtis_pairs_f64.wgsl` | 4 (+2 capability) |

**Grand total: 48/48 quantitative checks pass.**

---

## File Inventory

Files written by wetSpring that may be absorbed or referenced:

| File | Lines | Purpose |
|------|:-----:|---------|
| `src/gpu.rs` | 272 | `GpuF64` device bridge |
| `src/bio/diversity_gpu.rs` | 234 | GPU diversity dispatch functions |
| `src/tolerances.rs` | 85 | Centralized validation tolerances |
| `src/shaders/shannon_map_f64.wgsl` | 107 | Shannon map + `log_f64` |
| `src/shaders/simpson_map_f64.wgsl` | 51 | Simpson map |
| `src/shaders/bray_curtis_pairs_f64.wgsl` | 95 | All-pairs BC distance |
| `src/bin/validate_diversity_gpu.rs` | 243 | GPU validation binary |
| `src/error.rs` | 104 | `Error::Gpu` variant |

---

## Summary

wetSpring proves that `BarraCUDA` can serve life science and analytical
chemistry — domains very different from hotSpring's nuclear physics. The
key findings for ToadStool evolution:

1. **`log_f64` coefficients need halving** — this is a bug, not a design choice
2. **`zero + literal` pattern is mandatory** for f64 constants in WGSL
3. **Map-reduce is the dominant GPU pattern** for ecology/chemistry (not eigensolve)
4. **Three new shaders ready for absorption** — Shannon, Simpson, Bray-Curtis
5. **Consumer GPU f64 works for biological diversity computation** — RTX 4070, $599

The thesis holds: sovereign compute on consumer hardware can replicate
institutional bioinformatics and analytical chemistry, then accelerate it
via Rust + GPU.

---

*February 16, 2026 — Initial handoff. Three WGSL f64 shaders, corrected
`log_f64`, 48/48 quantitative checks pass. Tier A diversity GPU complete.
Tier B (tolerance search, peak fitting, sparse ops) is the next evolution.*
