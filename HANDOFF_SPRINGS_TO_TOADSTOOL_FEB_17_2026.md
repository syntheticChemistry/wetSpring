# Handoff: All Springs → ToadStool Wiring Team

> **SUPERSEDED** by [`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_19_2026_v3.md`](HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_19_2026_v3.md)
> (645 checks, 430 tests, 22 experiments, full evolution readiness).

**Date:** February 17, 2026
**Updated:** February 18, 2026 — **Nearly all items RESOLVED by ToadStool deep debt work**
**From:** hotSpring, wetSpring, airSpring validation teams
**To:** ToadStool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## Executive Summary

> **Update Feb 18:** ToadStool completed a massive evolution since this
> handoff was drafted. Of the 14 high-priority items listed below,
> **13 are now wired with full Rust orchestrators**. The remaining item
> (Kriging GPU-side LU) is a known low-priority optimization.
>
> Additionally, ToadStool completed:
> - wgpu 0.19 → v22 migration across all crates
> - Fully concurrent test infrastructure with shared device pool
> - Capability-based dispatch refactoring
> - 5 new statistical f64 orchestrators (correlation, covariance, variance, beta, digamma)
> - 10+ new f64 scientific orchestrators (bessel, legendre, spherical harmonics)
> - Sparse solver architecture split (spmv, dot_reduce, vector_ops, cg_kernels)
> - Coulomb GPU energy path, tridiagonal serial kernel
>
> **wetSpring rewire result:** 9 ToadStool primitives, 38/38 GPU PASS, 101/101 total.

~~ToadStool has **~500 WGSL shaders** but only **~95 have Rust orchestrators**.~~
ToadStool has wired nearly all requested orchestrators.

**Combined validation after wiring:** 101 wetSpring + hotSpring + airSpring checks.

---

## Quick Win: Export Missing Module — RESOLVED

~~`ops/cyclic_reduction_wgsl.rs` exists and works but is **not in `ops/mod.rs`**.~~

**Status:** DONE — `cyclic_reduction_f64` is now exported in `ops/mod.rs` and has
a full orchestrator with `solve()` and `solve_batch()` methods.

---

## Priority 1: Cross-Spring (all three benefit)

### 1.1 `cyclic_reduction_f64.wgsl` — f64 Tridiagonal Solve — RESOLVED

**Shader:** `shaders/linalg/cyclic_reduction_f64.wgsl`
**Entry points:** `reduction_f64`, `substitution_f64`, `solve_serial_f64`, `reduction_batch_f64`, `substitution_batch_f64`, `solve_batch_serial_f64`
**Bindings:** `params` (n, step, phase), `a` (sub-diag), `b` (main diag), `c` (super-diag), `d` (RHS → solution in-place)
**Effort:** Easy — `cyclic_reduction_wgsl.rs` already wires the f32 version; point it at the f64 shader + change buffer types

| Consumer | Use Case |
|----------|----------|
| hotSpring | Schrodinger equation, diffusion, implicit time stepping |
| wetSpring | 1D diffusion models, Richards equation |
| airSpring | Soil heat transport, Richards equation (critical path) |

### 1.2 `kriging_f64.wgsl` — GPU Kriging

**Shader:** `shaders/interpolation/kriging_f64.wgsl`
**Entry points:** `build_covariance_matrix`, `build_rhs_vector`, `interpolate`, `simple_kriging_interpolate`
**Bindings:** `known_points` (x,y,z), `target_points` (x,y), `params` (n_known, n_target, variogram_model, nugget, sill, range_param), work buffers for covariance matrix + weights, `output` (interpolated values + variances)
**Effort:** Moderate — `ops/kriging_f64.rs` exists but does CPU LU solve; wire the shader for the covariance build + interpolation steps, keep LU on CPU or chain with `lu_gpu`

| Consumer | Use Case |
|----------|----------|
| airSpring | Soil moisture mapping from sparse sensors to field grid (primary) |
| wetSpring | Diversity metric interpolation across sampling sites |
| hotSpring | Spatial data analysis |

### 1.3 `weighted_dot_f64.wgsl` — Weighted Dot Product — RESOLVED

**Shader:** `shaders/reduce/weighted_dot_f64.wgsl`
**Entry points:** `weighted_dot_simple`, `weighted_dot_parallel`, `final_reduce`, `weighted_dot_batched`, `dot_parallel`, `norm_squared_parallel`
**Bindings:** `weights`, `vec_a`, `vec_b`, `params` (n), `result` (scalar or partial sums)
**Effort:** Easy — matches `sum_reduce_f64` pattern, already validated by hotSpring

| Consumer | Use Case |
|----------|----------|
| hotSpring | Nuclear EOS, energy integrals |
| wetSpring | Galerkin inner products, weighted correlation |
| airSpring | Weighted sensor fusion |

### 1.4 `correlation.wgsl` + `covariance.wgsl` — Statistical Kernels — RESOLVED

**Shader:** `shaders/special/correlation.wgsl`
**Entry point:** `main`
**Bindings:** `x`, `y` (f32), `params` (size, num_pairs, stride), `output` (Pearson r per pair)
**Effort:** Easy — single kernel, embarrassingly parallel

**Shader:** `shaders/special/covariance.wgsl`
**Entry point:** `main`
**Bindings:** `x`, `y` (f32), `params` (size, num_pairs, stride, ddof), `output` (covariance per pair)
**Effort:** Easy — identical pattern to correlation

**Note:** These are f32. f64 variants would be valuable for all springs.

| Consumer | Use Case |
|----------|----------|
| wetSpring | Feature correlation, sensor cross-correlation |
| airSpring | Sensor network analysis, spatial statistics |
| hotSpring | Observable correlation |

---

## Priority 2: hotSpring Critical Path

### 2.1 `eigh_f64.wgsl` — Single-Matrix Eigendecomposition — RESOLVED (batched_eigh_gpu fixed)

**Shader:** `shaders/linalg/eigh_f64.wgsl`
**Entry points:** `init_V`, `compute_jacobi_angle`, `jacobi_rotate_A`, `jacobi_update_block`, `jacobi_rotate_V`, `extract_eigenvalues`, `find_max_off_diag`
**Bindings:** `A` (symmetric n×n), `V` (eigenvectors out), `eigenvalues` (out), `params` (n)
**Effort:** Hard — multi-pass Jacobi with CPU convergence loop (init → find pivot → angle → rotate A/V → extract), but `batched_eigh_gpu.rs` already does this for batched case

**Why critical:** hotSpring's HFB eigensolve is the #1 bottleneck (760K queue submissions for batched 12×12). This single-matrix variant could be the basis for the single-dispatch Jacobi kernel identified in the previous handoff.

### 2.2 `hermite_f64.wgsl` — Hermite Polynomials — RESOLVED

**Shader:** `shaders/special/hermite_f64.wgsl`
**Entry points:** `main`, `hermite_function_kernel`
**Bindings:** `input` (f64), `params` (size, n), `output` (f64)
**Effort:** Trivial — `hermite_wgsl` exists for f32; f64 variant is identical pattern

| Consumer | Use Case |
|----------|----------|
| hotSpring | Quantum oscillator basis functions, nuclear structure |

### 2.3 `laguerre_f64.wgsl` — Laguerre Polynomials — RESOLVED

**Shader:** `shaders/special/laguerre_f64.wgsl`
**Entry points:** `main`, `radial_laguerre`
**Bindings:** `input` (f64), `params` (size, n, alpha), `output` (f64)
**Effort:** Trivial — `laguerre_wgsl` exists for f32; add alpha parameter

| Consumer | Use Case |
|----------|----------|
| hotSpring | Hydrogen/helium radial wavefunctions |

---

## Priority 3: PDE/ODE Infrastructure (hotSpring + airSpring)

### 3.1 `crank_nicolson.wgsl` — Implicit PDE Solver — RESOLVED

**Shader:** `shaders/pde/crank_nicolson.wgsl`
**Entry points:** `compute_rhs`, `build_matrix`, `apply_source`, `adi_rhs_x_sweep`, `adi_rhs_y_sweep`, `compute_laplacian`
**Bindings:** 1D: `u` (current field), `rhs` (out), `a_diag`/`b_diag`/`c_diag` (tridiagonal). 2D ADI: `u_2d`, `rhs_2d`. Source: `source`, `u_src` (in-place)
**Effort:** Moderate — multiple kernels, chains with `cyclic_reduction` for the tridiagonal solve, CPU time-stepping loop

| Consumer | Use Case |
|----------|----------|
| airSpring | Richards equation (unsaturated flow), soil heat (critical path) |
| hotSpring | Time-dependent Schrodinger, Two-Temperature Model |

### 3.2 `rk_stage.wgsl` — Runge-Kutta ODE Integrator — RESOLVED

**Shader:** `shaders/numerical/rk_stage.wgsl`
**Entry points:** `prepare_stage`, `update_solution`, `compute_error`, `error_norm`, `axpy`
**Bindings:** `y` (state), `k_stages` (RHS evaluations), `params` (n, h, stage), `y_stage` (for RHS eval), `y_new` (result), error buffers. Dormand-Prince RK45 coefficients embedded.
**Effort:** Moderate — CPU loop for stages + adaptive step control; `f(t,y)` evaluation via custom kernel or CPU callback

| Consumer | Use Case |
|----------|----------|
| hotSpring | Time-dependent nuclear physics ODEs |
| airSpring | Soil dynamics, crop growth models |

---

## Priority 4: wetSpring Specific

### 4.1 `cosine_similarity_f64.wgsl` — f64 Spectral Matching — RESOLVED

**Shader:** `shaders/math/cosine_similarity_f64.wgsl`
**Entry points:** `main` (all-pairs, workgroup 16x16), `cosine_single_pair` (single pair, workgroup 256, shared-memory reduction)
**Bindings:** `vectors_a`, `vectors_b` (f64), `params` (num_vectors_a, num_vectors_b, vector_dim), `output` (similarity matrix)
**Effort:** Easy — `cosine_similarity.rs` exists for f32; f64 variant follows same pattern

**Note:** wetSpring currently uses `GemmF64 + FusedMapReduceF64` for pairwise cosine, which is superior for large batches. This dedicated shader would be useful for **single-pair** or **small-batch** queries where GEMM overhead dominates.

| Consumer | Use Case |
|----------|----------|
| wetSpring | MS2 spectral matching, library search (small queries) |

---

## Priority 5: MD Forces (hotSpring + Future) — ALL RESOLVED

These shaders now have full Rust orchestrators:

| Shader | Entry Points | Effort |
|--------|-------------|--------|
| `ops/md/forces/coulomb_f64.wgsl` | `coulomb_force_f64` | Easy — f32 version wired |
| `ops/md/forces/morse_f64.wgsl` | `morse_force_f64` | Easy — f32 version wired |
| `ops/md/forces/yukawa_celllist_f64.wgsl` | `yukawa_celllist_f64` | Moderate — cell list management |
| `ops/md/forces/born_mayer.wgsl` | `born_mayer_force` | Easy — no f32 exists |
| `ops/md/observables/rdf_histogram.wgsl` | `rdf_histogram` | Easy — histogram accumulation |
| `ops/md/integrators/velocity_verlet.wgsl` | `velocity_verlet_step` | Easy |
| `ops/md/integrators/rk4.wgsl` | `rk4_step` | Moderate |
| `ops/md/pbc.wgsl` | `apply_pbc` | Trivial |

---

## Wiring Pattern (for reference)

Every wired shader follows this pattern:

```rust
// 1. Include shader source
let shader_source = include_str!("../shaders/path/to/shader.wgsl");

// 2. Create pipeline
let module = device.create_shader_module(ShaderModuleDescriptor {
    label: Some("MyOp"),
    source: ShaderSource::Wgsl(shader_source.into()),
});
let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("MyOp Pipeline"),
    layout: None,
    module: &module,
    entry_point: "main",
});

// 3. Create buffers, bind group, dispatch, read back
// (see any existing orchestrator for the full pattern)
```

Existing orchestrators to copy from:
- **Simple (single dispatch):** `sum_reduce_f64.rs`, `bray_curtis_f64.rs`
- **Multi-pass (CPU loop):** `batched_eigh_gpu.rs`, `cg_gpu.rs`
- **Batched:** `batched_elementwise_f64.rs`

---

## Effort Summary — Resolution Status

| Priority | Shader | Status | Wired By |
|----------|--------|--------|----------|
| Quick win | Export `cyclic_reduction_wgsl` in `mod.rs` | **DONE** | ToadStool `d69c472b` |
| P1.1 | `cyclic_reduction_f64` | **DONE** | ToadStool `d69c472b` |
| P1.2 | `kriging_f64` GPU | Partial (CPU LU) | Original `0c477306` |
| P1.3 | `weighted_dot_f64` | **DONE** | ToadStool `d69c472b` |
| P1.4 | `correlation` + `covariance` + f64 | **DONE** | ToadStool `d69c472b` + `d67b4ee0` |
| P2.1 | `eigh_f64` single-matrix | **DONE** | ToadStool `728838fa` |
| P2.2 | `hermite_f64` | **DONE** | ToadStool `acfaa454` |
| P2.3 | `laguerre_f64` | **DONE** | ToadStool `acfaa454` |
| P3.1 | `crank_nicolson` | **DONE** | ToadStool `d69c472b` |
| P3.2 | `rk_stage` | **DONE** | ToadStool `d69c472b` |
| P4.1 | `cosine_similarity_f64` | **DONE** | ToadStool `d69c472b` |
| P5 | MD forces f64 (6 shaders) | **DONE** | ToadStool `acfaa454` + `5ae47207` |

**Result: 13/14 fully wired, 1 partial (Kriging LU on CPU — low priority).**

Additional bonus items wired (not in original handoff):
- `VarianceF64` (population, sample, std dev)
- `CorrelationF64` (Pearson f64)
- `CovarianceF64` (population, sample, custom ddof)
- `BetaF64`, `DigammaF64` (special functions)
- Bessel I₀/J₀/J₁/K₀ f64, Legendre f64, Spherical Harmonics f64
- Born-Mayer f64, Velocity Verlet f64, Kinetic Energy f64, RDF f64
- Sparse solver split: spmv, dot_reduce, vector_ops, cg_kernels

---

## Validation Plan

Each spring team will:

1. Wire the new orchestrator into their GPU modules
2. Add GPU-vs-CPU validation checks to their existing binaries
3. Report results back with tolerance and pass/fail

Current baselines:
- hotSpring: 195 checks
- wetSpring: **101 checks** (63 CPU + 38 GPU) — **7 new GPU stats checks since handoff**
- airSpring: 70 checks

---

## Git Commands

```bash
# ToadStool team
cd /path/to/toadStool
# Wire orchestrators...
cargo test -p barracuda
cargo clippy -p barracuda -- -D warnings
git push origin master

# Spring teams pull and validate
cd /path/to/{hot,wet,air}Spring
git -C ../phase1/toadstool pull origin master
cargo test --features gpu
cargo run --features gpu --bin validate_*_gpu
```

---

*February 17, 2026 — Shaders are done. Wire them and the springs flow.*

*February 18, 2026 — They wired them. The springs flow. wetSpring: wgpu v22,
9 ToadStool primitives, 38/38 GPU PASS, 101/101 total. Zero custom WGSL.*
