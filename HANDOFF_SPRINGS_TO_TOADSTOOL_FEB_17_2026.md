# Handoff: All Springs → ToadStool Wiring Team

**Date:** February 17, 2026
**From:** hotSpring, wetSpring, airSpring validation teams
**To:** ToadStool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## Executive Summary

ToadStool has **~500 WGSL shaders** but only **~95 have Rust orchestrators**.
The shaders are complete, tested in WGSL, and ready to wire. The springs are
blocked on missing orchestrators for ~15 high-value shaders that would unlock
the next phase of GPU acceleration across nuclear physics, life science, and
precision agriculture.

**Ask:** Wire the shaders listed below. Each entry includes the shader path,
binding layout, entry points, which spring needs it, and estimated effort.
Most are easy — the hard work (shader authoring, f64 precision, math
correctness) is already done.

**Combined validation after wiring:** 313+ existing checks will be joined by
new GPU-vs-CPU checks from each spring team. We will validate everything.

---

## Quick Win: Export Missing Module

`ops/cyclic_reduction_wgsl.rs` exists and works but is **not in `ops/mod.rs`**.

```rust
// ops/mod.rs — add this line:
pub mod cyclic_reduction_wgsl;
```

This immediately unblocks tridiagonal solves for all springs. Takes 10 seconds.

---

## Priority 1: Cross-Spring (all three benefit)

### 1.1 `cyclic_reduction_f64.wgsl` — f64 Tridiagonal Solve

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

### 1.3 `weighted_dot_f64.wgsl` — Weighted Dot Product

**Shader:** `shaders/reduce/weighted_dot_f64.wgsl`
**Entry points:** `weighted_dot_simple`, `weighted_dot_parallel`, `final_reduce`, `weighted_dot_batched`, `dot_parallel`, `norm_squared_parallel`
**Bindings:** `weights`, `vec_a`, `vec_b`, `params` (n), `result` (scalar or partial sums)
**Effort:** Easy — matches `sum_reduce_f64` pattern, already validated by hotSpring

| Consumer | Use Case |
|----------|----------|
| hotSpring | Nuclear EOS, energy integrals |
| wetSpring | Galerkin inner products, weighted correlation |
| airSpring | Weighted sensor fusion |

### 1.4 `correlation.wgsl` + `covariance.wgsl` — Statistical Kernels

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

### 2.1 `eigh_f64.wgsl` — Single-Matrix Eigendecomposition

**Shader:** `shaders/linalg/eigh_f64.wgsl`
**Entry points:** `init_V`, `compute_jacobi_angle`, `jacobi_rotate_A`, `jacobi_update_block`, `jacobi_rotate_V`, `extract_eigenvalues`, `find_max_off_diag`
**Bindings:** `A` (symmetric n×n), `V` (eigenvectors out), `eigenvalues` (out), `params` (n)
**Effort:** Hard — multi-pass Jacobi with CPU convergence loop (init → find pivot → angle → rotate A/V → extract), but `batched_eigh_gpu.rs` already does this for batched case

**Why critical:** hotSpring's HFB eigensolve is the #1 bottleneck (760K queue submissions for batched 12×12). This single-matrix variant could be the basis for the single-dispatch Jacobi kernel identified in the previous handoff.

### 2.2 `hermite_f64.wgsl` — Hermite Polynomials

**Shader:** `shaders/special/hermite_f64.wgsl`
**Entry points:** `main`, `hermite_function_kernel`
**Bindings:** `input` (f64), `params` (size, n), `output` (f64)
**Effort:** Trivial — `hermite_wgsl` exists for f32; f64 variant is identical pattern

| Consumer | Use Case |
|----------|----------|
| hotSpring | Quantum oscillator basis functions, nuclear structure |

### 2.3 `laguerre_f64.wgsl` — Laguerre Polynomials

**Shader:** `shaders/special/laguerre_f64.wgsl`
**Entry points:** `main`, `radial_laguerre`
**Bindings:** `input` (f64), `params` (size, n, alpha), `output` (f64)
**Effort:** Trivial — `laguerre_wgsl` exists for f32; add alpha parameter

| Consumer | Use Case |
|----------|----------|
| hotSpring | Hydrogen/helium radial wavefunctions |

---

## Priority 3: PDE/ODE Infrastructure (hotSpring + airSpring)

### 3.1 `crank_nicolson.wgsl` — Implicit PDE Solver

**Shader:** `shaders/pde/crank_nicolson.wgsl`
**Entry points:** `compute_rhs`, `build_matrix`, `apply_source`, `adi_rhs_x_sweep`, `adi_rhs_y_sweep`, `compute_laplacian`
**Bindings:** 1D: `u` (current field), `rhs` (out), `a_diag`/`b_diag`/`c_diag` (tridiagonal). 2D ADI: `u_2d`, `rhs_2d`. Source: `source`, `u_src` (in-place)
**Effort:** Moderate — multiple kernels, chains with `cyclic_reduction` for the tridiagonal solve, CPU time-stepping loop

| Consumer | Use Case |
|----------|----------|
| airSpring | Richards equation (unsaturated flow), soil heat (critical path) |
| hotSpring | Time-dependent Schrodinger, Two-Temperature Model |

### 3.2 `rk_stage.wgsl` — Runge-Kutta ODE Integrator

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

### 4.1 `cosine_similarity_f64.wgsl` — f64 Spectral Matching

**Shader:** `shaders/math/cosine_similarity_f64.wgsl`
**Entry points:** `main` (all-pairs, workgroup 16x16), `cosine_single_pair` (single pair, workgroup 256, shared-memory reduction)
**Bindings:** `vectors_a`, `vectors_b` (f64), `params` (num_vectors_a, num_vectors_b, vector_dim), `output` (similarity matrix)
**Effort:** Easy — `cosine_similarity.rs` exists for f32; f64 variant follows same pattern

**Note:** wetSpring currently uses `GemmF64 + FusedMapReduceF64` for pairwise cosine, which is superior for large batches. This dedicated shader would be useful for **single-pair** or **small-batch** queries where GEMM overhead dominates.

| Consumer | Use Case |
|----------|----------|
| wetSpring | MS2 spectral matching, library search (small queries) |

---

## Priority 5: MD Forces (hotSpring + Future)

These are already-written f64 force shaders with no orchestrators:

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

## Effort Summary

| Priority | Shader | Effort | Springs |
|----------|--------|--------|---------|
| Quick win | Export `cyclic_reduction_wgsl` in `mod.rs` | 10 seconds | All |
| P1.1 | `cyclic_reduction_f64` | Easy | All |
| P1.2 | `kriging_f64` GPU | Moderate | airSpring, wetSpring |
| P1.3 | `weighted_dot_f64` | Easy | All |
| P1.4 | `correlation` + `covariance` | Easy | All |
| P2.1 | `eigh_f64` single-matrix | Hard | hotSpring (critical) |
| P2.2 | `hermite_f64` | Trivial | hotSpring |
| P2.3 | `laguerre_f64` | Trivial | hotSpring |
| P3.1 | `crank_nicolson` | Moderate | airSpring, hotSpring |
| P3.2 | `rk_stage` | Moderate | airSpring, hotSpring |
| P4.1 | `cosine_similarity_f64` | Easy | wetSpring |
| P5 | MD forces f64 (6 shaders) | Easy–Moderate | hotSpring |

**Total: 14 orchestrators** to unlock the next GPU evolution phase.

Estimated total effort: ~3–5 days for one developer familiar with the codebase.
The 5 trivial/easy P1+P2 items could ship in an afternoon.

---

## Validation Plan

Each spring team will:

1. Wire the new orchestrator into their GPU modules
2. Add GPU-vs-CPU validation checks to their existing binaries
3. Report results back with tolerance and pass/fail

Current baselines:
- hotSpring: 195 checks
- wetSpring: 94 checks (63 CPU + 31 GPU)
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
