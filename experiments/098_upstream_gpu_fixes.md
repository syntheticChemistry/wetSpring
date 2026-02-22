# Exp098: Upstream GPU Fixes — SNP Binding, ODE f64 Builtins, Jacobi Eigenvectors

**Date:** February 22, 2026
**Binary:** `validate_barracuda_gpu_full`, `validate_local_wgsl_compile`, `validate_gpu_extended`, `validate_diversity_gpu`
**Status:** COMPLETE

---

## Objective

Fix three upstream ToadStool bugs discovered during full control validation of
BarraCuda CPU vs GPU vs metalForge mixed hardware validation pass.

## Bugs Found and Fixed

### 1. SNP Calling — Bind Group Layout Mismatch

**File:** `toadstool/crates/barracuda/src/ops/bio/snp.rs`

The `SnpCallingF64` BGL declared 7 entries (uniform + 6 storage) but the shader
and dispatch only use 6 entries (uniform + 5 storage). Additionally, binding 2
(`is_variant`) was marked `read_only: true` but the shader declares it `read_write`.

| Before | After |
|--------|-------|
| `make_bgl(&device, &[true, true, false, false, false, false])` | `make_bgl(&device, &[true, false, false, false, false])` |

### 2. Batched ODE RK4 — f64 Literal Type Mismatch

**File:** `toadstool/crates/barracuda/src/shaders/numerical/batched_qs_ode_rk4_f64.wgsl`

The QS/c-di-GMP ODE shader used WGSL builtins `max()`, `pow()`, and `clamp()`
with f64 arguments. These builtins only support f32/f16 in naga/wgpu 22.1.0.
AbstractFloat literals (`0.0`, `1.0`, `2.0`, `1e-14`) in arithmetic and function
calls also failed naga type resolution for f64.

**Fix:** Replaced builtins with manual f64-safe functions (`fmax`, `fpow`, `fclamp`)
and applied the `(zero + literal)` pattern for all f64 constants:
```wgsl
let z = h - h; // f64 zero
let half = z + 0.5;
let two  = z + 2.0;
```

### 3. Jacobi Eigensolver — Eigenvector Rotation Bug

**File:** `toadstool/crates/barracuda/src/shaders/linalg/batched_eigh_single_dispatch_f64.wgsl`

The V (eigenvector) rotation was inside the `if (k != p && k != q)` guard
that's correct for A (the 2×2 block handled separately) but wrong for V.
Eigenvectors require rotation for ALL rows k, including k == p and k == q.

Additionally, a commented-out `// @unroll_hint 32` triggered the WgslOptimizer's
loop unroller, producing AbstractInt literals (`0`) where `u32` (`0u`) was required.

| Eigenvalues | Eigenvectors (before) | Eigenvectors (after) |
|-------------|----------------------|---------------------|
| Correct | Zero or wrong values | Correct (PCoA matches CPU) |

## Validation

| Suite | Before | After |
|-------|--------|-------|
| `validate_barracuda_gpu_full` | CRASH (SNP binding) | 24/24 PASS |
| `validate_local_wgsl_compile` | 10/11 (ODE panic) | 13/13 PASS |
| `validate_gpu_ode_sweep` | 6/12 (ODE skip) | 12/12 PASS |
| `validate_gpu_extended` | CRASH (unroll hint) | 53/53 PASS |
| `validate_diversity_gpu` | CRASH | 38/38 PASS |
| `validate_pure_gpu_pipeline` | CRASH | 31/31 PASS |

## Full Control Validation (All Green)

| Tier | Binary | Checks |
|------|--------|--------|
| CPU | `validate_barracuda_cpu_full` | 50/50 |
| CPU | `validate_barracuda_cpu_v7` | 43/43 |
| GPU | `validate_barracuda_gpu_full` | 24/24 |
| GPU | `validate_gpu_ode_sweep` | 12/12 |
| GPU | `validate_gpu_extended` | 53/53 |
| GPU | `validate_gpu_track1c` | 27/27 |
| GPU | `validate_gpu_rf` | 13/13 |
| GPU | `validate_gpu_hmm_forward` | 13/13 |
| GPU | `validate_gpu_phylo_compose` | 15/15 |
| GPU | `validate_local_wgsl_compile` | 13/13 |
| GPU | `validate_diversity_gpu` | 38/38 |
| GPU | `validate_pure_gpu_streaming` | 80/80 |
| GPU | `validate_pure_gpu_pipeline` | 31/31 |
| CPU vs GPU | `validate_cpu_vs_gpu_all_domains` | 48/48 |
| Mixed HW | `validate_metalforge_full_v3` | 28/28 |
| Mixed HW | `validate_metalforge_pipeline` | 45/45 |
| Mixed HW | `validate_pcie_direct` | 32/32 |
| Mixed HW | `validate_streaming_dispatch` | 25/25 |
| Cross-spring | `validate_cross_spring_evolution` | 39/39 |
| ToadStool | `validate_toadstool_bio` | 12/12 |
| ToadStool | `validate_cross_substrate` | 20/20 |
| ToadStool | `validate_cross_substrate_pipeline` | 17/17 |
| ToadStool | `validate_dispatch_overhead_proof` | 21/21 |
| ToadStool | `validate_substrate_router` | 20/20 |
| **Unit tests** | `cargo test` | **666/666** |

## Impact

These fixes unlock correct GPU-accelerated computation for all 16 validated
science domains. The PCoA eigenvector bug specifically affected any downstream
consumer that relied on GPU eigenvalue decomposition for coordinate extraction
(PCoA, spectral clustering, etc.).
