# wetSpring → ToadStool Handoff v14 — Upstream GPU Fixes

**Date:** February 22, 2026
**Phase:** 26
**Author:** wetSpring validation pipeline
**Previous:** [v13 — Lean Phase Complete](WETSPRING_TOADSTOOL_V13_FEB22_2026.md)

---

## Executive Summary

Full control validation of BarraCuda CPU vs GPU vs metalForge mixed hardware
uncovered **three upstream ToadStool bugs** that were preventing correct GPU
execution for SNP calling, batched ODE integration, and eigenvalue decomposition.
All three have been fixed in the local ToadStool checkout and validated end-to-end.

**Key metrics after fix:**

| Metric | v13 | v14 |
|--------|-----|-----|
| Experiments | 97 | 98 |
| GPU bins passing | ~20 | 24/24 (all) |
| CPU vs GPU (16 domains) | 48/48 | 48/48 |
| metalForge mixed HW | PASS | PASS |
| ODE sweep (Exp049-050) | 6/12 (ODE skip) | 12/12 |
| PCoA eigenvectors | CRASH | 53/53 |
| `cargo test` | 666 lib | 666 lib |

---

## Part 1: Bug Fixes (Critical for ToadStool to Absorb)

### Bug 1: `SnpCallingF64` Bind Group Layout Mismatch

**File:** `crates/barracuda/src/ops/bio/snp.rs` line 39

The BGL used `&[true, true, false, false, false, false]` (6 storage entries = 7
total with uniform), but the shader only declares 6 bindings (0-5). Binding 2
(`is_variant`) was also incorrectly marked `read_only` when the shader declares
`read_write`.

```rust
// Before:
let bgl = make_bgl(&device, &[true, true, false, false, false, false]);
// After:
let bgl = make_bgl(&device, &[true, false, false, false, false]);
```

### Bug 2: ODE Shader f64 Builtin Incompatibility

**File:** `crates/barracuda/src/shaders/numerical/batched_qs_ode_rk4_f64.wgsl`

WGSL builtins `max()`, `pow()`, `clamp()` don't support f64 in naga/wgpu 22.1.0.
The shader used these with f64 arguments, causing naga validation errors.
Additionally, AbstractFloat literals (`0.0`, `1.0`, `2.0`) don't auto-promote
to f64 in function call arguments.

**Fix pattern:** Manual f64-safe functions + `(zero + literal)` pattern:

```wgsl
fn fmax(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}

fn hill(x: f64, K: f64, n: f64) -> f64 {
    let z = x - x; // f64 zero
    let xc = fmax(x, z);
    let xn = fpow(xc, n);
    ...
}
```

**Key insight:** Function names must NOT use `_f64` suffix (e.g., `fmax` not
`max_f64`) because `ShaderTemplate::substitute_fossil_f64()` rewrites `max_f64(`
→ `max(`, defeating the polyfill. Use `fmax`, `fpow`, `fclamp` instead.

### Bug 3: Jacobi Eigenvector Rotation Missing Rows p, q

**File:** `crates/barracuda/src/shaders/linalg/batched_eigh_single_dispatch_f64.wgsl`

The V (eigenvector) rotation was inside the `if (k != p && k != q)` guard that's
correct for A (the 2×2 diagonal block is updated separately), but eigenvectors
require rotation for **all** rows k including k == p and k == q.

```wgsl
// Before: V rotation inside A guard (WRONG)
for (var k = 0u; k < n; k++) {
    if (k != p && k != q) {
        // A rotation...
        // V rotation...  ← skips rows p, q
    }
}

// After: separate loops
for (var k = 0u; k < n; k++) {
    if (k != p && k != q) { /* A rotation only */ }
}
for (var k = 0u; k < n; k++) {
    /* V rotation for ALL rows */
}
```

**Also fixed:** Removed commented-out `// @unroll_hint 32` that matched
`contains("@unroll_hint")` and triggered the WgslLoopUnroller, producing
AbstractInt literals where u32 was required.

---

## Part 2: f64 WGSL Patterns for ToadStool Evolution

These patterns should be applied to all f64 shaders:

1. **No WGSL builtins for f64 math:** `max()`, `pow()`, `clamp()` are f32-only
   in naga. Use manual polyfills (`fmax`, `fpow`, `fclamp`).

2. **`(zero + literal)` for f64 constants:** Bare `0.0`, `1.0`, `2.0` literals
   are AbstractFloat and fail as function arguments or may cause type mismatch.
   ```wgsl
   let z = x - x; // f64 zero from existing f64
   let one = z + 1.0;
   let eps = z + 1e-14;
   ```

3. **Avoid `_f64` function suffixes:** `substitute_fossil_f64()` rewrites
   `max_f64(` → `max(`. Name polyfills `fmax`, `fpow`, etc.

4. **`inject_missing_math_f64` for `exp_f64`/`log_f64`:** These are correctly
   auto-injected. Calling `exp_f64(e * log_f64(base))` is the correct pattern
   for f64 power.

5. **`@ilp_region` / `@unroll_hint` in comments:** The `contains()` check
   matches inside comments. Either remove the comment or use a different marker.

---

## Part 3: Full Control Validation Results

Every validation binary passes after fixes:

| Tier | Binary | Checks | Status |
|------|--------|--------|--------|
| CPU | `validate_barracuda_cpu_full` | 50/50 | PASS |
| CPU | `validate_barracuda_cpu_v7` | 43/43 | PASS |
| GPU | `validate_barracuda_gpu_full` | 24/24 | PASS |
| GPU | `validate_gpu_ode_sweep` | 12/12 | PASS |
| GPU | `validate_gpu_extended` | 53/53 | PASS |
| GPU | `validate_gpu_track1c` | 27/27 | PASS |
| GPU | `validate_gpu_rf` | 13/13 | PASS |
| GPU | `validate_gpu_hmm_forward` | 13/13 | PASS |
| GPU | `validate_gpu_phylo_compose` | 15/15 | PASS |
| GPU | `validate_local_wgsl_compile` | 13/13 | PASS |
| GPU | `validate_diversity_gpu` | 38/38 | PASS |
| GPU | `validate_pure_gpu_streaming` | 80/80 | PASS |
| GPU | `validate_pure_gpu_pipeline` | 31/31 | PASS |
| CPU vs GPU | `validate_cpu_vs_gpu_all_domains` | 48/48 | PASS |
| Mixed HW | `validate_metalforge_full_v3` | 28/28 | PASS |
| Mixed HW | `validate_metalforge_pipeline` | 45/45 | PASS |
| Mixed HW | `validate_pcie_direct` | 32/32 | PASS |
| Mixed HW | `validate_streaming_dispatch` | 25/25 | PASS |
| Cross-spring | `validate_cross_spring_evolution` | 39/39 | PASS |
| ToadStool | `validate_toadstool_bio` | 12/12 | PASS |
| ToadStool | `validate_cross_substrate` | 20/20 | PASS |
| ToadStool | `validate_cross_substrate_pipeline` | 17/17 | PASS |
| ToadStool | `validate_dispatch_overhead_proof` | 21/21 | PASS |
| ToadStool | `validate_substrate_router` | 20/20 | PASS |

---

## Part 4: Action Items for ToadStool

### Critical (merge these fixes)

1. **`snp.rs` line 39:** Change BGL to `&[true, false, false, false, false]`
2. **`batched_qs_ode_rk4_f64.wgsl`:** Replace `max/pow/clamp` with `fmax/fpow/fclamp` polyfills using `(zero + literal)` pattern
3. **`batched_eigh_single_dispatch_f64.wgsl`:** Separate V rotation into its own all-rows loop; remove `@unroll_hint` from comments

### Recommended (improve robustness)

4. Audit ALL f64 shaders for bare `max/pow/clamp` calls — replace with polyfills
5. Fix `WgslLoopUnroller` to emit `0u` not `0` for integer loop variables
6. Fix `substitute_fossil_f64` to not match inside comments
7. Add `--check` flag to `compile_shader_f64` that catches naga errors early

---

## Appendix: Files Changed

### ToadStool (upstream fixes)
- `crates/barracuda/src/ops/bio/snp.rs` — BGL binding count + read_only fix
- `crates/barracuda/src/shaders/numerical/batched_qs_ode_rk4_f64.wgsl` — f64 polyfills
- `crates/barracuda/src/shaders/linalg/batched_eigh_single_dispatch_f64.wgsl` — V rotation + @unroll_hint removal

### wetSpring (local)
- `barracuda/src/bio/gemm_cached.rs` — `const fn` (clippy)
- `barracuda/src/bio/ode_sweep_gpu.rs` — doc backtick fix
- `barracuda/src/gpu.rs` — doc backtick fix
- `experiments/098_upstream_gpu_fixes.md` — new experiment
