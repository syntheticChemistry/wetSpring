# SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring V93 → barraCuda/toadStool Evolution Feedback

**Date:** 2026-03-03
**From:** wetSpring team (V93)
**To:** barraCuda team, toadStool team
**barraCuda version:** v0.3.1 (standalone)
**wetSpring tests:** 1,044 passed, 0 failed
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring successfully rewired to standalone barraCuda v0.3.1 with zero code
changes (path swap only). This document captures everything we learned during
the rewire and deep debt evolution that may be useful for barraCuda and
toadStool's continued evolution.

---

## 1. Rewire Experience

### What Went Right

- **Zero API breakage**: 1,044 tests pass with only a Cargo.toml path change.
  This validates the clean extraction — the API surface is stable.
- **`WgpuDevice::from_existing()`**: wetSpring creates its own wgpu device
  and wraps it. This pattern works perfectly with standalone barraCuda.
- **Feature gates**: `default-features = false` on barraCuda works correctly.
  CPU-only builds are clean. GPU feature enables GPU ops as expected.
- **MSRV 1.87**: No issues. wetSpring was on 1.85; bump was trivial.

### What Could Be Better

- **`cargo fmt --all` follows path deps**: When running `cargo fmt --all` from
  wetSpring, it follows the path dependency into `barraCuda/` and tries to
  resolve *its* workspace members (barracuda-core → sourdough-core). This fails
  if sourDough isn't checked out. Workaround: `cargo fmt -p wetspring-barracuda`.
  Consider adding `[workspace]` exclusion patterns or documenting this.

---

## 2. Primitives Consumed (144)

wetSpring consumes 144 barraCuda primitives across these domains:

| Domain | Count | Key Primitives |
|--------|:-----:|----------------|
| Bio diversity | 12 | FusedMapReduceF64, BrayCurtisF64, DiversityFusionGpu |
| Bio ODE | 10 | BatchedOdeRK4 (5 systems × generate_shader + integrate_cpu) |
| Linalg | 8 | GemmF64, GemmCachedF64, BatchedEighGpu, graph_laplacian |
| Spectral | 6 | anderson_eigenvalues, lanczos, lanczos_eigenvalues, level_spacing |
| Stats | 8 | shannon, simpson, hill, pearson, bootstrap_ci, fit_* |
| Sample | 4 | boltzmann_sampling, metropolis, latin_hypercube, sobol |
| Bio GPU | 40+ | ANI, SNP, dN/dS, HMM, kmer, UniFrac, Felsenstein, etc. |
| Kriging | 2 | KrigingF64 spatial interpolation |
| Special | 4 | erf, ln_gamma, regularized_gamma, normal_cdf |
| Numerical | 4 | numerical_hessian, brent_minimize, nelder_mead |
| Device | 6 | WgpuDevice, TensorContext, GpuDriverProfile, Fp64Strategy |

### Primitives We'd Like to See Evolve

| Request | Priority | Notes |
|---------|----------|-------|
| `ComputeDispatch` tarpc adoption | P3 | wetSpring currently calls primitives directly; tarpc dispatch would enable transparent hardware routing |
| DF64 GEMM public API | P3 | `wgsl_shader_for_device()` is private; wetSpring can't access DF64 GEMM paths directly |
| `BandwidthTier` in device profile | P3 | metalForge could use this for smarter substrate routing |
| `domain-genomics` extraction | P1 | If barraCuda extracts genomics domain models, wetSpring would consume them |

---

## 3. Precision Observations

wetSpring validates f64 precision across 36+ domains. Observations for the
precision team:

- **FusedMapReduceF64**: GPU ↔ CPU parity within 1e-6 for Shannon/Simpson.
  The `GPU_VS_CPU_F64` tolerance of 1e-6 is tight but consistently achieved
  on RTX 4070 (Vulkan) and Titan V.
- **BatchedOdeRK4**: CPU `integrate_cpu()` matches scipy.integrate.odeint
  to within 4.44e-16 (machine epsilon) for all 5 bio ODE systems. GPU path
  matches CPU to within 1e-6.
- **GemmF64**: 7.1x GPU speedup on 256x256 matrices. Parity within 1e-10.
- **DF64 roundtrip**: `pack → unpack` preserves to 1e-13 (documented as
  `DF64_ROUNDTRIP` tolerance).
- **Transcendental functions**: GPU `exp`/`log` polyfills on NVK drivers
  need workarounds (detected via `GpuDriverProfile::needs_exp_f64_workaround()`).
  This works well — thank you for the driver profile infrastructure.

---

## 4. Deep Debt Patterns (Lessons Learned)

### Tolerance Centralization

wetSpring maintains 106 named tolerance constants with scientific justification.
Key insight: tolerance constants should be organized by **domain** (bio, gpu,
spectral, instrument) rather than by **magnitude**. We use:

```
tolerances/
├── mod.rs          (machine precision, special functions, numerical guards)
├── gpu.rs          (GPU vs CPU parity)
├── spectral.rs     (Anderson, dynamic systems)
├── instrument.rs   (measurement error)
└── bio/
    ├── alignment.rs, diversity.rs, esn.rs, ode.rs, phylogeny.rs, misc.rs, brain.rs
```

### Hardcoding Evolution

All NCBI API URLs are now configurable via environment variables with sensible
defaults. Data directories follow XDG standards. This pattern is recommended
for any primal that accesses external services:

```rust
fn api_base() -> String {
    std::env::var("PRIMAL_API_URL").unwrap_or_else(|_| DEFAULT_URL.to_owned())
}
```

### Test Extraction Pattern

For files where tests exceed 40% of content, extracting to `module/tests.rs`
with `#[cfg(test)] mod tests;` in `module/mod.rs` keeps production code
readable. Critical: use `#![allow(...)]` (inner attribute) not `#[allow(...)]`
(outer attribute) in the extracted test file.

---

## 5. Architecture After Rewire

```
wetSpring (validation Spring)
  ├── barracuda/     → barraCuda v0.3.1 (standalone math primal, direct dep)
  │   ├── 144 primitives consumed, 767+ WGSL shaders available
  │   └── Universal precision: f64/f32/f16/Df64 per hardware
  ├── metalForge/    → barraCuda v0.3.1 (substrate routing, direct dep)
  └── akida-driver   → toadStool neuromorphic (independent, optional)
```

No reverse dependencies. No toadStool dependency. Springs depend on barraCuda
directly for math, and toadStool orchestrates hardware routing separately.

---

## 6. What wetSpring Can Contribute Back

| Contribution | Type | Status |
|-------------|------|--------|
| 47 bio algorithm implementations | Domain knowledge | Available for `domain-genomics` extraction |
| 106 tolerance constants with justification | Quality infrastructure | Pattern for other primals |
| 57 Python baseline scripts | Validation reference | Reproducible with `requirements.txt` |
| 280 experiment protocols | Methodology | Documented in `experiments/` |
| Precision validation data (36+ domains) | Benchmark data | Available for regression testing |
| metalForge substrate routing | Hardware discovery | Could inform toadStool dispatch |
