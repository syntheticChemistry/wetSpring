# wetSpring → ToadStool/BarraCUDA Handoff V92E — S86 Rewire

**Date:** March 2, 2026
**From:** wetSpring (V92E)
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S86 (`2fee1969`)
**Previous pin:** S79 (`f97fc2ae`)
**Primitives consumed:** 93 → 144

---

## Summary

wetSpring rewired from ToadStool S79 to S86, absorbing 7 commits of evolution.
During the rewire, we discovered and fixed **3 feature-gate bugs** in barracuda
where pure CPU modules were incorrectly gated behind `#[cfg(feature = "gpu")]`:

1. **`spectral` module** — Anderson localization, Lanczos, Hofstadter, tridiagonal
   eigensolve, level spacing statistics. All pure CPU code, no GPU dependencies.
   Fix: ungated module, gated only `batch_ipr` (which uses wgpu).

2. **`linalg::graph` module** — Graph Laplacian, belief propagation, disordered
   Laplacian, effective rank. All pure CPU code.
   Fix: ungated module and re-exports.

3. **`sample` module** — Boltzmann/Metropolis sampling, Latin hypercube, Sobol
   sequences, random uniform. CPU samplers incorrectly blocked by GPU-gated
   WGSL statics in `mod.rs`.
   Fix: gated only WGSL statics and GPU-dependent submodules (`direct`, `sparsity`),
   kept CPU samplers (`lhs`, `metropolis`, `sobol`, `maximin`) always available.

## ToadStool S80-S86 Evolution Absorbed

| Session | Key Changes |
|---------|-------------|
| S80 | Nautilus reservoir computing (22 tests), BatchedEncoder (46-78× fused), fused_mlp, StatefulPipeline, Batch Nelder-Mead GPU, NVK driver workarounds, NeighborMode::PrecomputedBuffer |
| S81 | InterconnectTopology, SubstratePipeline, 4 ET₀ methods (Thornthwaite/Makkink/Turc/Hamon), anderson_eigenvalues, complex_polyakov_average, FitResult named accessors, BarracudaError::Io+Json |
| S82 | 16 ComputeDispatch ops (FHE, lattice QCD, audio/signal, bio), OS memory detection (/proc/meminfo), creation.rs DRY refactor |
| S83 | BrentGpu, anderson_4d, OmelyanIntegrator, RichardsGpu, L-BFGS, BatchedStatefulF64, SpectralBridge, HeadKind generalization |
| S84-S86 | +33 ComputeDispatch ops (matmul_tiled, gemm_f64, losses, ML ops), hydrology dir split, experimental real probes, 2,866 barracuda tests |

## New Primitives Available to wetSpring

### CPU (always available, no feature gate)
- `spectral::anderson_eigenvalues`, `anderson_2d`, `anderson_3d`, `anderson_hamiltonian`
- `spectral::lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`
- `spectral::classify_spectral_phase`, `SpectralAnalysis`, `SpectralPhase`
- `spectral::detect_bands`, `spectral_bandwidth`, `spectral_condition_number`
- `spectral::hofstadter_butterfly`, `almost_mathieu_hamiltonian`, `GOLDEN_RATIO`
- `spectral::find_all_eigenvalues`, `sturm_count`
- `linalg::graph_laplacian`, `disordered_laplacian`, `effective_rank`
- `linalg::belief_propagation_chain`
- `sample::boltzmann_sampling`, `BoltzmannResult`
- `sample::latin_hypercube`, `random_uniform`
- `sample::sobol_scaled`, `sobol_sequence`, `SobolGenerator`
- `stats::thornthwaite_et0`, `thornthwaite_heat_index`
- `stats::makkink_et0`, `turc_et0`, `hamon_et0`
- `stats::FitResult::slope()`, `intercept()`, `coefficients()`

### GPU (requires `gpu` feature)
- `spectral::BatchIprGpu`
- `sample::direct_sampler`, `sparsity` module
- All 144 ComputeDispatch ops

## Validation

- **Exp296**: `validate_cross_spring_s86` — 64/64 PASS
  - Spectral: Anderson 1D/2D/3D, Lanczos, level spacing, Hofstadter
  - Graph: Laplacian, belief propagation, effective rank
  - Sample: Boltzmann, LHS, Sobol
  - Hydrology: 4 new ET₀ methods
  - Regression: FitResult named accessors, fit_all
  - Tridiagonal eigensolve
  - Post-rewire regression: erf, ln_gamma, Hargreaves, FAO-56, norm_cdf, pearson

## Quality Gates

| Gate | Status |
|------|--------|
| `cargo check` (CPU) | CLEAN |
| `cargo check --all-features` (GPU) | CLEAN |
| `cargo clippy -- -W clippy::pedantic` | 0 warnings |
| `cargo fmt --all` | CLEAN |
| `cargo test` | 1,044 PASS, 0 FAIL |
| Exp296 validation | 64/64 PASS |

## Action Items for ToadStool

1. **Absorb feature-gate fixes** — the 3 bugs fixed in this rewire should be
   committed to ToadStool main:
   - `src/lib.rs`: `spectral` ungated, `sample` ungated
   - `src/spectral/mod.rs`: `batch_ipr` gated, rest ungated
   - `src/sample/mod.rs`: WGSL statics + `direct`/`sparsity` gated, rest ungated
   - `src/linalg/mod.rs`: `graph` ungated
   - `src/stats/mod.rs`: hydrology GPU types properly gated

2. **Consider CPU/GPU split** for more modules — the hydrology split (S84) is a
   good pattern. Other candidates: `spectral` (most CPU, only `batch_ipr` GPU),
   `stats` (most CPU, only GPU kernels need gating).

3. **WGSL shader static pattern** — `LazyLock` statics that reference
   `crate::shaders::precision` should be consistently `#[cfg(feature = "gpu")]`
   to avoid blocking CPU-only builds.

## Provenance

| Field | Value |
|-------|-------|
| ToadStool pin | S86 (`2fee1969`) |
| Previous pin | S79 (`f97fc2ae`) |
| Commits absorbed | 7 (S80-S86) |
| Validation | Exp296: 64/64 PASS |
| Tests | 1,044 PASS, 0 FAIL |
| Feature-gate bugs fixed | 3 (spectral, graph, sample) |
