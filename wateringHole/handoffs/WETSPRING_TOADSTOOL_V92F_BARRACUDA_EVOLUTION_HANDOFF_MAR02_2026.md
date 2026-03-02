# wetSpring → ToadStool/BarraCuda: V92F Comprehensive Evolution Handoff

**Date:** March 2, 2026
**From:** wetSpring V92F
**To:** ToadStool/BarraCuda team
**ToadStool pin:** S86 (`2fee1969`)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring has completed a full validation cycle at ToadStool S86, proving the
cross-spring evolution model works end-to-end. This handoff documents:
1. Everything wetSpring consumes from barracuda (144 primitives, 264 dispatch ops)
2. What we found that needs ToadStool attention (API breakage, feature-gate bugs)
3. Performance benchmarks on real hardware (RTX 4070, DF64 Hybrid)
4. Learnings about cross-spring shader evolution that benefit all springs
5. Paper control chain validation (52 papers × 5 tiers)

---

## Part 1: What wetSpring Consumes

### Primitive Count: 144 (always-on, zero fallback)

wetSpring uses `barracuda` as an always-on dependency: `default-features = false`
for CPU builds, `barracuda/gpu` for GPU builds. There is zero `#[cfg(not(feature = "gpu"))]`
fallback code — every math operation delegates to ToadStool.

### CPU Primitives (47 modules)

| Domain | Functions Used | Origin |
|--------|--------------|--------|
| **Stats** | mean, variance, std, median, bootstrap_ci, jackknife_mean_variance, fit_all, pearson, spearman, correlation_matrix, mae, rmse, nse, kriging_weights, r_squared | groundSpring V54 |
| **Stats (hydrology)** | hargreaves_et0, fao56_et0, thornthwaite_et0, makkink_et0, turc_et0, hamon_et0, thornthwaite_heat_index | airSpring V039→S81 |
| **Stats (diversity)** | chao1_classic, chao2, ace, shannon, simpson, evenness, bray_curtis, unifrac_unweighted, unifrac_weighted | wetSpring V6 |
| **Linalg** | ridge_regression, nmf, cosine_similarity, graph_laplacian, effective_rank, belief_propagation_chain, disordered_laplacian | neuralSpring V64 |
| **Special** | erf, erfc, ln_gamma, regularized_gamma, dot, l2_norm | hotSpring S58 |
| **Numerical** | trapz, numerical_hessian | airSpring |
| **Spectral** | anderson_eigenvalues, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio, classify_spectral_phase, spectral_bandwidth, hofstadter_butterfly, almost_mathieu_hamiltonian, tridiagonal_eigensolve | hotSpring v0.6.0→S83 |
| **Sample** | boltzmann_sampling, latin_hypercube, sobol_scaled, maximin_lhs | wateringHole V69 |
| **ODE** | All 5 bio systems via trait-generated WGSL (QS, phage, cooperation, bistable, multi-signal) | wetSpring V8→S58 |

### GPU Primitives (42 modules, all lean)

| Domain | GPU Ops | ComputeDispatch Names |
|--------|---------|----------------------|
| **Bio diversity** | DiversityFusionGpu | `diversity_fusion` |
| **Bio alignment** | SmithWatermanGpu | `SW BandedF64` |
| **Bio phylo** | FelsensteinGpu, HmmBatchForwardF64 | `felsenstein`, `hmm_forward_f64` |
| **Bio ecology** | BatchedMultinomialGpu, WrightFisherGpu | `batched_multinomial`, `wright_fisher` |
| **Linalg** | GemmF64, GemmCached, BrayCurtisF64 | `GemmF64`, `bray_curtis_f64` |
| **ODE** | 5 × BatchedOdeRK4 | `rk4_{system}` |
| **Drug repurposing** | NMF via GEMM, TranseScoreF64, TopK | `transe_score_f64`, `TopK` |
| **Stats** | JackknifeMeanGpu, HistogramGpu, KimuraGpu, ChiSquaredBatchGpu | `bootstrap_mean`, `chi_squared_f64` |
| **Signal** | PeakDetectF64 | `peak_detect_f64` |
| **Precision** | DF64 pack/unpack, compile_shader_universal | (infrastructure) |

---

## Part 2: Issues Found and Fixed

### Feature-Gate Bugs (CRITICAL — V92E)

We found and fixed 4 feature-gate bugs in ToadStool S86. These were committed
directly to ToadStool (`7e01ac7e`):

| File | Bug | Fix |
|------|-----|-----|
| `src/lib.rs` | `pub mod spectral` gated behind `#[cfg(feature = "gpu")]` | Ungated (pure CPU code) |
| `src/lib.rs` | `pub mod sample` gated behind `#[cfg(feature = "gpu")]` | Ungated (mixed CPU/GPU) |
| `src/linalg/mod.rs` | `pub mod graph` gated behind `#[cfg(feature = "gpu")]` | Ungated (pure CPU code) |
| `src/spectral/mod.rs` | `pub mod batch_ipr` was ungated | Correctly gated (uses wgpu) |
| `src/sample/mod.rs` | WGSL statics at module root blocked CPU builds | GPU statics + submodules gated |
| `src/stats/mod.rs` | GPU-only hydrology types unconditionally re-exported | GPU types gated |

**Root cause:** When new GPU-dependent code was added to modules, the entire
module was gated rather than just the GPU-dependent parts. This is easy to miss
because `cargo check --all-features` passes (it's the default-features build that breaks).

**Recommendation:** Add a CI job that runs `cargo check` (no features) to catch
this class of bug. Consider splitting GPU-dependent code into separate submodules
(`spectral::gpu`, `sample::gpu`) to make feature boundaries explicit.

### API Breakage (V92F)

`BatchedMultinomialGpu::sample` changed signature between S79 and S86:
- `seeds: &mut Vec<u32>` → `seeds: Option<&mut Vec<u32>>`
- New required parameter: `config: BatchedMultinomialConfig`

This broke `wetspring_barracuda::bio::rarefaction_gpu`. We fixed it, but
downstream springs have no way to know about breaking changes without trying to compile.

**Recommendation:** Consider semver-aware changelogs for breaking API changes,
or a `BREAKING_CHANGES.md` per session.

---

## Part 3: Performance Benchmarks (RTX 4070)

### Hardware: NVIDIA GeForce RTX 4070
- Fp64Strategy: **Hybrid** (DF64 double-float on 5888 FP32 cores)
- Precision: **Df64** (no native f64; uses double-float pairs)
- NVK workarounds: Not needed (proprietary driver)

### Exp297 Benchmark Results

| Operation | GPU (ms) | CPU (ms) | Speedup | Notes |
|-----------|---------|---------|---------|-------|
| GEMM 128×128 | 18.1 | 3.7 | 0.2× | First dispatch; pipeline compile overhead |
| GemmCached 64×32×16 | 13.5 | — | — | B-matrix stays on device |
| DiversityFusion 500 taxa | 74.6 | 0.004 | — | Pipeline compile dominates |
| BrayCurtis 20×200 | 2.7 | — | — | Condensed distance matrix |
| Anderson 1D n=1000 | — | 348 | — | Dense eigensolve, CPU-only |
| Lanczos 200 steps | — | 9.8 | — | Sparse iterative |
| Bootstrap 200×50k | — | 24.2 | — | Resampling |
| Boltzmann 5k×2D | — | 0.26 | — | Metropolis MCMC |
| Sobol 10k×5D | — | 0.30 | — | Low-discrepancy |
| LHS 10k×5D | — | 0.33 | — | Latin hypercube |

### Key performance observations

1. **Pipeline compile overhead dominates small dispatches.** DiversityFusion on
   500 taxa is 0.004 ms on CPU but 75 ms on first GPU dispatch. For repeated
   calls (rarefaction loops), GPU amortizes well.

2. **DF64 max error: 1.75e-5 for GEMM 128×128.** This is expected for double-float
   on FP32 cores. For ML (neuralSpring), this is fine. For spectral theory
   (hotSpring eigensolve), CPU f64 is preferred.

3. **GPU wins at scale.** BrayCurtis on 20×200 at 2.7 ms benefits from O(n²)
   parallelism. The crossover where GPU beats CPU depends on problem size and
   pipeline compile caching.

---

## Part 4: Cross-Spring Evolution Learnings

### The bidirectional flow works

Every spring contributed primitives that other springs now use:

| Origin | → ToadStool | Back to other springs |
|--------|------------|----------------------|
| hotSpring | DF64 precision, NVK workarounds, Anderson spectral | ALL springs use DF64; neuralSpring uses SpectralBridge |
| wetSpring | Bio diversity, 5 ODE systems, alignment, phylo | neuralSpring brain diversity; groundSpring ecological |
| neuralSpring | GemmF64, graph linalg, AlphaFold2, BatchedEncoder | wetSpring NMF via GEMM; all springs use dispatch |
| airSpring | 6 ET₀ methods, seasonal pipeline | groundSpring soil moisture; wetSpring environmental |
| groundSpring | Bootstrap, Wright-Fisher, topology | ALL springs use bootstrap |
| wateringHole | Boltzmann, Sobol, LHS, chi-squared | ALL springs use sampling |

### What evolved well

1. **ComputeDispatch builder** — eliminated 80+ lines of bind-group boilerplate
   per GPU op. Every spring benefits from this S72 improvement.

2. **compile_shader_universal()** — single call handles F16/F32/F64/DF64. The
   precision layer is completely transparent to consumers.

3. **FitResult named accessors (S81)** — `slope()`, `intercept()`, `coefficients()`
   make regression results ergonomic across all domains.

4. **The Write → Absorb → Lean cycle** — wetSpring has ZERO local WGSL, ZERO
   local ODE derivative code, ZERO local regression math. Everything delegates
   to ToadStool. The elimination of 30,424 bytes of local WGSL was clean.

### What needs attention

1. **Feature-gate discipline** — see Part 2. CPU modules ending up behind GPU
   gates is a systemic risk.

2. **API stability** — `BatchedMultinomialGpu::sample` breakage is not the only
   one possible. As ToadStool evolves rapidly (7 sessions S80-S86), downstream
   springs need a way to discover breaking changes.

3. **Anderson eigensolve at scale** — `anderson_eigenvalues(n, W, seed)` does a
   full dense eigensolve (O(n³)). For n=2000, this is 1.3s. The GPU path
   (BatchIprGpu) would help, but it requires the full CSR matrix pipeline.

---

## Part 5: Paper Control Chain

wetSpring validated 52 papers through the complete 5-tier control chain:

```
Paper math → BarraCuda CPU → BarraCuda GPU → Streaming → metalForge
  Exp291        Exp292          Exp293          Exp294       Exp295
  45/45         40/40           35/35           16/16        28/28
```

All 39 actionable papers (25 Tracks 1-2 + 5 Track 3 + 9 Track 4) have full
three-tier coverage (CPU + GPU + metalForge). This validates that ToadStool's
abstraction layers preserve mathematical correctness from published equations
through every substrate.

---

## Part 6: Action Items for ToadStool

### P0 (Breaking)
1. **CI: `cargo check` without features** — catches the feature-gate bug class
2. **BREAKING_CHANGES.md** — per-session log of API changes

### P1 (Performance)
3. **Pipeline compile caching** — first-dispatch penalty dominates small problems
4. **Anderson GPU batch eigensolve** — n=2000 at 1.3s CPU could benefit from
   BatchIprGpu for disorder-averaged studies

### P2 (Evolution)
5. **Separate GPU submodules** — `spectral::gpu`, `sample::gpu`, `linalg::gpu`
   instead of mixing CPU and GPU code in the same module
6. **Cross-spring provenance in code** — the `//! Provenance:` comments are
   excellent; consider making them machine-readable for tooling

### P3 (Documentation)
7. **SPRING_ABSORPTION.md** is comprehensive — keep it updated per session
8. **EVOLUTION_TRACKER.md** principles are solid — the "deep debt over shortcuts"
   philosophy pays off measurably

---

## Quality Gates

| Gate | Status |
|------|--------|
| Exp296 (CPU rewire) | 64/64 PASS |
| Exp297 (GPU cross-spring) | 46/46 PASS |
| 1,089 unit tests | PASS |
| `cargo clippy --all-features -W pedantic` | CLEAN |
| `cargo fmt --all -- --check` | CLEAN |
| `cargo test` (full suite) | PASS |
| Paper-math chain (Exp291-295) | 164/164 PASS |
