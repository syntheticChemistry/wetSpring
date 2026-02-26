# wetSpring → ToadStool/BarraCuda V53 Cross-Spring Evolution Handoff

**Date:** February 26, 2026
**From:** wetSpring (biomeGate)
**To:** ToadStool/BarraCuda team
**Phase:** 53 (V53 — cross-spring evolution benchmarks, doc cleanup, archive)
**ToadStool pin:** `045103a7` (S66 Wave 5)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring V53 completes a full audit-and-benchmark pass confirming cross-spring
evolution is working as designed. 7 benchmark/validator binaries PASS on RTX 4070,
documenting provenance, GPU speedup, and upstream benefit from the Write → Absorb →
Lean cycle. 79 primitives consumed. Zero local WGSL, derivative, or regression math.

This handoff documents what ToadStool should know for its next evolution session:
observed GPU performance, tolerance learnings, and absorption opportunities.

---

## Part 1: GPU Performance Data (RTX 4070, Ada, Feb 26 2026)

### Cross-Spring Scaling (Exp095)

| Primitive | Origin | Problem Size | CPU (µs) | GPU (µs) | Speedup |
|-----------|--------|-------------|----------|----------|---------|
| PairwiseJaccard | neuralSpring | 200×2000 (20K pairs) | 41,777 | 342 | **122×** |
| SpatialPayoff | neuralSpring | 256×256 (65K cells) | 1,026 | 47 | **22×** |
| PairwiseHamming | neuralSpring | 500×1000 (125K pairs) | 10,383 | 1,039 | **10×** |
| LocusVariance | neuralSpring | 100×10K (1M elems) | 1,079 | 154 | **7×** |
| BatchFitness | neuralSpring | 4096×256 (1M elems) | 470 | 79 | **6×** |
| GemmF64 | wetSpring | 256×256 | 3,479 | 3,643 | 1.0× |

### ODE CPU Parity (5 systems, upstream vs local)

| System | Vars | Local CPU (µs) | Upstream CPU (µs) | Speedup | Max Diff |
|--------|------|---------------|-------------------|---------|----------|
| Capacitor | 6 | 2,020 | 1,581 | 1.28× | 0.00 |
| Cooperation | 4 | 825 | 696 | 1.19× | 4.44e-16 |
| MultiSignal | 7 | 1,592 | 1,214 | 1.31× | 4.44e-16 |
| Bistable | 5 | 1,738 | 1,424 | 1.22× | 0.00 |
| PhageDefense | 4 | 82 | 63 | 1.30× | extreme params |

Upstream `integrate_cpu` is **18-31% faster** than local — ToadStool's optimized
hot loop (post-absorption) benefits wetSpring measurably.

### GPU ODE (128 batches each, release mode)

| System | GPU Time (ms) |
|--------|--------------|
| Bistable | 57.6 |
| MultiSignal | 28.9 |
| Capacitor | 11.3 |
| Cooperation | 6.3 |
| PhageDefense | 5.1 |

---

## Part 2: Tolerance Learnings for ToadStool

### Chained Transcendentals Need Wider Tolerance

V51 GPU validation revealed that chained transcendental operations (Shannon
entropy = `sum(p * log(p))`, spectral cosine = `dot / (norm * norm)` where
norms involve `sqrt(sum(x²))`) accumulate error beyond `GPU_VS_CPU_TRANSCENDENTAL`
(1e-10) on Ada architecture (RTX 4070).

**Recommendation:** ToadStool should document or provide a tolerance constant for
chained transcendental operations: **`GPU_LOG_POLYFILL` = 1e-7** (wetSpring's name).
This is empirically safe for:
- Shannon entropy from raw counts (log + multiply + sum)
- Simpson index from raw counts (sum of squares)
- Spectral cosine similarity (dot + sqrt + divide)
- Pielou evenness (Shannon / log(S))

Single transcendental calls (`log(x)`, `exp(x)`) remain within 1e-10.

### f32 → f64 Migration Pattern

When upstream shaders evolve buffer types (e.g., `BatchFitnessGpu` moving from
`f32` to `f64` in S65+), downstream consumers silently produce garbage unless
they update buffer allocation (`size * 4` → `size * 8`) and readback types.

**Recommendation:** ToadStool could add a compile-time or runtime check that
buffer element size matches shader expectations, or document breaking changes
in shader type evolution more prominently.

---

## Part 3: Absorption Candidates

### Ready to absorb (wetSpring → ToadStool)

| Item | Location | Effort | Benefit |
|------|----------|--------|---------|
| `QsBiofilmOde` system | `bio/qs_biofilm.rs` | Low | 6th bio ODE for trait ecosystem |
| `rk4_trajectory` helper | `bio/ode_solvers.rs` | Low | Returns full trajectory, not just final state |
| `ConvergenceGuard` pattern | `bio/multi_signal_gpu.rs` | Medium | c-di-GMP stability for stiff ODE |
| `DiversityFusionGpu` config presets | `bio/diversity_fusion_gpu.rs` | Low | Pre-built configs for common community sizes |
| Streaming FASTQ/mzML/MS2 parsers | `bio/io/` | Medium | Zero-copy I/O for bioinformatics |

### Patterns worth absorbing (architecture, not code)

| Pattern | What it demonstrates |
|---------|---------------------|
| Tolerance constant hierarchy | `ANALYTICAL_F64` (1e-12) → `GPU_VS_CPU_TRANSCENDENTAL` (1e-10) → `GPU_LOG_POLYFILL` (1e-7) → `MATRIX_EPS` (1e-14) |
| Three-tier benchmark harness | Python baseline → Rust CPU → GPU, wall-time capture per tier |
| `check()` macro for validation | `(label, actual, expected, tolerance)` → structured pass/fail with report |
| metalForge cross-substrate validation | Same math on CPU/GPU/NPU, diff < tolerance |

### NOT candidates (domain-specific)

- Pangenome analysis, taxonomy classifier, phylogenetic bootstrap
- FASTQ/mzML domain-specific logic (only parsers are absorption candidates)
- NCBI HTTP client (infrastructure, not compute)

---

## Part 4: Cross-Spring Evolution Provenance

### What each spring contributed → ToadStool → what others consumed

```
hotSpring (physics)
  → f64 precision polyfills, DF64 core-streaming (14 shaders)
  → Anderson Hamiltonian, Lanczos eigensolver, find_w_c
  → ESN reservoir, RK4/RK45 adaptive ODE
  → Jacobi eigendecomposition (BatchedEighGpu)
  CONSUMED BY: wetSpring (GPU ODE, PCoA, Anderson spectral, ESN NPU)

wetSpring (biology)
  → 5 bio ODE systems → OdeSystem trait + BatchedOdeRK4
  → DiversityFusion WGSL (fused Shannon+Simpson+J')
  → 11 diversity metrics → stats::diversity
  → GemmCachedF64 (60× taxonomy speedup)
  → Tolerance constant pattern
  CONSUMED BY: airSpring (crop biodiversity), neuralSpring (population genetics ODE)

neuralSpring (ML/population)
  → PairwiseHamming, Jaccard, L2 distance matrices
  → SpatialPayoff, BatchFitness, LocusVariance
  → graph_laplacian, belief_propagation
  → xoshiro128ss PRNG, logsumexp_reduce
  CONSUMED BY: wetSpring (SNP distance, fitness eval, community networks)

airSpring (agriculture)
  → Richards PDE, moving_window_stats, van Genuchten
  → Kriging spatial interpolation, IoT sensor fusion
  → fit_linear, fit_quadratic, fit_exponential regression suite
  CONSUMED BY: wetSpring (Heap's law via fit_linear), groundSpring (hydrology)
```

### Session Timeline (S39 → S66)

| Session | Key Evolution |
|---------|--------------|
| S39-S44 | hotSpring precision → ToadStool core (ShaderTemplate, Fp64Strategy, NVK workarounds) |
| S45-S50 | neuralSpring ML → ToadStool ops (pairwise metrics, evolutionary computation) |
| S51-S58 | wetSpring bio → ToadStool (ODE trait, Gillespie, SmithWaterman, GemmF64, diversity) |
| S58 | hotSpring DF64 → ToadStool (14 double-float f32-pair shaders for consumer GPU) |
| S60-S62 | ToadStool consolidation (SparseGemm, TranseScore, TopK, PeakDetect, BGL helpers) |
| S63 | wetSpring Write→Absorb→Lean completes (DiversityFusion absorbed, local WGSL deleted) |
| S64 | Cross-spring stats absorption (11 diversity + 2 metrics + 8 lattice shaders) |
| S65 | Smart refactoring (compute_graph 819→522, files < 1000 LOC, 2490 tests) |
| S66 | Multi-precision expansion (regression, metrics, bootstrap, hydrology, DF64 compiler) |

---

## Part 5: Verification

```
cargo test --lib                  → 823 pass, 0 fail
cargo clippy --features gpu       → 0 warnings (pedantic + nursery)
cargo fmt                         → 0 diffs
GPU validators (70)               → 1,578 checks PASS (RTX 4070)
CPU validators (57)               → 917 checks PASS
Cross-spring benchmarks (7)       → 95/95 checks PASS
Total validation checks           → 4,494+
ToadStool primitives consumed     → 79
Local WGSL                        → 0
Local derivative math             → 0
Local regression math             → 0
```

---

## Part 6: What's Next

wetSpring is fully lean. Remaining evolution opportunities are incremental:

1. **DF64 path for FP32-only GPUs** — wetSpring ODE shaders currently require native f64; the `compile_shader_df64` path (S66) could enable consumer GPUs without f64 hardware
2. **`HillFunctionF64` GPU op** — ToadStool now has a GPU Hill shader (`ops::hill_f64`); wetSpring could use it for batch Hill activation in QS parameter sweeps instead of CPU-only `barracuda::stats::hill`
3. **`stats::bootstrap_ci`** — could replace local rarefaction bootstrap variance estimation if API aligns
4. **`moving_window_stats_f64`** — potential use for bloom time-series analysis if wetSpring adds temporal ecology

No blocking requests for ToadStool. All 9/9 P0-P3 evolution requests remain DONE.
