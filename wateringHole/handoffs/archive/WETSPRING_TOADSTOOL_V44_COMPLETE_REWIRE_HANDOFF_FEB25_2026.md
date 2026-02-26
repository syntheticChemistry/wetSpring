# wetSpring → ToadStool V44 Handoff: Complete Cross-Spring Rewire

**Date:** February 25, 2026
**Phase:** 48 (V44 — complete cross-spring rewire + modern benchmark)
**Primitives consumed:** 53 (was 50 at V43)
**Evolution requests:** 8/9 P0-P3 delivered (unchanged)
**Experiment:** Exp169 `benchmark_cross_spring_modern` — 12/12 PASS

---

## Executive Summary

V44 completes the rewiring of wetSpring to modern ToadStool/BarraCuda primitives.
Three new upstream delegations replace local code in 6 validation binaries.
A new benchmark experiment (Exp169) validates the full CPU primitive surface with
per-primitive cross-spring provenance tracking.

---

## Part 1: Rewires Executed

### 1.1 Anderson Spectral: `find_last_downward_crossing` → `barracuda::spectral::find_w_c`

| File | Before | After |
|------|--------|-------|
| `validate_finite_size_scaling.rs` | Local `find_last_downward_crossing(&[(f64,f64)])` | `find_w_c(&[AndersonSweepPoint])` |
| `validate_geometry_zoo.rs` | Same local function | Same upstream delegation |
| `validate_correlated_disorder.rs` | Inline W_c loop (12 lines) | `find_w_c(&[AndersonSweepPoint])` |
| `validate_finite_size_scaling_v2.rs` | Inline W_c loop | `find_w_c(&[AndersonSweepPoint])` |

**Provenance:** `find_w_c` originated in hotSpring spectral theory (Kachkovskiy),
absorbed into ToadStool in S59. Linear interpolation on `⟨r⟩(W)` sweep to
find metal-insulator transition W_c.

### 1.2 Stats: `correlation_cpu`/`variance_cpu` → `barracuda::stats::pearson_correlation`

| File | Before | After |
|------|--------|-------|
| `validate_pure_gpu_pipeline.rs` | Local 20-line `correlation_cpu` + `variance_cpu` | `pearson_correlation(x, y).unwrap_or(0.0)` |

**Provenance:** `pearson_correlation` is ToadStool-native (stats module, S59).

### 1.3 Not Rewired (by design)

| Item | Reason |
|------|--------|
| 6 local `hill()` functions | CPU ODE derivative helpers — GPU equivalent is `generate_shader()` |
| Local `Validator` | API mismatch with upstream `ValidationHarness`, 158 binaries, no benefit |
| `tolerances.rs` (77 constants) | Complementary to upstream `barracuda::tolerances` struct system |
| `dot`/`l2_norm` | No upstream CPU export; local helpers used by 15+ binaries |

---

## Part 2: New Experiment — Exp169

`benchmark_cross_spring_modern` exercises the full CPU primitive surface:

| Section | Primitives | Origin |
|---------|-----------|--------|
| S1: CPU Math | `erf`, `ln_gamma`, `regularized_gamma_p` | ToadStool (A&S, Lanczos) |
| S2: CPU Stats | `norm_cdf`, `pearson_correlation` | ToadStool S59 |
| S3: V43 Rewire | `normal_cdf` delegation (bit-exact) | wetSpring V43 |
| S4: Numerical | `trapz` | ToadStool core |
| S5: Graph (GPU) | `graph_laplacian` | neuralSpring → ToadStool S54 |
| S6: Anderson (GPU) | `anderson_3d`, `lanczos`, `level_spacing_ratio` | hotSpring → ToadStool |
| S7: find_w_c (GPU) | `find_w_c` | hotSpring → ToadStool S59 |
| S8: Ridge (GPU) | `ridge_regression` | wetSpring → ToadStool S59 |
| S9: Provenance | Full 4-spring map | Documentation |

**Result:** 12/12 PASS (CPU path). GPU path exercises additional sections.

---

## Part 3: Cross-Spring Evolution Provenance

### Where Things Evolved to Be Helpful

**hotSpring precision → wetSpring biology:**
- f64 polyfills (naga workarounds) → all GPU shaders
- `PeakDetectF64` → LC-MS signal processing
- `BatchedEighGpu` (NAK-optimized) → PCoA ordination
- Anderson 2D/3D + Lanczos → QS-disorder coupling in biofilms
- `find_w_c` → phase transition detection (V44 rewire)

**wetSpring biology → neuralSpring population genetics:**
- ODE trait + `generate_shader()` → population dynamics models
- 15 bio GPU shaders → available for neuralSpring/airSpring consumption
- `ridge_regression`, `trapz`, `erf` → core CPU math for all springs
- Tolerance constant pattern → adopted by ToadStool (S52)

**neuralSpring ML → wetSpring ecology:**
- `PairwiseHammingGpu` → SNP-based strain typing
- `PairwiseJaccardGpu` → gene presence/absence ecology
- `SpatialPayoffGpu` → cooperation game theory in biofilms
- `graph_laplacian` → community network spectral analysis
- `belief_propagation_chain` → hierarchical taxonomy

**ToadStool infrastructure → all springs:**
- `FusedMapReduceF64` (FMR) → universal GPU primitive (12+ wetSpring modules)
- `GemmF64`/`GemmCachedF64` → matrix multiply (kriging, chimera, NMF)
- `barracuda::stats` → norm_cdf, pearson_correlation (V43-V44 rewires)
- `barracuda::tolerances` → complementary to spring-local constants

---

## Part 4: Updated Dependency Surface

| Category | Count | Details |
|----------|-------|---------|
| CPU math (always-on) | 7 | erf, ln_gamma, regularized_gamma_p, norm_cdf, trapz, ridge_regression, pearson_correlation |
| GPU bio ops | 15 | FelsensteinGpu, HmmBatchForwardF64, SnpCallingF64, etc. |
| GPU physics (cross-spring) | 8 | Anderson 2D/3D, Lanczos, level_spacing_ratio, find_w_c, anderson_sweep_averaged |
| GPU ML (cross-spring) | 8 | PairwiseHamming/Jaccard, SpatialPayoff, graph_laplacian, etc. |
| GPU core ops | 11 | FMR, GEMM, BatchedEigh, PeakDetect, TransE, SparseGemm, TopK, etc. |
| BGL helpers | 2 | storage_bgl_entry, uniform_bgl_entry |
| Trait-generated WGSL | 5 | ODE systems via BatchedOdeRK4::generate_shader() |
| Local WGSL (Write) | 1 | diversity_fusion_f64.wgsl |
| **Total consumed** | **53** | + 2 BGL helpers + 1 WGSL extension |

---

## Part 5: Recommendations for ToadStool

1. **Add V44 items to ABSORPTION_TRACKER**: `find_w_c` consumption (4 files), `pearson_correlation` consumption.
2. **Export `dot`/`l2_norm` as CPU functions**: wetSpring has 15+ binaries using local `dot`/`l2_norm` helpers. If exported from `barracuda::linalg`, wetSpring could eliminate more local code.
3. **`diversity_fusion_f64.wgsl` absorption**: Still the only open P0 item (8/9 → 9/9).

---

## Part 6: Quality Gates

- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery` — 0 warnings
- `cargo test --lib` — 819 passed, 1 ignored, 0 failed
- Exp169 — 12/12 PASS
