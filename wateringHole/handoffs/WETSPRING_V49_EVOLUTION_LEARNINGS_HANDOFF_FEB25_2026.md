# wetSpring → ToadStool V49 Handoff: Evolution Learnings & Future Roadmap

**Date:** February 25, 2026
**Phase:** 50 (V49 — doc cleanup + evolution learnings)
**Primitives consumed:** 66 + 2 BGL helpers (zero local WGSL)
**ToadStool pin:** `17932267` (S65)
**wetSpring status:** Fully lean — zero local WGSL, zero local math, 9/9 evolution requests DONE

---

## Purpose

This handoff is **not** a feature request. wetSpring is fully lean and has no
pending absorption needs. This document captures what we learned during the
Write → Absorb → Lean cycle that may help ToadStool's ongoing evolution and
other springs' adoption.

---

## Part 1: BarraCuda Primitive Usage Review (66 + 2 BGL)

### Usage by Category

| Category | Count | wetSpring Consumers | Origin |
|----------|------:|---------------------|--------|
| GPU bio ops | 15 | 22 GPU modules (alignment, diversity, ODE, phylo, etc.) | wetSpring V16-V40 → S42-S58 |
| GPU core (linalg) | 11 | gemm_cached, pcoa_gpu, kriging, esn | hotSpring/wetSpring → S40-S62 |
| CPU special | 7 | erf, ln_gamma, gamma, beta, digamma, reg_gamma_{p,q} | hotSpring precision → S39-S50 |
| CPU stats | 4 | norm_cdf, pearson_correlation, dot, l2_norm | hotSpring+wetSpring → S44-S64 |
| CPU diversity | 11 | shannon, simpson, chao1, pielou, bray_curtis, etc. | wetSpring → S64 |
| Spectral | 5 | anderson_3d, lanczos, level_spacing_ratio, GOE_R, POISSON_R | hotSpring → S50-S62 |
| Cross-spring | 8 | find_w_c, anderson_sweep_averaged, hamming, jaccard, etc. | hotSpring+neuralSpring → S50-S62 |
| Linalg/NMF | 5 | NMF, ridge, cosine_similarity, cholesky_solve | wetSpring → S58 |
| **Total** | **66** | + 2 BGL helpers | |

### Usage Patterns

1. **Heavy hitters** (used by 5+ modules): `GemmF64`, `FusedMapReduceF64`, `BatchedOdeRK4`
2. **Single consumer** (7 primitives): `DiversityFusionGpu`, `Dada2EStepGpu`, `GillespieGpu`,
   `KrigingF64`, `SnpCallingF64`, `PeakDetectF64`, `PangenomeClassifyGpu`
3. **Delegation only** (thin re-exports): `diversity_fusion_gpu`, `bio::diversity`, `special::{dot, l2_norm}`
4. **Compose pattern** (7 modules): kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, molecular_clock

---

## Part 2: Cross-Spring Evolution Insights

### Timeline of Cross-Spring Contributions

```
Session  Origin        Contribution              → Consumers
───────  ──────────    ────────────────────────   ──────────────────
S39-40   hotSpring     erf, ln_gamma, gamma,      wetSpring, neuralSpring
                       erfc, digamma, beta
S42      hotSpring     anderson_3d, lanczos        wetSpring (Track 4)
S44      hotSpring     norm_cdf                    wetSpring, airSpring
S50      hotSpring     level_spacing_ratio,        wetSpring (Track 4)
                       GOE_R, POISSON_R
S50      neuralSpring  pearson_correlation          wetSpring
S54-58   wetSpring     5 ODE systems, NMF,         (self, shared via ToadStool)
                       ridge, cosine_similarity
S60      hotSpring     DF64 FMA + transcendentals  all springs
S62      hotSpring     anderson_sweep_averaged,    wetSpring (Track 4)
                       find_w_c (soil presets)
S63      ToadStool     diversity_fusion absorption  wetSpring (Lean)
S64      airSpring     stats::metrics (dot, l2,    wetSpring (Lean)
         groundSpring  mean, rmse, mbe, etc.)
S64      wetSpring     stats::diversity (shannon,  (shared via ToadStool)
                       simpson, chao1, etc.)
S65      ToadStool     smart refactoring (30-40%   all springs (perf)
                       code reduction in core)
```

### Patterns That Work Well

1. **Biome independence**: Springs never import each other. All sharing goes
   through ToadStool. This prevents circular dependencies and makes absorption
   clean. wetSpring's entire lean transition (V48) required zero changes to
   hotSpring, neuralSpring, or airSpring.

2. **Trait-generated WGSL**: The `BatchedOdeRK4<S: OdeSystem>` pattern (S58)
   is the best example. Springs define domain science (params, derivatives);
   ToadStool generates the GPU shader. wetSpring's 5 ODE systems each get
   custom WGSL without maintaining any .wgsl files.

3. **CPU-first, GPU-second**: Every GPU path in wetSpring has a validated CPU
   baseline. This caught precision issues early (e.g., erf A&S 7.1.26
   approximation has ~1.5e-7 max error — discovered during Exp166/168 parity
   checks, not production failures).

4. **Named tolerances**: wetSpring's 77 named tolerance constants made the
   lean transition seamless. When delegation changed implementation details,
   tolerance names stayed stable. No magic-number hunts.

5. **Handoff-driven absorption**: Every ToadStool absorption in S42-S65 was
   preceded by a `wateringHole/handoffs/` document specifying exact code
   locations, binding layouts, test expectations, and tolerance rationale.
   This eliminated back-and-forth during absorption.

### Patterns to Improve

1. **Status line drift**: With 15+ files carrying status lines, counts drift
   between versions. A single canonical source (e.g., generated from
   `Cargo.toml` metadata) would prevent the 182→183 / 53→66 staleness
   we cleaned up in V49.

2. **Binary count tracking**: With 173 binaries, manually updating counts is
   fragile. Consider `cargo metadata` or a simple `ls src/bin/*.rs | wc -l`
   in CI.

---

## Part 3: Exp183 Performance Benchmarks (Cross-Spring S65)

Benchmark ran on the development GPU (release mode, `--features gpu`).

| Benchmark | Origin | Description |
|-----------|--------|-------------|
| GPU ODE × 5 | wetSpring → S58 | 128-batch RK4 for bistable, cooperation, phage, capacitor, multi-signal |
| GPU DiversityFusion | wetSpring → S63 | Fused Shannon + Simpson + Pielou, CPU parity verified |
| CPU diversity × 6 | wetSpring → S64 | Delegation parity: `bio::diversity::*` → `barracuda::stats::*` |
| CPU math × 2 | wetSpring → S64 | Delegation parity: `special::{dot, l2_norm}` → `barracuda::stats::*` |
| CPU special × 5 | hotSpring → S39-50 | erf, ln_gamma, norm_cdf, trapz, pearson_correlation |
| GEMM pipeline | wetSpring → S62 | GemmCached compile + dispatch (1024×1024 F64) |
| Anderson spectral | hotSpring → S50-62 | anderson_3d + lanczos, find_w_c + sweep |
| NMF + Ridge | wetSpring → S58 | nmf_mu (100×20→5 factors) + ridge regression |

**Result:** 36/36 checks PASS, all delegation chains verified, all cross-spring
primitives functional.

---

## Part 4: Future Opportunities (Not Blocking)

These are observations, not requests. wetSpring's current 9/9 evolution
requests are all DONE.

| Opportunity | Description | Benefit |
|-------------|-------------|---------|
| Soil-specific Anderson presets | `anderson_3d` with pre-configured pore size distributions for common soil types (clay, loam, sand) | Reduce boilerplate for Track 4 soil papers |
| ODE sensitivity helpers | IC perturbation sweep + Lyapunov exponent estimation built into `BatchedOdeRK4` | Would benefit wetSpring (bio) + hotSpring (physics) ODE users |
| GPU diversity fusion F32 variant | F32 path for large metagenomic datasets where F64 precision isn't needed | Performance for large datasets |
| `stats::diversity` → GPU | GPU versions of shannon, simpson, chao1 for very large abundance vectors | Currently CPU-only; GPU would help at >1M samples |
| Status-line generation | `barracuda::meta` module exporting version, primitive count, session for CI | Would eliminate cross-file status drift |

---

## Part 5: Lessons for Other Springs

For hotSpring, neuralSpring, airSpring teams considering the lean transition:

1. **Start with the absorption manifest**: Document every local primitive, its
   upstream equivalent, and the test that validates parity. wetSpring's
   `ABSORPTION_MANIFEST.md` tracked all 66 primitives through Write → Lean.

2. **Lean in order**: GPU shaders first (biggest code/maintenance savings),
   then CPU math, then validation adjustments. Don't try to do everything
   in one pass.

3. **Tolerance constants are load-bearing**: When switching from local to
   upstream implementations, floating-point results may differ at the ULP
   level. Named tolerances with documented rationale make this painless.

4. **Keep the CPU reference**: Even after leaning to upstream GPU, keep the
   local CPU integration function (e.g., `rk4_integrate` for ODE). It's
   essential for deterministic testing and python parity checks.

5. **Archive aggressively**: Old handoffs accumulate. Archive superseded
   versions (we archived V7-V45) so the active handoff set stays small and
   current.

---

## Acceptance Criteria

- [x] All 66 + 2 BGL primitives documented with origin and consumers
- [x] Cross-spring timeline from S39 to S65
- [x] Exp183 benchmark results summarized
- [x] Future opportunities catalogued (non-blocking)
- [x] Lessons for other springs documented
- [x] V44, V45 handoffs archived
- [x] All stale references cleaned (182→183, 53→66, S62→S65, Write→Lean)
