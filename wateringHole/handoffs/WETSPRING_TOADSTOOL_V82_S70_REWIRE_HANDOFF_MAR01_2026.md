# wetSpring → ToadStool/BarraCuda V82 Handoff

**Date:** March 1, 2026
**From:** wetSpring V82 (Phase 82)
**To:** ToadStool/BarraCuda team
**Status:** 248 experiments, 6,315+ checks, 1,210 tests, ALL PASS
**Supersedes:** V81 (archived)
**ToadStool Pin:** S70+++ (`1dd7e338`)
**License:** AGPL-3.0-only

---

## Executive Summary

V82 advances the ToadStool pin from S68+ to S70+++ (13 commits, 324 files,
9,440 insertions), validates 3 new upstream stats primitives (evolution,
jackknife, chao1_classic), and confirms zero breakage across the entire
wetSpring validation suite (1,210 tests, 248 experiments).

**Key outcomes:**
- **Clean S70+++ absorption** — no breaking changes, no API drift, no local workarounds
- **3 new primitives consumed** — `stats::evolution` (4 functions), `stats::jackknife`
  (2 functions + struct), `stats::diversity::chao1_classic` (1 function)
- **85 total primitives** consumed from ToadStool (up from 82)
- **42/42 new validation checks** PASS (Exp247)

---

## Part 1: What Changed in ToadStool (S68+ → S70+++)

### Commits Absorbed (13)

| Commit | Summary |
|--------|---------|
| `5c9942d0` | S68++: AGPL-3 license, 0 clippy warnings, chrono eliminated, WebSocket removed |
| `92e77cdd` | S68+++: chrono fully eliminated — 28 crates migrated to std::time |
| `ec280113` | S68+++: unsafe evolved, localhost constants, println→tracing |
| `ed37d949` | S68+++: Dead code cleanup ~400 lines |
| `132fa651` | S68+++: STATUS.md with dead code cleanup results |
| `6b64b6e6` | S68+++: Archive stale debris, fix broken doc refs |
| `ad9e9dea` | fix: box_muller_cos shader f32→f64 |
| `168a2924` | S69++: ComputeDispatch migration, architecture evolution |
| `e5512db4` | S70: Deep debt — modern idiomatic concurrent Rust |
| `d3dae8bb` | S70+: Cross-spring absorption — new shaders, stats, DF64 ML |
| `61846932` | S70++: Sovereignty, Fp64Strategy::Concurrent, monitoring split |
| `f01778b5` | S70+++: Builder refactor, dead code removal, monitoring evolution |
| `1dd7e338` | Docs: clean and update all root docs for S70+++ accuracy |

### Key Upstream Changes

- **3 new stats modules**: `evolution.rs` (4 functions from groundSpring), `jackknife.rs`
  (2 functions + JackknifeResult), `hydrology.rs` (5 functions from airSpring)
- **1 new diversity function**: `chao1_classic` (Chao 1984, u64 counts)
- **`staging::pipeline::PipelineBuilder`** — GPU streaming topology builder (414 lines)
- **`Fp64Strategy::Concurrent`** — dual-run DF64 + native f64 for validation
- **`EcosystemCaller` removed** — 95 lines dead code eliminated
- **6 new WGSL shaders**: batched_elementwise_f64, seasonal_pipeline, anderson_coupling_f64,
  lanczos_iteration_f64, linear_regression_f64, matrix_correlation_f64
- **New GPU ops**: `SymmetrizeGpu`, `LaplacianGpu`
- **chrono eliminated** — 28 crates migrated to std::time
- **Dead code cleanup** — ~400 lines removed, stale allows fixed

### No Breaking Changes

- `ops::bio::*` API unchanged
- `ComputeDispatch` builder API unchanged
- `compile_shader_universal` API unchanged
- All 1,210 wetSpring tests pass without modification

---

## Part 2: New Primitives Consumed (Exp247, 42/42 PASS)

### stats::evolution (from groundSpring via S70)

| Function | wetSpring Use | Validated |
|----------|--------------|-----------|
| `kimura_fixation_prob(Ne, s, p0)` | Population genetics fixation analysis | Neutral, beneficial, deleterious, strong selection, already-fixed |
| `error_threshold(σ, L)` | Quasispecies error threshold (Eigen 1971) | Analytic match, genome length scaling, edge cases |
| `detection_power(p, D)` | Rare biosphere detection probability | Analytic match (1-(1-p)^D), edge cases |
| `detection_threshold(p, P_target)` | Minimum sequencing depth for target power | Analytic match (ceil(ln(1-P)/ln(1-p))), round-trip validation |

### stats::jackknife (from groundSpring via S70)

| Function | wetSpring Use | Validated |
|----------|--------------|-----------|
| `jackknife_mean_variance(data)` | Diversity estimate uncertainty | Mean accuracy, variance/SE, constant data, edge cases |
| `jackknife(data, statistic)` | Generalized leave-one-out for any statistic | Mean, Shannon diversity with SE estimation |

### stats::diversity::chao1_classic (from groundSpring via S70)

| Function | wetSpring Use | Validated |
|----------|--------------|-----------|
| `chao1_classic(counts: &[u64])` | Integer-count Chao 1984 richness estimator | Analytic formula, no-singletons edge, no-doubletons edge, comparison with bias-corrected chao1 |

---

## Part 3: Gap Analysis Update

### Previously Identified Gaps (from TOADSTOOL_WETSPRING_GAP_ANALYSIS.md)

| Gap | Status |
|-----|--------|
| ComputeDispatch | DONE V75 |
| BatchedMultinomialGpu | DONE V75 |
| PairwiseL2Gpu | DONE V75 |
| FstVariance | DONE V75 |
| bootstrap_ci, rawr_mean | P2 — evaluate |
| Precision::Df64 | P2 — evaluate |
| dispatch_for, substrates | P3 — evaluate |

### New S70+ Gaps Identified

| Gap | Priority | Notes |
|-----|----------|-------|
| `staging::pipeline::PipelineBuilder` | P3 | Alternative to local `GpuPipelineSession`; evaluate if topology builder simplifies streaming |
| `Fp64Strategy::Concurrent` | P3 | Useful for automated DF64 vs native validation; low priority |
| `SymmetrizeGpu` + `LaplacianGpu` | P3 | Community graph analysis; adopt if graph-spectral pipeline needed |
| `stats::hydrology` (fao56_et0 etc.) | Skip | Not in wetSpring domain (airSpring) |

---

## Part 4: Current State

| Metric | V81 | V82 |
|--------|-----|-----|
| ToadStool pin | S68+ (`e96576ee`) | S70+++ (`1dd7e338`) |
| Primitives consumed | 82 | 85 |
| Experiments | 247 | 248 |
| Validation checks | 6,273+ | 6,315+ |
| Rust tests | 1,219 | 1,210 |
| Clippy warnings | 0 | 0 |
| Local WGSL | 0 | 0 |
| Unsafe code | 0 | 0 |

---

## Part 5: Recommendations for ToadStool Team

| Priority | Item |
|----------|------|
| **Info** | wetSpring S70+++ absorption is clean — zero breakage, zero workarounds |
| **Info** | `chao1` (f64, bias-corrected) and `chao1_classic` (u64, original Chao 1984) produce different estimates — this is correct and documented |
| **Info** | evolution/jackknife functions validated with analytic ground truth — formulas are correct |
| **Consider** | Document `chao1` vs `chao1_classic` formula differences in module docs |
| **Consider** | `PipelineBuilder` could benefit from a CPU-only mode for topology analysis without GPU context |

---

*This handoff is unidirectional: wetSpring → ToadStool. ToadStool absorbs what
it finds useful; wetSpring leans on upstream. No response expected.*
