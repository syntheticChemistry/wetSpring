# wetSpring → ToadStool Handoff V43: ToadStool Catch-Up Review

**Date:** February 25, 2026
**From:** wetSpring (Phase 48, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team + all springs
**Supersedes:** V42 (deep debt round 2)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring reviewed ToadStool's full commit evolution (S42–S62+DF64, 80+ commits)
and cross-referenced against the ToadStool ABSORPTION_TRACKER. All 46 items
from wetSpring handoffs V16–V22 are confirmed DONE. This handoff documents
the catch-up, a new rewire (`normal_cdf` → upstream), updated priority tracking,
and items ToadStool should add to their tracker from our V40–V42 handoffs.

**Post-V43 numbers:** 898 tests (819 lib + 47 forge + 32 integration/doc),
**96.78%** library coverage, **50** ToadStool primitives + 2 BGL helpers consumed,
**77** named tolerance constants, zero bare tolerance literals, zero clippy warnings.

---

## Part 1: ToadStool Absorption Status (Verified)

### 1.1 All V16–V22 Items DONE (46/46)

ToadStool's `ABSORPTION_TRACKER.md` (last updated Session 57) confirms every
wetSpring request from handoffs V16–V22 has been absorbed:

| ToadStool ID | Item | Session | Status |
|-------------|------|---------|--------|
| H-005 | ESN NPU Weight Export | S51 | DONE |
| H-006 | `BatchedOdeRK4Generic` (5 ODE shaders) | S51 | DONE |
| M-005 | Root re-exports (`QualityConfig`, `UniFracConfig`) | S51 | DONE |
| M-006 | `FlatTree::from_newick()`, `from_edges()` | S52 | DONE |
| M-007 | `FusedMapReduceF64::dot(a, b)` | S51 | DONE |
| M-008 | ESN ridge regression | S52 | DONE |
| M-010 | `barracuda::tolerances` module | S52 | DONE |
| M-014 | `regularized_gamma_lower` | ALREADY PRESENT | — |
| L-002 | GPU ESN reservoir shader | S52 | DONE |
| L-003 | `chi_squared_f64` alias | S52 | DONE |
| L-004 | `GpuSessionBuilder` pre-warm | S52 | DONE |
| L-007 | Anderson transport/conductance | S52 | DONE |
| L-008 | NCBI data cache module | S52 | DONE |

Plus 33 items from hotSpring, neuralSpring, and airSpring (H-001 through L-014).

### 1.2 Items ToadStool Should Add from V40–V42

ToadStool's tracker stops at V22. The following items from V40–V42 should be
added to their next tracker update:

| Priority | Item | Source | Status |
|----------|------|--------|--------|
| **P0** | Absorb `diversity_fusion_f64.wgsl` as `ops::bio::diversity_fusion` | V41 §P0-1, V42 §P0-1 | **OPEN** — sole remaining local WGSL |
| **P1** | Unify `PhyloTree` (CPU+GPU single type) | V41 §P1-1, V42 §P1-1 | **OPEN** |
| **P1** | Export `barracuda::special::{dot, l2_norm}` CPU helpers | V41 §P1-2, V42 §P1-2 | **OPEN** — trivial (1-line functions) |
| **P1** | `Fp64Strategy` documentation page | V41 §P1-3, V42 §P1-3 | **OPEN** |
| **P1** | `wgsl_shader_for_device()` pub access | V41 §P1-4, V42 §P1-4 | **OPEN** |
| **P2** | ODE trait pattern documentation | V42 §P2-1 | **OPEN** |
| **P2** | metalForge absorption path | V42 §P2-2 | **OPEN** |
| **P2** | NPU primitive exploration | V42 §P2-3 | **OPEN** |
| **P2** | Per-domain GPU polyfill tracking | V42 §P2-4 | **OPEN** |

### 1.3 Evolution Requests: Updated Score

| # | Request | Status |
|---|---------|--------|
| 1 | `GemmF64::wgsl_shader_for_device()` public | **DELIVERED** (S62+DF64) |
| 2 | `PeakDetectF64` f64 end-to-end | **DELIVERED** (S62) |
| 3 | `ComputeDispatch` builder | **DELIVERED** (S62+DF64) |
| 4 | `dot`/`l2_norm` as GPU ops | **DELIVERED** (S60) |
| 5 | Absorb `diversity_fusion_f64.wgsl` | **OPEN** |
| 6 | `BatchedOdeRK4` via `OdeSystem` trait | **DELIVERED** (S58) |
| 7 | GPU Top-K selection | **DELIVERED** (S60) |
| 8 | NPU int8 quantization helpers | **DELIVERED** (S39) |
| 9 | Tolerance module pattern | **DELIVERED** (S52) |

**Score: 8/9 delivered.** Only `diversity_fusion_f64.wgsl` absorption remains.

---

## Part 2: Rewire — `normal_cdf` → `barracuda::stats::norm_cdf`

ToadStool S59 added `barracuda::stats::norm_cdf` (same formula: `Φ(x) = (1+erf(x/√2))/2`).
wetSpring's `special::normal_cdf` previously computed this locally. Now delegates:

```rust
pub fn normal_cdf(x: f64) -> f64 {
    barracuda::stats::norm_cdf(x)
}
```

This is wetSpring's 50th ToadStool primitive consumed. The delegation pattern
matches `erf`, `ln_gamma`, and `regularized_gamma_lower` — thin wrapper
preserving the local API while leaning on upstream math.

**Tests:** All 21 `special::tests` pass. The `normal_cdf_known_values` and
`normal_cdf_symmetry` tests validate the delegation end-to-end.

---

## Part 3: What wetSpring Does NOT Rewire (and Why)

### 3.1 `ValidationHarness` (Available, Not Consumed)

ToadStool S59 added `barracuda::validation::ValidationHarness` with a richer
API (`check_abs`, `check_rel`, `check_upper`, `check_lower`, `check_bool`,
`require`, `require!` macro, `finish`).

wetSpring keeps its local `validation::Validator` because:

1. **API mismatch** — Our `check(label, actual, expected, tolerance)` maps to
   their `check_abs(label, observed, expected, tolerance)` but method names differ.
2. **158 binaries** — Rewiring every validation binary for a name change yields
   no functional benefit.
3. **Extra methods** — Our `check_count(label, actual, expected)` and
   `check_count_u64(label, actual, expected)` have no upstream equivalent.
   `check_pass(label, pass)` maps to `check_bool` but with different semantics.
4. **`data_dir()` discovery** — Our `validation.rs` includes capability-based
   data directory resolution (`data_dir`, `resolve_data_dir`) which is
   wetSpring-specific and has no upstream counterpart.

Both implementations follow the hotSpring validation pattern. They are
complementary, not duplicative.

### 3.2 `barracuda::tolerances` (Complementary, Not Replacing)

ToadStool's `barracuda::tolerances` module provides a `Tolerance` struct with
combined absolute-or-relative checking and 12 named constants for barracuda's
internal validation. wetSpring's `tolerances.rs` provides 77 flat `f64` constants
for domain-specific Python-baseline comparison.

The two systems serve different purposes:
- **barracuda's** — cross-spring GPU-vs-CPU parity checking
- **wetSpring's** — per-experiment tolerance thresholds with scientific provenance

No rewiring needed. Confirmed P2-9 delivered.

### 3.3 `special::{dot, l2_norm}` (Local CPU Helpers)

ToadStool provides `dot` and `l2_norm` only as GPU operations
(`FusedMapReduceF64::dot()`, `NormReduceF64::l2()`). There are no CPU-level
`&[f64]` → `f64` equivalents in barracuda. wetSpring keeps these as 2-line
local helpers used by 16+ files. P1-2 request to export them from
`barracuda::special` remains open.

---

## Part 4: New Upstream APIs Available (Not Yet Consumed)

ToadStool S42–S62+DF64 added capabilities wetSpring doesn't currently use
but are available for future wiring:

| API | Module | Potential Use |
|-----|--------|---------------|
| `norm_pdf`, `norm_ppf` | `barracuda::stats::normal` | Statistical testing, quantile computation |
| `norm_cdf_batch`, `norm_pdf_batch` | `barracuda::stats::normal` | Batch CDF/PDF for enrichment arrays |
| `pearson_correlation` | `barracuda::stats::correlation` | Pairwise feature correlation |
| `bootstrap_ci` | `barracuda::stats::bootstrap` | Confidence intervals for diversity metrics |
| `chi2_decomposed` | `barracuda::stats::chi2` | Weighted chi-squared decomposition |
| `empirical_spectral_density` | `barracuda::stats::spectral_density` | Random matrix theory for gene expression |
| `marchenko_pastur_bounds` | `barracuda::stats::spectral_density` | RMT bulk edge bounds |
| `HillGateGpu` | `barracuda::ops::bio` | Hill-function cooperative gating |
| `MultiObjFitnessGpu` | `barracuda::ops::bio` | Multi-objective evolutionary fitness |
| `PairwiseL2Gpu` | `barracuda::ops::bio` | GPU pairwise L2 distance matrices |
| `SwarmNnGpu` | `barracuda::ops::bio` | Swarm neural network scoring |
| `anderson_sweep_averaged` | `barracuda::spectral` | Averaged Anderson disorder sweep |
| `find_w_c` | `barracuda::spectral` | Critical disorder threshold finder |
| `ComputeDispatch` builder | `barracuda::device` | Pipeline boilerplate elimination |
| `Fp64Strategy` auto-detect | `barracuda::device` | DF64 Native/Hybrid per GPU |
| DF64 GEMM | `barracuda::shaders` | ~10× throughput on FP32 cores |

These are documented for future reference. No wiring is needed now.

---

## Part 5: Updated Dependency Surface

### 5.1 ToadStool Primitives Consumed (50 + 2 BGL helpers)

| Category | Count | Δ from V42 |
|----------|:-----:|:----------:|
| GPU bio ops | 15 | — |
| GPU math/infra | 12 | — |
| GPU phylo | 6 | — |
| GPU ODE | 5+1 | — |
| GPU infra | 8+2 BGL | — |
| CPU special | 4 | **+1** (`norm_cdf`) |
| CPU numerical | 5 | — |
| CPU linalg | 4 | — |
| CPU spectral | 6+ | — |
| CPU stats | 1 | **+1** (`norm_cdf` via stats) |
| CPU sampling | 1 | — |
| **Total** | **50+2** | **+1** |

### 5.2 Quality Gates (Post-V43)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | 0 diffs |
| `cargo clippy --all-targets -W pedantic -W nursery` | 0 warnings |
| `cargo test --lib` | 819 passed, 1 ignored, 0 failed |
| `cargo test` (all) | 898 passed, 0 failed |
| `cargo doc --no-deps` | 0 warnings |
| `cargo llvm-cov --lib` | 96.78% |
| `#![deny(unsafe_code)]` | Enforced crate-wide |
| Named tolerance constants | 77 |
| External C dependencies | 0 |
| Max file size | All under 1000 LOC |

---

## Part 6: Recommendations for ToadStool

### For the ABSORPTION_TRACKER

Add the following to the next tracker update:

```
### H-015: diversity_fusion_f64.wgsl (wetSpring V41-V42)
Source: wetSpring barracuda/src/bio/diversity_fusion_gpu.rs
Target: barracuda::ops::bio::diversity_fusion_f64
Status: OPEN
Validation: Exp167 18/18 PASS
Binding layout: 3 buffers (counts, params, output), dispatch N/64
```

### For Cross-Spring Evolution

1. **`barracuda::stats` is underused** — wetSpring now consumes `norm_cdf`
   but the full stats module (`correlation_matrix`, `bootstrap_ci`,
   `chi2_decomposed`, `empirical_spectral_density`) is rich. Consider
   promoting it in cross-spring documentation.

2. **The tolerance pattern converged** — ToadStool (S52) and wetSpring (V41)
   independently arrived at centralized tolerance modules with named constants
   and hierarchical testing. This pattern should be recommended for all springs.

3. **`ValidationHarness` vs `Validator`** — Two implementations of the same
   hotSpring pattern exist. Not harmful (different APIs, different audiences)
   but worth noting for future unification if desired.

---

## Part 7: Acceptance Criteria

- [x] ToadStool commit evolution reviewed (S42–S62+DF64, 80 commits)
- [x] ABSORPTION_TRACKER verified: 46/46 V16–V22 items DONE
- [x] `normal_cdf` rewired to `barracuda::stats::norm_cdf`
- [x] All 898 tests pass
- [x] Zero clippy warnings (pedantic + nursery)
- [x] `ValidationHarness` decision documented (available, not consumed)
- [x] Evolution requests updated: 8/9 delivered
- [x] P0–P2 items for ToadStool tracker documented
- [x] ABSORPTION_MANIFEST and EVOLUTION_READINESS synchronized
- [x] V43 handoff submitted to wateringHole/handoffs/
