# wetSpring → ToadStool Handoff V26: Sync + Code Audit + Revalidation

**Date:** February 24, 2026
**From:** wetSpring
**To:** ToadStool / BarraCuda team
**License:** AGPL-3.0-or-later
**Purpose:** Acknowledge ToadStool S39-S53 absorption (all 26 items), report wetSpring code audit + cleanup, revalidate GPU paths against current ToadStool, update absorption roadmap

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Phase | 39 (stable — no new experiments since V25) |
| ToadStool sessions absorbed | S39–S53 (all 26 items from V16–V22) |
| wetSpring tests | 770 (GPU feature) / 755 (CPU-only) / 47 metalForge |
| GPU clippy | 0 warnings against ToadStool S53 |
| GPU build | clean (`cargo check --features gpu`) |
| Pre-existing GPU test defects fixed | 4 (GBM stump data, KMD assertion, 2 missing `#[ignore]`) |
| Deprecated APIs eliminated | 3 callers migrated from `parse_fastq` → `FastqIter` |
| Magic numbers promoted | 6 new named constants in `tolerances.rs` |
| Runtime dependencies corrected | 2 (flate2 + bytemuck), not 1 |
| Paper queue | 43 papers, ALL DONE |
| CPU validation sweep | 30/30 self-contained PASS |
| metalForge tests | 38 → 47 (+9 new) |

---

## Part 1: Absorption Acknowledgment

ToadStool S39–S53 absorbed **all 26 items** from wetSpring V16–V22.
We confirm clean compilation and test passage against current ToadStool HEAD (`9abd6857`).

### Verified Lean Points

| Item | ToadStool Module | wetSpring Usage | Status |
|------|-----------------|-----------------|--------|
| H-005 ESN NPU | `esn_v2::npu` | Local sovereign ESN (no lean needed — CPU path) | Verified |
| H-006 BatchedOdeRK4 | `numerical::ode_generic` | 5 GPU ODE modules import directly | **Leaning** |
| M-005 Root re-exports | `lib.rs` re-exports | `QualityConfig`, `UniFracConfig` | **Leaning** |
| M-006 FlatTree | `genomics::flat_tree` | Used in cross-spring binaries | **Leaning** |
| M-007 dot | `ops::fused_map_reduce_f64` | Used in streaming + GPU modules | **Leaning** |
| M-008 ESN ridge | `esn_v2::train_ridge_regression` | Local sovereign (CPU path) | Verified |
| L-007 Anderson transport | `special::anderson_transport` | No local duplicate — clean | Verified |
| L-008 NCBI cache | `NcbiCache` | Local `ncbi.rs` for validation (different scope) | Verified |

### No Local Duplicates Found

Audit confirmed zero duplication between wetSpring sovereign CPU code and ToadStool absorbed modules. wetSpring's local implementations (`erf`, `ln_gamma`, `cholesky_factor`, `solve_ridge`, `integrate_peak`) exist **specifically** for the CPU-only path (`cfg(not(feature = "gpu"))`). When the GPU feature is enabled, `special.rs` already delegates to `barracuda::special::*`.

---

## Part 2: Code Audit Cleanup (This Session)

### Deprecated API Elimination

| File | Old | New |
|------|-----|-----|
| `validate_extended_algae.rs` | `fastq::parse_fastq()` | `FastqIter::open().collect()` |
| `validate_algae_16s.rs` | `fastq::parse_fastq()` | `FastqIter::open().collect()` |
| `validate_public_benchmarks.rs` | `fastq::parse_fastq()` | `FastqIter::open().collect()` |

Zero `#[allow(deprecated)]` annotations remain.

### Magic Numbers → Named Constants

Added to `tolerances.rs`:

| Constant | Value | Used By |
|----------|-------|---------|
| `MATRIX_EPS` | `1e-15` | `nmf.rs` cosine/norm denominators |
| `NMF_INIT_FLOOR` | `1e-10` | `nmf.rs` W/H init positivity |
| `BOX_MULLER_U1_FLOOR` | `1e-15` | `esn.rs` Gaussian generation |
| `GAMMA_RIGHT_TAIL_OFFSET` | `200.0` | `special.rs` early exit |
| `ODE_DIVISION_GUARD` | `1e-30` | `ode_systems.rs` WGSL/CPU |
| `ERROR_BODY_PREVIEW_LEN` | `200` | `ncbi.rs` error truncation |

### Test Coverage Expansion

- `ncbi.rs`: +8 tests (XML parsing via extracted `parse_esearch_count`)
- `special.rs`: +15 tests (`l2_norm`, `dot`, gamma edge cases, erf symmetry)
- metalForge: +9 tests (streaming, dispatch, workloads)

### Pre-Existing GPU Test Fixes

| Test | Issue | Fix |
|------|-------|-----|
| `gbm_gpu::gbm_batch_matches_individual` | Stump data had 1 node + 2 values (inconsistent lengths) | Fixed to 3-node tree (split + 2 leaves) |
| `kmd_gpu::kmd_gpu_matches_cpu_small` | Asserted KMD ≈ 0 (wrong — homologues have equal, not zero, KMD) | Fixed to assert KMD consistency within series |
| `batch_fitness_gpu::batch_fitness_dot_product` | Missing `#[ignore]` for GPU hardware test | Added `#[ignore = "requires GPU hardware"]` |
| `locus_variance_gpu::locus_variance_uniform_is_zero` | Missing `#[ignore]` for GPU hardware test | Added `#[ignore = "requires GPU hardware"]` |

---

## Part 3: Current Gate Status

| Gate | CPU | GPU | metalForge |
|------|:---:|:---:|:----------:|
| `cargo fmt --check` | PASS | PASS | PASS |
| `cargo clippy --all-targets` | 0 warnings | 0 warnings | 0 warnings |
| `cargo doc -D warnings` | PASS | — | — |
| `cargo test` | 755 pass / 1 ignored | 770 pass / 9 ignored | 47 pass |
| CPU validation sweep | 30/30 | — | — |

---

## Part 4: Updated Absorption Roadmap

### Items from V25 Still Pending Absorption

| Priority | Item | Source | Notes |
|:--------:|------|--------|-------|
| P1 | `compile_shader` → `compile_shader_f64` | V25 §P1 | f64 preamble bug in `batched_ode_rk4.rs:209` |
| P2 | Correlated Anderson 3D | V25 Exp151 | `build_correlated_anderson_3d(l, w, xi_corr, seed)` |
| P2 | Disorder averaging | V25 Exp150 | Multi-realization ⟨r⟩ with stderr |
| P3 | CPU math feature gate | V25 §P3 | `[features] math = []` for CPU-only modules |
| P3 | Passthrough → GPU | V25 §P4 | GBM, feature table, signal peak detect |
| P3 | metalForge absorption | V25 §P5 | `probe`, `inventory`, `dispatch` → barracuda |

### New Items from V26 (This Handoff)

| Priority | Item | Description |
|:--------:|------|-------------|
| P2 | Named tolerance pattern | wetSpring now has 53+ named constants in `tolerances.rs`. ToadStool's `barracuda::tolerances` has 12. Consider expanding upstream. |
| P3 | `MATRIX_EPS` / `NMF_INIT_FLOOR` | Generic numerical stability constants — useful for any NMF/cosine/norm consumer |
| Info | Dependency audit | wetSpring has 2 runtime deps (flate2 + bytemuck), not 1. bytemuck is required for GPU uniform buffers. |

---

## Part 5: Paper Queue Status

**43 papers, ALL DONE.** Breakdown:

| Track | Papers | Status |
|-------|:------:|--------|
| Track 1 (Microbial ecology) | 14 | 14/14 DONE |
| Track 1b (Comparative genomics) | 6 | 6/6 DONE |
| Track 1c (Deep-sea metagenomics) | 6 | 6/6 DONE |
| Track 2 (Analytical chemistry) | 4 | 4/4 DONE |
| Track 3 (Drug repurposing) | 5 | 5/5 DONE |
| Phase 37 extension | 9 | 9/9 DONE |
| Cross-spring (Kachkovskiy) | 1 | 1/1 DONE |

Total validation checks: 3,132+ across 161 experiments.

---

## Part 6: Evolution Path

```
Python baseline (41 scripts) ─── DONE
        ↓
BarraCUDA CPU (1,476 checks) ─── DONE, 22.5× faster than Python
        ↓
BarraCUDA GPU (702 checks) ──── DONE, pure GPU math proven
        ↓
Pure GPU streaming (152 checks) ── DONE, 441-837× vs round-trip
        ↓
metalForge cross-system (37 domains) ── DONE, CPU↔GPU↔NPU
        ↓
NEXT: Expand CPU benchmarks against open data
      → GPU portability proof (same answer, faster)
      → Full GPU workload validation
```

---

*wetSpring V26 — February 24, 2026*
*AGPL-3.0-or-later*
