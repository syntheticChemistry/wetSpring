# wetSpring → ToadStool Handoff V37: Deep Debt Cleanup + Revalidation

**Date:** February 25, 2026
**From:** wetSpring (Phase 44, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring completed a comprehensive deep-debt cleanup and revalidated against
ToadStool's latest barracuda crate (commit `02207c4a`, S62+DF64). All 838 tests
pass, clippy is clean (pedantic + nursery), coverage is 95.74%, and zero unsafe
blocks. This handoff documents the cleanup, confirms compatibility with
ToadStool's recent S54-S62 evolution, and identifies the remaining evolution path.

**Key numbers (post-cleanup):** 838 tests (791 barracuda + 47 forge), 167
experiments, 3,279+ validation checks, 157 binaries, 44 ToadStool primitives
+ 2 BGL helpers consumed, 1 Write-phase WGSL extension, 0 clippy warnings,
95.74% library coverage.

---

## Part 1: Deep Debt Cleanup (This Session)

### 1.1 ncbi.rs — Sovereignty Evolution

**Before:** Hardcoded `curl` shell-out, legacy relative dev paths
(`../../../testing-secrets/`), hardcoded `CARGO_MANIFEST_DIR` in `cache_file()`.

**After:** Capability-based HTTP transport discovery chain
(`WETSPRING_HTTP_CMD` > `curl` > `wget`). All legacy paths removed. `cache_file()`
uses `WETSPRING_DATA_ROOT` cascade. URL encoding extended for `&` and `#`. 7 new
tests for backend discovery, `which_exists`, error formatting.

**Primal principle:** Code has self-knowledge only, discovers capabilities at runtime.

### 1.2 I/O Parsers — Deprecated Code Removal

Deprecated buffering functions (`parse_fastq`, `parse_mzml`, `parse_ms2`) evolved
from ~120 lines of duplicate buffering implementations to thin wrappers over
streaming iterators. `ms2.rs` reduced from 765 → 657 lines.

### 1.3 Binary Safety — 56 Files Modernized

- All `partial_cmp().unwrap()` → NaN-safe `.unwrap_or(Ordering::Equal)`
- All bare `.unwrap()` → descriptive `.expect("reason")` messages
- GPU init, channel operations, sort comparisons all modernized

### 1.4 Tolerance Provenance

Added commit-hash provenance to remaining gaps:
- `GC_CONTENT` → commit `504b0a8` (FastQC validated)
- `MEAN_QUALITY` → commit `cf15167` (full FastQC run history)
- `GALAXY_SHANNON/SIMPSON/BRAY_CURTIS_RANGE` → commit `21d43a0` (Exp002 complete)

### 1.5 CI Pipeline Expanded

| Job | New? | Purpose |
|-----|:----:|---------|
| `coverage` | Yes | `cargo-llvm-cov --fail-under-lines 90` |
| `forge-clippy` | Yes | metalForge pedantic + nursery |
| `forge-test` | Yes | metalForge 47 tests |

---

## Part 2: ToadStool Compatibility Verification

### Revalidation Results (against `02207c4a`)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D pedantic -D nursery` | PASS (0 warnings) |
| `cargo test` | 838 passed, 0 failed, 1 ignored |
| `cargo doc --no-deps` (-D warnings) | PASS |
| `cargo llvm-cov --lib` | 95.74% line coverage |
| metalForge `cargo test` | 47 passed |
| metalForge `cargo clippy` | PASS |

### API Compatibility

All 44 ToadStool primitives + 2 BGL helpers compile and test cleanly against
the latest barracuda. No API breakage detected in S54-S62 evolution.

### Primitives Consumed (Confirmed Working)

**CPU (always-on):**
`erf`, `ln_gamma`, `regularized_gamma_p`, `trapz`, `ridge_regression`, `nmf`,
`graph_laplacian`, `effective_rank`, `numerical_hessian`, `disordered_laplacian`,
`belief_propagation_chain`, `boltzmann_sampling`

**GPU (feature = "gpu"):**
`FusedMapReduceF64`, `BrayCurtisF64`, `BatchedEighGpu`, `BatchedOdeRK4<S>`,
`SmithWatermanGpu`, `FelsensteinGpu`, `GillespieGpu`, `HmmBatchForwardF64`,
`KmerHistogramGpu`, `UniFracPropagateGpu`, `TaxonomyFcGpu`, `TreeInferenceGpu`,
`AniBatchF64`, `DnDsBatchF64`, `SnpCallingF64`, `PangenomeClassifyGpu`,
`GemmF64`, `TranseScoreF64`, `SparseMmF64`, `PeakDetectF64`,
`PairwiseHammingGpu`, `PairwiseJaccardGpu`, `SpatialPayoffGpu`,
`BatchFitnessGpu`, `LocusVarianceGpu`, `KrigingF64`, `VarianceF64`,
`CorrelationF64`, `CovarianceF64`, `WeightedDotF64`, `RfBatchInferenceGpu`,
`QualityFilterGpu`, `Dada2EStepGpu`, `BatchedOdeRK4F64`

**Infrastructure:** `WgpuDevice`, `compile_shader_f64`, `read_buffer_f64`,
`storage_bgl_entry`, `uniform_bgl_entry`, `GpuDriverProfile`, `TensorContext`,
`BufferPool`, `PooledBuffer`

---

## Part 3: Local Implementations Audit

### Legitimate Local Code (CPU baselines — KEEP)

| Module | Purpose | Why Local |
|--------|---------|-----------|
| `bio/ode.rs` | Local RK4 integrator | CPU reference for validation against Python scipy |
| `bio/gillespie.rs` | Gillespie SSA | CPU reference with seeded PRNG for determinism tests |
| `bio/pcoa.rs` | Jacobi eigendecomposition | CPU reference for PCoA validation |
| `bio/alignment.rs` | Smith-Waterman | CPU reference with affine gaps + traceback |
| `bio/esn.rs` | Echo State Network | CPU reference for NPU validation |
| `bio/chimera.rs` | UCHIME chimera detection | Domain-specific, no ToadStool equivalent |
| `special.rs` | `normal_cdf`, `dot`, `l2_norm` | Thin helpers not in barracuda CPU |

### Passthrough Stubs (3 modules — DOCUMENT)

| Module | CPU Kernel | Blocked On |
|--------|-----------|------------|
| `gbm_gpu` | Sequential GBM inference | `GbmBatchInferenceGpu` (not in ToadStool) |
| `feature_table_gpu` | LC-MS feature extraction | `FeatureExtractionGpu` (not in ToadStool) |
| `signal_gpu` | Peak detection (1D) | PeakDetectF64 WGSL bug (V36 §3.4 still open) |

### No Duplicate Math

All `special.rs` functions delegate to `barracuda::special::*`. No local
reimplementations of math that barracuda provides.

---

## Part 4: Outstanding Bug Reports (from V36, still open)

| Issue | File | Fix | Status |
|-------|------|-----|--------|
| PeakDetectF64 f32 literal | `shaders/signal/peak_detect_f64.wgsl:49` | `0.0` → `0.0lf` | **Open** |
| `log()` narrowing in f64 shaders | `shaders/precision/mod.rs` | Always replace bare `log(` | **Open** |
| Private `wgsl_shader_for_device()` | `ops/linalg/gemm_f64.rs` | Make `pub` or add `cached_pipeline()` | **Open** |

---

## Part 5: Evolution Readiness

### Ready for GPU Promotion (when ToadStool adds primitives)

| Local Module | Needed Primitive | Complexity |
|-------------|-----------------|------------|
| `signal_gpu` | PeakDetectF64 bug fix | One-line WGSL fix |
| `feature_table_gpu` | Composed EIC + PeakDetect pipeline | Medium |
| `gbm_gpu` | Sequential tree traversal shader | Medium |

### Write-Phase Extension (ready for absorption)

| Module | WGSL | Tests | Status |
|--------|------|:-----:|--------|
| `diversity_fusion_gpu` | `diversity_fusion_f64.wgsl` | 6 + Exp167 | Ready for `ops::bio::diversity_fusion` |

---

## Supersedes

V36 (Write-phase handoff) and all prior versions for absorption accounting.
V36 §3.4 bug reports remain open and are carried forward.
