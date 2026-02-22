# wetSpring → ToadStool/BarraCUDA Handoff v9

**Date:** February 22, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Context:** Phase 22 — Write → Absorb → Lean cycle, post-ToadStool session 39 review

---

## Executive Summary

wetSpring has completed **93 experiments, 2,173+ validation checks, 728 Rust
tests, and 83 binaries** — all passing. 19 of 20 GPU modules lean on upstream
ToadStool primitives. 4 local WGSL shaders are in Write phase pending absorption.

This handoff covers:

1. **ToadStool session 39 review** — what's changed since v4 absorption
2. **ODE blocker update** — `enable f64;` cleared, new `compile_shader` issue found
3. **5 new bio primitives** available that wetSpring can leverage
4. **4 local WGSL shaders** in Write phase (absorption candidates)
5. **Concrete feedback** for ToadStool team

---

## Part 1: ToadStool Session 39 Review

### What ToadStool absorbed from wetSpring (v4)

ToadStool absorbed wetSpring's handoff v4 (`cce8fe7c`) and has since done
extensive work through sessions 25–39:

- **Session 25**: unit test expansion, FFT f64, error system debt
- **Sessions 31–31c**: executor wiring, smart refactoring, dead code cleanup
- **Session 31f**: MathOp wiring, orphan shader integration, f64 linalg
- **Session 31g**: RF inference, HMM f32, optimizer shaders, safety audit
- **Session 31h**: clippy clean, dead code audit, PollConfig refactor
- **Session 39**: dead code sweep, false-positive allow cleanup, archive stale

### What ToadStool now provides (ops::bio)

17 bio primitives in `barracuda::ops::bio`:

| Primitive | Status | wetSpring uses? |
|-----------|--------|:---------------:|
| `SmithWatermanGpu` | Stable | ✅ |
| `GillespieGpu` | Stable | ✅ |
| `TreeInferenceGpu` / `FlatForest` | Stable | ✅ |
| `FelsensteinGpu` | Stable | ✅ |
| `HmmBatchForwardF64` | Stable | ✅ |
| `AniBatchF64` | Stable | ✅ |
| `SnpCallingF64` | Stable | ✅ |
| `DnDsBatchF64` | Stable | ✅ |
| `PangenomeClassifyGpu` | Stable | ✅ |
| `QualityFilterGpu` | Stable | ✅ |
| `Dada2EStepGpu` | Stable | ✅ |
| `RfBatchInferenceGpu` | Stable | ✅ |
| `LocusVarianceGpu` | New (session 31f) | Future |
| `PairwiseHammingGpu` | New (session 31f) | Future |
| `PairwiseJaccardGpu` | New (session 31f) | Future |
| `SpatialPayoffGpu` | New (session 31f) | Future |
| `BatchFitnessGpu` | New (session 31f) | Future |

wetSpring consumes 12 directly from `ops::bio` + 7 from other ops modules
(FMR, BrayCurtis, WeightedDot, Kriging, Correlation, Covariance, Variance,
GEMM, Eigh, TolSearch, Xoshiro) = **19 lean GPU modules**.

---

## Part 2: ODE Blocker Update

### What was blocked

wetSpring's local `ode_sweep_gpu.rs` wraps `batched_qs_ode_rk4_f64.wgsl`
because ToadStool's upstream shader contained `enable f64;` (naga rejects it).

### What ToadStool fixed (session 31h+)

Line 35 of `shaders/numerical/batched_qs_ode_rk4_f64.wgsl` is now:
```
// f64 is enabled by compile_shader_f64() preamble injection — do not use `enable f64;`
```

The `enable f64;` directive is **removed**. The shader is correct.

### Remaining issue

`batched_ode_rk4.rs:209` calls `dev.compile_shader(...)` instead of
`dev.compile_shader_f64(...)`. The shader comment says preamble injection
should happen, but the Rust code doesn't invoke it.

**Fix**: Change line 209 from:
```rust
let shader = dev.compile_shader(Self::wgsl_shader(), Some("BatchedOdeRK4"));
```
to:
```rust
let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("BatchedOdeRK4"));
```

Or use `ShaderTemplate::for_driver_auto(source, true)` to get the
`pow_f64`/`exp_f64`/`log_f64` polyfills needed on Ada Lovelace GPUs.

Once fixed, wetSpring can delete `ode_sweep_gpu.rs` (~230 lines) and
`batched_qs_ode_rk4_f64.wgsl` (~130 lines), replacing with a thin
wrapper around `barracuda::ops::BatchedOdeRK4F64`.

---

## Part 3: 5 New Bio Primitives

wetSpring can leverage these for future domain expansion:

| Primitive | Life Science Use Case |
|-----------|----------------------|
| `LocusVarianceGpu` | Per-locus allele frequency variance for FST/population structure |
| `PairwiseHammingGpu` | SNP-based strain distance matrices (Track 1c) |
| `PairwiseJaccardGpu` | Gene presence/absence overlap for pangenomics |
| `SpatialPayoffGpu` | Spatial prisoner's dilemma for cooperation models (Exp025) |
| `BatchFitnessGpu` | Batch fitness evaluation for evolutionary simulation sweeps |

---

## Part 4: Local WGSL Shaders (Write Phase — 4 shaders)

These shaders are written locally, validated against CPU baselines, and
ready for ToadStool to absorb as `ops::bio::*` primitives:

### 1. `batched_qs_ode_rk4_f64.wgsl` (ODE sweep)
- **File**: `barracuda/src/shaders/batched_qs_ode_rk4_f64.wgsl`
- **Domain**: QS/c-di-GMP 5-variable RK4 parameter sweep
- **CPU ref**: `bio::ode`, `bio::qs_biofilm`
- **Validation**: Exp049, 7/7 GPU checks
- **Blocker**: Upstream `compile_shader` issue (see Part 2)
- **Bindings**: uniform(0), storage(1-3), workgroup 256

### 2. `kmer_histogram_f64.wgsl` (k-mer counting)
- **File**: `barracuda/src/shaders/kmer_histogram_f64.wgsl`
- **Domain**: Atomic histogram into 4^k flat buffer
- **CPU ref**: `bio::kmer::count_kmers` (Exp081)
- **Target**: `ops::bio::kmer_histogram`
- **Bindings**: uniform(0), storage(1-2), workgroup 256

### 3. `unifrac_propagate_f64.wgsl` (UniFrac)
- **File**: `barracuda/src/shaders/unifrac_propagate_f64.wgsl`
- **Domain**: Bottom-up CSR tree propagation for phylogenetic distances
- **CPU ref**: `bio::unifrac::unweighted_unifrac` (Exp082)
- **Target**: `ops::bio::unifrac_propagate`
- **Bindings**: uniform(0), storage(1-4), workgroup 64
- **Note**: Multi-pass design (one dispatch per tree level)

### 4. `taxonomy_fc_f64.wgsl` (taxonomy scoring)
- **File**: `barracuda/src/shaders/taxonomy_fc_f64.wgsl`
- **Domain**: Naive Bayes log-posterior scoring, GEMM-like
- **CPU ref**: `bio::taxonomy::NaiveBayesClassifier::classify` (Exp083)
- **Target**: `ops::bio::taxonomy_fc`
- **Bindings**: uniform(0), storage(1-4), workgroup (16,16)
- **Note**: NPU int8 variant planned via `to_int8_weights()`

---

## Part 5: Feedback for ToadStool Team

### Critical Fix

1. **`batched_ode_rk4.rs:209`** — switch `compile_shader` to `compile_shader_f64`

### Suggestions

2. **Crate-level re-exports** — `lib.rs:126-129` only re-exports the original
   v4 primitives (SW, Gillespie, TreeInference, Felsenstein). The 12 additional
   bio primitives (HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF, LocusVar,
   Hamming, Jaccard, SpatialPayoff, BatchFitness) require deep `ops::bio::`
   imports. Adding them to the crate-level `pub use` would match the ergonomics
   Springs expect.

3. **`compile_shader_f64` audit** — ensure all `_f64.wgsl` shaders in
   `shaders/numerical/` are compiled with `compile_shader_f64` (not `compile_shader`).

4. **PCoA / `BatchedEighGpu`** — still triggers a naga validation error on some
   drivers ("invalid function call" in the Eigh shader). wetSpring wraps with
   `catch_unwind` and falls back to CPU. May warrant a `ShaderTemplate` workaround.

---

## Validation Summary

| Category | Checks | Status |
|----------|:------:|--------|
| CPU parity (Python → Rust) | 1,392 | ALL PASS |
| GPU parity (CPU → GPU) | 609 | ALL PASS |
| Streaming dispatch | 80 | ALL PASS |
| Layout fidelity (Tier A) | 35 | ALL PASS |
| Transfer/streaming | 57 | ALL PASS |
| CPU vs GPU all 16 domains | 48 | ALL PASS (Exp092) |
| metalForge 16 domains | 28 | ALL PASS (Exp093) |
| **Total** | **2,173+** | **ALL PASS** |

---

## File Locations

| Item | Path |
|------|------|
| ABSORPTION_MANIFEST | `barracuda/ABSORPTION_MANIFEST.md` |
| EVOLUTION_READINESS | `barracuda/EVOLUTION_READINESS.md` |
| Local WGSL shaders | `barracuda/src/shaders/*.wgsl` (4 files) |
| ODE sweep GPU | `barracuda/src/bio/ode_sweep_gpu.rs` |
| Forge crate | `metalForge/forge/` (v0.2.0, 32 tests) |
| Experiment records | `experiments/001-093` |
| Binaries | `barracuda/src/bin/` (83 total) |
