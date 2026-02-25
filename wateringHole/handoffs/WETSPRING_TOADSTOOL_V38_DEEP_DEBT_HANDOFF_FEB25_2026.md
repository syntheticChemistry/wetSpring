# wetSpring → ToadStool Handoff V38: Deep Debt Resolution + Evolution Requests

**Date:** February 25, 2026
**From:** wetSpring (Phase 45, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**Supersedes:** V37 (revalidation handoff)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring completed comprehensive deep debt resolution: all tolerance literals
centralized (200+ replacements across 35 binaries), all I/O paths evolved to
zero per-line allocation, all 157 binaries carry complete provenance with real
commit hashes and commands, and library APIs annotated with `#[must_use]` and
`Vec::with_capacity` pre-allocation. The codebase now passes `cargo clippy
--all-targets -W pedantic -W nursery -D warnings` with **zero diagnostics**.

**Post-V38 numbers:** 759 lib tests, 47 forge tests, 95.75% library coverage,
62 named tolerance constants, 0 clippy warnings, 0 TODO/FIXME, 0 unsafe,
0 bare `.unwrap()`, 167 experiments, 3,279+ validation checks, 157 binaries,
44 ToadStool primitives + 2 BGL helpers consumed, 1 Write-phase WGSL extension.

---

## Part 1: What Changed (V38 Session)

### 1.1 Tolerance Centralization (Complete)

All validation binaries now use named constants from `tolerances.rs` instead of
ad-hoc numeric literals. This makes tolerance rationale traceable and auditable.

| Before | After | Occurrences |
|--------|-------|:-----------:|
| `0.0` (tolerance arg) | `tolerances::EXACT` | ~180 |
| `1e-10` (GPU parity) | `tolerances::GPU_VS_CPU_TRANSCENDENTAL` | 6 |
| `1e-10` (Python parity) | `tolerances::PYTHON_PARITY` | 3 |
| `0.001` (Simpson) | `tolerances::ODE_METHOD_PARITY` | 1 |

**3 new constants added:**

| Constant | Value | Justification |
|----------|-------|---------------|
| `GPU_LOG_POLYFILL` | 1e-7 | WGSL `log_f64` polyfill precision (~1e-8), looser than native transcendentals |
| `ODE_NEAR_ZERO_RELATIVE` | 1.5 | 150% relative error for near-zero ODE variables (e.g., dormant phage defense) |
| Strengthened `EXACT` usage | 0.0 | All exact-match checks now use the named constant |

### 1.2 I/O Performance Evolution

All `reader.lines()` patterns (which allocate a new `String` per line) replaced
with `read_line()` into a reusable `String` buffer:

| File | Function | Pattern |
|------|----------|---------|
| `io/ms2.rs` | `Ms2Iter::next()` | `Lines<Box<dyn BufRead>>` → `Box<dyn BufRead>` + buffer |
| `io/ms2.rs` | `stats_from_file()` | `reader.lines()` → `read_line()` loop |
| `bio/validation_helpers.rs` | `stream_taxonomy_tsv()` | `reader.lines()` → `read_line()` loop |
| `bio/validation_helpers.rs` | `stream_fasta_subsampled()` | `reader.lines()` → `read_line()` loop |
| `bench/power.rs` | `spawn_nvidia_smi_poller()` | `reader.lines()` → `read_line()` loop |

### 1.3 Provenance Hardening

| Action | Count |
|--------|:-----:|
| Placeholder `current HEAD` → `1f9f80e` | 24 binaries |
| Placeholder `(current)` → `1f9f80e` | 1 binary |
| Added missing `| Command |` row | 11 binaries |
| Reformatted to standard `//! # Provenance` table | 2 binaries |
| **Binaries with complete provenance** | **157/157** |

### 1.4 API Quality

| Annotation | Files | Items |
|------------|:-----:|:-----:|
| `#[must_use]` on public fns | 5 | 7 functions |
| `Vec::with_capacity` | 5 | 7 allocations |

---

## Part 2: What wetSpring Consumes from ToadStool

**44 ToadStool primitives + 2 BGL helpers + 1 Write-phase WGSL extension.**

All consumption unchanged from V37. barracuda is always-on (`default-features = false`
for CPU, `barracuda/gpu` for GPU builds). Zero fallback code. Zero local WGSL
shaders for absorbed domains.

### Absorption Accounting (unchanged from V37)

| Category | Count | Examples |
|----------|:-----:|---------|
| GPU bio primitives | 8 | HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF |
| GPU linalg | 7 | GEMM, Jacobi, NMF, Ridge, SpMM, BatchedEigh, BrayCurtis |
| GPU ops | 6 | FMR, TransE, PeakDetect, SparseGemm, KmerHistogram, UniFracPropagate |
| ODE trait shaders | 5 | `BatchedOdeRK4<S>::generate_shader()` for 5 bio ODE systems |
| Cross-spring (neuralSpring) | 11 | Hamming, Jaccard, SpatialPayoff, BatchFitness, LocusVariance, GraphLaplacian, EffectiveRank, NumericalHessian, DisorderedLaplacian, BeliefPropagation, BoltzmannSampling |
| Cross-spring (hotSpring) | 5 | Anderson Hamiltonians (1D/2D/3D), Lanczos, AlmostMathieu |
| BGL helpers | 2 | `storage_bgl_entry`, `uniform_bgl_entry` |
| Write-phase WGSL | 1 | `diversity_fusion_f64.wgsl` (Shannon + Simpson + evenness) |

---

## Part 3: Evolution Requests for ToadStool/BarraCuda

### P0 — Blocking

| # | Request | Context | Benefit |
|---|---------|---------|---------|
| 1 | Make `GemmF64::wgsl_shader_for_device()` public | wetSpring's `gemm_cached.rs` cannot auto-select DF64 vs native path | Enables DF64 GEMM on consumer GPUs without manual selection |

### P1 — High Value

| # | Request | Context | Benefit |
|---|---------|---------|---------|
| 2 | Fix `PeakDetectF64` WGSL shader (f32 literal → f64 array, line 49) | Reported V35; shader compiles but produces wrong results on mixed-precision | Correct f64 peak detection |
| 3 | `ComputeDispatch` with cached-pipeline variant | wetSpring's `GpuPipelineSession` manually caches pipelines + BGLs | Returns (pipeline, BGL) for reuse — eliminates 100ms cold-start per dispatch |
| 4 | `barracuda::math::{dot, l2_norm}` CPU primitives | Currently in `wetspring::special`, thin wrappers. Used by `spectral_match` cosine similarity | Shared CPU math for all Springs |

### P2 — Nice to Have

| # | Request | Context | Benefit |
|---|---------|---------|---------|
| 5 | Absorb `diversity_fusion_f64.wgsl` as `ops::bio::diversity_fusion` | Write-phase extension, 18/18 checks, documented binding layout | Fused Shannon + Simpson + evenness in single dispatch |
| 6 | `BatchedOdeRK4Generic<N, P>` for arbitrary ODE dimensions | Currently trait-generated; 5 bio ODE systems use different (N, P) pairs | Eliminates per-system shader generation overhead |

### P3 — Future

| # | Request | Context | Benefit |
|---|---------|---------|---------|
| 7 | GPU Top-K selection primitive | Used in drug repurposing scoring; currently CPU-only | Enables end-to-end GPU drug-disease ranking |
| 8 | NPU int8 quantization helpers | ESN reservoir → Akida deployment pattern | Standardized CPU→NPU weight conversion |

---

## Part 4: Three-Tier Control Matrix (Paper Queue)

**All 30 actionable papers carry full three-tier validation (CPU + GPU + metalForge).**

| Track | Papers | CPU | GPU | metalForge |
|-------|:------:|:---:|:---:|:----------:|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 |
| **Actionable total** | **30** | **30/30** | **30/30** | **30/30** |
| Cross-spring | 1 | 1/1 | 1/1 | — |
| Extensions (analytical) | 9 | 9/9 | — | — |
| Track 4 (Soil QS) | 9 | — | — | — |
| **Grand total** | **43+9** | **43/43** | **31/31** | **30/30** |

Track 4 papers are queued for reproduction — they extend the Anderson-QS
framework to agricultural soil pore networks.

---

## Part 5: Lessons Learned (for ToadStool Evolution)

### 5.1 Tolerance Naming Prevents Silent Regressions

When tolerance literals like `1e-10` are scattered across 35 binaries, it's
impossible to audit whether each value is appropriate. After centralization:
- Every tolerance traces to a named constant with a doc comment explaining its physics
- Changing a tolerance value requires updating one location and understanding its justification
- The hierarchy test in `tolerances.rs` ensures `EXACT < PYTHON_PARITY < GPU_VS_CPU_F64 < ...`

**Recommendation for ToadStool:** If barracuda validation binaries use tolerance
literals, consider a similar centralization. The `tolerances` module pattern works
well — constants grouped by domain with `#[doc]` comments explaining the physics.

### 5.2 `reader.lines()` Is a Hidden Allocation Tax

`BufReader::lines()` allocates a new `String` for every line. For parsers processing
millions of lines (FASTQ, mzML, MS2), this is a significant allocation tax.
`read_line()` with a reusable buffer is measurably faster and generates zero
per-line garbage.

**Recommendation for ToadStool:** If any upstream parsers use `reader.lines()`,
consider the `read_line()` + reusable buffer pattern.

### 5.3 Provenance Completeness Enables Reproducibility

Every validation binary needs: (1) a commit hash pinning the baseline, (2) the
exact command to reproduce, and (3) references to the papers/algorithms being
validated. Without these, validation results are assertions without evidence.

### 5.4 `#[must_use]` on Fallible APIs Catches Bugs at Compile Time

Adding `#[must_use]` to `parse_*` and `stats_from_file` functions catches cases
where the caller discards the `Result` — even though `Result` itself has
`#[must_use]`, the custom message provides domain-specific context.

---

## Part 6: Quality Gates (All Green)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --all-targets -W pedantic -W nursery -D warnings` | **0 diagnostics** |
| `cargo test --lib` | 759 passed, 0 failed, 1 ignored |
| `cargo doc --no-deps` | 0 warnings, 88 files |
| `cargo llvm-cov --lib --summary-only` | 95.75% line coverage |
| `#![deny(unsafe_code)]` | Enforced crate-wide |
| `#![deny(clippy::expect_used, unwrap_used)]` | Enforced crate-wide (except test modules) |
| Named tolerance constants | 62 (up from 59) |
| Provenance headers | 157/157 binaries |
| TODO/FIXME/HACK markers | 0 |

---

## Part 7: File Locations

| Item | Path |
|------|------|
| tolerances module | `barracuda/src/tolerances.rs` |
| MS2 streaming parser | `barracuda/src/io/ms2.rs` |
| validation helpers | `barracuda/src/bio/validation_helpers.rs` |
| power monitoring | `barracuda/src/bench/power.rs` |
| absorption manifest | `barracuda/ABSORPTION_MANIFEST.md` |
| evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| paper review queue | `specs/PAPER_REVIEW_QUEUE.md` |
| this handoff | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V38_DEEP_DEBT_HANDOFF_FEB25_2026.md` |

---

## Related Handoffs

- **Supersedes:** `WETSPRING_TOADSTOOL_V37_REVALIDATION_HANDOFF_FEB25_2026.md`
- **Builds on:** V34-V36 (absorption accounting, DF64 lean, write-phase extension)
- **See:** `CROSS_SPRING_SHADER_EVOLUTION.md` (660+ shader provenance)
- **See:** `barracuda/EVOLUTION_READINESS.md` (full upstream request list)
