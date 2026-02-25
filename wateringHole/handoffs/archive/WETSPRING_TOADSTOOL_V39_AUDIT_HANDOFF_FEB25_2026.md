# wetSpring → ToadStool Handoff V39: Comprehensive Audit + Tolerance Completion

**Date:** February 25, 2026
**From:** wetSpring (Phase 45, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**Supersedes:** V38 (deep debt handoff)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring underwent a comprehensive codebase audit covering every dimension:
linting, formatting, clippy (pedantic + nursery), test coverage, unsafe code,
zero-copy I/O, validation fidelity, barracuda dependency health, evolution
readiness, data provenance, and wateringHole compliance. The audit found
the codebase in excellent health (95/100) with 8 remaining ad-hoc tolerance
literals as the only gap. All 8 have been centralized into named constants.

**Post-V39 numbers:** 759 lib tests, 47 forge tests, 95.75% library coverage,
**70** named tolerance constants (up from 62), 0 ad-hoc tolerance literals
remaining, 0 clippy warnings, 0 TODO/FIXME, 0 unsafe, 0 production mocks,
167 experiments, 3,279+ validation checks, 157 binaries, 44 ToadStool
primitives + 2 BGL helpers consumed, 1 Write-phase WGSL extension.

---

## Part 1: What Changed (V39 Session)

### 1.1 Tolerance Completion (8 New Named Constants)

The V38 session centralized ~200 tolerance replacements. The V39 audit found
8 remaining ad-hoc literals across 5 validation binaries. All have been
promoted to named constants with scientific justification.

| Constant | Value | Binary | Justification |
|----------|-------|--------|---------------|
| `RAREFACTION_MONOTONIC` | 1e-10 | `validate_diversity` | Hypergeometric rounding in rarefaction curves |
| `PCOA_EIGENVALUE_FLOOR` | 1e-10 | `validate_diversity` | Jacobi eigenvalue numerical noise from centering |
| `KMD_NON_HOMOLOGUE` | 0.005 | `validate_pfas` | Tighter KMD for non-homologue separation |
| `HMM_FORWARD_PARITY` | 1e-6 | `validate_barracuda_cpu` | Log-sum-exp accumulation across T×N states |
| `GILLESPIE_PYTHON_RANGE_REL` | 0.15 | `validate_gillespie` | Different PRNG → 15% ensemble range |
| `GILLESPIE_FANO_PHYSICAL` | 1.0 | `validate_gillespie` | Fano factor physical range [0, 2] for Poisson |
| `ASARI_CROSS_MATCH_PCT` | 70.0 | `validate_features` | Single-file vs 8-file asari extraction |
| `ASARI_MZ_RANGE_PCT` | 10.0 | `validate_features` | m/z range coverage ≥ 90% of asari |

Additionally, ~15 bare `0.0` tolerance arguments in `validate_barracuda_cpu`
and `validate_gillespie` were replaced with `tolerances::EXACT` for
consistency with the "no ad-hoc magic numbers" rule.

### 1.2 Comprehensive Audit Results

| Dimension | Finding | Score |
|-----------|---------|:-----:|
| Linting / fmt / clippy (pedantic + nursery) | All green, zero warnings | 10/10 |
| Test coverage (llvm-cov) | 95.75% library, 90%+ gate passes | 9/10 |
| Validation fidelity | All baselines documented with provenance | 9/10 |
| Zero-copy / streaming I/O | All parsers stream; FASTQ has zero-copy path | 9/10 |
| Unsafe code | `#![deny(unsafe_code)]` crate-wide, zero blocks | 10/10 |
| BarraCuda integration | 44 primitives, no reinvention, correct delegation | 10/10 |
| Evolution readiness | Track 3 awaiting 3 ToadStool primitives | 9/10 |
| License / sovereignty | AGPL-3.0-or-later, zero violations | 9/10 |
| Code organization | All files < 1000 LOC, single-responsibility | 10/10 |
| Data provenance | Public repos, accession numbers, SHA-256 | 10/10 |

**Lower-coverage modules** (below 90% individual target):
- `bio/ncbi_data.rs`: 68.73% (network-dependent code paths)
- `ncbi.rs`: 77.47% (HTTP transport discovery)
- `io/fastq/mod.rs`: 83.67% (gzip edge cases)

These are the only modules below 90%. All involve external I/O or network
code that is inherently difficult to unit test. Overall library coverage
remains 95.75%.

### 1.3 Zero Debt Confirmation

| Category | Count |
|----------|:-----:|
| TODO/FIXME/HACK markers | 0 |
| Production mocks or stubs | 0 |
| `unsafe` blocks | 0 |
| `.unwrap()` / `.expect()` in library code | 0 |
| Ad-hoc tolerance literals | 0 |
| Files > 1000 LOC | 0 |
| Validation binaries without provenance | 0 |

---

## Part 2: What wetSpring Consumes from ToadStool

**Unchanged from V38.** 44 ToadStool primitives + 2 BGL helpers + 1
Write-phase WGSL extension. See V38 Part 2 for full accounting.

### Delegation Audit (Confirmed)

| wetSpring module | Delegates to | Pattern |
|-----------------|-------------|---------|
| `special::erf()` | `barracuda::special::erf()` | Direct delegation |
| `special::ln_gamma()` | `barracuda::special::ln_gamma()` | Unwrap with `f64::INFINITY` fallback |
| `special::regularized_gamma_lower()` | `barracuda::special::regularized_gamma_p()` | Unwrap with `0.0` fallback |
| `forge::bridge::create_device()` | `barracuda::device::WgpuDevice` | GPU device creation |

**No duplicate math.** `normal_cdf`, `l2_norm`, `dot` are local helpers with
no barracuda equivalents (P1 request #4 asks ToadStool to absorb `dot`/`l2_norm`).

---

## Part 3: Evolution Requests for ToadStool/BarraCuda

**Unchanged from V38.** All P0-P3 requests remain open:

| Priority | # | Request | Status |
|----------|---|---------|--------|
| P0 | 1 | `GemmF64::wgsl_shader_for_device()` public | Open |
| P1 | 2 | Fix `PeakDetectF64` f32→f64 literal | Open |
| P1 | 3 | `ComputeDispatch` cached-pipeline variant | Open |
| P1 | 4 | `barracuda::math::{dot, l2_norm}` | Open |
| P2 | 5 | Absorb `diversity_fusion_f64.wgsl` | Open |
| P2 | 6 | `BatchedOdeRK4Generic<N, P>` | Open |
| P3 | 7 | GPU Top-K selection primitive | Open |
| P3 | 8 | NPU int8 quantization helpers | Open |

### New Recommendation from Audit

**P2 request #9: Tolerance module pattern for ToadStool validation.**
wetSpring's `tolerances.rs` pattern (70 named constants, hierarchy tests,
doc comments with physics justification) has proven effective at preventing
silent regressions. If ToadStool/BarraCuda has validation binaries with
ad-hoc tolerance literals, consider adopting this pattern.

---

## Part 4: Three-Tier Control Matrix (Paper Queue)

**Unchanged from V38.** All 30 actionable papers carry full three-tier
validation (CPU + GPU + metalForge). 43/43 total papers reproduced.

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

### Open Data Provenance (Confirmed)

All 43 reproductions use publicly accessible data or published model
parameters. No proprietary data dependencies. Sources:

- **ODE/Stochastic**: Model parameters from published equations (journals)
- **16S amplicon**: NCBI SRA (PRJNA488170, PRJNA382322, PRJNA1114688, etc.)
- **PFAS**: asari test data, Zenodo 14341321, EPA public data, MassBank
- **Genomics**: NCBI SRA (PRJNA283159, PRJEB5293), MBL, MG-RAST, Figshare, OSF
- **Drug repurposing**: repoDB (PMC7153111), published equations, ROBOKOP KG

Track 4 (Soil QS) papers are queued — all use open data (Nature Comms,
ISWCR, Soil Biol Biochem, etc.).

---

## Part 5: Lessons Learned (for ToadStool Evolution)

### 5.1 Comprehensive Audits Find What Incremental Work Misses

V38 centralized ~200 tolerance replacements across 35 binaries. The V39
audit (reviewing every validation binary systematically) found 8 more
ad-hoc literals that had been missed. Lesson: periodic full-codebase
audits catch residual debt that incremental development overlooks.

### 5.2 The hotSpring Validation Pattern Is Battle-Tested

All 145 wetSpring validation binaries follow the hotSpring pattern:
hardcoded expected values, explicit pass/fail, exit code 0/1, provenance
comments. This pattern scales to 3,279+ checks across 167 experiments.
No binary was found to be missing provenance or using incorrect tolerances.

### 5.3 `#![deny(unsafe_code)]` Eliminates an Entire Bug Class

With `deny(unsafe_code)` on both `wetspring-barracuda` and `wetspring-forge`,
zero unsafe blocks exist in the codebase. The entire crate compiles with
Rust's full safety guarantees. Performance-critical paths (I/O parsers,
matrix operations) achieve competitive speed without unsafe.

### 5.4 Zero-Copy Where Possible, Streaming Always

The FASTQ parser demonstrates the right layering:
- `for_each_record()` — zero-copy borrowed slices, O(1) memory
- `FastqIter` — owned records for multi-pass analysis
- `stats_from_file()` — single-pass statistics, zero per-record allocation

All three parsers (FASTQ, MS2, mzML) stream via `BufReader` — no parser
buffers an entire file. Binary decoding (mzML base64+zlib) inherently
requires allocation; the `DecodeBuffer` reuse pattern minimizes it.

### 5.5 Barracuda Dependency Is Clean

The audit confirmed zero reinvented math, correct delegation to
`barracuda::special`, and no duplicate GPU ops. The 44-primitive consumption
is well-documented in `ABSORPTION_MANIFEST.md` with per-primitive provenance.

---

## Part 6: Quality Gates (All Green)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --all-targets -W pedantic -W nursery -D warnings` | **0 diagnostics** |
| `cargo test` | 806 passed (759 lib + 47 forge), 0 failed |
| `cargo doc --no-deps` | 0 warnings, 88 files |
| `cargo llvm-cov --lib --fail-under-lines 90` | 95.75% line coverage |
| `#![deny(unsafe_code)]` | Enforced crate-wide (barracuda + forge) |
| Named tolerance constants | **70** (up from 62) |
| Ad-hoc tolerance literals | **0** (down from 8) |
| Provenance headers | 157/157 binaries |
| TODO/FIXME/HACK markers | 0 |
| Production mocks/stubs | 0 |

---

## Part 7: File Locations

| Item | Path |
|------|------|
| tolerances module (70 constants) | `barracuda/src/tolerances.rs` |
| Updated validation binaries | `barracuda/src/bin/validate_{diversity,pfas,barracuda_cpu,gillespie,features}.rs` |
| absorption manifest | `barracuda/ABSORPTION_MANIFEST.md` |
| evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| paper review queue | `specs/PAPER_REVIEW_QUEUE.md` |
| primitive map (metalForge) | `metalForge/PRIMITIVE_MAP.md` |
| this handoff | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V39_AUDIT_HANDOFF_FEB25_2026.md` |

---

## Part 8: Evolution Readiness by Module

### Tier A — Ready for GPU Shader Promotion (Already Lean)

27 modules already consume ToadStool GPU primitives. No promotion needed.

### Tier B — Adapt (Compose/Passthrough)

10 modules wire ToadStool primitives or accept GPU buffers:
`kmd_gpu`, `merge_pairs_gpu`, `robinson_foulds_gpu`, `derep_gpu`,
`neighbor_joining_gpu`, `reconciliation_gpu`, `molecular_clock_gpu`,
`gbm_gpu`, `feature_table_gpu`, `signal_gpu`.

### Write-Phase Extension

1 local WGSL shader: `diversity_fusion_f64.wgsl` (fused Shannon + Simpson +
evenness). P2 request for ToadStool absorption.

### CPU-Only (No GPU Benefit)

FASTQ parsing (I/O-bound), NCBI HTTP (network-bound).

### Blocking GPU Promotion

Track 3 drug repurposing awaiting 3 ToadStool primitives:
Top-K selection (P3 #7), NPU int8 quantization helpers (P3 #8).
NMF, SpGEMM, TransE, cosine already upstream.

### Mapping: Rust Module → WGSL Shader → Pipeline Stage

| Rust Module | GPU Primitive | Pipeline Stage |
|-------------|--------------|----------------|
| `diversity` | `ops::bio::diversity` (lean) | 16S → diversity |
| `diversity_fusion_gpu` | `diversity_fusion_f64.wgsl` (local) | Fused diversity |
| `ode::*_gpu` × 5 | `BatchedOdeRK4<S>::generate_shader()` | ODE parameter sweep |
| `hmm_gpu` | `ops::bio::hmm_forward_f64` | Phylo-HMM |
| `ani_gpu` | `ops::bio::ani_f64` | Metagenomics ANI |
| `pcoa_gpu` | `linalg::jacobi` | Ordination |
| `gemm_cached` | `linalg::gemm_f64` | Linear algebra |
| `nmf_gpu` | `linalg::nmf` | Drug repurposing |
| `streaming_gpu` | `pipeline::StreamingSession` | Zero round-trip |

---

## Related Handoffs

- **Supersedes:** `WETSPRING_TOADSTOOL_V38_DEEP_DEBT_HANDOFF_FEB25_2026.md`
- **Builds on:** V34-V38 (absorption accounting, DF64 lean, write-phase, revalidation, deep debt)
- **See:** `CROSS_SPRING_SHADER_EVOLUTION.md` (660+ shader provenance)
- **See:** `barracuda/EVOLUTION_READINESS.md` (full upstream request list)
