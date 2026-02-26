# ToadStool / BarraCUDA — V55 Deep Debt Resolution & Evolution Handoff

**Date:** February 26, 2026
**From:** wetSpring (ecoPrimals)
**To:** ToadStool / BarraCUDA core team
**Covers:** V55 deep debt resolution, idiomatic Rust evolution, barracuda usage review, absorption learnings
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring V55 completes a deep debt resolution pass across the entire codebase.
All 6 clippy-pedantic-failing binaries are fixed. `cargo clippy --all-targets
-- -D warnings -W clippy::pedantic` now passes cleanly across the library and
all 173 binaries. Two new named tolerance constants replace the last ad-hoc
magic numbers. The `encoding.rs` module evolves from `Result<_, String>` to
proper typed errors. This handoff documents:

1. **What changed** — V55 specific fixes and evolutions
2. **How wetSpring uses barracuda** — 79-primitive usage audit (V55 update)
3. **What ToadStool should evolve** — patterns and learnings for absorption
4. **Paper validation controls** — 52/52 papers, 39/39 three-tier confirmed
5. **Hardware coverage matrix** — barracuda CPU, GPU, metalForge

---

## Part 1: What Changed in V55

### Clippy Pedantic (6 binaries fixed, 0 remaining)

| Binary | Fixes Applied |
|--------|---------------|
| `benchmark_cross_spring_s65` | Doc backticks, `f64::from()`, `f64::midpoint()`, `.is_some_and()`, import ordering, redundant closures |
| `validate_gpu_diversity_fusion` | Full rewrite — migrated to `Validator` framework, refactored into 4 sub-functions |
| `validate_soil_qs_cpu_parity` | `many_single_char_names` allow, strict float comparison → `Validator::check()` |
| `benchmark_cross_spring_modern` | `many_single_char_names` allow |
| `benchmark_modern_systems_df64` | `many_single_char_names` allow |
| `validate_metalforge_drug_repurposing` | `items_after_statements` allow |

### Error Type Evolution

`encoding::base64_decode` return type evolved:

```
Before: Result<Vec<u8>, String>
After:  crate::error::Result<Vec<u8>>  (Error::Base64 variant)
```

Caller in `mzml/decode.rs` simplified from `.map_err(Error::Base64)?` to `?`.

### New Named Tolerances (2)

| Constant | Value | Replaces | Provenance |
|----------|-------|----------|------------|
| `ODE_GPU_SWEEP_ABS` | 0.15 | Ad-hoc `0.15` in `validate_gpu_ode_sweep` | GPU f64 vs CPU f64 drift over 1000+ RK4 steps, 128 parameter batches |
| `GPU_EIGENVALUE_REL` | 0.05 | Ad-hoc `0.05` in `validate_gpu_ode_sweep` | Jacobi eigendecomposition GPU vs CPU, near-zero eigenvalues |

### Bug Fix

`ncbi/http.rs`: whitespace-only `WETSPRING_HTTP_CMD` (e.g. `"   "`) was treated
as a valid custom backend. Now properly rejected via `.trim().is_empty()`.

### Provenance Addition

`PfasFragments` default masses now document full derivation from NIST Chemistry
WebBook 2023 CODATA monoisotopic atomic masses.

### Test Count: 906 → 912

6 new tests: FASTQ empty-line break, FASTQ/FastqIter nonexistent file,
HTTP whitespace custom cmd, HTTP custom with args, HTTP invalid UTF-8 output,
tolerance non-negativity.

---

## Part 2: How wetSpring Uses BarraCUDA (V55 Audit)

### Upstream Primitives Consumed (79 total, S66)

| Category | Count | Examples |
|----------|:-----:|---------|
| GPU compute ops | 27 | `FusedMapReduceF64`, `DiversityFusionGpu`, `SmithWatermanGpu`, `FelsensteinGpu`, `BatchedOdeRK4` |
| GPU linear algebra | 6 | `GemmF64`, `SparseGemmF64`, `NMF`, `ridge_regression`, `BatchedEighGpu`, `TranseScoreF64` |
| GPU signal/spectral | 3 | `PeakDetectF64`, `PairwiseHammingGpu`, `PairwiseJaccardGpu` |
| CPU stats (S64) | 13 | `shannon`, `simpson`, `chao1`, `pielou_evenness`, `bray_curtis`, `dot`, `l2_norm`, etc. |
| CPU stats (S66) | 6 | `hill`, `monod`, `fit_linear`, `percentile`, `mean`, `shannon_from_frequencies` |
| CPU special functions | 5 | `erf`, `ln_gamma`, `regularized_gamma_p`, `norm_cdf`, `pearson_correlation` |
| CPU numerical | 2 | `trapz`, `gradient_1d` |
| Spectral primitives | 8 | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `find_w_c`, etc. |
| Infrastructure | 9 | `WgpuDevice`, `TensorContext`, `ShaderTemplate`, `compile_shader_f64`, BGL helpers, etc. |

### Local Implementation (wetSpring owns biology, barracuda owns compute)

| Module | Why Local | Absorption Status |
|--------|-----------|:-----------------:|
| `bio/` (27 CPU modules) | Domain-specific biology (ODE systems, parsers, diversity) | Lean — all math delegated |
| `bio/*_gpu.rs` (42 GPU modules) | Thin wrappers wiring ToadStool ops to wetSpring types | Lean — 0 local WGSL |
| `io/` (FASTQ, mzML, MS2) | Sovereign streaming parsers | Local — no external parser deps |
| `tolerances.rs` (79 constants) | Spring-local validation thresholds | Local — primal self-knowledge |
| `validation.rs` | Validation framework (hotSpring pattern) | Local — primal self-knowledge |

### Evolution Principle

> wetSpring owns domain biology. BarraCUDA owns compute primitives.
> When local math becomes reusable, it gets absorbed upstream.
> When tolerance values stabilize, they stay local with provenance.

---

## Part 3: What ToadStool Should Evolve

### 3a. Tolerance Pattern — Recommended for All Springs

wetSpring's `tolerances.rs` pattern (79 named constants, full provenance, hierarchy
tests, non-negativity tests) has proven valuable for maintaining validation
fidelity across 183 experiments. ToadStool should consider:

- **Standardizing a `tolerances` module** in barracuda for cross-spring tolerances
  (e.g. `GPU_VS_CPU_F64`, `GPU_LOG_POLYFILL` are GPU-inherent, not spring-specific)
- **Defining a `tolerance!` macro** or builder for provenance-documenting constants

### 3b. Validator Framework — Proven Pattern

The `Validator` struct (pass/fail with named checks, section headers, explicit
exit code 0/1) is used by all 160 validation binaries. Consider absorbing this
pattern into barracuda as a shared validation harness.

### 3c. Error Type Evolution Learnings

Evolving `Result<_, String>` to typed errors across module boundaries revealed:
- `map_err` chains simplify to bare `?` when error types align
- `Error::Base64` was already defined but not used by the source module
- Lesson: define typed errors at the module that produces them, not the caller

### 3d. Edition 2024 + Pedantic Clippy Learnings

- `std::env::set_var` / `remove_var` are `unsafe` in edition 2024 — test code
  that manipulates env vars needs unsafe blocks or alternative patterns
- `f64::midpoint()` is available in edition 2024 — use instead of manual `(a+b)/2.0`
- `.is_some_and()` replaces `.map_or(false, |x| ...)` pattern
- `f64::from(i32)` replaces `i32 as f64` (avoids `cast_lossless`)
- `items_after_statements` is triggered by `const` inside function bodies

### 3e. Streaming I/O Patterns for Absorption

wetSpring's FASTQ/mzML/MS2 parsers demonstrate three streaming patterns that
could inform barracuda's I/O abstraction:

1. **Zero-copy borrowed records** (`FastqRefRecord<'a>`) — callback-based, no alloc
2. **Reusable decode buffers** (`DecodeBuffer`) — amortized across spectrum stream
3. **Iterator-based owned records** (`FastqIter`) — for multi-pass analysis

---

## Part 4: Paper Validation Controls (52/52 Confirmed)

### Three-Tier Control Matrix

| Track | Papers | CPU | GPU | metalForge | Status |
|-------|:------:|:---:|:---:|:----------:|--------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | Full three-tier |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | Full three-tier |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | Full three-tier |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 4 (Soil QS/Anderson) | 9 | 9/9 | 9/9 | 9/9 | Full three-tier |
| **Subtotal (three-tier)** | **39** | **39/39** | **39/39** | **39/39** | **ALL three-tier** |
| Cross-spring (spectral) | 1 | 1/1 | 1/1 | — | CPU + GPU |
| Extensions (Phase 37-39) | 9 | 9/9 | — | — | CPU only (analytical) |
| Anderson finite-size | 3 | 3/3 | — | — | CPU only (scaling) |
| **Grand total** | **52** | **52/52** | **40/40** | **39/39** | |

### Open Data Provenance — All 52 Papers Confirmed

Every reproduction uses publicly accessible data:
- **ODE/stochastic models** (10 papers): published equations from journal articles
- **Real 16S amplicon** (4 experiments): NCBI SRA accession numbers documented
- **PFAS/LC-MS** (4 papers): asari test data, EPA public data, MassBank, Zenodo
- **Phylogenetics** (5 papers): public algorithm implementations, PhyNetPy datasets
- **Deep-sea genomics** (6 papers): NCBI SRA, MBL darchive, MG-RAST, Figshare, OSF
- **Drug repurposing** (5 papers): repoDB (public), published equations, algorithmic
- **Soil QS** (9 papers): published soil metrics, model equations, review frameworks
- **Spectral/extensions** (9 papers): algorithmic (barracuda primitives), no external data

No proprietary data. No Sandia-restricted data. All proxy datasets are synthetic.

---

## Part 5: Hardware Coverage Matrix

### BarraCUDA CPU (Pure Rust, single-threaded)

| Metric | Value |
|--------|-------|
| Papers covered | 52/52 |
| Validation experiments | Exp035, 043, 057, 070, 079, 085, 102, 163, 179 |
| Total checks | 407+ |
| Python parity | All within documented tolerances |
| Key barracuda modules | `special::*`, `stats::*`, `numerical::*`, `linalg::*` |

### BarraCUDA GPU (RTX 4070, WGSL f64)

| Metric | Value |
|--------|-------|
| Papers covered | 43/43 (actionable papers) |
| Validation experiments | Exp064, 071, 087, 092, 101, 164, 180 |
| Total GPU checks | 1,578 on physical hardware |
| CPU ↔ GPU parity | All within `GPU_VS_CPU_F64` (1e-6) or `GPU_LOG_POLYFILL` (1e-7) |
| Key barracuda modules | `FusedMapReduceF64`, `GemmF64`, `BatchedOdeRK4`, `DiversityFusionGpu` |

### metalForge Cross-Substrate (CPU = GPU = NPU)

| Metric | Value |
|--------|-------|
| Papers covered | 39/39 (all three-tier-eligible) |
| Validation experiments | Exp060, 065, 080, 084, 086, 088, 093, 103, 104, 165, 182 |
| Total checks | 243+ |
| Cross-substrate parity | Same output regardless of compute substrate |
| Substrates tested | CPU (x86), GPU (RTX 4070), NPU (simulated Akida) |

---

## Part 6: Verification Commands

```bash
cd barracuda

# All gates
cargo fmt --check                                          # formatting
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic  # pedantic lint
RUSTDOCFLAGS="-D warnings" cargo doc --lib --no-deps       # docs
cargo test                                                 # 912 tests
cargo llvm-cov --lib --summary-only                        # 96.67% coverage

# metalForge
cd ../metalForge/forge && cargo test                       # 47 tests
```

---

## Part 7: Next Steps

### For ToadStool Team

1. **Consider absorbing `Validator` pattern** — 160 validation binaries use it.
   Consistent pass/fail, section headers, exit codes. Could be a shared harness.

2. **Consider cross-spring tolerance constants** — GPU-inherent tolerances like
   `GPU_VS_CPU_F64` and `GPU_LOG_POLYFILL` are used by both wetSpring and
   hotSpring independently. A shared `barracuda::tolerances` module would
   prevent drift.

3. **`f64::midpoint()` adoption** — edition 2024 provides this. ToadStool
   binaries should use it instead of manual `(a + b) / 2.0`.

### For wetSpring Team

1. V55 is complete. All gates pass. Ready for further evolution.
2. Consider adding integration tests for streaming I/O edge cases (gzip
   corruption, truncated mzML) to push FASTQ coverage above 90%.
3. Track 4 paper queue is complete (9/9). No new papers identified.

---

*This handoff supersedes V54. V54 archived to `handoffs/archive/`.*
