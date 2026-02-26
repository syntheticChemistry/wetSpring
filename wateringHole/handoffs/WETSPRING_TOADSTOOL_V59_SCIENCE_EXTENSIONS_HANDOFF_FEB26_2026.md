# wetSpring → ToadStool/BarraCUDA Handoff: V59 Science Extensions + Deep Debt Resolution

**Date:** February 26, 2026
**From:** wetSpring V59
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`) — 700 shaders, 0 f32-only, universal precision
**wetSpring:** 1,008 tests (882 lib + 60 integration + 19 doc + 47 forge), 197 experiments, 184 binaries, 4,688+ validation checks (1,578 GPU on RTX 4070), 86 named tolerances, clippy pedantic+nursery CLEAN, typed NCBI errors, 0 local WGSL

---

## Executive Summary

V59 extends wetSpring in two directions: (1) five new science extension
experiments (Exp184-188) that exercise the sovereign NCBI pipeline, dynamic
Anderson physics, DF64 readiness, and NPU sentinel deployment; and (2) deep
debt resolution that evolved NCBI error handling to typed errors, eliminated
all inline tolerance literals, and achieved zero clippy pedantic+nursery
warnings across the entire workspace including fuzz targets. This handoff
identifies concrete absorption candidates and DF64 readiness data for
ToadStool's evolution.

---

## Part 1: Science Extensions (Exp184-188)

### New Validation Binaries

| Exp | Binary | Checks | Domain | BarraCuda Usage |
|-----|--------|:------:|--------|-----------------|
| 184 | `validate_real_ncbi_pipeline` | 25 | NCBI → diversity → Anderson | `ncbi::*`, `bio::diversity::*`, `barracuda::spectral::*` (GPU) |
| 185 | `validate_cold_seep_pipeline` | 8 | 50 cold seep communities | `bio::diversity::*`, Bray-Curtis condensed, Anderson 3D (GPU) |
| 186 | `validate_dynamic_anderson` | 7 | Time-varying disorder W(t) | `barracuda::spectral::{anderson_3d, lanczos, level_spacing_ratio}` (GPU) |
| 187 | `validate_df64_anderson` | 4 | L=6-14 f64, DF64 Phase 2 ready | `barracuda::spectral::{anderson_3d, lanczos, find_w_c}` (GPU) |
| 188 | `validate_npu_sentinel_stream` | 10 | Int8 sentinel pipeline | `bio::diversity::*`, CPU int8 inference (NPU sim) |

### What This Teaches ToadStool

1. **Dynamic Anderson (Exp186)** uses `anderson_3d` at many time points with
   varying W — embarrassingly parallel across (t, realization). A batched
   `anderson_3d_batch(L, W_array, seed_array)` primitive could eliminate
   per-invocation overhead. Three W(t) functions tested: exponential decay
   (tillage→no-till), spike+recovery (antibiotic), sinusoidal (seasonal).

2. **DF64 readiness (Exp187)** proves that L=14 (N=2,744 matrix) works with
   standard f64 Lanczos. At L=24 (N=13,824), f64 precision becomes marginal
   for eigenvalue separation near the band center. A DF64 Lanczos kernel
   would unlock L=24+ with ~30 digits of precision on FP32 cores.

3. **NPU sentinel (Exp188)** validates the int8 quantization + threshold
   classifier pipeline at 64M inferences/sec on CPU. The same pipeline would
   run on AKD1000 spiking neural network at ~0.5ms/inference. The key
   features (Shannon, Simpson, dominant fraction) are all computed by
   BarraCuda diversity primitives upstream.

---

## Part 2: Deep Debt Resolution — What Changed

### NCBI Typed Errors

Migrated 6 NCBI modules from `Result<T, String>` to `crate::error::Result<T>`
with a new `Error::Ncbi(String)` variant:

| Module | Functions migrated | Pattern |
|--------|-------------------|---------|
| `ncbi/http.rs` | `get`, `interpret_output` | `Error::Ncbi(format!(...))` |
| `ncbi/entrez.rs` | `esearch_count`, `parse_count` | `Error::Ncbi(...)` |
| `ncbi/efetch.rs` | `efetch_fasta`, `efetch_genbank`, `efetch_fasta_batch`, `validate_fasta` | `Error::Ncbi(...)` |
| `ncbi/cache.rs` | `accession_dir`, `write_with_integrity`, `verify_integrity` | `Error::Ncbi(...)` |
| `ncbi/sra.rs` | `download_sra_run`, `validate_accession`, `run_fasterq_dump`, `run_fastq_dump` | `ok_or_else(\|\| Error::Ncbi(...))` |
| `ncbi/nestgate.rs` | `store`, `retrieve`, `exists`, `health`, `fetch_or_fallback`, `rpc_call` | `Error::Ncbi(...)` |

**Pattern for ToadStool**: The `Error` enum with domain-specific variants
(`Io`, `Parse`, `Gpu`, `Ncbi`) is a clean pattern for typed error handling
in Rust. If ToadStool evolves its error types, this pattern scales well.

### Tolerance Hygiene

Added 4 new named constants to `tolerances.rs`:

| Constant | Value | Provenance |
|----------|-------|------------|
| `PPM_FACTOR` | `1e-6` | Parts-per-million conversion (mass spectrometry) |
| `ERF_PARITY` | `1e-14` | Error function CPU ↔ barracuda::special parity |
| `NORM_CDF_PARITY` | `1e-14` | Normal CDF CPU ↔ barracuda::special parity |
| `NORM_CDF_TAIL` | `1e-10` | Normal CDF tail accuracy (|x| > 3) |

Replaced all inline tolerance literals in validation binaries with named
constants from `tolerances::*`. Total: **86 named tolerances, 0 ad-hoc**.

### Clippy Evolution

- Fixed `redundant_closure` in fuzz FASTQ target
- Fixed `match_same_arms` in fuzz XML target
- Migrated `validate_neighbor_joining` from custom `check!` macro to `Validator` harness
- Rewired `feature_table_gpu` to compose `signal_gpu::find_peaks_gpu` (uses `PeakDetectF64`)
- All 184 binaries pass clippy pedantic + nursery with 0 warnings

---

## Part 3: BarraCuda Primitive Usage Audit

### Current Consumption (79 primitives, S68)

wetSpring uses BarraCuda primitives in three tiers:

**Tier 1 — Core math (always-on, no feature gate)**
- `barracuda::special::{erf, ln_gamma, regularized_gamma_lower}` → `crate::special` delegates
- `barracuda::stats::{dot, l2_norm, mean, percentile, shannon_from_frequencies, hill, monod, fit_linear}`
- `barracuda::linalg::{nmf, ridge, cosine_similarity}`
- `barracuda::tolerances::*`

**Tier 2 — GPU compute (feature = "gpu")**
- `barracuda::ops::bio::*` (ANI, SNP, dN/dS, pangenome, HMM, QF, DADA2, RF, chimera, derep, merge, NJ, reconciliation, clock, signal, feature_table, cooperation, capacitor)
- `barracuda::spectral::{anderson_3d, anderson_2d, anderson_hamiltonian, lanczos, lanczos_eigenvalues, level_spacing_ratio, find_w_c, GOE_R, POISSON_R}`
- `barracuda::ops::{FusedMapReduceF64, GemmF64, SparseGemmF64, TranseScoreF64, PeakDetectF64, TopK}`
- `barracuda::gpu::{WgpuDevice, GpuF64, compile_shader_universal, Precision}`

**Tier 3 — Cross-spring absorbed (via ToadStool)**
- `diversity_fusion_f64` (S63 — Write→Absorb→Lean complete)
- ODE systems via `BatchedOdeRK4` (S58 — 5 bio systems)

### Primitives NOT Yet Used (Available in S68)

| Primitive | Why Not Used | Recommendation |
|-----------|-------------|----------------|
| `ComputeDispatch` | Manual BGL still works | Migrate for cleaner dispatch (P3) |
| `Fp64Strategy::Hybrid` | Using `Precision::F64` directly | Adopt for RTX 4070 GEMM (P3) |
| `BandwidthTier` | metalForge has its own routing | Wire into forge dispatch (P3) |
| `DF64 Lanczos` | Not yet in ToadStool | **Requested** — enables Exp187 Phase 2 |

---

## Part 4: Absorption Candidates for ToadStool

### High Priority — Patterns

| Pattern | Location | Lines | Why |
|---------|----------|:-----:|-----|
| `Validator` harness | `src/validation.rs` | 250 | Structured f64 comparison, provenance tables, section headers, exit 0/1 |
| `tolerances.rs` | `src/tolerances.rs` | 620 | 86 named constants with paper/algorithm provenance |
| `Error` enum | `src/error.rs` | 165 | Domain-specific variants (Io, Parse, Gpu, Ncbi) with typed Display/source |

### Medium Priority — Primitives

| Primitive | What it does | Exp | Checks |
|-----------|-------------|-----|--------|
| `anderson_3d_batch` | Batch multiple (W, seed) Anderson lattices | 186 | Would eliminate per-call overhead for dynamic W(t) |
| DF64 Lanczos | Lanczos with double-float arithmetic | 187 | Enables L=24+ lattice (13,824 × 13,824) |
| `int8_quantize_batch` | Batch feature quantization for NPU | 188 | SIMD-friendly int8 conversion |

### Low Priority — Domain-Specific

These are validated but too application-specific for ToadStool absorption:

| Module | Domain | Reason to Keep Local |
|--------|--------|---------------------|
| `soil_qs_*` | Soil pore QS | Track 4 specific |
| `drug_repurposing_*` | NMF + pathway scoring | Track 3 specific |
| `ncbi/*` | NCBI API wrappers | wetSpring data acquisition |

---

## Part 5: Three-Tier Validation Status

### Paper Queue Controls (52/52 papers, 39/39 three-tier)

| Tier | What it proves | Experiments | Checks |
|------|---------------|:-----------:|:------:|
| **Python baseline** | Published paper math is reproducible | 52 scripts | 52/52 GREEN |
| **BarraCuda CPU** | Pure Rust math matches Python (no GPU needed) | Exp035,043,057,070,079,085,102,163 | 380/380 |
| **BarraCuda GPU** | GPU math matches CPU (portable) | Exp064,071,087,092,101,164 | 1,578 |
| **metalForge** | CPU = GPU = same answer (substrate-independent) | Exp060,065,080,084,086,088,093,103,104,165,182 | 243+ |
| **Pure GPU streaming** | Zero CPU round-trips, unidirectional dispatch | Exp072,073,075,089,090,091,105,106,181 | 252+ |

### Evolution Path Proven

```
Python baseline (interpreted, slow)
  → BarraCuda CPU (pure Rust, 22.5× faster)
    → BarraCuda GPU (portable math, 10-926× faster)
      → Pure GPU streaming (zero round-trips, 441-837× throughput)
        → metalForge (CPU/GPU/NPU substrate-independent)
```

### Controls by Track

| Track | Papers | CPU | GPU | metalForge | Status |
|-------|:------:|:---:|:---:|:----------:|--------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | Full three-tier |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | Full three-tier |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | Full three-tier |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 4 (Soil QS/Anderson) | 9 | 9/9 | 9/9 | 9/9 | Full three-tier |
| **Total three-tier** | **39** | **39/39** | **39/39** | **39/39** | **ALL** |
| Cross-spring (spectral) | 1 | 1/1 | 1/1 | — | CPU + GPU |
| Extensions (Phase 37-39) | 9 | 9/9 | — | — | CPU only (analytical) |
| Science extensions (V59) | 5 | 5/5 | — | — | CPU (GPU deferred) |

---

## Part 6: Benchmark Data for ToadStool

### GPU vs CPU Crossover (RTX 4070, updated V59)

| Domain | Crossover N | GPU Speedup at Large N | Source |
|--------|:----------:|:---------------------:|--------|
| Spectral cosine | ~100 | 926× at 2.1M pairs | Exp087 |
| Bray-Curtis | ~50 | 200×+ at 500 samples | Exp092 |
| Smith-Waterman | ~10 | 625× at 1000 pairs | Exp059 |
| ODE parameter sweep | ~100 | 30×+ at 1000 combos | Exp049 |
| Anderson 3D (L=8) | Always | Proportional to L³ | Exp127 |
| GEMM | Always | Proportional to N² | Exp066 |
| HMM forward | ~50 | 15× at 500 seqs | Exp047 |

### Streaming vs Round-Trip

Streaming dispatch gives **441–837×** throughput over per-dispatch round-trips
at batch sizes ≥ 1000 (Exp091). This validates ToadStool's unidirectional
streaming architecture.

---

## Part 7: Recommended Next Steps for ToadStool

### Immediate

1. **DF64 Lanczos kernel** — enables Exp187 Phase 2 (L=24+ Anderson lattice).
   wetSpring's `anderson_3d` + `lanczos` already runs at f64; DF64 variant
   would use `compile_shader_universal(source, Precision::Df64)`.

2. **`anderson_3d_batch`** — batch multiple (W, seed) lattice constructions.
   Exp186 dynamic Anderson calls `anderson_3d` at 20-365 time points ×
   4-8 realizations. A batched version would eliminate dispatch overhead.

### Short-Term

3. **Validate `BatchedOdeRK4` with `Precision::Df64`** — wetSpring's 5 bio
   ODE systems are already universal-precision-ready. A single DF64 test
   with `PhageDefenseOde` (4 vars, 11 params) would prove the path.

4. **`--no-default-features` CI target** — wetSpring is the primary consumer
   of `default-features = false`. Any new shader-dependent code in CPU-only
   modules breaks the build.

### Medium-Term

5. **`ComputeDispatch` migration guide** — wetSpring still uses manual BGL
   for GPU ops. A migration example from manual BGL to `ComputeDispatch`
   would help all Springs adopt.

6. **Cross-spring tolerance merge** — wetSpring has 86 domain-specific
   tolerance constants; ToadStool has 12 infrastructure constants. A shared
   tolerance vocabulary would improve cross-spring consistency.

---

## Part 8: Files Changed in V59

| File | Change |
|------|--------|
| `barracuda/src/error.rs` | Added `Error::Ncbi(String)` variant + Display + source |
| `barracuda/src/tolerances.rs` | +4 constants (PPM_FACTOR, ERF_PARITY, NORM_CDF_PARITY, NORM_CDF_TAIL) |
| `barracuda/src/ncbi/*.rs` | 6 modules migrated to typed errors |
| `barracuda/src/bio/tolerance_search.rs` | Replaced `1e-6` with `tolerances::PPM_FACTOR` |
| `barracuda/src/bio/eic.rs` | Replaced `1e-6` with `tolerances::PPM_FACTOR` |
| `barracuda/src/bio/validation_helpers.rs` | Named constants for SILVA parameters |
| `barracuda/src/bin/validate_real_ncbi_pipeline.rs` | **NEW** — Exp184 (25 checks) |
| `barracuda/src/bin/validate_cold_seep_pipeline.rs` | **NEW** — Exp185 (8 checks) |
| `barracuda/src/bin/validate_dynamic_anderson.rs` | **NEW** — Exp186 (7 checks) |
| `barracuda/src/bin/validate_df64_anderson.rs` | **NEW** — Exp187 (4 checks) |
| `barracuda/src/bin/validate_npu_sentinel_stream.rs` | **NEW** — Exp188 (10 checks) |
| `barracuda/src/bin/validate_barracuda_cpu_v10.rs` | **NEW** — Exp190 (75 checks) |
| `barracuda/src/bin/validate_gpu_v59_science.rs` | **NEW** — Exp191 (29 checks) |
| `barracuda/src/bin/validate_metalforge_v59_science.rs` | **NEW** — Exp192 (36 checks) |
| `barracuda/src/bin/validate_neighbor_joining.rs` | Migrated to Validator harness |
| `barracuda/src/bin/validate_soil_*.rs` (10 files) | Replaced hardcoded tolerances |
| `barracuda/src/bin/validate_notill_*.rs` (3 files) | Replaced hardcoded tolerances |
| `barracuda/src/bin/validate_barracuda_cpu_*.rs` (2 files) | Replaced hardcoded tolerances |
| `barracuda/src/bin/validate_local_wgsl_compile.rs` | Expanded provenance table |
| `barracuda/fuzz/fuzz_targets/*.rs` (2 files) | Clippy pedantic fixes |
| All root `.md` files | Phase 58 → 59 metrics |
| `wateringHole/handoffs/` | This handoff (V59) |

---

## Part 9: Pin History

| Version | ToadStool Pin | Session | Key Changes |
|---------|--------------|---------|-------------|
| **V59** | `f0feb226` | S68 | Science extensions (Exp184-188), typed NCBI errors, 86 tolerances, deep debt |
| V58 | `f0feb226` | S68 | Evolution learnings handoff, full doc sync |
| V57 | `f0feb226` | S68 | Universal precision catch-up, feature-gate fix |
| V56 | `045103a7` | S66 | Science pipeline, NCBI, NestGate, biomeOS |
| V55 | `045103a7` | S66 | Deep debt, idiomatic Rust |
| V54 | `045103a7` | S66 | Codebase audit, supply-chain |
| V53 | `045103a7` | S66 | Cross-spring evolution benchmarks |

---

## License

SPDX-License-Identifier: AGPL-3.0-or-later
