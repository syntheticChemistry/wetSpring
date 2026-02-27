# toadStool / barracuda — V61 Nanopore Field Genomics + Evolution Handoff

**Date:** February 27, 2026
**From:** wetSpring
**To:** toadStool / barracuda core team
**Covers:** V61 `io::nanopore` module + barracuda evolution audit + deep code quality audit + absorption roadmap + three-tier paper controls
**ToadStool Pin:** S68 (`f0feb226`)
**License:** AGPL-3.0-only

---

## Executive Summary

wetSpring V61 completes the first field genomics software path and closes a
comprehensive barracuda evolution audit:

1. **`io::nanopore` module operational** — POD5/NRS signal parsing, streaming
   iterator, synthetic read generation. 52 pre-hardware checks PASS (Exp196a-c).
   Awaiting MinION hardware for real sequencer integration.

2. **Barracuda evolution audit** — 79 ToadStool primitives consumed, 0 local
   WGSL, 0 local derivative/regression math. One documented passthrough
   (`reconciliation_gpu` — dispatches to CPU, needs `BatchReconcileGpu`).
   Three absorption candidates ready for upstream.

3. **Three-tier paper controls confirmed** — 39/39 actionable papers validated
   at CPU + GPU + metalForge levels. All use open data and open systems.
   Controls verified for BarraCuda CPU, BarraCuda GPU, and metalForge
   mixed-hardware substrates.

4. **Learnings for ToadStool evolution** — int8 quantization patterns,
   streaming I/O architecture, ESN reservoir computing, NPU inference bridge,
   and field deployment constraints that inform barracuda crate design.

5. **Deep code quality audit** — `partial_cmp` → `f64::total_cmp` migration
   (10 library sites), vestigial dead code removed, HMM/quality iterators
   modernized, baseline manifest regenerated (41/41 match, 0 drift), coverage
   measured at 95.46% line / 93.54% fn / 94.99% branch.

---

## Part 1: How wetSpring Uses BarraCUDA (V61 State)

### Key Metrics

| Metric | Value |
|--------|-------|
| ToadStool primitives consumed | **79** (barracuda always-on, zero fallback) |
| Local WGSL shaders | **0** (all absorbed by ToadStool S63) |
| Local derivative/regression math | **0** (all lean on upstream) |
| GPU bio modules | 42 (27 lean + 7 compose + 5 write→lean + 0 passthrough*) |
| CPU bio modules | 47 |
| I/O modules | 5 (fastq, mzml, ms2, xml, **nanopore** ← new V61) |
| Rust tests | 1,022 (896 lib + 60 integration + 19 doc + 47 forge) |
| Validation checks | 4,800+ (1,578 GPU on RTX 4070, 60 NPU on AKD1000) |
| Named tolerances | 92 (scientifically justified, hierarchy-tested) |
| Experiments | 203 (203 PASS) |
| Papers with three-tier | 39/39 |
| Coverage | 95.46% line / 93.54% fn / 94.99% branch (cargo-llvm-cov) |
| Clippy | pedantic + nursery CLEAN |

*`reconciliation_gpu` is documented as Compose but validates GPU device then
dispatches to CPU `reconcile_dtl()`. True GPU promotion needs `BatchReconcileGpu`.

### Upstream Dependencies (lean)

| barracuda Module | wetSpring Usage | Checks |
|------------------|-----------------|:------:|
| `ops::bio::*` | All 42 GPU bio modules | 1,578 GPU |
| `ops::linalg::*` | BatchedEighGpu, lu_solve, GEMM, NMF, Ridge | 200+ |
| `spectral::*` | Anderson 1D/2D/3D, Lanczos, level spacing | 3,400+ |
| `stats::*` | diversity, hill, mean, percentile, dot, l2_norm | 500+ |
| `numerical::*` | trapz, gradient_1d, ODE derivative, integrate_cpu | 300+ |
| `pipeline::*` | FMR, streaming, dispatch | 152 |
| `device::WgpuDevice` | GPU adapter, shader compilation | all GPU |
| `compile_shader_universal` | All 42 GPU modules use `Precision::F64` | all GPU |

### Local Implementation (wetSpring-specific)

| Module | Purpose | Lines | Absorption? |
|--------|---------|:-----:|:-----------:|
| `bio::esn` | Echo state network reservoir computing | ~400 | **Yes** |
| `bio::anderson_qs` | QS-disorder mapping | ~200 | No (domain) |
| `io::nanopore` | POD5/NRS signal parser (V61 new) | ~350 | **Partial** |
| `io::fastq` | Streaming FASTQ/gzip | ~300 | No (NestGate) |
| `io::mzml` | Streaming mzML/base64 | ~250 | No (NestGate) |
| `io::ms2` | Streaming MS2 | ~150 | No (NestGate) |
| `npu` | NPU inference bridge (DMA, quantization) | ~300 | **Partial** |
| `tolerances` | 92 named constants | ~200 | No (local) |
| `validation` | Validator harness (hotSpring pattern) | ~150 | **Yes** |
| `special` | erf, ln_gamma, normal_cdf | ~100 | No (local) |
| `ncbi/*` | EFetch, SRA, NestGate, cache | ~500 | No (NestGate) |

---

## Part 2: Absorption Candidates for ToadStool

### Ready Now (3 items)

#### 1. `bio::esn` — Echo State Network Reservoir Computing

**Why upstream:** ESN is used by 4+ Springs (wetSpring NPU, hotSpring precision,
neuralSpring ML, airSpring IoT). The reservoir computing pattern is generic:
build reservoir → train readout → infer. Only the readout task varies by domain.

**What to absorb:**
- `EsnConfig` — reservoir size, spectral radius, input scaling, leak rate
- `Esn::new()` — sparse reservoir generation with controlled spectral radius
- `Esn::train()` — Cholesky ridge regression readout (already uses `barracuda::linalg`)
- `Esn::infer()` — single-step reservoir update + readout multiplication
- `w_in()`, `w_res()`, `w_out()`, `w_out_mut()`, `config()` — raw weight access

**Binding layout (GPU candidate):**
- Group 0: Reservoir weights (NxN f64), input weights (NxD f64)
- Group 1: Readout weights (CxN f64), state vector (N f64)
- Group 2: Input vector (D f64), output vector (C f64)
- Workgroup: 64 (reservoir state update is matrix-vector multiply)

**Validation:** 896 lib tests, Exp114 (QS classifier 100%), Exp194 (3 classifiers
on real AKD1000), Exp196c (int8 quantization pipeline).

#### 2. NPU Inference Bridge (`npu_infer_i8`, `load_reservoir_weights`)

**Why upstream:** Any Spring deploying to AKD1000 needs these primitives.
The bridge is hardware-agnostic at the API level — `npu_infer_i8` takes int8
tensors and returns int8 results. AKD1000-specific DMA is behind the
`akida-driver` crate.

**What to absorb:**
- `npu_infer_i8` — single int8 inference via DMA round-trip
- `load_reservoir_weights` — f64 → f32 → NPU SRAM (with capacity check)
- `load_readout_weights` — online readout switching (weight mutation)
- `npu_batch_infer` — batch int8 inference with aggregate metrics
- `quantize_community_profile_int8` — f64 abundance → int8 affine quantization

**Validation:** Exp193-195 (60 checks on real AKD1000 hardware), Exp196c (13
int8 quantization checks).

#### 3. Validator Harness Pattern

**Why upstream:** hotSpring and wetSpring independently developed the same
validation pattern. ToadStool already has `ValidationHarness` upstream; the
pattern alignment should be documented and harmonized.

**What to absorb:** Pattern documentation only — local `Validator` struct stays
local, but conventions (tolerance naming, provenance headers, check format)
should be standardized.

### Not Ready Yet (deferred)

| Item | Blocker | When |
|------|---------|------|
| `reconciliation_gpu` full GPU | Needs `BatchReconcileGpu` primitive | ToadStool adds DTL shader |
| `io::nanopore` upstream | Pre-hardware — real POD5 validation pending | After MinION arrives |
| `bio::basecall` | Not yet implemented | After Exp197+ |

---

## Part 3: Barracuda Evolution — What We Learned

### 3.1 The Write → Absorb → Lean Cycle Works

wetSpring has completed 8 full absorption cycles (Phase 7, 20, 43, 50-52):

| Phase | What Was Absorbed | Result |
|-------|-------------------|--------|
| 7 | 4 bio GPU primitives (SW, Gillespie, Tree, Felsenstein) | 10/10 rewire |
| 20 | 8 bio WGSL shaders (HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF) | 25 KB deleted |
| 43 | DF64 BGL helpers (storage_bgl_entry, uniform_bgl_entry) | 258 lines removed |
| 50 | diversity_fusion_f64.wgsl → upstream | 0 local WGSL |
| 50 | bio::diversity (11 functions) → barracuda::stats::diversity | delegation |
| 50 | 5 ODE RHS → barracuda::numerical::ode_bio::*Ode::cpu_derivative | ~200 lines |
| 52 | hill, fit_heaps_law, compute_ci → upstream | 79 primitives |
| 57 | compile_shader_f64 → compile_shader_universal (S68) | universal precision |

**Key learning:** The cycle velocity increased over time. Early absorptions took
weeks; later ones completed in hours because ToadStool's absorption machinery
(shader templating, type-generic `generate_shader()`, `Precision` enum) was
well-established.

### 3.2 Int8 Quantization Patterns

From ESN + NPU work (Exp114-119, 193-195, 196c):

- **Affine quantization** (scale + zero-point) preserves >95% of community info
- **Regime classification** (3-5 classes) is robust to int8 — you lose magnitude
  precision but preserve ordering, which is what classifiers need
- **Online readout mutation** — swapping NPU weights at 136 gen/sec enables
  real-time model evolution on hardware
- **toadStool action:** Consider `barracuda::quantize` module for int8/int4
  quantization primitives (affine, symmetric, per-channel)

### 3.3 Streaming I/O Architecture

From `io::fastq`, `io::mzml`, `io::nanopore`:

- **Iterator-based streaming** eliminates whole-file buffering — critical for
  field deployment where RAM is constrained
- **Reusable buffer pattern** (`read_line()` + `String` reuse) eliminates
  per-line allocation across all I/O modules
- **Typed errors** (`Error::Ncbi`, `Error::Nanopore`) instead of `String` —
  every I/O failure classifiable without string parsing
- **toadStool action:** When NestGate absorbs I/O, the streaming pattern should
  be the canonical approach — no whole-file deserialization

### 3.4 Named Tolerance Discipline

92 named constants in `tolerances.rs`, zero ad-hoc magic numbers:

- **Hierarchy-tested** — unit tests verify `EXACT ≤ BITWISE ≤ GPU_*`
- **Scientifically justified** — each tolerance documents why that value
  (e.g., `GPU_LOG_POLYFILL = 1e-7` because chained `log()` in polyfill
  WGSL accumulates 7 digits of error on consumer Ampere)
- **toadStool action:** Recommend cross-spring tolerance naming standard in
  wateringHole — all Springs use the same names for the same concepts

### 3.5 Field Deployment Constraints

From Sub-thesis 06 architecture and NPU work:

- **Power budget:** 10 mW (NPU) + 5W (MinION) + variable (host CPU/GPU)
- **Coin-cell lifetime:** >11 years at 1 Hz inference (AKD1000)
- **Latency:** 53 µs per NPU classification, 37× headroom over MinION read rate
- **Memory:** Field units have 1-8 GB — streaming I/O + int8 models mandatory
- **Connectivity:** Optional — sovereign pipeline runs offline indefinitely
- **toadStool action:** `metalForge` power-budget-aware routing. Dispatch to
  NPU for classification (microwatts), GPU for basecalling (watts), CPU for
  fallback. Power budget as first-class constraint alongside latency and accuracy.

---

## Part 4: Three-Tier Paper Controls

### Confirmed: 39/39 Papers — Open Data, Open Systems

Every three-tier-eligible paper has been validated at all three levels:

| Track | Papers | CPU | GPU | metalForge | Data |
|-------|:------:|:---:|:---:|:----------:|------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | NCBI SRA, published ODE params |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | PhyNetPy, SATé, dendropy |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | NCBI SRA, MBL, Figshare, OSF |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | Zenodo, EPA, Michigan EGLE |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | repoDB, PMC, ROBOKOP |
| Track 4 (Soil QS/Anderson) | 9 | 9/9 | 9/9 | 9/9 | Published tables, models |
| **Total** | **39** | **39/39** | **39/39** | **39/39** | **All open** |

### Controls by Hardware Tier

#### BarraCuda CPU (39/39)

All 39 papers validated as pure Rust math — no Python, no C, no unsafe code.
Key validation experiments: Exp035, 043, 057, 070, 079, 085, 102, 163, 179, 190.
Combined: 407+ CPU parity checks.

#### BarraCuda GPU (39/39)

All 39 papers validated on RTX 4070 via ToadStool S68 `compile_shader_universal`.
Key experiments: Exp064, 071, 087, 092, 101, 164, 180, 191.
Combined: 1,578 GPU checks. Math parity with CPU within named tolerances.

#### metalForge Mixed Hardware (39/39)

All 39 papers validated substrate-independent — same answer on CPU, GPU, or
mixed dispatch. Key experiments: Exp060, 065, 080, 103, 104, 165, 182, 192.
Combined: 243+ metalForge checks. CPU↔GPU↔NPU parity proven.

### Data Provenance

No proprietary data. All sources: NCBI SRA, Zenodo, EPA, Michigan EGLE, PMC,
MBL darchive, MG-RAST, Figshare, OSF, MassBank, published model equations.
Full provenance audit in `specs/PAPER_REVIEW_QUEUE.md`.

---

## Part 5: Hardware Validation Matrix

| Substrate | Device | Checks | Key Finding |
|-----------|--------|:------:|-------------|
| CPU | i9-12900K | 1,476 | 22.5× vs Python, 380/380 CPU parity |
| GPU (Ampere) | RTX 4070 | 1,578 | f64 via polyfill, ~1:64 throughput |
| GPU (Volta HBM2) | Titan V | — | Native f64 1:2, NVK pipeline |
| NPU | AKD1000 | 60 | 18.8K infer/sec, 1.4 µJ/infer, PUF |
| metalForge | CPU+GPU+NPU | 243+ | Substrate-independent dispatch |
| Streaming | GPU pipeline | 204 | 441-837× vs round-trip |

---

## Part 6: Recommended Actions for ToadStool

| Priority | Action | Impact |
|:--------:|--------|--------|
| **1** | Absorb `bio::esn` as `barracuda::ml::esn` | 4+ Springs benefit |
| **2** | Absorb NPU inference bridge (`npu_infer_i8`, `load_reservoir_weights`) | Standard NPU dispatch |
| **3** | Add `BatchReconcileGpu` primitive (DTL reconciliation) | Clears last passthrough |
| **4** | Consider `barracuda::quantize` module (int8/int4 affine) | Field deployment standard |
| **5** | Power-budget-aware metalForge routing | Field genomics constraint |
| **6** | Cross-spring tolerance naming standard | Consistency across all Springs |

---

## Part 7: Deep Code Quality Audit (V61 continuation)

### Audit Scope

Comprehensive sweep of the entire barracuda library (26,922 coverage regions,
98 bio modules, 11 I/O files, 174 validation binaries).

### Findings and Fixes

| Category | Before | After | Impact |
|----------|--------|-------|--------|
| `partial_cmp().unwrap_or(Ordering::Equal)` | 10 library sites | `f64::total_cmp()` | Deterministic NaN ordering; no transitive-comparison risk |
| `f64::total_cmp` closures | 3 sites with `\|a, b\| a.total_cmp(b)` | `f64::total_cmp` method reference | Clippy pedantic clean |
| Vestigial dead code | 7 lines in `io::nanopore/mod.rs` | Removed | Transmute scaffold from unsafe approach |
| HMM backward init | `for i in 0..n` assignment loop | `slice.fill(0.0)` | Idiomatic |
| HMM Viterbi termination | Manual max loop | `fold()` | Expressive, same perf |
| Quality trim | `for i in 0..n` with parallel indexing | `.zip()` iterator | Bounds-check elision |
| `unwrap_or(&"").to_string()` | taxonomy/kmers.rs | `copied().unwrap_or_default().to_string()` | Idiomatic |
| Baseline manifest | 41/41 drift (stale hashes from pre-SPDX) | 41/41 match, 0 drift | Integrity verified |
| Coverage measurement | 96.67% (stale) | **95.46% line / 93.54% fn / 94.99% branch** | Authoritative |

### What Clippy Pedantic CLEAN Means

- `#![deny(unsafe_code)]` — zero unsafe blocks crate-wide
- `#![deny(clippy::expect_used, clippy::unwrap_used)]` — zero panics in library code
- `#![warn(clippy::pedantic, clippy::nursery)]` — zero warnings
- `f64::total_cmp` throughout — no more `partial_cmp` + `unwrap_or` patterns
- All validation binaries compile and pass (174 binaries, 4,800+ checks)

### Coverage by Module

| Module Group | Line Coverage |
|-------------|:------------:|
| `bio/*` | 96%+ |
| `io/*` | 87–99% |
| `ncbi/*` | 65–99% (network-dependent modules lower) |
| `error.rs` | 100% |
| `encoding.rs` | 100% |
| `special.rs` | 99% |
| `tolerances.rs` | 100% |
| `validation.rs` | 95% |

### toadStool Action: `total_cmp` Standard

Recommend all barracuda crate code use `f64::total_cmp` instead of
`partial_cmp().unwrap_or(Ordering::Equal)`. The `total_cmp` function:
- Is deterministic for NaN (NaN sorts after all reals)
- Avoids transitive-ordering violations in sorts
- Is a single function call (no closure needed for `sort_by`)
- Is available since Rust 1.62 (stable since 2022)

This should be a clippy lint or crate-level standard for ToadStool.

---

## Supersedes

- `WETSPRING_TOADSTOOL_V60_NPU_FIELD_GENOMICS_HANDOFF_FEB26_2026.md` (archived)

## References

- `barracuda/EVOLUTION_READINESS.md` — absorption map
- `barracuda/ABSORPTION_MANIFEST.md` — module-by-module tracking
- `specs/PAPER_REVIEW_QUEUE.md` — three-tier control audit
- `whitePaper/baseCamp/sub_thesis_06_field_genomics.md` — field architecture
- `experiments/196a-c` — nanopore pre-hardware validation
