# wetSpring → ToadStool Handoff V42: Deep Debt Evolution + BarraCuda Absorption Handoff

**Date:** February 25, 2026
**From:** wetSpring (Phase 47, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team + all springs
**Supersedes:** V41 (deep audit + coverage improvements)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring completed a second round of deep debt evolution targeting testability,
tolerance completeness, and architectural clarity. This handoff documents what
we learned about barracuda's API surface, what patterns work well, what should
be evolved, and a concrete absorption plan for the ToadStool/BarraCuda team.

**Post-V42 numbers:** 898 tests (819 lib + 47 forge + 32 integration/doc),
**96.78%** library coverage, **49** ToadStool primitives + 2 BGL helpers consumed,
**77** named tolerance constants, zero bare tolerance literals, zero clippy warnings.

**Key finding:** barracuda's CPU math and GPU primitive API surfaces are mature
and well-designed. wetSpring's remaining evolution opportunities are mostly at
the integration boundary (PhyloTree unification, metalForge absorption, NPU
pipeline), not in the core math.

---

## Part 1: What Changed in V42

### 1.1 Testability Refactoring (ncbi.rs)

wetSpring's `ncbi.rs` module provides capability-based HTTP transport discovery,
NCBI API key resolution, and cache file path discovery. All three functions
depended on `std::env::var()`, making branches untestable without `unsafe`
env-var mutation (which Rust edition 2024 correctly flags).

**Solution:** Extracted pure-logic functions from env-dependent wrappers:

| Public wrapper | Extracted pure-logic function | What it tests |
|---------------|------------------------------|---------------|
| `api_key()` | `api_key_from_paths(data_root, home)` | TOML file discovery cascade |
| `discover_http_backend()` | `select_backend(custom, has_curl, has_wget)` | Backend priority chain |
| `cache_file()` | `resolve_cache_path(data_root, manifest, filename)` | Data dir resolution |

16 new tests cover all branches. Coverage: **86.39% → 93.38%**.

**Pattern for other springs:** When env-dependent code has low coverage, extract
the decision logic into a pure function that accepts parameters, test the pure
function exhaustively, and let the env-reading wrapper remain a thin untestable shim.

### 1.2 Tolerance Completeness

| Constant | Value | Provenance |
|----------|-------|-----------|
| `GPU_VS_CPU_HMM_BATCH` | 1e-3 | 256 seqs × 100 steps × 3 states; warp-level vs sequential reduction |
| `ODE_BISTABLE_LOW_B` | 0.005 | Fernandez 2020 bistable, `B_ss` ≈ 0.040, RK4 vs LSODA |
| `ODE_SIGNAL_SS` | 0.02 | c-di-GMP / autoinducer steady-state tolerance |
| `HMM_INVARIANT_SLACK` | 1e-10 | Viterbi ≤ Forward invariant check |
| `PHAGE_LARGE_POPULATION` | 1000.0 | Large phage population counts |
| `PHAGE_CRASH_FLOOR` | 1.0 | Near-zero crashed phage population |
| `NPU_PASS_RATE_CEILING` | 0.30 | NPU triage acceptance: max 30% pass rate |
| `NPU_RECALL_FLOOR` | 0.90 | NPU triage: min 90% recall |
| `NPU_TOP1_FLOOR` | 0.80 | NPU triage: min 80% top-1 match |
| `GEMM_COMPILE_TIMEOUT_MS` | 30000.0 | GPU GEMM shader compilation timeout |

**Semantic fixes:** `GC_CONTENT` was misused for ODE bistable tolerance;
`KMD_SPREAD` was misused for ODE signal tolerance. Both replaced with
purpose-built constants.

**14 bare `0.0` tolerance params** replaced with `tolerances::EXACT` across
3 validation binaries (`validate_metalforge_v5`, `validate_cpu_vs_gpu_all_domains`,
`benchmark_phylo_hmm_gpu`).

### 1.3 Smart Refactoring (ncbi_data.rs)

Monolithic 724-line `bio/ncbi_data.rs` split into domain-focused submodule:

```
bio/ncbi_data/
├── mod.rs      ← shared JSON helpers, data_dir, re-exports
├── vibrio.rs   ← VibrioAssembly struct + load + synthetic fallback
├── campy.rs    ← CampyAssembly struct + load + synthetic fallback
└── biome.rs    ← BiomeProject struct + load + diversity params
```

Shared helpers (`json_str_value`, `json_int_value`, `split_json_objects`) live
in `mod.rs`. Each domain struct has its own `from_json_obj`, loader, and
synthetic fallback in its file.

### 1.4 Hardcoding Evolution

| File | Before | After |
|------|--------|-------|
| `validation_helpers.rs` | `"silva_138_99_seqs.fasta"` | `SILVA_FASTA` constant |
| `validation_helpers.rs` | `"silva_138_99_taxonomy.tsv"` | `SILVA_TAX_TSV` constant |
| `benchmark_phylo_hmm_gpu.rs` | `1e-3` | `tolerances::GPU_VS_CPU_HMM_BATCH` |
| 6 ODE/triage binaries | Mismatched tolerance constants | Purpose-built constants |

---

## Part 2: Complete BarraCuda Dependency Surface

### 2.1 What We Consume (49 GPU + CPU math + infrastructure)

**CPU Math (always-on, `default-features = false`):**

| Domain | Primitive | wetSpring Files | Quality |
|--------|-----------|----------------|---------|
| Special | `erf`, `ln_gamma`, `regularized_gamma_p` | `special.rs` | Excellent — thin delegation, tested to 1e-10 |
| Numerical | `trapz` | `bio/eic.rs` | Excellent — trapezoidal integration |
| Numerical | `numerical_hessian` | Cross-spring bins | Good — ML curvature analysis |
| Numerical | `ridge_regression` | `bio/esn.rs` | Excellent — proper Cholesky solve |
| Numerical | `BatchedOdeRK4<S>` | 5 ODE GPU modules | Excellent — `generate_shader()` is brilliant |
| Numerical | 5 ODE system structs | ODE GPU modules | Excellent — trait-based, zero boilerplate |
| LinAlg | `nmf` (NMF factorization) | Drug repurposing | Good — `NmfConfig`, `top_k_predictions` |
| LinAlg | `sparse::CsrMatrix` | GPU drug bins | Good |
| LinAlg | Graph/spectral ops | Cross-spring bins | Excellent — `graph_laplacian`, `effective_rank`, etc. |
| Spectral | Anderson/Lanczos/level stats | 15+ cross-spring bins | Excellent — comprehensive spectral theory |
| Sampling | `boltzmann_sampling` | Cross-spring bins | Good — MCMC for ODE params |

**GPU Ops (49 primitives):**

| Category | Count | Primitives | Quality |
|----------|:-----:|-----------|---------|
| Math | 12 | FMR, BrayCurtis, GEMM, BatchedEigh, PeakDetect, SparseGEMM, TransE, Kriging, WeightedDot, Var, Corr, Cov | Excellent |
| Bio | 15 | UniFrac, RF, HMM, DADA2, ANI, dN/dS, SNP, QF, Pangenome, K-mer, Jaccard, Hamming, SpatialPayoff, LocusVar, BatchFitness | Excellent |
| Phylo | 6 | Felsenstein, PhyloTree, TreeInference, FlatForest, Gillespie, SW | Excellent |
| ODE | 5+1 | BatchedOdeRK4 × 5 systems + generate_shader() | Excellent |
| Infra | 8+2 | WgpuDevice, TensorContext, GpuDriverProfile, WgslOpClass, Rarefaction, TaxonomyFC, GemmCached, NmfConfig + 2 BGL helpers | Good |

### 2.2 What Works Exceptionally Well

1. **`BatchedOdeRK4<S>::generate_shader()`** — The ODE trait-based WGSL
   generation pattern is the best API decision in barracuda. wetSpring deleted
   30,424 bytes of local WGSL when this landed. All 5 biological ODE systems
   "just work" by implementing the `OdeSystem` trait. Other springs should use
   this pattern for any new ODE domains.

2. **`FusedMapReduceF64`** — The workhorse GPU primitive. wetSpring uses FMR
   in 12+ GPU modules (diversity, spectral, kmd, merge_pairs, molecular_clock,
   neighbor_joining, eic, rarefaction, spectral_match, etc.). Its flexibility
   (custom map function + parallel reduction) covers an enormous range of
   computational biology workloads.

3. **CPU math delegation** — `barracuda::special::{erf, ln_gamma, regularized_gamma_p}`
   and `barracuda::numerical::{trapz, ridge_regression}` work perfectly as
   CPU-only math. The `default-features = false` pattern is exactly right —
   springs get CPU math without pulling in wgpu.

4. **Bio op API consistency** — All 15 bio GPU ops follow the same pattern:
   `new(device)` → `call(inputs)` → `Result<Output>`. This makes GPU module
   wrappers in wetSpring trivially thin.

5. **`storage_bgl_entry` / `uniform_bgl_entry` helpers** — Eliminated ~258
   lines of BGL boilerplate across 6 wetSpring files when these landed in S62.

### 2.3 What Could Be Better

| Issue | Impact | Suggestion |
|-------|--------|-----------|
| `PhyloTree` split | Medium — wetSpring has `bio::unifrac::PhyloTree` (CPU) and barracuda has `PhyloTree` (GPU). Validation binaries convert between them. | Unify into a single `barracuda::PhyloTree` that both CPU and GPU code use. |
| `Fp64Strategy` docs | Low — consumer GPU users need guidance on Hybrid vs Native DF64 selection. | A doc page on `Fp64Strategy` auto-detection would help all springs. |
| `diversity_fusion` absorption | Medium — sole remaining local WGSL shader in wetSpring. | Absorb as `barracuda::ops::bio::diversity_fusion_f64`. |
| `dot`/`l2_norm` CPU export | Low — thin local helpers that duplicate trivial math. | Export from `barracuda::special` for consistency. 1-line functions, low effort. |
| `wgsl_shader_for_device()` is private | Low — blocks DF64 GEMM adoption in `gemm_cached.rs`. | Make pub or provide alternative API for springs that cache compiled shaders. |

### 2.4 What We Do NOT Use (and why)

| Capability | Reason |
|------------|--------|
| barracuda `Tensor` API | wetSpring's GPU modules wire primitives directly; Tensor abstraction is for higher-level spring patterns |
| barracuda `log_f64` polyfill | Consumed indirectly through `needs_f64_exp_log_workaround()` capability check |
| barracuda NVVM backend | Not needed — Vulkan/Metal path via wgpu sufficient for validation |

---

## Part 3: Absorption Recommendations for ToadStool

### P0 — Immediate (blocks wetSpring lean completion)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | **`diversity_fusion_f64.wgsl` absorption** | Medium | Eliminates sole remaining local WGSL shader. CPU reference: `bio/diversity_fusion_gpu.rs`. Binding layout: 3 buffers (counts, params, output). Dispatch: N/64 workgroups. Validation: Exp167 18/18 PASS. |

### P1 — High Value (improves spring developer experience)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | **`PhyloTree` unification** | Medium | Single `barracuda::PhyloTree` for CPU and GPU eliminates conversion boilerplate in validation binaries. |
| 2 | **`barracuda::special::{dot, l2_norm}`** | Trivial | 1-line CPU helpers — eliminates last local math in springs. |
| 3 | **`Fp64Strategy` documentation page** | Low | Helps all springs understand DF64 auto-detection on consumer GPUs. |
| 4 | **`wgsl_shader_for_device()` pub access** | Low | Enables cached shader compilation in springs (wetSpring `gemm_cached.rs`). |

### P2 — Ecosystem Value (benefits all springs)

| # | Item | Detail |
|---|------|--------|
| 1 | **ODE trait pattern documentation** | `BatchedOdeRK4<S>::generate_shader()` is the best pattern in barracuda. A "How to add an ODE system" guide would help airSpring (Richards PDE), neuralSpring (population dynamics), and future springs. |
| 2 | **metalForge absorption path** | wetSpring's `metalForge/forge/` crate (substrate discovery, dispatch routing, bridge to barracuda) is ready for upstream absorption. `bridge.rs` is the integration point. 47 tests, `forbid(unsafe_code)`. |
| 3 | **NPU primitive exploration** | wetSpring has a working ESN → int8 → NPU pipeline for reservoir computing. If barracuda adds NPU dispatch, wetSpring's ESN module is a ready test case. |
| 4 | **Per-domain GPU polyfill tracking** | As DF64 transcendentals improve, track which domains still need `needs_f64_exp_log_workaround()`. |

### P3 — Cross-Spring Insights

| # | Observation | Value to ToadStool |
|---|------------|-------------------|
| 1 | **Spectral theory is cross-domain** | wetSpring uses `barracuda::spectral::*` for Anderson localization in microbial ecology. hotSpring uses it for condensed matter. Same math, different physics. Consider promoting spectral to a first-class barracuda module family. |
| 2 | **FMR is the universal GPU primitive** | 12+ wetSpring modules, many hotSpring modules use FMR. Its flexibility (custom map + reduce) makes it the most reusable primitive. Performance tuning FMR benefits everyone. |
| 3 | **Tolerance system architecture** | wetSpring's 77-constant `tolerances.rs` with hierarchy testing (`all_tolerances_are_non_negative`) is a pattern other springs should adopt. Consider making `Validator` + tolerance-constant pattern a barracuda contribution. |
| 4 | **Synthetic fallback pattern** | wetSpring's `bio/ncbi_data/` modules use `gen_synthetic_*()` fallbacks when real NCBI data isn't available. This enables deterministic CI without network access. Pattern is reusable. |

---

## Part 4: What wetSpring Learned for BarraCuda Evolution

### 4.1 The Write → Absorb → Lean Cycle Works

wetSpring wrote 8 bio WGSL shaders + 5 ODE WGSL shaders. All 8 bio shaders
were absorbed by ToadStool (sessions 31d/31g). All 5 ODE shaders were replaced
by the superior `BatchedOdeRK4<S>::generate_shader()` pattern. The cycle
works exactly as designed: springs write validated extensions, ToadStool
absorbs, springs lean on upstream.

**Total local code deleted after absorption:** 55,424 bytes (25 KB bio + 30 KB ODE).

### 4.2 Environment-Dependent Code Needs Seams

The testability refactoring in `ncbi.rs` (V42) revealed a general pattern:
code that reads `std::env::var()` or probes the filesystem is inherently
untestable in Rust 2024 (where `set_var` is `unsafe`). The solution is always
the same: extract a pure function, pass the env values as parameters.

This pattern should be documented in a barracuda contributor guide.

### 4.3 Tolerance Constants Should Be Hierarchical and Tested

wetSpring evolved from ad-hoc `0.0`, `1e-6`, `0.001` scattered across 158
binaries to 77 named constants in a single `tolerances.rs` module with:
- Scientific justification per constant (doc comments with experiment/commit refs)
- Hierarchy testing (`all_tolerances_are_non_negative`)
- Semantic naming (`GPU_VS_CPU_HMM_BATCH`, not `TOLERANCE_3`)

**Lesson:** Bare numeric literals in tolerance positions are tech debt. Named
constants are searchable, documentable, and auditable.

### 4.4 `default-features = false` Is the Right Default

wetSpring's `barracuda = { ..., default-features = false }` pattern gives CPU
math without pulling in wgpu/tokio/pollster. This is essential for:
- Fast compilation (CPU-only builds)
- CI environments without GPU
- Deterministic testing (no GPU hardware variance)
- WASM/no_std future (if barracuda evolves there)

---

## Part 5: Quality Report

### 5.1 Test Coverage

| Family | Coverage | Tests | Delta from V41 |
|--------|----------|-------|----------------|
| bio/* (47 modules) | ~97% | ~560 | +10 (ncbi_data submodule) |
| io/* (FASTQ, mzML, MS2, XML) | 87-97% | ~80 | — |
| special, tolerances, validation | 99%+ | ~40 | +1 (GPU_VS_CPU_HMM_BATCH) |
| bench/power, bench/hardware | 71-81% | ~25 | — |
| ncbi | **93%** | ~46 | **+16** (from 86%) |
| encoding, error, phred | 99%+ | ~20 | — |
| **Total** | **96.78%** | **819** | **+27 tests, +0.30% coverage** |

### 5.2 Quality Gates

| Gate | Status |
|------|--------|
| `cargo fmt --check` | 0 diffs |
| `cargo clippy --all-targets -W pedantic -W nursery` | 0 warnings |
| `cargo test` | 898 passed, 0 failed |
| `cargo doc --no-deps` | 0 warnings |
| `cargo llvm-cov --lib` | 96.78% |
| `#![deny(unsafe_code)]` | Enforced crate-wide |
| `#![deny(missing_docs)]` | Enforced crate-wide |
| Inline tolerance literals | 0 remaining |
| Named tolerance constants | 77 |
| External C dependencies | 0 |
| Max file size | All under 1000 LOC |

---

## Part 6: Acceptance Criteria

- [x] All 898 tests pass
- [x] Coverage ≥ 96% overall
- [x] ncbi.rs coverage ≥ 90%
- [x] Zero bare tolerance literals in validation binaries
- [x] All tolerance constants scientifically justified with provenance
- [x] Zero clippy warnings (pedantic + nursery)
- [x] Zero TODO/FIXME markers
- [x] All 77 tolerance constants in `all_tolerances_are_non_negative` test
- [x] V42 handoff submitted to wateringHole/handoffs/
- [x] Root docs, whitePaper, experiments, CHANGELOG synchronized
