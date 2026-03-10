# SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring V93+ → barraCuda/toadStool Deep Debt Evolution Handoff

**Date:** 2026-03-04
**From:** wetSpring team (V93+)
**To:** barraCuda team, toadStool team
**barraCuda version:** v0.3.1 (standalone)
**wetSpring tests:** 1,054 lib + 175 forge + 27 doc-tests — 0 failures
**License:** AGPL-3.0-or-later

---

## Executive Summary

- Comprehensive deep debt audit and execution completed across the full wetSpring codebase.
- All 164 named tolerance constants centralized (was 106); zero inline tolerance literals remain.
- 6 library files refactored via test extraction (production code under 1000 LOC).
- All 284 validation binaries carry complete provenance documentation.
- Zero `unreachable!()` in library code; zero unsafe; zero clippy warnings (pedantic + nursery); zero doc warnings.
- All external dependencies confirmed pure Rust (no C evolution targets).
- Clone audit, `read_to_string` audit, and streaming I/O audit completed — all clean.
- CPU Jacobi eigendecomposition in `pcoa.rs` documented as absorption candidate.

---

## Part 1: What Changed (for barraCuda/toadStool awareness)

### Tolerance Centralization (106 → 164 constants)

~82 inline numeric literals (`1e-6`, `1e-10`, `0.0`, `1e-8`, etc.) migrated to
named `tolerances::` constants across 16 validation binaries. 4 new constants:

| Constant | Value | Justification |
|----------|-------|---------------|
| `LIMIT_CONVERGENCE` | 1e-8 | Bounded values approaching theoretical limits (norm_cdf, P_fix, eigenvalue checks) |
| `VARIANCE_EXACT` | 1e-20 | Variance of exactly constant data (jackknife delete-1 cancellation) |
| `NMF_SPARSITY_THRESHOLD` | 1e-8 | W/H matrix sparsity counting threshold |
| `NMF_CONVERGENCE_RANK_SEARCH` | 1e-5 | NMF convergence for rank-sensitivity sweeps |

**Pattern recommendation for barraCuda**: Organize tolerance constants by domain
rather than magnitude. wetSpring's hierarchy (`bio/`, `gpu/`, `spectral/`,
`instrument/`) scales well to 164 constants.

### Test Extraction Pattern

6 library files had `#[cfg(test)]` blocks extracted to separate files:

| Source | Tests File | Lines Saved |
|--------|-----------|:-----------:|
| `bench/power.rs` | `bench/power_tests.rs` | ~200 |
| `bench/hardware.rs` | `bench/hardware_tests.rs` | ~250 |
| `bio/gbm.rs` | `bio/gbm_tests.rs` | ~150 |
| `bio/merge_pairs.rs` | `bio/merge_pairs_tests.rs` | ~180 |
| `bio/felsenstein.rs` | `bio/felsenstein_tests.rs` | ~200 |
| `metalForge/forge/ncbi.rs` | `metalForge/forge/ncbi_tests.rs` | ~120 |

Pattern: `#[cfg(test)] #[path = "module_tests.rs"] mod tests;` in the source
file. Use `#![allow(...)]` (inner attribute) in the test file, not `#[allow(...)]`.

### kmer.rs: unreachable!() → Lookup Table

The 2-bit DNA decode match with `unreachable!()` catch-all was replaced with a
`const BASES: [u8; 4]` lookup table. Eliminates the panic path and is marginally
faster (direct index vs. branch table).

**Pattern recommendation**: For bit-masked enumerations where the mask guarantees
the range, prefer `const` lookup tables over match-with-unreachable.

### pcoa.rs: CPU Jacobi Absorption Path Documented

`bio::pcoa::jacobi_eigen()` is a local CPU eigendecomposition. barraCuda provides
GPU `BatchedEighGpu` (consumed by `pcoa_gpu`) but no CPU Jacobi in its public API.
If barraCuda upstreams a CPU Jacobi, wetSpring will absorb it (Write → Absorb → Lean).

**barraCuda action (P3):** Consider upstreaming a CPU symmetric eigendecomposition
for small matrices (N < 200), complementing the GPU `BatchedEighGpu`.

---

## Part 2: Audits Completed (No Action Needed)

### Clone Audit

All 60+ `clone()` calls in library code fall into justified categories:
- `device.clone()` / `ctx.clone()` — `Arc<T>` (cheap pointer increment)
- `..params.clone()` — struct update syntax creating intentional copies
- Ownership transfers from borrowed inputs (annotated with comments)

No zero-copy evolution opportunities identified.

### `read_to_string` Audit

All 17 library usages are on small files:
- `/proc/self/status`, `/proc/cpuinfo`, `/proc/meminfo` — kernel proc files
- `/sys/class/powercap/...` — sysfs single values
- TOML config files, SHA256 sidecars — tiny files
- `bio/ncbi_data/mod.rs` JSON loader — the only candidate for streaming, but
  the parsing uses `content.find()` which requires the full string. Would need
  a streaming JSON parser to evolve (currently `serde_json` is optional).

### External Dependencies

All deps confirmed pure Rust or Rust-first:

| Dep | Status |
|-----|--------|
| `barracuda` | Ecosystem (path dep, pure Rust) |
| `wgpu` | Pure Rust (renderdoc-sys transitive, optional debug) |
| `tokio` | Pure Rust, optional (GPU feature only) |
| `flate2` | Pure Rust (`rust_backend` feature, miniz_oxide) |
| `chacha20poly1305`, `ed25519-dalek`, `blake3` | Pure Rust (RustCrypto) |
| `serde_json` | Pure Rust, optional (JSON feature only) |
| `bytemuck` | Pure Rust |

No C/C++ dependencies to evolve.

---

## Part 3: Current Primitive Consumption (144 barraCuda primitives)

Unchanged from V93 rewire handoff. Key domains:

| Domain | Count | Key Primitives |
|--------|:-----:|----------------|
| Bio diversity | 12 | FusedMapReduceF64, BrayCurtisF64, DiversityFusionGpu |
| Bio ODE | 10 | BatchedOdeRK4 (5 systems × generate_shader + integrate_cpu) |
| Linalg | 8 | GemmF64, GemmCachedF64, BatchedEighGpu, graph_laplacian |
| Spectral | 6 | anderson_eigenvalues, lanczos, lanczos_eigenvalues, level_spacing |
| Stats | 8 | shannon, simpson, hill, pearson, bootstrap_ci, fit_* |
| Sample | 4 | boltzmann_sampling, metropolis, latin_hypercube, sobol |
| Bio GPU | 40+ | ANI, SNP, dN/dS, HMM, kmer, UniFrac, Felsenstein, etc. |

### Evolution Requests (unchanged)

| Request | Priority | Notes |
|---------|----------|-------|
| `ComputeDispatch` tarpc adoption | P3 | Enable transparent hardware routing |
| DF64 GEMM public API | P3 | `wgsl_shader_for_device()` is private |
| `BandwidthTier` in device profile | P3 | metalForge substrate routing |
| CPU Jacobi eigendecomposition | P3 | Small-matrix complement to GPU `BatchedEighGpu` |
| `domain-genomics` extraction | P1 | wetSpring's 47 bio algorithms as barraCuda domain |

---

## Part 4: Quality State

| Metric | Value |
|--------|-------|
| Library unit tests | 1,054 |
| metalForge forge tests | 175 |
| Doc-tests | 27 |
| Named tolerances | 164 |
| Inline tolerance literals | 0 |
| Validation/benchmark binaries | 284 |
| Provenance-documented binaries | 284 (100%) |
| Experiments | 280 |
| Validation checks | 8,241+ |
| `cargo clippy --pedantic --nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `cargo fmt --check` | 0 diffs |
| `unreachable!()` in library | 0 |
| `unsafe` in library | 0 |
| `TODO`/`FIXME` markers | 0 |
| Files over 1000 LOC | 0 (library code) |
| External C dependencies | 0 |
| Library code coverage | 95.86% line / 93.54% fn / 94.99% branch |

---

## Part 5: What wetSpring Learned (Relevant to barraCuda Evolution)

### 1. Tolerance Architecture Scales

Starting at 79 constants (V42), growing to 164 (V93+), the domain-organized
hierarchy (`bio/alignment.rs`, `bio/diversity.rs`, `gpu.rs`, `spectral.rs`)
scales without naming conflicts. The `all_tolerances_are_non_negative` meta-test
catches regressions.

### 2. Test Extraction Keeps Production Code Focused

For files where tests exceed 40% of content, extracting to `*_tests.rs` with
`#[cfg(test)] #[path = "..."] mod tests;` reduces cognitive load without
splitting domain logic. Critical: inner attributes (`#![allow(...)]`) in the
test file, not outer.

### 3. Hardcoding Evolution is Straightforward

Environment variable cascades with sensible defaults:
```
PRIMAL_SPECIFIC_VAR → XDG_DATA_HOME/primal → ~/.local/share/primal → compile-time default
```
Converts every hardcoded path into a capability-based discovery. wetSpring evolved
NCBI URLs, data directories, and temp paths. No performance impact.

### 4. Streaming I/O Boundaries Are Clear

FASTQ, mzML, MS2 all stream (iterator-based, no full-file buffering). Small
config files (TOML, sysfs, proc) are fine with `read_to_string`. JSON validation
data is the gray zone — would need `serde_json` streaming to evolve, but
validation datasets are bounded. Not worth the dependency for validation code.

### 5. Clone Discipline

Annotating necessary clones with ownership-transfer comments (`// ownership
transfer: borrowed input requires clone`) prevents false-positive cleanup
attempts. All GPU device clones are `Arc<T>` (cheap).
