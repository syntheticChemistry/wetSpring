# wetSpring → ToadStool/BarraCUDA Handoff: V56 Science Extension Pipeline

**Date:** February 26, 2026
**From:** wetSpring V56
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S66 (`045103a7`)
**wetSpring:** 961 tests, 882 barracuda lib, 96.67% llvm-cov, 82 named tolerances, 188 experiments

---

## Executive Summary

V56 extends wetSpring from validated synthetic/table-value science to real NCBI
data pipelines orchestrated through the primal ecosystem. New infrastructure:
NCBI EFetch/SRA download, NestGate JSON-RPC integration, SHA-256 integrity
caching, GPU Anderson finite-size scaling at L=14-20, and biomeOS science
pipeline orchestration. 49 new tests, 3 new named tolerances, 5 new experiment
protocols. All existing validation remains green.

---

## Part 1: What Changed in V56

### New NCBI Pipeline Modules

| Module | Purpose | Tests |
|--------|---------|:-----:|
| `ncbi/efetch.rs` | EFetch FASTA/GenBank download with response validation | 11 |
| `ncbi/sra.rs` | SRA run download via capability-discovered `fasterq-dump`/`fastq-dump` | 14 |
| `ncbi/cache.rs` (extended) | Accession-based directory trees, SHA-256 integrity sidecar files | 11 |
| `ncbi/nestgate.rs` | Optional NestGate data provider via JSON-RPC 2.0 over Unix sockets | 13 |

**Total new tests:** 49 (882 - 833)

### New Tolerances

| Constant | Value | Provenance |
|----------|:-----:|-----------|
| `GPU_LANCZOS_EIGENVALUE_ABS` | 0.03 | SpMV summation order on GPU, 50% margin over worst case |
| `FINITE_SIZE_SCALING_REL` | 0.08 | W_c variation across L=6-20, largest deviation at L=6 |
| `LEVEL_SPACING_STDERR_MAX` | 0.015 | 8-16 disorder realizations, well-sampled regime |

### New Validation Binary

- `validate_anderson_gpu_scaling` — Exp184b: GPU Anderson finite-size scaling L=14-20
  (16 realizations, 15 W-points, scaling collapse for ν). Requires `--features gpu`.

### biomeOS Science Graph

- `graphs/science_pipeline.toml` — NestGate → wetSpring → ToadStool pipeline
- `capability_registry.toml` — `science` domain with 5 capabilities

---

## Part 2: How wetSpring Uses BarraCUDA (V56 Audit)

### Upstream Primitives Consumed (79 total, unchanged from V55)

V56 adds no new ToadStool primitive consumption — all new code uses wetSpring's
sovereign `ncbi/` module and existing `barracuda::spectral` primitives.

**Key upstream dependencies for science pipeline:**

| Category | Primitives | V56 Usage |
|----------|-----------|-----------|
| Spectral | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `find_w_c` | GPU Anderson L=14-20 binary |
| GPU Compute | `FusedMapReduceF64`, `BrayCurtisF64`, `GemmF64` | Diversity pipeline |
| ODE | `BatchedOdeRK4F64` | QS model (planned Exp186) |
| Stats | `shannon`, `simpson`, `observed_features` | Diversity → Anderson mapping |

### New Sovereign Code (no upstream impact)

| Module | Lines | Purpose |
|--------|:-----:|---------|
| `ncbi/efetch.rs` | ~140 | NCBI EFetch HTTP wrapper |
| `ncbi/sra.rs` | ~170 | SRA Toolkit capability discovery |
| `ncbi/cache.rs` (new code) | ~200 | SHA-256 + accession dirs |
| `ncbi/nestgate.rs` | ~260 | JSON-RPC 2.0 client |

All new code is **sovereign** — zero new ToadStool dependencies. The NestGate
integration uses standard JSON-RPC over Unix sockets, not ToadStool APIs.

---

## Part 3: What ToadStool Should Evolve

### GPU Lanczos (Priority: High)

wetSpring's `validate_anderson_gpu_scaling` currently runs CPU Lanczos for
L=14-20 lattices. At L=20, the matrix is 8000×8000 — GPU SpMV + Lanczos
would provide 10-50× speedup.

**Request:** GPU Lanczos kernel wrapping existing `barracuda::spectral::lanczos`
with GPU SpMV for the matrix-vector products. The tridiagonal decomposition
itself stays on CPU (tiny matrix).

### DF64 Lanczos (Priority: Medium)

Exp187 proposes DF64 Anderson at L=24+ for refined W_c. hotSpring has validated
DF64 arithmetic. A DF64 variant of GPU Lanczos would enable ~30-digit precision
eigenvalue computation on FP32 cores.

### Tolerance Module Pattern (Priority: Low, unchanged from V55)

wetSpring's `tolerances.rs` pattern (82 named constants with provenance docs,
hierarchy test) has proven effective. ToadStool could adopt this pattern for
GPU parity testing.

---

## Part 4: Paper Validation Controls (unchanged from V55)

- 52/52 CPU papers validated
- 40/40 GPU papers validated
- 39/39 metalForge three-tier papers validated
- All papers use publicly accessible data (NCBI SRA, Zenodo, EPA)

---

## Part 5: Science Pipeline Architecture

```
biomeOS orchestrator
    ├── NestGate (data provider)
    │   ├── NCBI ESearch/EFetch
    │   ├── SRA download (fasterq-dump)
    │   └── Storage (SHA-256 integrity)
    ├── wetSpring (science)
    │   ├── FASTQ parser (zero-copy streaming)
    │   ├── Diversity pipeline (Shannon, Simpson, etc.)
    │   └── Anderson spectral analysis
    └── ToadStool (GPU compute)
        ├── Lanczos eigensolver
        ├── SpMV (sparse matrix-vector)
        ├── Batched ODE RK4
        └── FusedMapReduce
```

wetSpring discovers NestGate at runtime (`WETSPRING_DATA_PROVIDER=nestgate`),
uses it if available, falls back to sovereign HTTP if not. No hard dependency.
This is the Songbird pattern — capability-based, not import-based.

---

## Part 6: Verification Commands

```bash
cd barracuda

# Full test suite (961 tests)
cargo test

# Clippy pedantic (0 warnings)
cargo clippy --all-targets -- -D warnings -W clippy::pedantic

# NCBI module tests only (147 tests)
cargo test --lib ncbi

# GPU Anderson scaling (requires GPU)
cargo run --features gpu --release --bin validate_anderson_gpu_scaling

# Line coverage
cargo llvm-cov --lib --summary-only
```

---

## Part 7: Next Steps

### For ToadStool Team
1. **GPU Lanczos kernel** — wrap CPU Lanczos with GPU SpMV (highest impact)
2. **DF64 Lanczos variant** — extended precision for L=24+ lattices
3. **Tolerance module adoption** — 82-constant pattern from wetSpring

### For wetSpring Team
1. **Exp184 execution** — run real NCBI 16S through pipeline (Tier 1: 5 accessions)
2. **Exp185 execution** — cold seep metagenomes (170 samples)
3. **Exp186 implementation** — dynamic Anderson W(t) binary
4. **biomeOS integration testing** — science_pipeline.toml with live NestGate
