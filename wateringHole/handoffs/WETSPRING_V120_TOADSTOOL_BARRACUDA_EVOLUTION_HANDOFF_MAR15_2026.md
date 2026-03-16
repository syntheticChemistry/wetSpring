# wetSpring V120 — toadStool / barraCuda Evolution Handoff

**Date:** March 15, 2026
**From:** wetSpring V120 (ecoPrimals)
**To:** toadStool (S155+) and barraCuda (v0.3.5) teams
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring is the largest consumer of barraCuda primitives (44 GPU modules,
150+ primitives, 1,638 tests, 328 validation binaries across 376 experiments).
This handoff documents: (1) what we consume and how, (2) patterns we evolved
that benefit the ecosystem, (3) specific absorption/evolution targets for
toadStool and barraCuda.

---

## Part 1: barraCuda Primitive Consumption

### GPU Ops (44 modules)

| Category | barracuda Op | wetSpring Module | Domain |
|----------|-------------|-----------------|--------|
| **GEMM** | `ops::linalg::gemm_f64::GemmF64` | `derep_gpu`, `chimera_gpu`, `spectral_match_gpu`, `neighbor_joining_gpu`, `gemm_cached` | Pairwise distance, cosine similarity |
| **Map-Reduce** | `ops::fused_map_reduce_f64::FusedMapReduceF64` | `derep_gpu`, `chimera_gpu`, `spectral_match_gpu`, `neighbor_joining_gpu` | Aggregation chains |
| **Eigendecomp** | `ops::linalg::batched_eigh_gpu::BatchedEighGpu` | `pcoa_gpu` | PCoA ordination |
| **Peak Detect** | `ops::peak_detect_f64::PeakDetectF64` | `signal_gpu` | LC-MS peak finding |
| **Tolerance Search** | `ops::batch_tolerance_search_f64::BatchToleranceSearchF64` | `tolerance_search_gpu` | PFAS m/z matching |
| **Quality Filter** | `ops::bio::quality_filter::QualityConfig` | `quality_gpu` | FASTQ QC |
| **Kmer Histogram** | `ops::bio::kmer_histogram::KmerHistogramGpu` | `kmer_gpu` | K-mer profiling |
| **HMM Forward** | `ops::bio::hmm::HmmBatchForwardF64` | `hmm_gpu` | Gene prediction |
| **Pairwise Hamming** | `ops::bio::pairwise_hamming::PairwiseHammingGpu` | `hamming_gpu` | Sequence distance |
| **Pangenome** | `ops::bio::pangenome::PangenomeClassifyGpu` | `pangenome_gpu` | Gene family classification |
| **Batch Fitness** | `ops::bio::batch_fitness::BatchFitnessGpu` | `batch_fitness_gpu` | Evolutionary game theory |

### Special Functions (CPU)

| Function | Usage |
|----------|-------|
| `barracuda::special::erf` | Anderson W disorder parameter, norm_cdf |
| `barracuda::special::ln_gamma` | Chi-squared tests, regularized gamma |
| `barracuda::special::regularized_gamma_p` | P-value computations |

### Linear Algebra

| Module | Usage |
|--------|-------|
| `barracuda::linalg::nmf` | NMF (Lee & Seung 2001) for feature discovery |
| `barracuda::linalg::solve_f64_cpu` | Ridge regression readout (ESN) |

### Spectral (Anderson Localization)

| Function | Usage |
|----------|-------|
| `barracuda::spectral::anderson_2d/3d` | Anderson disorder lattice construction |
| `barracuda::spectral::lanczos` | Tridiagonalization for eigenvalues |
| `barracuda::spectral::lanczos_eigenvalues` | Level spacing ratio (LSR) |
| `barracuda::spectral::level_spacing_ratio` | GOE→Poisson transition detection |

### ESN (Echo State Network)

| Type | Usage |
|------|-------|
| `barracuda::esn_v2::MultiHeadEsn` | 6-head bio classifier (diversity, taxonomy, AMR, bloom, disorder, urgency) |
| `barracuda::tensor::Tensor` | Feature vector I/O |

### Feature Flags

```toml
gpu = ["barracuda/gpu", "barracuda/domain-esn", "dep:wgpu", "dep:tokio"]
```

---

## Part 2: Patterns Evolved in wetSpring (Absorb These)

### 2a. Typed Error Enums (Replace String Errors)

wetSpring V119–V120 systematically replaced all `Result<_, String>` with
typed enums. This pattern should propagate to barraCuda and toadStool:

```rust
#[derive(Debug)]
pub enum NcbiError {
    HttpRequest(String),
    InvalidUtf8(String),
    AssemblyNotFound(String),
}
impl std::fmt::Display for NcbiError { /* ... */ }
impl std::error::Error for NcbiError {}
```

**Why:** Callers can match on variants instead of parsing error strings.
The `Display` impl preserves backward-compatible messages.

### 2b. `#[expect(reason)]` Instead of `#[allow()]`

All lint suppressions in wetSpring use `#[expect(lint, reason = "...")]`
which enforces documentation of WHY a lint is suppressed and fails if the
suppression becomes unnecessary.

### 2c. Centralized Named Tolerances (200+ constants)

```rust
pub mod tolerances {
    pub const ANALYTICAL_F64: f64 = 1e-12;   // IEEE 754 f64 chain rounding
    pub const GPU_VS_CPU_F64: f64 = 1e-6;    // GPU instruction reorder
    pub const ODE_DIVISION_GUARD: f64 = 1e-30; // Hill denominator floor
}
```

**Pattern:** Every tolerance has a doc comment citing its scientific basis
and validation experiment. No ad-hoc magic numbers. Shared Python module
mirrors all constants (`scripts/tolerances.py`).

### 2d. Capability-Based Discovery (`primal_names.rs`)

```rust
pub mod primal_names {
    pub const SELF: &str = "wetspring";
    pub const BIOMEOS: &str = "biomeos";
    pub const SONGBIRD: &str = "songbird";
    pub const TOADSTOOL: &str = "toadstool";
    pub const SQUIRREL: &str = "squirrel";
}
```

Zero hardcoded primal names in production code. All discovery uses
`discover_socket(primal_names::TOADSTOOL)`.

### 2e. Deploy Graph with `fallback = "skip"`

Optional primals (Squirrel, ToadStool, NestGate, petalTongue) use
`optional = true` + `fallback = "skip"` so biomeOS can deploy wetSpring
even when some primals are absent.

### 2f. Niche Self-Knowledge (`niche.rs` + BYOB YAML)

Every spring should export:
- `NICHE_NAME`, `CAPABILITIES`, `DEPENDENCIES`
- A `.yaml` BYOB manifest in `niches/`
- Cost estimates and semantic mappings (feature-gated behind `json`)

### 2g. `proptest` for Stochastic Algorithms

Property-based testing catches edge cases that unit tests miss:
```rust
proptest! {
    #[test]
    fn bootstrap_ci_covers_mean(seed in 0u64..1000, n in 5usize..50) {
        // 95% CI should contain the true mean
    }
}
```

---

## Part 3: Evolution Targets for toadStool

### 3a. Typed Errors

toadStool S155 uses `forbid(unsafe_code)` (good). Next: replace any
remaining `Result<_, String>` with typed enums following wetSpring pattern.

### 3b. `#[expect(reason)]` Migration

If toadStool still uses `#[allow()]`, migrate to `#[expect(reason)]` in
Edition 2024.

### 3c. Deploy Graph Integration

wetSpring's deploy graph (`graphs/wetspring_deploy.toml`) includes toadStool
as an optional compute node. toadStool should verify it can be discovered
via `by_capability = "compute"` in biomeOS orchestration.

### 3d. Tolerance Centralization

If toadStool has any hardcoded numeric thresholds (dispatch thresholds,
timeout values), centralize them with scientific documentation.

---

## Part 4: Evolution Targets for barraCuda

### 4a. GPU Hardware Tolerance Failures

3 pre-existing GPU tests fail on specific hardware with assertion errors.
These are tolerance issues, not algorithm bugs:

- `jaccard_gpu::tests::jaccard_gpu_basic` — f32 parity
- `spatial_payoff_gpu::tests::spatial_payoff_all_cooperators` — f32 spatial
- `hamming_gpu::tests::hamming_gpu_matches_cpu` — f64 cast to f32

**Suggestion:** Review `GPU_F32_PARITY` (1e-5) and `GPU_F32_SPATIAL` (1e-4)
tolerances for these tests, or conditionally skip on hardware where f32
precision differs.

### 4b. Feature Module Gating

`barracuda::spectral::stats` and `barracuda::ops` require `gpu` feature
even for CPU-only code paths. Consider splitting CPU-only spectral stats
behind a separate feature gate so springs can `cargo test --lib` without
`gpu`.

### 4c. Typed Error Adoption

barraCuda's `BarracudaError` already exists but some ops may still return
`String` errors. Ensure all public APIs return typed errors.

### 4d. ESN Evolution

wetSpring's `MultiHeadBioEsn` demonstrates a 6-head bio classifier pattern.
The reservoir is shared, heads are domain-specific. This pattern could be
generalized in barraCuda for other springs.

---

## Part 5: Metrics

| Metric | Value |
|--------|-------|
| wetSpring version | V120 |
| barraCuda version | v0.3.5 |
| toadStool version | S155 |
| GPU modules | 44 |
| Primitives consumed | 150+ |
| Library tests | 1,638 (1,404 barracuda + 234 forge) |
| Validation binaries | 328 (306 barracuda + 22 forge) |
| Experiments | 376 |
| Capability domains | 16 |
| Methods | 22 |
| Tolerance constants | 200+ |
| LOC (Rust) | 199,125 (182K barracuda + 17K forge) |
| Python scripts | 58 |
| Papers reproduced | 63 |
| Clippy warnings | 0 (pedantic + nursery) |
| Unsafe blocks | 0 |

---

## Part 6: Cross-Spring Learnings

| Learning | Source | Relevance |
|----------|--------|-----------|
| DI for socket/env eliminates unsafe `set_var` | airSpring v0.8.2 | Already clean in wetSpring; pattern useful for toadStool |
| `fallback = "skip"` in deploy graph | neuralSpring S147 | Applied in wetSpring V120 |
| `TissueContext` GPU uniform buffers | healthSpring V21 | Reduces kernel arg count — applicable to barraCuda |
| Typed `tarpc` IPC client | groundSpring V101 | Compile-time checked IPC — applicable to toadStool |
| Shared tolerance module (Python) | wetSpring V120 | Pattern for any spring with Python baselines |
