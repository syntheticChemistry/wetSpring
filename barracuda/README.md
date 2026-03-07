# wetspring-barracuda

**Crate:** `wetspring-barracuda` v0.1.0
**License:** AGPL-3.0-or-later
**Updated:** March 7, 2026 (Phase 97e — barraCuda v0.3.3 `2a6c072`, wgpu 28)

---

## What This Is

Rust validation crate for wetSpring — life science, analytical chemistry,
and environmental monitoring algorithms. Proves the full path from Python
baseline through Rust CPU to GPU acceleration via barraCuda.

## Architecture

```
wetspring-barracuda
├── 47 CPU bio modules          (pure Rust math, no external C deps)
├── 45 GPU bio modules          (45 Lean + 7 Compose, 0 Passthrough)
├── 1 provenance module         (barracuda::shaders::provenance wiring)
├── 3 streaming I/O parsers     (FASTQ/gzip, mzML/base64, MS2)
├── 291 validation/benchmark binaries
└── depends on: barracuda via path dependency
```

45 Lean GPU modules delegate to `barracuda::ops::*` primitives.
Builder patterns wired: `HmmForwardArgs`, `Dada2DispatchArgs`, `GillespieModel`.
`PrecisionRoutingAdvice` for shared-memory f64 safety.

## barraCuda Primitives Consumed (150+, 264 ComputeDispatch ops)

| # | Primitive | Category | Origin | Consumed Since |
|---|-----------|----------|--------|:--------------:|
| 1 | `FusedMapReduceF64` (×3: Shannon, Simpson, observed) | stats | core | Feb 16 |
| 2 | `BrayCurtisF64` | diversity | wetSpring | Feb 16 |
| 3 | `GemmF64` | linalg | core | Feb 16 |
| 4 | `BatchedEighGpu` | linalg | neuralSpring | Feb 16 |
| 5 | `KrigingF64` | spatial | core | Feb 16 |
| 6 | `VarianceF64` | stats | core | Feb 16 |
| 7 | `CorrelationF64` | stats | core | Feb 16 |
| 8 | `CovarianceF64` | stats | core | Feb 16 |
| 9 | `WeightedDotF64` | linalg | core | Feb 16 |
| 10 | `PrngXoshiro` | rng | core | Feb 16 |
| 11 | `SmithWatermanGpu` | bio | wetSpring | Feb 20 |
| 12 | `GillespieGpu` | bio | wetSpring | Feb 20 |
| 13 | `TreeInferenceGpu` | bio | wetSpring | Feb 20 |
| 14 | `FelsensteinGpu` | bio | wetSpring | Feb 20 |
| 15 | `HmmBatchForwardF64` | bio | wetSpring | Feb 22 |
| 16 | `AniBatchF64` | bio | wetSpring | Feb 22 |
| 17 | `SnpCallingF64` | bio | wetSpring | Feb 22 |
| 18 | `DnDsBatchF64` | bio | wetSpring | Feb 22 |
| 19 | `PangenomeClassifyGpu` | bio | wetSpring | Feb 22 |
| 20 | `QualityFilterGpu` | bio | wetSpring | Feb 22 |
| 21 | `Dada2EStepGpu` | bio | wetSpring | Feb 22 |
| 22 | `RfBatchInferenceGpu` | bio | wetSpring | Feb 22 |
| 23 | `BatchTolSearchF64` | search | core | Feb 16 |

## Module Map

### CPU Bio (41 modules)

| Module | Domain | Validated Against |
|--------|--------|-------------------|
| `alignment` | Smith-Waterman (affine gaps) | Pure Python SW |
| `ani` | Average Nucleotide Identity | Pure Python |
| `bistable` | Bistable phenotypic switching ODE | scipy ODE |
| `bootstrap` | RAWR bootstrap resampling | Pure Python |
| `capacitor` | Phenotypic capacitor ODE | scipy ODE |
| `chimera` | UCHIME chimera detection | DADA2-R |
| `cooperation` | QS game theory ODE | scipy ODE |
| `dada2` | ASV denoising | DADA2-R |
| `decision_tree` | Decision tree inference | sklearn |
| `derep` | Dereplication + abundance | VSEARCH |
| `diversity` | Shannon, Simpson, Chao1, Bray-Curtis, Pielou, rarefaction | QIIME2 |
| `dnds` | Nei-Gojobori dN/dS | Pure Python + JC |
| `eic` | EIC extraction + peak integration | asari |
| `feature_table` | LC-MS feature extraction | asari |
| `felsenstein` | Felsenstein pruning | Pure Python JC69 |
| `gbm` | GBM inference (binary + multi-class) | sklearn |
| `gillespie` | Gillespie SSA | numpy ensemble |
| `hmm` | HMM (forward/backward/Viterbi/posterior) | numpy sovereign |
| `kmd` | Kendrick mass defect | pyOpenMS |
| `kmer` | K-mer counting (2-bit canonical) | QIIME2 |
| `merge_pairs` | Paired-end overlap merging | VSEARCH |
| `molecular_clock` | Strict/relaxed clock, calibration | Pure Python |
| `multi_signal` | Multi-input QS network ODE | scipy ODE |
| `neighbor_joining` | NJ tree construction | Pure Python |
| `ode` | Generic RK4 integrator | scipy odeint |
| `pangenome` | Gene clustering, Heap's law, enrichment | Pure Python |
| `pcoa` | PCoA (Jacobi eigendecomposition) | QIIME2 |
| `phage_defense` | Phage defense deaminase ODE | scipy ODE |
| `phred` | Phred quality decode/encode | Biopython |
| `placement` | Phylogenetic placement | Pure Python |
| `qs_biofilm` | QS/c-di-GMP ODE | scipy ODE |
| `quality` | Quality filtering | Trimmomatic |
| `random_forest` | RF ensemble inference | sklearn |
| `reconciliation` | DTL reconciliation | Pure Python |
| `robinson_foulds` | RF tree distance | dendropy |
| `signal` | 1D peak detection | scipy.signal |
| `snp` | SNP calling | Pure Python |
| `spectral_match` | MS2 cosine similarity | pyOpenMS |
| `taxonomy` | Naive Bayes classifier | QIIME2 |
| `tolerance_search` | ppm/Da m/z search | FindPFAS |
| `unifrac` | UniFrac + Newick parser | QIIME2 |

### GPU Bio (20 modules)

| Module | barraCuda Primitive | Status |
|--------|-------------------|--------|
| `ani_gpu` | `AniBatchF64` | ✅ Lean |
| `chimera_gpu` | `FusedMapReduceF64` | ✅ Lean |
| `dada2_gpu` | `Dada2EStepGpu` | ✅ Lean |
| `diversity_gpu` | `FusedMapReduceF64` (×4) | ✅ Lean |
| `dnds_gpu` | `DnDsBatchF64` | ✅ Lean |
| `eic_gpu` | `FusedMapReduceF64` | ✅ Lean |
| `gemm_cached` | `GemmF64` | ✅ Lean |
| `hmm_gpu` | `HmmBatchForwardF64` | ✅ Lean |
| `kriging` | `KrigingF64` | ✅ Lean |
| `ode_sweep_gpu` | `BatchedOdeRK4F64` | ✅ Lean |
| `pangenome_gpu` | `PangenomeClassifyGpu` | ✅ Lean |
| `pcoa_gpu` | `BatchedEighGpu` | ✅ Lean |
| `quality_gpu` | `QualityFilterGpu` | ✅ Lean |
| `rarefaction_gpu` | `PrngXoshiro` | ✅ Lean |
| `random_forest_gpu` | `RfBatchInferenceGpu` | ✅ Lean |
| `snp_gpu` | `SnpCallingF64` | ✅ Lean |
| `spectral_match_gpu` | `WeightedDotF64` | ✅ Lean |
| `stats_gpu` | `VarianceF64` etc. | ✅ Lean |
| `streaming_gpu` | orchestrator | ✅ Lean |
| `taxonomy_gpu` | `GemmCached` | ✅ Lean |

### I/O

| Module | Format | Pattern |
|--------|--------|---------|
| `io::fastq` | FASTQ/gzip | Streaming, zero-copy `FastqRefRecord` |
| `io::mzml` | mzML/base64 | Streaming with `DecodeBuffer` reuse |
| `io::ms2` | MS2 text | Streaming iterator |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `special` | Sovereign math (erf, ln_gamma, regularized_gamma, normal_cdf) |
| `tolerances` | 164 named constants (scientifically justified) |
| `validation` | Structured test harness with provenance |
| `encoding` | Sovereign base64 (zero dependencies) |
| `error` | Error types (no external crates) |
| `gpu` | `GpuF64` device abstraction over barraCuda's `WgpuDevice` |
| `provenance` | Cross-spring shader provenance (wires `barracuda::shaders::provenance`) |

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean |
| `cargo clippy --pedantic --nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `#![deny(unsafe_code)]` | Enforced crate-wide (test-only `allow` for `env::set_var` in edition 2024) |
| `#![deny(clippy::expect_used, unwrap_used)]` | Enforced |
| External C dependencies | 0 (`flate2` uses `rust_backend`) |
| Tests | 1,346 pass (1,047 lib + 200 forge + 99 doc) |
| ESN ridge regression | Proper Cholesky solve (not diagonal approximation) |
| I/O parsers | Streaming-first; buffering APIs deprecated |
| Validation checks | 8,431+ across 291 binaries |

## Quick Start

```bash
# All tests
cargo test

# GPU tests (requires NVIDIA GPU)
cargo test --features gpu

# Quality gate
cargo fmt -- --check
cargo clippy --lib -- -W clippy::pedantic -W clippy::nursery
cargo doc --no-deps

# CPU validation (1,291 checks)
cargo run --release --bin validate_barracuda_cpu_full

# GPU validation (451 checks, needs --features gpu)
cargo run --features gpu --release --bin validate_barracuda_gpu_full
```

## Related Documents

| Document | Purpose |
|----------|---------|
| `EVOLUTION_READINESS.md` | Absorption tiers, shader inventory, ecosystem status |
| `ABSORPTION_MANIFEST.md` | Write → Absorb → Lean lifecycle tracking |
| `../CHANGELOG.md` | Version history and evolution log |
| `../metalForge/PRIMITIVE_MAP.md` | Module ↔ barraCuda primitive mapping |
| `../wateringHole/handoffs/` | barraCuda/toadStool handoff documents |

## Dependencies

- `barracuda` — standalone math primal via path: `../../barraCuda/crates/barracuda` (v0.3.3, `2a6c072`)
- `flate2` (pure Rust backend) — gzip decompression
- `serde_json` (optional, `json` feature) — model import for 2 binaries
- `wgpu` 28 (via `gpu` feature) — GPU device management
- `bytemuck` — zero-copy GPU buffer casting
