# wetspring-barracuda

**Crate:** `wetspring-barracuda` v0.1.0
**License:** AGPL-3.0-or-later
**Updated:** February 22, 2026

---

## What This Is

Rust validation crate for wetSpring — life science, analytical chemistry,
and environmental monitoring algorithms. Proves the full path from Python
baseline through Rust CPU to GPU acceleration via ToadStool/BarraCUDA.

## Architecture

```
wetspring-barracuda
├── 41 CPU bio modules          (pure Rust math, no external C deps)
├── 20 GPU bio modules          (19 lean on ToadStool, 4 local WGSL shaders (ODE, kmer, unifrac, taxonomy))
├── 3 streaming I/O parsers     (FASTQ/gzip, mzML/base64, MS2)
├── 73 validation/benchmark bins (50 CPU + 18 GPU + 5 benchmark)
└── depends on: barracuda (ToadStool) via path dependency
```

All GPU modules delegate to `barracuda::ops::*` primitives from ToadStool.
No local WGSL shader compilation except for 4 Write-phase shaders (ODE, kmer, unifrac, taxonomy; ODE blocked: upstream uses `compile_shader` not `compile_shader_f64`).

## ToadStool Primitives Consumed (23)

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

| Module | ToadStool Primitive | Status |
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
| `ode_sweep_gpu` | local WGSL | ⚠️ Local |
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
| `tolerances` | 39 named constants (scientifically justified) |
| `validation` | Structured test harness with provenance |
| `encoding` | Sovereign base64 (zero dependencies) |
| `error` | Error types (no external crates) |
| `gpu` | `GpuF64` device abstraction over ToadStool's `WgpuDevice` |

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean |
| `cargo clippy --pedantic --nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `#![forbid(unsafe_code)]` | Enforced crate-wide |
| `#![deny(clippy::expect_used, unwrap_used)]` | Enforced |
| External C dependencies | 0 (`flate2` uses `rust_backend`) |
| Line coverage | 96.21% |
| Tests | 730 (654 lib + 60 integration + 14 doc + 2 bench) |
| Validation checks | 1,835 (1,349 CPU + 451 GPU + 35 dispatch) |

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
| `EVOLUTION_READINESS.md` | Absorption tiers, shader inventory, ToadStool status |
| `ABSORPTION_MANIFEST.md` | Write → Absorb → Lean lifecycle tracking |
| `DEPRECATION_MIGRATION.md` | Deprecated APIs and migration paths |
| `../metalForge/PRIMITIVE_MAP.md` | Module ↔ ToadStool primitive mapping |
| `../wateringHole/handoffs/` | ToadStool handoff documents |

## Dependencies

- `barracuda` (ToadStool) — GPU primitives, via path: `../../phase1/toadstool/crates/barracuda`
- `flate2` (pure Rust backend) — gzip decompression
- `serde` + `serde_json` — serialization
- `rand` + `rand_xoshiro` — RNG for Gillespie/rarefaction
- `wgpu` (via `gpu` feature) — GPU device management
