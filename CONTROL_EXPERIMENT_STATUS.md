# wetSpring Control Experiment Status

**Date:** February 22, 2026
**Status:** 93 experiments, 2,173+ validation checks, all PASS (728 Rust tests)

---

## Experiment Status

| Exp | Name | Track | Status | Checks |
|-----|------|-------|--------|--------|
| 001 | Galaxy Bootstrap (QIIME2/DADA2) | 1 | COMPLETE | 28 |
| 002 | Phytoplankton 16S (PRJNA1195978) | 1 | COMPLETE | — |
| 003 | Phage Assembly (SPAdes/Pharokka) | 1 | COMPLETE | — |
| 004 | Rust FASTQ + Diversity | 1 | COMPLETE | 55 |
| 005 | asari LC-MS Bootstrap | 2 | COMPLETE | 7 |
| 006 | PFAScreen Validation (FindPFAS) | 2 | COMPLETE | 10 |
| 007 | Rust mzML + PFAS | 2 | COMPLETE | — |
| 008 | PFAS ML Water Monitoring | 2 | COMPLETE (Phase 3) | 7 |
| 009 | Feature Pipeline (asari MT02) | 2 | COMPLETE | 8 |
| 010 | Peak Detection (scipy baseline) | cross | COMPLETE | 17 |
| 011 | 16S Pipeline End-to-End | 1 | COMPLETE | 37 |
| 012 | Algae Pond 16S (PRJNA488170) | 1 | COMPLETE | 34 |
| 013 | VOC Peak Validation (Reese 2019) | 1/cross | COMPLETE | 22 |
| 014 | Public Data Benchmarks (4 BioProjects) | 1 | COMPLETE | 202 |
| 015 | Pipeline Benchmark (Rust vs Galaxy) | 1 | COMPLETE | Benchmark |
| 016 | GPU Pipeline Parity (CPU↔GPU) | 1 | COMPLETE | 88 |
| 017 | Extended Algae (PRJNA382322) | 1 | COMPLETE | 35 |
| 018 | PFAS Library (Jones Lab Zenodo) | 2 | COMPLETE | 26 |
| 019 | Phylogenetic Validation (Phase 1) | 1b | COMPLETE | 30 |
| 020 | Waters 2008 QS/c-di-GMP ODE | 1 | COMPLETE | 16 |
| 021 | Robinson-Foulds Validation | 1b | COMPLETE | 23 |
| 022 | Massie 2012 Gillespie SSA | 1 | COMPLETE | 13 |
| 023 | Fernandez 2020 Bistable Switching | 1 | COMPLETE | 14 |
| 024 | Srivastava 2011 Multi-Signal QS | 1 | COMPLETE | 19 |
| 025 | Bruger & Waters 2018 Cooperation | 1 | COMPLETE | 20 |
| 026 | Liu 2014 HMM Primitives | 1b | COMPLETE | 21 |
| 027 | Mhatre 2020 Phenotypic Capacitor | 1 | COMPLETE | 18 |
| 028 | Smith-Waterman Alignment | 1b | COMPLETE | 15 |
| 029 | Felsenstein Pruning Likelihood | 1b/c | COMPLETE | 16 |
| 030 | Hsueh 2022 Phage Defense Deaminase | 1 | COMPLETE | 12 |
| 031 | Wang 2021 RAWR Bootstrap | 1b | COMPLETE | 11 |
| 032 | Alamin & Liu 2024 Placement | 1b | COMPLETE | 12 |
| 033 | Liu 2009 Neighbor-Joining (SATé core) | 1b | COMPLETE | 16 |
| 034 | Zheng 2023 DTL Reconciliation | 1b | COMPLETE | 14 |
| 035 | BarraCUDA CPU Parity v2 | cross | COMPLETE | 18 |
| 036 | PhyNetPy RF distances (Exp019 Phase 2) | 1b | COMPLETE | 15 |
| 037 | PhyloNet-HMM discordance (Exp019 Phase 3) | 1b | COMPLETE | 10 |
| 038 | SATe pipeline benchmark (Exp019 Phase 4) | 1b | COMPLETE | 17 |
| 039 | Algal pond time-series (Cahill proxy) | 1 | COMPLETE | 11 |
| 040 | Bloom surveillance (Smallwood proxy) | 1 | COMPLETE | 15 |
| 041 | EPA PFAS ML (Jones F&T proxy) | 2 | COMPLETE | 14 |
| 042 | MassBank spectral (Jones MS proxy) | 2 | COMPLETE | 9 |
| 043 | BarraCUDA CPU Parity v3 | cross | COMPLETE | 45 |
| 044 | BarraCUDA GPU v3 | cross | COMPLETE | 14 |
| 045 | ToadStool Bio Absorption | cross/GPU | COMPLETE | 10 |
| 046 | GPU Phylogenetic Composition | GPU | COMPLETE | 15 |
| 047 | GPU HMM Batch Forward | GPU | COMPLETE | 13 |
| 048 | CPU vs GPU Benchmark (Phylo + HMM) | GPU | COMPLETE | 6 |
| 049 | GPU ODE Parameter Sweep | GPU | COMPLETE | 7 |
| 050 | GPU Bifurcation Eigenvalue Analysis | GPU | COMPLETE | 5 |
| 051 | Anderson 2015 Rare Biosphere | 1c | COMPLETE | 35 |
| 052 | Anderson 2014 Viral Metagenomics | 1c | COMPLETE | 22 |
| 053 | Mateos 2023 Sulfur Phylogenomics | 1c | COMPLETE | 15 |
| 054 | Boden 2024 Phosphorus Phylogenomics | 1c | COMPLETE | 13 |
| 055 | Anderson 2017 Population Genomics | 1c | COMPLETE | 24 |
| 056 | Moulana 2020 Pangenomics | 1c | COMPLETE | 24 |
| 057 | BarraCUDA CPU Parity v4 (Track 1c) | cross | COMPLETE | 44 |
| 058 | GPU Track 1c (ANI + SNP + Pangenome + dN/dS) | GPU | COMPLETE | 27 |
| 059 | 25-Domain Rust vs Python Benchmark | cross | COMPLETE | Benchmark |
| 060 | metalForge Cross-Substrate Validation | cross/GPU | COMPLETE | 20 |
| 061 | Random Forest Ensemble Inference | cross | COMPLETE | 13 |
| 062 | GBM Inference (Binary + Multi-Class) | cross | COMPLETE | 16 |
| 063 | GPU Random Forest Batch Inference | GPU | COMPLETE | 13 |
| 064 | BarraCUDA GPU Parity v1 (all GPU domains) | cross/GPU | COMPLETE | 26 |
| 065 | metalForge Full Cross-System Validation | cross/GPU | COMPLETE | 35 |
| 066 | CPU vs GPU Scaling Benchmark (all GPU domains) | GPU | COMPLETE | Benchmark |
| 067 | ToadStool Dispatch Overhead Profiling | GPU | COMPLETE | Benchmark |
| 068 | Pipeline Caching Optimization | GPU | COMPLETE | Optimization/Benchmark |
| 069 | Python → Rust CPU → GPU Three-Tier Benchmark | cross | COMPLETE | Benchmark |
| 070 | BarraCUDA CPU 25-Domain Pure Rust Math Proof | cross | COMPLETE | 50 |
| 071 | BarraCUDA GPU Math Portability Proof | GPU | COMPLETE | 24 |
| 072 | GPU Streaming Pipeline Proof | GPU | COMPLETE | 17 |
| 073 | Compute Dispatch Overhead Proof | GPU | COMPLETE | 21 |
| 074 | metalForge Substrate Router | cross/GPU | COMPLETE | 20 |
| 075 | Pure GPU Analytics Pipeline | GPU | COMPLETE | 31 |
| 076 | Cross-Substrate Pipeline | cross/GPU | COMPLETE | 17 |
| 077 | ToadStool Bio Rewire | GPU/cross | COMPLETE | 451 (re-validated) |
| 078 | ODE GPU Sweep Readiness | cross/GPU | COMPLETE | 10 (round-trip + parity) |
| 079 | BarraCUDA CPU v6 — ODE Flat Param | CPU/cross | COMPLETE | 48 (flat RT + ODE + Python) |
| 080 | metalForge Dispatch Routing | cross/dispatch | COMPLETE | 35 (7 sections × 5 configs) |
| 081 | K-mer GPU Histogram Prep | GPU/refactor | COMPLETE | 4 (round-trip + GPU sizing) |
| 082 | UniFrac Flat Tree (CSR) | GPU/refactor | COMPLETE | 4 (CSR + parity + matrix) |
| 083 | Taxonomy NPU Quantization | NPU/refactor | COMPLETE | 3 (int8 RT + parity + sizing) |
| 084 | metalForge Full Cross-Substrate v2 | metalForge | COMPLETE | 35+ (12 domains CPU ↔ GPU) |
| 085 | BarraCUDA CPU v7 — Tier A Layouts | CPU/layout | COMPLETE | 43 (kmer/unifrac/taxonomy flat) |
| 086 | metalForge Pipeline Proof | metalForge | COMPLETE | 45 (5-stage dispatch + parity) |
| 087 | GPU Extended Domains (EIC/PCoA/Kriging/Rarefaction) | GPU | COMPLETE | 50+ (4 new GPU domains) |
| 088 | metalForge PCIe Direct Transfer | metalForge | COMPLETE | 32 (6 paths + buffer contracts) |
| 089 | ToadStool Streaming Dispatch | streaming | COMPLETE | 25 (5 patterns + determinism) |
| 090 | Pure GPU Streaming Pipeline | GPU/streaming | COMPLETE | 80 (4 modes: RT, stream, parity, scaling) |
| 091 | Streaming vs Round-Trip Benchmark | GPU/benchmark | COMPLETE | 2 (parity + Bray-Curtis error) |
| 092 | CPU vs GPU All 16 Domains | GPU/parity | COMPLETE | 48 (16 domains head-to-head) |
| 093 | metalForge Full v3 (16 domains) | metalForge | COMPLETE | 28 (16 domains substrate-independent) |

---

## Totals

| Category | Count |
|----------|-------|
| Experiments completed | 89 |
| CPU validation checks | 1,349 |
| GPU validation checks | 451 |
| Dispatch validation checks | 35 |
| **Total validation checks** | **1,835** |
| Rust tests | 730 (654 lib + 60 integration + 14 doc + 2 bench) |
| BarraCUDA CPU parity | 205/205 (25 domains + 6 ODE flat) |
| BarraCUDA GPU parity | 8 domains consolidated (Exp064) |
| metalForge cross-system | 8 domains CPU↔GPU proven (Exp065) |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| ToadStool primitives consumed | 23 (15 original + 8 bio) |

---

## Python Baseline Status

| Script | Tool | Date | Status |
|--------|------|------|--------|
| `benchmark_python_baseline.py` | QIIME2/DADA2-R | Feb 2026 | GREEN |
| `validate_public_16s_python.py` | BioPython + NCBI | Feb 2026 | GREEN |
| `waters2008_qs_ode.py` | scipy.integrate.odeint | Feb 2026 | GREEN (35/35) |
| `gillespie_baseline.py` | numpy SSA ensemble | Feb 2026 | GREEN (8/8) |
| `rf_distance_baseline.py` | dendropy RF distance | Feb 2026 | GREEN (10/10) |
| `newick_parse_baseline.py` | dendropy tree stats | Feb 2026 | GREEN (10/10) |
| `pfas_tree_export.py` | sklearn DecisionTree | Feb 2026 | GREEN (acc=0.989) |
| `exp008_pfas_ml_baseline.py` | sklearn RF+GBM | Feb 2026 | GREEN (RF F1=0.978) |
| `fernandez2020_bistable.py` | scipy ODE (bifurcation) | Feb 2026 | GREEN |
| `srivastava2011_multi_signal.py` | scipy ODE (multi-signal) | Feb 2026 | GREEN |
| `bruger2018_cooperation.py` | scipy ODE (game theory) | Feb 2026 | GREEN |
| `liu2014_hmm_baseline.py` | numpy HMM (sovereign) | Feb 2026 | GREEN |
| `mhatre2020_capacitor.py` | scipy ODE (capacitor) | Feb 2026 | GREEN |
| `smith_waterman_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `felsenstein_pruning_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `hsueh2022_phage_defense.py` | scipy ODE (phage defense) | Feb 2026 | GREEN |
| `wang2021_rawr_bootstrap.py` | pure Python (bootstrap) | Feb 2026 | GREEN |
| `alamin2024_placement.py` | pure Python (placement) | Feb 2026 | GREEN |
| `liu2009_neighbor_joining.py` | pure Python (NJ) | Feb 2026 | GREEN |
| `zheng2023_dtl_reconciliation.py` | pure Python (DTL) | Feb 2026 | GREEN |
| `phynetpy_rf_baseline.py` | PhyNetPy gene trees | Feb 2026 | GREEN |
| `phylohmm_introgression_baseline.py` | PhyloNet-HMM | Feb 2026 | GREEN |
| `sate_alignment_baseline.py` | SATe pipeline | Feb 2026 | GREEN |
| `algae_timeseries_baseline.py` | Cahill proxy | Feb 2026 | GREEN |
| `bloom_surveillance_baseline.py` | Smallwood proxy | Feb 2026 | GREEN |
| `epa_pfas_ml_baseline.py` | Jones F&T proxy | Feb 2026 | GREEN |
| `massbank_spectral_baseline.py` | Jones MS proxy | Feb 2026 | GREEN |
| `benchmark_rust_vs_python.py` | 18-domain timing (Exp043) | Feb 2026 | GREEN |
| `anderson2015_rare_biosphere.py` | diversity/rarefaction (Exp051) | Feb 2026 | GREEN |
| `anderson2014_viral_metagenomics.py` | dN/dS + diversity (Exp052) | Feb 2026 | GREEN |
| `mateos2023_sulfur_phylogenomics.py` | clock/reconciliation (Exp053) | Feb 2026 | GREEN |
| `boden2024_phosphorus_phylogenomics.py` | clock/reconciliation (Exp054) | Feb 2026 | GREEN |
| `anderson2017_population_genomics.py` | ANI/SNP (Exp055) | Feb 2026 | GREEN |
| `moulana2020_pangenomics.py` | pangenome/enrichment (Exp056) | Feb 2026 | GREEN |
| `barracuda_cpu_v4_baseline.py` | 5 Track 1c domain timing (Exp057) | Feb 2026 | GREEN |

---

## Remaining Work

### Deferred (not blocking)
- Exp002 raw data: 70 FASTQ pairs from SRA (Galaxy bootstrap, not needed for validation)
- Trimmomatic/pyteomics baselines: superseded by sovereign Rust implementations

### Completed
- Exp019 Phases 2-4 (Phylogenetic): All COMPLETE
- Exp008 Full ML Pipeline: All COMPLETE
- Tolerance centralization: **DONE** — 39 named constants in `tolerances.rs`
- Code quality hardening: **DONE** — `forbid(unsafe_code)`, `deny(expect_used, unwrap_used)`, pedantic + nursery clippy
- metalForge forge crate: **DONE** — `wetspring-forge` (24 tests, substrate discovery + dispatch)
- GPU workgroup constants: **DONE** — all GPU modules use named `WORKGROUP_SIZE` matching ToadStool shaders
- Hardware abstraction: **DONE** — `HardwareInventory::from_content()`, injectable `/proc` parsing
- I/O streaming: **DONE** — zero-copy FASTQ (`FastqRefRecord`), mzML buffer reuse (`DecodeBuffer`)
- Determinism tests: **DONE** — 16 bitwise-exact tests across non-stochastic algorithms
- Fuzz testing: **DONE** — 4 harnesses (FASTQ, mzML, MS2, XML) via cargo-fuzz
- Doc strictness: **DONE** — `-D missing_docs -D rustdoc::broken_intra_doc_links` pass
- Math extraction: **DONE** — `bio::special` → `crate::special` (top-level, re-export for compat)
- Absorption batch APIs: **DONE** — `snp::call_snps_batch`, `quality::filter_reads_flat`, `pangenome::analyze_batch`

---

## Track Coverage

### Track 1: Microbial Ecology (16S rRNA)
**Status:** Comprehensive. 9 experiments (001, 004, 011, 012, 014, 017, 020, 039, 040)
cover the full 16S pipeline from FASTQ to diversity metrics, validated against
QIIME2, DADA2, and 4 BioProjects with 22 samples.

### Track 1b: Comparative Genomics & Phylogenetics
**Status:** Comprehensive. 12 experiments covering the full phylogenetic
pipeline: Newick parsing (Exp019), Robinson-Foulds (Exp021), HMM (Exp026),
Smith-Waterman (Exp028), Felsenstein pruning (Exp029), bootstrap (Exp031),
placement (Exp032), Neighbor-Joining tree construction (Exp033), DTL
reconciliation (Exp034), PhyNetPy RF (Exp036), PhyloNet-HMM (Exp037), and
SATe pipeline (Exp038). This provides a complete
toolkit from sequence alignment through tree construction, evaluation,
statistical confidence, and cophylogenetic analysis.

### Track 1c: Deep-Sea Metagenomics (Anderson)
**Status:** Comprehensive. 6 experiments (051-056) plus CPU parity (Exp057).
Covers ANI, SNP calling, dN/dS, molecular clock, pangenome analysis,
phylogenomics, and rare biosphere diversity — all validated against Python
baselines and proved as pure Rust math via BarraCUDA CPU v4.

### Track 2: Analytical Chemistry (LC-MS, PFAS)
**Status:** Comprehensive. 7 experiments (005, 006, 009, 013, 018, 041, 042) cover
mzML parsing, feature extraction, peak detection, PFAS screening, and library
matching. Exp008 adds sovereign ML for environmental monitoring.

---

## Code Quality (Feb 22, 2026)

```
cargo fmt --check              → clean (0 diffs, both crates)
cargo clippy --pedantic        → 0 warnings (pedantic + nursery, default + GPU features)
cargo doc --features gpu       → clean (0 warnings, strict: -D missing_docs -D broken_intra_doc_links)
cargo test --lib               → 633 passed, 0 failed, 1 ignored (hardware-dependent)
cargo test --tests             → 60 integration (23 bio + 16 determinism + 21 I/O)
cargo test --doc               → 14 passed, 0 failed
#![forbid(unsafe_code)]        → enforced crate-wide
#![deny(expect_used, unwrap_used)] → enforced crate-wide (test modules #[allow])
partial_cmp().unwrap()         → 0 (all migrated to f64::total_cmp)
inline tolerance literals      → 0 (39 named constants in tolerances.rs)
GPU workgroup sizes            → named constants in all 9 *_gpu.rs (match WGSL shaders)
shared math (bio::special)     → erf, ln_gamma, regularized_gamma (no duplication)
hardware detection             → injectable (from_content / parse_*), no direct /proc in library
SPDX headers                   → all .rs files
max file size                  → all under 1000 LOC (fastq.rs: 907 largest)
external C dependencies        → 0 (flate2 uses rust_backend)
provenance headers             → all 79 binaries (commit, command, hardware)
Python baselines               → scripts/requirements.txt (pinned numpy, scipy, sklearn)
barracuda_cpu                  → 205/205 checks PASS (25 domains + 6 ODE flat)
barracuda_gpu                  → 451 GPU checks PASS
fuzz harnesses                 → 4 (FASTQ, mzML, MS2, XML)
zero-copy I/O                  → FastqRefRecord, DecodeBuffer reuse, streaming iterators
```

## BarraCUDA CPU Parity

The `validate_barracuda_cpu` v1-v5 binaries prove pure Rust math matches
Python across all 25 algorithmic domains:
- v1 (Exp035): 9 core domains
- v2 (Exp035): +5 batch/flat APIs
- v3 (Exp043): +9 domains (QS, phage, bootstrap, placement, decision tree, spectral, diversity, k-mer, pipeline)
- v4 (Exp057): +5 Track 1c domains (ANI, SNP, dN/dS, molecular clock, pangenome)
- v5 (Exp061/062): +2 ML domains (Random Forest, GBM)

Combined: 205/205 CPU parity checks. This is the bridge to pure GPU execution.

```
Total CPU time: ~85ms (release build, all 25 domains, v4 adds ~0.4ms, v5 adds ~62µs)
```

## BarraCUDA GPU Parity

Exp064 consolidates ALL GPU-eligible domains into a single validation binary,
proving pure GPU math matches CPU reference truth across the full portfolio:

- Diversity (Shannon, Simpson, Bray-Curtis) — via `FusedMapReduceF64`
- ANI — via `barracuda::ops::bio::ani::AniBatchF64` (ToadStool)
- SNP — via `barracuda::ops::bio::snp::SnpCallingF64` (ToadStool)
- dN/dS — via `barracuda::ops::bio::dnds::DnDsBatchF64` (ToadStool)
- Pangenome — via `barracuda::ops::bio::pangenome::PangenomeClassifyGpu` (ToadStool)
- Random Forest — via `barracuda::ops::bio::rf_inference::RfBatchInferenceGpu` (ToadStool)
- HMM forward — via `barracuda::ops::bio::hmm::HmmBatchForwardF64` (ToadStool)

This is the GPU analogue of barracuda_cpu_v1-v5: CPU → GPU → same answer.
All 8 bio primitives now flow through ToadStool's absorbed shaders, which
benefit from cross-spring precision evolution (hotSpring f64 polyfills,
neuralSpring eigensolvers).

## metalForge Cross-System Proof

Exp065 extends Exp060 to ALL domains, proving substrate-independence:
for every GPU-eligible algorithm, the metalForge router can dispatch to
CPU or GPU and get the same answer. This is the foundation for CPU/GPU/NPU
routing in production.

## ToadStool Evolution (Feb 22, 2026)

### Write → Absorb → Lean Status

Following hotSpring's pattern for ToadStool integration:

| Phase | Count | Status |
|-------|:-----:|--------|
| **Lean** (consumed upstream) | 20 GPU modules, 23 primitives | Active — 15 original + 8 bio (HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF) |
| **Write** (local WGSL, pending absorption) | 4 shaders (ODE, kmer, unifrac, taxonomy) | ODE blocked: upstream `compile_shader` needs `compile_shader_f64`; others pending validation |
| **CPU math** (`bio::special`) | 3 functions (erf, ln_gamma, regularized_gamma) | Consolidated; shaped for extraction to `barracuda::math` |
| **CPU-only** (no GPU path) | 12 modules | Stable — chimera, derep, GBM, merge_pairs, etc. |
| **Blocked** (needs upstream) | 3 modules | kmer hash, UniFrac tree traversal, taxonomy NPU |
| **metalForge** (absorption eng.) | 32 tolerances, SoA patterns, `#[repr(C)]` | Shaping all modules for ToadStool absorption |

### Feb 22 Rewire: 8 Bio Primitives Absorbed

ToadStool sessions 31d/31g absorbed all 8 wetSpring bio WGSL shaders. On Feb 22,
wetSpring rewired all 8 GPU modules to delegate to `barracuda::ops::bio::*`,
deleted the local shaders (25 KB), and verified 633 tests pass with 0 clippy
warnings. Two ToadStool bugs found and fixed during validation:

1. **SNP binding layout** — `is_variant` (binding 2) was declared `read_only` but
   the shader writes to it; extra phantom binding 6. Fixed in ToadStool `snp.rs`.
2. **AdapterInfo propagation** — wetSpring's `GpuF64::new()` used
   `WgpuDevice::from_existing_simple()` which sets synthetic adapter info, breaking
   ToadStool's RTX 4070 Ada Lovelace detection and f64 exp/log polyfill. Fixed to
   use `WgpuDevice::from_existing()` with real `AdapterInfo`.

### Cross-Spring Evolution

ToadStool `barracuda` v0.2.0 is the convergence hub for three springs:

| Spring | Contribution | Primitives |
|--------|-------------|-----------|
| **hotSpring** | Precision shaders, lattice QCD, spectral theory | Dirac, CG, plaquette, Higgs U(1), SU(3) HMC, Lanczos, Anderson, Hofstadter |
| **wetSpring** | Bio/genomics WGSL shaders, math_f64, Hill kinetics | HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF, SW, Gillespie, Felsenstein |
| **neuralSpring** | ML inference, eigensolvers, TensorSession | Batch IPR, RF inference, HMM forward log, pairwise Hamming/Jaccard, batch fitness, Eigh |

All three springs now lean on the same ToadStool primitives, benefiting from
cross-spring evolution: hotSpring's precision fixes improve wetSpring's bio
shaders; neuralSpring's eigensolver powers wetSpring's PCoA; wetSpring's
bio primitives are available to neuralSpring's metalForge pipeline.

### Streaming & Dispatch Validation (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 072 | `validate_gpu_streaming_pipeline` | 17 | Pre-warmed FMR eliminates per-stage dispatch; 1.22x streaming speedup |
| 073 | `validate_dispatch_overhead_proof` | 21 | Streaming beats individual at all batch sizes; overhead quantified |
| 074 | `validate_substrate_router` | 20 | GPU↔NPU↔CPU routing; PCIe topology; fallback parity |

### ToadStool Bio Rewire (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 077 | (all GPU binaries) | 451 | Full revalidation after 8-module rewire to ToadStool primitives |

Bugs found and fixed: SNP binding layout (ToadStool), AdapterInfo propagation (wetSpring).

### ODE Flat API + Dispatch Routing (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 079 | `validate_barracuda_cpu_v6` | 48 | GPU-compatible flat param APIs preserve bitwise ODE math across all 6 bio models |
| 080 | `validate_dispatch_routing` (forge) | 35 | Forge router correctly classifies 11 workloads across 5 substrate configs |

### Tier B → A Module Refactoring (Feb 22, 2026)

| Exp | Module | Tests | What it proves |
|-----|--------|:-----:|----------------|
| 081 | `kmer` | 4 | Histogram (4^k) + sorted pairs GPU layouts, round-trip fidelity |
| 082 | `unifrac` | 4 | CSR flat tree + sample matrix, UniFrac parity through flat path |
| 083 | `taxonomy` | 3 | Int8 affine quantization, argmax parity with f64 for NPU inference |
| 084 | metalForge full | 35+ | 12-domain cross-substrate (extends Exp065: +SW, Gillespie, DT, spectral) |
| 085 | Tier A layouts | 43 | kmer histogram/sorted-pairs RT, unifrac CSR RT, taxonomy int8 parity |
| 086 | metalForge pipeline | 45 | 5-stage dispatch routing + CPU/NPU parity + flat buffer readiness |
| 087 | GPU Extended Domains | 50+ | EIC, PCoA, Kriging, Rarefaction — 4 new GPU domains (--features gpu) |
| 088 | metalForge PCIe Direct | 32 | 6 paths + buffer contracts (CPU-only binary) |
| 089 | ToadStool Streaming Dispatch | 25 | 5 patterns + determinism (CPU-only binary) |
| 090 | Pure GPU Streaming Pipeline | 80 | 4 modes: round-trip, streaming, parity, batch scaling (--features gpu) |
| 091 | Streaming vs Round-Trip Benchmark | 2 | CPU ↔ RT ↔ streaming parity + Bray-Curtis error (--features gpu) |
| 092 | CPU vs GPU All 16 Domains | 48 | 16 domains CPU↔GPU parity (--features gpu) |
| 093 | metalForge Full v3 (16 domains) | 28 | 16 domains substrate-independent (--features gpu) |

### Handoff Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Tier A shader specs | `../wateringHole/handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md` | Original binding layouts, dispatch geometry |
| Rewire results | `wateringHole/handoffs/WETSPRING_TOADSTOOL_REWIRE_FEB22_2026.md` | Rewire outcomes, bugs, cross-spring evolution |
| Cross-spring evolution | `wateringHole/handoffs/CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md` | wetSpring perspective on biome model |
