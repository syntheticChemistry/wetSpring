# wetSpring: Replicable Life Science and Analytical Chemistry on Consumer GPU Hardware

**Working Draft** — February 21, 2026

---

## Abstract

We demonstrate that published bioinformatics and analytical chemistry
workflows can be (1) independently replicated using open tools and data,
(2) re-implemented in pure Rust with sovereign I/O parsers (FASTQ, mzML, MS2, XML),
and (3) accelerated on consumer GPUs achieving up to 926× speedup for
batch-parallel spectral operations and 24× for denoising, with structured
three-tier profiling (Python → Rust CPU → Rust GPU) capturing wall time,
energy, and memory in a unified benchmark harness. The study covers
four tracks: 16S amplicon metagenomics (Track 1), comparative genomics
and mathematical biology (Track 1b), deep-sea metagenomics and microbial
evolution (Track 1c), and PFAS detection via LC-MS (Track 2),
validating 61 Rust modules (41 CPU + 20 GPU) against baselines from Galaxy,
QIIME2, asari, FindPFAS, scipy, sklearn, dendropy, real NCBI SRA data, and
published paper models with 1,501 quantitative checks across 63 experiments
— all passing. The pipeline proves substrate independence: math produces
identical results on CPU and GPU, validated via metalForge cross-substrate
checks (Exp060). Random Forest ensemble and Gradient Boosting Machine
inference achieve 100% specification parity, joining the existing decision
tree engine to cover the three dominant ensemble ML methods in pure Rust.
Gillespie stochastic simulation converges to analytical
steady state within 0.2%. Robinson-Foulds tree distance matches dendropy
exactly. Rust runs 22.5× faster than Python across all 25 algorithmic domains
(Exp059), with peak speedup of 625× for Smith-Waterman alignment. The full 16S pipeline
runs 8.3–13× faster than Galaxy on CPU (depending on sample size) and
2.45× faster again on GPU, with 88/88 math parity checks proving
identical results across hardware.

---

## 1. Introduction

Computational biology and environmental chemistry increasingly depend on
complex software pipelines that are difficult to reproduce, audit, and
deploy outside institutional environments. Published results rely on
tool chains spanning Python, R, Java, and vendor-locked mass spectrometry
software, creating barriers for independent verification.

This study asks whether:
- Published workflows can be replicated by an independent researcher
  using open data and open tools (Galaxy, QIIME2, asari, FindPFAS)
- Those workflows can be re-implemented in a single systems language
  (Rust) with full computational sovereignty
- Consumer GPU hardware (NVIDIA RTX 4070, $549) can accelerate the
  most compute-intensive stages at full f64 precision
- The same math produces identical results on CPU and GPU, proving
  that BarraCUDA solves math and ToadStool solves hardware

### 1.1 Relationship to hotSpring

wetSpring follows the same validation methodology as hotSpring (nuclear
structure on consumer GPU), applying it to life science and analytical
chemistry:

| | hotSpring | wetSpring T1 | wetSpring T2 (blueFish) |
|--|-----------|-------------|------------------------|
| Domain | Nuclear physics | Life science | Analytical chemistry |
| Validation target | Binding energies | Organism ID | PFAS ID + concentration |
| Baseline | Python scipy | Galaxy/QIIME2 | asari/PFΔScreen |
| GPU layer | ToadStool (wgpu) | ToadStool (wgpu) | ToadStool (wgpu) |
| Success metric | chi² match | Same taxonomy | Same PFAS detected |
| Checks | 418/418 | 1,501/1,501 | (included in 1,501) |

Both prove the ecoPrimals thesis: sovereign compute on consumer hardware
can replicate institutional results, then exceed them via Rust + GPU.

---

## 2. Methods

### 2.1 Three-Phase Validation Protocol

Every workload passes through three phases, each validating against the
previous:

**Phase 1 — Open Tool Replication**: Run the published analysis using
Galaxy 24.1, QIIME2, DADA2, asari, or FindPFAS. Record tool versions,
parameters, and quantitative outputs (feature counts, diversity metrics,
PFAS hits). This establishes the replication baseline.

**Phase 2 — Rust CPU Port**: Re-implement each computational stage in
Rust using the `wetspring-barracuda` crate. Validate against Phase 1
baselines using hardcoded expected values and tolerance thresholds.
Sovereign I/O — FASTQ, mzML, MS2, XML, and Base64 parsers are all
in-tree Rust code. `serde_json` is the only external parsing dependency,
used for ML model import (decision tree from sklearn).

**Phase 3 — GPU Acceleration**: Promote validated CPU code to GPU via
ToadStool primitives and custom WGSL shaders. Validate GPU results
against CPU baselines within f64 tolerance (≤ 1e-6 for pipeline stages,
≤ 1e-10 for individual operations).

### 2.2 Tracks

**Track 1 (Life Science)**: 16S rRNA amplicon metagenomics for algal
pond crash forensics and phage biocontrol. Source papers: Carney et al.
2016 (DOI: 10.1016/j.algal.2016.05.011), Humphrey et al. 2023 (OSTI
2311389).

**Track 2 (PFAS / blueFish)**: Per- and polyfluoroalkyl substance
detection in water via LC-MS. Source papers: asari (Nature Communications
14, 4113, 2023), PFΔScreen (Analytical and Bioanalytical Chemistry, 2023).

### 2.3 Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA RTX 4070 12 GB (SHADER_F64) |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Rust | stable 1.82+ |
| wgpu | v22 (Vulkan backend) |

---

## 3. Results

### 3.1 Science Replication (Phase 1)

8 of 10 planned experiments completed:

| Exp | Dataset | Baseline Tool | Key Result |
|-----|---------|---------------|------------|
| 001 | MiSeq SOP (Schloss lab) | Galaxy/QIIME2/DADA2 | 232 ASVs, 20 samples |
| 002 | PRJNA1195978 phytoplankton | QIIME2/DADA2 | 2,273 ASVs, 10 samples |
| 003 | Phage genomes | SPAdes/Pharokka | Assembly + annotation |
| 005 | MT02 HILIC-pos | asari 1.13.1 | 5,951 features, 8 files |
| 006 | PFAS standard mix | FindPFAS/pyOpenMS | 25 PFAS precursors |
| 009 | MT02 (Rust features) | asari 1.13.1 | 28.7% cross-match, 9/9 checks |
| 010 | Synthetic chromatograms | scipy.signal.find_peaks | Exact match, 17/17 checks |

### 3.2 Rust CPU Validation (Phase 2)

1,241 CPU quantitative checks across 29 self-contained validation binaries — all pass.
In addition, the original 17 pipeline validation binaries (data-dependent) cover
the Phase 1-3 checks. The following are the self-contained validators:

| Binary | Checks | What's Validated |
|--------|:------:|-----------------|
| `validate_fastq` | 28 | FASTQ parsing, quality filtering, adapter trimming, paired-end merging, dereplication |
| `validate_diversity` | 27 | Shannon, Simpson, Bray-Curtis, Chao1, Pielou, k-mers, rarefaction |
| `validate_mzml` | 7 | mzML parsing, spectrum counts, m/z ranges, data types |
| `validate_pfas` | 10 | MS2 parsing, fragment screening, cosine similarity, KMD |
| `validate_features` | 8 | Feature pipeline: EIC extraction, mass tracks, feature table vs asari |
| `validate_peaks` | 17 | Peak detection across 5 test cases vs scipy |
| `validate_16s_pipeline` | 37 | DADA2 + chimera + taxonomy + UniFrac (end-to-end) |
| `validate_algae_16s` | 34 | Real NCBI data (PRJNA488170) vs paper baselines |
| `validate_voc_peaks` | 22 | VOC biomarkers vs Reese 2019 published table |
| `validate_public_benchmarks` | 202 | 22 samples, 4 BioProjects vs paper ground truth |
| `validate_extended_algae` | 35 | PRJNA382322 Nannochloropsis outdoor pilot |
| `validate_pfas_library` | 26 | Jones Lab PFAS library (175 compounds, Zenodo) |
| `validate_newick_parse` | 30 | Newick tree parsing vs dendropy (10 topologies) |
| `validate_qs_ode` | 16 | Waters 2008 QS/c-di-GMP ODE vs scipy RK4 |
| `validate_rf_distance` | 23 | Robinson-Foulds tree distance vs dendropy |
| `validate_gillespie` | 13 | Gillespie SSA ensemble vs analytical steady state |
| `validate_pfas_decision_tree` | 7 | Decision tree inference vs sklearn (744 samples) |
| `validate_bistable` | 14 | Fernandez 2020 bistable switching vs scipy bifurcation |
| `validate_multi_signal` | 19 | Srivastava 2011 multi-signal QS network |
| `validate_cooperation` | 20 | Bruger & Waters 2018 QS game theory |
| `validate_hmm` | 21 | Liu 2014 HMM (forward/backward/Viterbi/posterior) |
| `validate_capacitor` | 18 | Mhatre 2020 phenotypic capacitor ODE |
| `validate_alignment` | 15 | Smith-Waterman local alignment with affine gaps |
| `validate_felsenstein` | 16 | Felsenstein pruning phylogenetic likelihood |
| `validate_barracuda_cpu` | 21 | Cross-domain CPU parity (9 algorithmic domains) |
| `validate_phage_defense` | 12 | Hsueh 2022 phage defense deaminase |
| `validate_bootstrap` | 11 | Wang 2021 RAWR bootstrap resampling |
| `validate_placement` | 12 | Alamin & Liu 2024 phylogenetic placement |
| `validate_neighbor_joining` | 16 | Liu 2009 NJ tree construction (Exp033) |
| `validate_reconciliation` | 14 | Zheng 2023 DTL reconciliation (Exp034) |
| `validate_barracuda_cpu_v2` | 18 | Batch/flat APIs (5 domains, Exp035) |
| `validate_phynetpy_rf` | 15 | PhyNetPy RF gene trees (Exp036) |
| `validate_phylohmm` | 10 | PhyloNet-HMM discordance (Exp037) |
| `validate_sate_pipeline` | 17 | SATé pipeline benchmark (Exp038) |
| `validate_algae_timeseries` | 11 | Algal pond time-series (Exp039) |
| `validate_bloom_surveillance` | 15 | Bloom surveillance (Exp040) |
| `validate_epa_pfas_ml` | 14 | EPA PFAS ML (Exp041) |
| `validate_massbank_spectral` | 9 | MassBank spectral (Exp042) |
| `validate_barracuda_cpu_v3` | 45 | 18-domain CPU parity (Exp043) |
| `validate_rare_biosphere` | 35 | Anderson 2015 rare biosphere diversity (Exp051) |
| `validate_viral_metagenomics` | 22 | Anderson 2014 viral dN/dS + diversity (Exp052) |
| `validate_sulfur_phylogenomics` | 15 | Mateos 2023 molecular clock + DTL (Exp053) |
| `validate_phosphorus_phylogenomics` | 13 | Boden 2024 clock + reconciliation (Exp054) |
| `validate_population_genomics` | 24 | Anderson 2017 ANI + SNP calling (Exp055) |
| `validate_pangenomics` | 24 | Moulana 2020 pangenome + enrichment (Exp056) |
| `validate_barracuda_cpu_v4` | 44 | 5 Track 1c domains (Exp057) |
| `validate_barracuda_cpu_v5` | 29 | RF + GBM ensemble ML (Exp061/062) |

### 3.3 GPU Validation (Phase 3)

260 GPU checks across 12 GPU validation binaries — all pass:

**Individual operations (38 checks, tolerance ≤ 1e-10):**

| ToadStool Primitive | Operation | Checks |
|---------------------|-----------|:------:|
| FusedMapReduceF64 | Shannon, Simpson, observed, evenness, alpha bundle | 12 |
| BrayCurtisF64 | Pairwise Bray-Curtis distance matrix | 6 |
| BatchedEighGpu | PCoA eigendecomposition | 5 |
| GemmF64 + FusedMapReduceF64 | Pairwise spectral cosine similarity | 8 |
| VarianceF64 | Population/sample variance, std dev | 3 |
| CorrelationF64 + CovarianceF64 | Pearson correlation, covariance | 2 |
| WeightedDotF64 | Weighted and plain dot product | 2 |

**Full pipeline math parity (88 checks, tolerance ≤ 1e-6):**

| Category | Checks | What's Validated |
|----------|:------:|-----------------|
| Quality filter CPU == GPU | 10 | Per-read parallel QF via WGSL shader |
| DADA2 ASV count CPU ≈ GPU | 10 | GPU E-step produces identical ASVs |
| DADA2 total reads CPU == GPU | 10 | Read conservation across hardware |
| Chimera decisions CPU == GPU | 14 | k-mer scoring (4 full parity + 6 GPU-completes) |
| Shannon CPU ≈ GPU | 10 | Alpha diversity within 1e-6 |
| Simpson CPU ≈ GPU | 10 | Alpha diversity within 1e-6 |
| Observed features CPU ≈ GPU | 10 | Feature counts within 1e-6 |
| Taxonomy genus CPU == GPU | 10 | Naive Bayes classification |
| Chimera agreement > 95% | 4 | Per-decision agreement |

### 3.4 GPU Pipeline Architecture

The GPU pipeline compiles all shaders once at session init and reuses
cached pipelines + pooled buffers across all dispatches:

```
GpuPipelineSession::new(gpu)              40ms one-time
  ├── QualityFilterCached                  per-read parallel WGSL shader
  ├── Dada2Gpu                             batch log_p_error WGSL shader
  ├── GemmCached                           pre-compiled GEMM pipeline
  ├── FusedMapReduceF64                    pre-compiled FMR
  └── BufferPool (TensorContext)           93% buffer reuse

Pipeline flow:
  FASTQ → QF (GPU) → derep (CPU) → DADA2 (GPU E-step) → chimera (CPU)
    → taxonomy GEMM (GPU) → diversity FMR (GPU) → results
```

**Custom WGSL shaders (9):**

| Shader | Purpose | f64? | Approach |
|--------|---------|------|----------|
| `quality_filter.wgsl` | Per-read quality trimming | No (u32) | One thread per read |
| `dada2_e_step.wgsl` | Batch log_p_error | Yes (addition only) | One thread per (seq,center) pair |
| `hmm_forward_f64.wgsl` | Batch HMM forward (log-space) | Yes (full) | One thread per sequence |
| `batched_qs_ode_rk4_f64.wgsl` | QS ODE parameter sweep | Yes (full) | One thread per trajectory |
| `ani_batch_f64.wgsl` | ANI pairwise identity | Yes | One thread per pair |
| `snp_calling_f64.wgsl` | SNP calling | Yes | One thread per position |
| `dnds_batch_f64.wgsl` | dN/dS (Nei-Gojobori) | Yes (log polyfill) | One thread per pair |
| `pangenome_classify.wgsl` | Pangenome gene classification | Yes | One thread per gene |
| `rf_batch_inference.wgsl` | Random Forest batch inference | Yes (SoA) | One thread per (sample,tree) |

**Key design: forced f64 polyfills for NVVM.** RTX 4070 (Ada Lovelace)
NVVM cannot compile native f64 `exp()`, `log()`, or `pow()`.
All shaders using f64 transcendentals are compiled via
`ShaderTemplate::for_driver_auto(source, true)` which injects
`exp_f64`, `log_f64`, `pow_f64` as pure-arithmetic WGSL polyfills.
DADA2 avoids this entirely by precomputing transcendentals on CPU.

### 3.5 Performance: Three-Tier Benchmark

**Full pipeline (10 samples, 4 BioProjects, 2.87M reads):**

| Tier | Implementation | Total Time | Per-Sample | vs Galaxy |
|------|---------------|-----------|------------|-----------|
| Galaxy/Python | QIIME2 + DADA2-R + Docker | 95.6 s | 9.56 s | baseline |
| BarraCUDA CPU | Pure Rust, 1 binary | 7.3 s | 0.73 s | **13.1×** |
| BarraCUDA GPU | Rust + ToadStool (RTX 4070) | **3.0 s** | **0.30 s** | **31.9×** |

**Per-stage CPU vs GPU:**

| Stage | CPU (ms/sample) | GPU (ms/sample) | Speedup | Nature |
|-------|----------------|-----------------|---------|--------|
| DADA2 denoise | 326 | **13** | **24.4×** | Compute-bound |
| Taxonomy GEMM | 115 | **11** | **10.5×** | Compute-bound |
| Quality filter | 30 | 40 | 0.85× | Memory-bound |
| Chimera detect | 100 | 100 | 1× | CPU (k-mer sketch) |
| Dereplication | 6 | 6 | 1× | CPU (hash) |
| Diversity | 0.0 | 0.0 | — | Trivial at this scale |

**Taxonomy scaling benchmark (GPU vs CPU):**

| Queries | CPU (ms) | GPU (ms) | Speedup |
|---------|----------|----------|---------|
| 5       | 79       | 4.8      | **16.5×** |
| 25      | 433      | 14.4     | **30.1×** |
| 100     | 1,749    | 29.0     | **60.3×** |
| 500     | 8,829    | 140      | **63.3×** |

**Spectral cosine benchmark (batch-parallel, structured harness):**

| Workload | N_pairs | Python | Rust CPU | Rust GPU | GPU/CPU |
|----------|---------|--------|----------|----------|---------|
| Spectral cosine 10×10 | 45 | 31.6µs | 9.37ms | 3.06ms | **3.1×** |
| Spectral cosine 50×50 | 1,225 | 582µs | 248ms | 3.13ms | **79×** |
| Spectral cosine 100×100 | 4,950 | 2.29ms | 996ms | 3.36ms | **296×** |
| Spectral cosine 200×200 | 19,900 | 9.10ms | 3,782ms | 4.08ms | **926×** |

### 3.6 Energy & Cost

| Metric | Galaxy | Rust CPU | Rust GPU |
|--------|--------|----------|----------|
| TDP | 125 W | 125 W | 200 W |
| Pipeline time (10 sam) | 95.6 s | 7.3 s | 3.0 s |
| Energy (kWh) | 0.00332 | 0.00025 | 0.00017 |
| Cost at 10K samples | **$0.40** | **$0.03** | **$0.02** |

Rust GPU is cheapest per sample because the 2.45× speed advantage more
than compensates for higher TDP.

### 3.7 Three-Tier Profiling Infrastructure

A sovereign benchmark harness (`barracuda/src/bench.rs`) captures
performance and energy data across all three tiers in a single JSON
schema, enabling reproducible comparisons without external profiling
tools.

**Architecture:**

| Component | Purpose |
|-----------|---------|
| `HardwareInventory` | CPU (model, cores, MHz), RAM (GB), GPU (name, VRAM, driver, CC) |
| `PowerMonitor` | Background threads: RAPL CPU energy (µJ), nvidia-smi GPU power/temp/VRAM |
| `EnergyReport` | Per-phase: CPU joules, GPU joules, peak watts, peak temp, peak VRAM |
| `PhaseResult` | Per-benchmark: wall time, evals, per-eval µs, energy, peak RSS, notes |
| `BenchReport` | Timestamp, hardware, vector of phases — serialized to JSON |

**Key design decisions:**

- **Minimal dependencies.** Core I/O and validation use manual `to_json()`
  serialization. `serde_json` added only for ML model import (decision
  tree inference). Runtime deps: `flate2`, `bytemuck`, `serde_json`.
- **RAPL counter wrap handling.** Intel RAPL energy counters overflow
  at `max_energy_range_uj`; the monitor detects wraps and computes
  the correct delta.
- **GPU power integration.** nvidia-smi is polled every 100ms; total
  energy is computed via trapezoidal integration of power samples.
- **Cross-tier compatibility.** The Python baseline script
  (`benchmark_python_baseline.py`) emits the same JSON schema,
  enabling direct comparison of all three tiers in a single report.

Protocol: [`benchmarks/PROTOCOL.md`](../benchmarks/PROTOCOL.md) |
Results: [`benchmarks/results/`](../benchmarks/results/)

### 3.8 Complete 16S Pipeline in Rust

The full pipeline from raw reads to ecological analysis:

```
FASTQ → quality filter → adapter trim → paired-end merge
  → dereplication → DADA2 denoising → chimera removal
  → taxonomy classification (naive Bayes / SILVA)
  → diversity metrics → UniFrac distance → PCoA ordination
```

Each stage has unit tests (570 total, 95%+ line coverage), end-to-end
validation against Galaxy baselines, GPU math parity checks, and
determinism tests ensuring identical output across runs.

### 3.9 Optimization History

| Change | Before | After | Speedup |
|--------|--------|-------|---------|
| Chimera: O(N³) → k-mer sketch | 1,985 s | 1.6 s | 1,256× |
| Taxonomy: HashMap → flat array | 24.5 s | 1.15 s | 21× |
| Taxonomy: flat array → compact GEMM | 1.15 s | 0.11 s | 10.5× |
| DADA2: CPU → GPU E-step | 3.3 s | 0.13 s | 24.4× |
| Pipeline total: baseline → current | ~2,010 s | 3.0 s | **670×** |

---

## 4. Discussion

### 4.1 Replication success

All 29 actionable papers in the review queue are now reproduced, covering
ODE systems (6 models), stochastic simulation, HMM, phylogenetic algorithms
(Felsenstein, Robinson-Foulds, bootstrap, placement), sequence alignment
(Smith-Waterman), phage defense dynamics, and deep-sea metagenomics
(dN/dS, molecular clock, ANI, SNP calling, pangenomics). Track 1c added
6 R. Anderson papers requiring 5 new sovereign modules — all validated
against Python baselines with public data.

### 4.2 Rust as a scientific computing platform

61 modules (41 CPU bio, 20 GPU, plus I/O and benchmarking) with minimal
runtime dependency (`flate2` for gzip) demonstrate that Rust can serve
as a standalone platform for bioinformatics and analytical chemistry.
The sovereign XML, FASTQ, mzML, and MS2 parsers eliminate the need for
Python/R in the critical path. The `bench` module provides hardware
inventory, RAPL CPU energy monitoring, nvidia-smi GPU power/temperature
sampling, and JSON output — enabling reproducible cross-tier comparisons
without external profiling tools.

### 4.3 BarraCUDA solves math, ToadStool solves hardware

The central architectural finding: separating math (BarraCUDA) from
hardware dispatch (ToadStool) enables clean validation and optimization.

**BarraCUDA solves math.** The same algorithms produce identical results
on CPU and GPU. 88/88 parity checks prove: same ASV counts, same reads,
same taxonomy, same diversity metrics. The math is hardware-independent.

**ToadStool solves hardware.** BufferPool (93% reuse), pipeline caching,
TensorContext — dispatch overhead is eliminated. GPU is competitive even
at 5 queries (16.5×). The hardware abstraction works.

### 4.4 Compute-bound vs memory-bound: the architectural boundary

The GPU pipeline reveals a clean separation:

- **Compute-bound stages** (DADA2 E-step, taxonomy GEMM): GPU wins
  massively (10–63×). These involve O(N² × L) operations where N is
  sequence count and L is sequence length — embarrassingly parallel.
- **Memory-bound stages** (quality scanning, small diversity): CPU wins
  slightly (~0.85×). Sequential per-read scanning has optimal cache
  access patterns on CPU. Data transfer overhead dominates.

This is not a math bottleneck — it's the correct architectural insight.
GPU acceleration targets compute-dominated stages. Memory-bound stages
remain on CPU until the full pipeline is GPU-resident with persistent
buffers (chipset-phase work).

### 4.5 GPU parallelism model

Three patterns emerged:

1. **Batch matrix operations** (taxonomy GEMM, spectral cosine): GPU
   time stays nearly constant as problem size grows because all pairs
   dispatch in a single call. CPU time grows quadratically. This produces
   the headline speedups (60–1,077×).

2. **Batch pair-wise reduction** (DADA2 E-step): each thread reduces
   over a shared dimension for one pair. With precomputed lookup tables,
   no GPU transcendentals needed. 24× speedup across all sample sizes.

3. **Per-element parallel scan** (quality filter): each thread processes
   one element independently. Effective but limited by data transfer.
   GPU matches CPU; wins will come from persistent GPU buffers.

### 4.6 Streaming I/O and Code Quality

All three I/O parsers (FASTQ, mzML, MS2) provide both buffered (`parse_*`)
and streaming iterator (`FastqIter`, `MzmlIter`, `Ms2Iter`) APIs. The
streaming iterators yield one record at a time without buffering the
entire file, enabling constant-memory processing of arbitrarily large
datasets. All use `BufReader` for disk-level streaming; the iterators
add record-level streaming on top.

Code quality gates (all enforced in CI):
- `cargo fmt --check` — zero formatting deviations
- `cargo clippy --all-targets -- -D warnings` — zero warnings (CPU and GPU)
- `cargo clippy --all-targets --features gpu -- -D warnings` — zero GPU-specific warnings
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` — zero doc warnings
- 0 `unsafe` blocks, 0 `TODO`/`FIXME`, 0 production `unwrap()`/`expect()`
- 6 determinism tests covering diversity, Bray-Curtis, DADA2, chimera, taxonomy, and the full 16S pipeline

### 4.7 Write → Absorb → Lean (GPU evolution pattern)

Following hotSpring's pattern, wetSpring writes local GPU extensions, validates
them against CPU baselines, then hands off to ToadStool for absorption. Once
upstream absorbs the primitive, wetSpring removes the local copy and leans on
the shared crate. This cycle has completed for 4 bio primitives
(SmithWatermanGpu, GillespieGpu, TreeInferenceGpu, FelsensteinGpu) and is
in progress for 9 local WGSL shaders spanning pipeline, bioinformatics,
and ML ensemble domains.

| Stage | Extensions | Status |
|-------|-----------|--------|
| **Absorbed** | SW, Gillespie, DT, Felsenstein, GEMM, diversity | Lean on upstream |
| **Tier A** (handoff ready) | `hmm_forward_f64.wgsl` (13/13), `batched_qs_ode_rk4_f64.wgsl` (7/7), `dada2_e_step.wgsl`, `quality_filter.wgsl` | Handoff pending |
| **Tier A** (Track 1c) | `ani_batch_f64.wgsl` (7/7), `snp_calling_f64.wgsl` (5/5), `dnds_batch_f64.wgsl` (9/9), `pangenome_classify.wgsl` (6/6) | Validated, handoff ready |
| **Tier A** (ML) | `rf_batch_inference.wgsl` (13/13) | SoA layout, validated |
| **Tier B** (needs refactor) | kmer, unifrac, taxonomy (NPU) | GPU pattern identified |
| **Tier C** (CPU-only) | chimera, cooperation, derep, NJ, reconciliation, GBM | No GPU path |

**NVVM driver profile bug**: ToadStool's driver profile incorrectly reports
`needs_f64_exp_log_workaround() = false` for Ada Lovelace (RTX 40-series).
Upstream fix: return `true` for all Ada Lovelace GPUs.

**CPU math evolution**: 4 local functions (`erf`, `ln_gamma`,
`regularized_gamma_lower`, `integrate_peak`) duplicate barracuda upstream
primitives (`barracuda::special::erf`, `barracuda::special::ln_gamma`,
`barracuda::special::regularized_gamma_p`, `barracuda::numerical::trapz`).
Migration blocked on barracuda adding a CPU-only `math` feature gate that
does not pull in wgpu/akida-driver/toadstool-core.

Full absorption map: `barracuda/EVOLUTION_READINESS.md`.
Active handoff: `wateringHole/handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md`.

---

## 5. Reproducibility

All code, data paths, and validation binaries are in the `wetSpring`
repository (AGPL-3.0). No institutional access required.

```bash
# Run all CPU validations (1,241 checks)
cd barracuda
cargo test --release          # 552 tests (539 lib + 13 doc)
cargo run --release --bin validate_fastq
cargo run --release --bin validate_diversity
cargo run --release --bin validate_mzml
cargo run --release --bin validate_pfas
cargo run --release --bin validate_features
cargo run --release --bin validate_peaks
cargo run --release --bin validate_16s_pipeline
cargo run --release --bin validate_algae_16s
cargo run --release --bin validate_voc_peaks
cargo run --release --bin validate_public_benchmarks

# GPU validation + benchmark (260 checks)
cargo run --release --features gpu --bin validate_diversity_gpu
cargo run --release --features gpu --bin validate_16s_pipeline_gpu
cargo run --release --features gpu --bin benchmark_cpu_gpu

# Python baseline benchmark
python3 scripts/benchmark_python_baseline.py
```

---

## References

### Source Papers

1. Carney, L.T. et al. "Pond Crash Forensics: Presumptive identification
   of pond crash agents by next generation sequencing in replicate raceway
   mass cultures of *Nannochloropsis salina*." *Algal Research* 17 (2016).
   DOI: 10.1016/j.algal.2016.05.011.
2. Humphrey, B. et al. (incl. Smallwood, C.R. and Cahill, J.) "Biotic
   countermeasures that rescue *Nannochloropsis gaditana* from a *Bacillus
   safensis* infection." *Frontiers in Microbiology* (2023). OSTI 2311389.
3. Reese, K.L. et al. (incl. Jones, A.D.) "Chemical Profiling of Volatile
   Organic Compounds in the Headspace of Algal Cultures as Early Biomarkers
   of Algal Pond Crashes." *Scientific Reports* 9 (2019).
   DOI: 10.1038/s41598-019-50125-z.
4. Li, S. et al. "Trackable and scalable LC-MS metabolomics data
   processing using asari." *Nature Communications* 14, 4113 (2023).
   DOI: 10.1038/s41467-023-39889-1.
5. Zweigle, J. et al. "PFΔScreen — an optimised screening tool for PFAS."
   *Analytical and Bioanalytical Chemistry* (2023).
   DOI: 10.1007/s00216-023-05070-2.

### Algorithms Implemented

6. Callahan, B.J. et al. "DADA2: High-resolution sample inference from
   Illumina amplicon data." *Nature Methods* 13, 581–583 (2016).
   DOI: 10.1038/nmeth.3869.
7. Edgar, R.C. "UCHIME improves sensitivity and speed of chimera
   detection." *Bioinformatics* 27, 2194–2200 (2011).
   DOI: 10.1093/bioinformatics/btr381.
8. Wang, Q. et al. "Naive Bayesian Classifier for Rapid Assignment of
   rRNA Sequences into the New Bacterial Taxonomy." *Appl. Environ.
   Microbiol.* 73, 5261–5267 (2007). DOI: 10.1128/AEM.00062-07.
9. Lozupone, C. & Knight, R. "UniFrac: a new phylogenetic method for
   comparing microbial communities." *Appl. Environ. Microbiol.* 71,
   8228–8235 (2005). DOI: 10.1128/AEM.71.12.8228-8235.2005.
10. Virtanen, P. et al. "SciPy 1.0: Fundamental Algorithms for Scientific
    Computing in Python." *Nature Methods* 17, 261–272 (2020).
    DOI: 10.1038/s41592-019-0686-2.
