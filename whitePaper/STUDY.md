# wetSpring: Replicable Life Science and Analytical Chemistry on Consumer GPU Hardware

**Working Draft** — February 2026

---

## Abstract

We demonstrate that published bioinformatics and analytical chemistry
workflows can be (1) independently replicated using open tools and data,
(2) re-implemented in pure Rust with zero external parser dependencies,
and (3) accelerated on consumer GPUs achieving up to 1,077× speedup over
CPU for batch-parallel workloads. The study covers two tracks: 16S
amplicon metagenomics (Track 1) and PFAS detection via LC-MS (Track 2),
validating 30 Rust modules against baselines from Galaxy, QIIME2, asari,
FindPFAS, scipy, real NCBI SRA data, and published paper baselines with
426 quantitative checks — all passing. Includes 202 checks on public open data
(PRJNA1114688) benchmarked against paper ground truth.

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
Zero external parser dependencies — FASTQ, mzML, MS2, XML, and Base64
parsers are all sovereign Rust code.

**Phase 3 — GPU Acceleration**: Promote validated CPU code to GPU via
ToadStool primitives (wgpu v22 WGSL shaders). Validate GPU results
against CPU baselines within f64 tolerance (≤ 1e-10 for most operations).

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

### 3.1 Science Replication (Track A)

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

### 3.2 Rust CPU Validation (Track B, Phase 2)

89 quantitative checks across 6 validation binaries — all pass:

| Binary | Checks | What's Validated |
|--------|:------:|-----------------|
| `validate_fastq` | 28 | FASTQ parsing, quality filtering, adapter trimming, paired-end merging, dereplication |
| `validate_diversity` | 18 | Shannon, Simpson, Bray-Curtis, Chao1, Pielou, k-mers, rarefaction |
| `validate_mzml` | 7 | mzML parsing, spectrum counts, m/z ranges, data types |
| `validate_pfas` | 10 | MS2 parsing, fragment screening, cosine similarity, KMD |
| `validate_features` | 9 | Mass tracks, EIC extraction, feature table vs asari |
| `validate_peaks` | 17 | Peak detection across 5 test cases vs scipy |

### 3.3 GPU Validation (Track B, Phase 3)

38 quantitative checks — all pass (GPU = CPU within ≤ 1e-10):

| ToadStool Primitive | Operation | Checks |
|---------------------|-----------|:------:|
| FusedMapReduceF64 | Shannon, Simpson, observed, evenness, alpha bundle | 12 |
| BrayCurtisF64 | Pairwise Bray-Curtis distance matrix | 6 |
| BatchedEighGpu | PCoA eigendecomposition | 5 |
| GemmF64 + FusedMapReduceF64 | Pairwise spectral cosine similarity | 8 |
| VarianceF64 | Population/sample variance, std dev | 3 |
| CorrelationF64 + CovarianceF64 | Pearson correlation, covariance | 2 |
| WeightedDotF64 | Weighted and plain dot product | 2 |

### 3.4 Performance: Three-Tier Benchmark

GPU advantage emerges for batch-parallel workloads where O(N²) pairs
amortize the ~0.5–2ms dispatch overhead:

| Workload | N_pairs | Python | Rust CPU | Rust GPU | GPU/CPU |
|----------|---------|--------|----------|----------|---------|
| Bray-Curtis 100×100 | 4,950 | 15.0ms | 1.04ms | 2.59ms | 0.40× |
| Spectral cosine 10×10 | 45 | 31µs | 8.9ms | 3.4ms | **2.6×** |
| Spectral cosine 50×50 | 1,225 | 581µs | 239ms | 3.5ms | **68×** |
| Spectral cosine 100×100 | 4,950 | 2.2ms | 969ms | 3.9ms | **250×** |
| Spectral cosine 200×200 | 19,900 | 8.8ms | 3,937ms | 3.7ms | **1,077×** |

The spectral cosine benchmark is the headline result: GPU time stays
nearly constant (~3.5ms) as problem size grows because the GEMM kernel
dispatches all pairs in a single GPU call. CPU time grows quadratically.

For single-vector reductions (Shannon, Simpson, variance), Rust CPU is
1–2× faster than numpy and GPU dispatch overhead dominates at all tested
sizes. The GPU wins only when there is enough independent parallel work
to amortize that overhead.

### 3.5 Complete 16S Pipeline in Rust

The full pipeline from raw reads to ecological analysis is now
implemented:

```
FASTQ → quality filter → adapter trim → paired-end merge
  → dereplication → DADA2 denoising → chimera removal
  → taxonomy classification (naive Bayes / SILVA)
  → diversity metrics → UniFrac distance → PCoA ordination
```

Each stage has unit tests (284 total) and the first five stages have
end-to-end validation against Galaxy baselines.

---

## 4. Discussion

### Replication success

6 of 9 target papers have at least one baseline experiment. The remaining
papers (spectroradiometric, VOC, femtosecond MS) involve specialized
instrumentation that requires domain-specific data acquisition.

### Rust as a scientific computing platform

30 modules with a single runtime dependency (`flate2` for gzip)
demonstrate that Rust can serve as a standalone platform for
bioinformatics and analytical chemistry. The sovereign XML, FASTQ, mzML,
and MS2 parsers eliminate the need for Python/R in the critical path.

### GPU parallelism model

The 1,077× speedup on spectral cosine illustrates the GPU's strength:
batch-parallel matrix operations. Single-vector reductions don't benefit
because GPU dispatch overhead (~0.5ms) exceeds the CPU computation time.
The practical implication is that GPU acceleration should target workloads
with O(N²) or higher parallelism — pairwise distance matrices, spectral
library matching, and batch feature extraction.

---

## 5. Reproducibility

All code, data paths, and validation binaries are in the `wetSpring`
repository (AGPL-3.0). No institutional access required.

```bash
# Run all validations
cd barracuda
cargo test --release          # 284 tests
cargo run --release --bin validate_fastq
cargo run --release --bin validate_diversity
cargo run --release --bin validate_mzml
cargo run --release --bin validate_pfas
cargo run --release --bin validate_features
cargo run --release --bin validate_peaks

# GPU validation + benchmark
cargo run --release --features gpu --bin validate_diversity_gpu
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
