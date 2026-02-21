# Validation Methodology

**Purpose**: Document the two-track validation approach used in the wetSpring study
**See also**: [README.md](README.md) for key results

---

## 1. Two-Track Approach

wetSpring validates two complementary scientific domains that exercise different
computational primitives. Each track follows the same three-phase evolution path.

### Track 1: Life Science (Algae / Metagenomics)

Reproduce published 16S rRNA amplicon microbiome profiling for algae pond
management. This exercises: FASTQ parsing, quality filtering, k-mer counting,
diversity metrics (Shannon, Simpson, Chao1), beta diversity (Bray-Curtis),
and ordination (PCoA).

**Source papers**: Pond Crash Forensics; Biotic Countermeasures (Sandia National Labs)

### Track 1b: Comparative Genomics / Mathematical Biology

Reproduce published mathematical models and phylogenetic algorithms from the
Liu Lab (CMSE, MSU), Waters Lab, and collaborators. This exercises: Newick tree
parsing, Robinson-Foulds distance, RK4 ODE integration, Gillespie stochastic
simulation, decision tree inference, HMM (forward/backward/Viterbi/posterior),
Smith-Waterman local alignment, Felsenstein pruning phylogenetic likelihood,
RAWR bootstrap resampling, phylogenetic placement, bistable phenotypic
switching, multi-signal QS networks, game-theoretic cooperation, phenotypic
capacitor models, and phage defense dynamics.

**Source papers**: Waters 2008, Massie 2012, Liu 2014, Fernandez 2020,
Srivastava 2011, Bruger & Waters 2018, Mhatre 2020, Hsueh/Severin 2022,
Wang 2021 (RAWR), Alamin & Liu 2024

### Track 1c: Deep-Sea Metagenomics & Microbial Evolution

Reproduce published metagenomic, population genomic, and pangenomic analyses
from deep-sea hydrothermal vent systems. This exercises: average nucleotide
identity (ANI), single nucleotide polymorphism (SNP) calling, pairwise dN/dS
estimation (Nei-Gojobori 1986), strict and relaxed molecular clock calibration,
pangenome analysis (core/accessory/unique gene partitioning), Heap's law
(pangenome openness), hypergeometric enrichment testing, and Benjamini-Hochberg
FDR correction.

**Source papers**: Anderson et al. (2014, 2015, 2017), Mateos et al. (2023),
Boden et al. (2024), Moulana et al. (2020)

### Track 2: PFAS Analytical Chemistry (codename blueFish)

Reproduce published PFAS non-target screening from LC-MS/HRMS data. This
exercises: mzML/MS2 parsing, peak detection, mass defect analysis, suspect
screening, and homologous series detection.

**Source papers**: Jones et al. (MSU), Michigan PFAS monitoring studies

### Why Four Tracks

| Track | Computational Primitives | GPU Shader Value |
|-------|-------------------------|------------------|
| Track 1 (Life Science) | Sequence I/O, k-mer ops, diversity metrics | Hash tables, reduce, distance matrices |
| Track 1b (Comp. Genomics) | Tree traversal, alignment, HMM, ODE | Felsenstein, SW, batch forward, ODE sweep |
| Track 1c (Metagenomics) | ANI, SNP, dN/dS, clock, pangenome | Pairwise alignment, batch comparison |
| Track 2 (Analytical Chemistry) | Spectral I/O, peak detection, mass math | Signal processing, tolerance search, spectral correlation |

All tracks produce GPU kernels useful **far beyond** their original domain —
the ecoPrimals convergent evolution thesis in action.

---

## 2. Three-Phase Evolution

### Phase 1: Python / Galaxy Control

Run the published analysis using the original tools on our hardware. This
establishes:

- **Correctness**: Our hardware produces results matching published outputs
- **Performance baseline**: Traditional Python scientific stack
- **Bug inventory**: Upstream issues found during reproduction
- **Provenance**: Exact tool versions, dataset accessions, commands

Phase 1 uses Galaxy (for bioinformatics tools), QIIME2 (for diversity),
asari (for LC-MS), and `PFΔScreen` (for PFAS screening).

### Phase 2: Rust Validation

Re-implement critical I/O and compute stages in sovereign Rust. Compare:

- **Accuracy**: Output matches Phase 1 within documented tolerances
- **Dependencies**: Minimize external crates (target: zero parser dependencies)
- **Streaming**: Zero-copy / streaming I/O where possible
- **Determinism**: Same input → identical output, always

### Phase 3: GPU Acceleration

Promote validated Rust code to WGSL f64 shaders via `ToadStool` / `BarraCUDA`.
Compare:

- **Accuracy**: GPU results match CPU within `GPU_VS_CPU_*` tolerances
- **Throughput**: GPU vs CPU wall time for identical workloads
- **Precision**: IEEE 754 f64 via `SHADER_F64` — no f32 compromise
- **Vendor neutrality**: Vulkan backend, any GPU with f64 support

---

## 3. Workloads

### 3.1 FASTQ Parsing and Quality Assessment (Track 1)

**Tool**: `io::fastq` (sovereign, gzip-aware streaming parser)
**Validates against**: `FastQC` quality reports from Galaxy

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Record count | Exact match | 0 | Deterministic count |
| GC content | Percentage | ± 0.5% | Rounding differences |
| Mean quality score | Phred average | ± 0.5 | Float precision |
| Min/max length | Exact match | 0 | Deterministic |

### 3.2 Mass Spectrometry Parsing (Track 2)

**Tools**: `io::mzml` + `io::xml` (sovereign XML pull parser), `io::ms2`
**Validates against**: pyteomics spectrum extraction

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Spectrum count | Exact match | 0 | Deterministic |
| m/z values | Float comparison | ± 0.01 | IEEE rounding |
| Intensity values | Float comparison | ± 0.01 | IEEE rounding |
| MS level | Exact match | 0 | Integer attribute |

### 3.3 K-mer Counting (Track 1)

**Tool**: `bio::kmer` (2-bit canonical k-mer engine)
**Validates against**: DADA2 dereplication counts

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Unique k-mers | Exact match | 0 | Deterministic |
| Total k-mers | Exact match | 0 | Deterministic |
| Canonical equivalence | Verified | — | Forward == reverse complement |

### 3.4 Diversity Metrics (Track 1 — CPU and GPU)

**Tools**: `bio::diversity` (CPU), `bio::diversity_gpu` (GPU WGSL shaders)
**Validates against**: scikit-bio analytical formulas

#### CPU Validation

| Metric | Formula | Tolerance | Rationale |
|--------|---------|-----------|-----------|
| Shannon | `-Σ p_i ln(p_i)` | `1e-12` | Analytical f64 |
| Simpson | `Σ p_i²` | `1e-12` | Analytical f64 |
| Chao1 | `S + f1(f1-1)/(2(f2+1))` | `1e-12` | Analytical f64 |
| Observed features | Count of non-zero | 0 | Integer |
| Bray-Curtis | `Σ|a-b| / Σ(a+b)` | `1e-15` | Symmetry check |

#### GPU vs CPU Validation

| Metric | Primitive / Shader | Tolerance | Rationale |
|--------|-------------------|-----------|-----------|
| Shannon | `FusedMapReduceF64` (ToadStool) | `1e-10` | Transcendental `log_f64` in WGSL |
| Simpson | `FusedMapReduceF64` (ToadStool) | `1e-6` | Pure arithmetic, generous margin |
| Bray-Curtis | `BrayCurtisF64` (ToadStool, absorbed) | `1e-10` | f64 absolute value and division |
| PCoA eigenvalues | `BatchedEighGpu` (ToadStool) | `1e-6` | Jacobi rotation convergence |
| PCoA distances | Reconstructed from eigenvectors | `1e-6` | Accumulated eigenvector error |
| PCoA proportions | Computed from eigenvalues | `1e-6` | Division by positive sum |

The GPU tolerances are deliberately looser than CPU because:
1. WGSL f64 may use fused multiply-add differently than CPU
2. `log_f64` is a polynomial approximation (not hardware `log`)
3. Workgroup-level reduction ordering may differ from sequential
4. Jacobi eigendecomposition convergence introduces small residuals

### 3.5 PFAS Fragment Screening (Track 2)

**Tool**: `bio::tolerance_search` (ppm/Da tolerance search)
**Validates against**: ismembertolerance (MATLAB equivalent)

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Fragment matches | ppm window | ± 10 ppm | Standard HRMS |
| Known PFAS fragments | Detected count | 0 | Exact match |
| False positive rate | Specificity | 0 | No spurious matches |

### 3.6 Feature Extraction Pipeline (Track 2)

**Tools**: `bio::eic`, `bio::signal`, `bio::feature_table`
**Validates against**: asari 1.13.1 feature table (MT02 demo)

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Mass track count | Count | ± 50% | Different m/z binning strategies |
| Peak count | Count | ± 50% | Different prominence thresholds |
| Feature count | Count | ± 50% | Different filtering criteria |
| m/z range | Float range | ± 10% | Different mass accuracy handling |
| RT range | Float range | ± 10% | Different retention time alignment |
| Cross-match | Fraction | ≥ 20% | 10 ppm m/z + 0.5 min RT window |

### 3.7 Peak Detection (Cross-Track)

**Tool**: `bio::signal::find_peaks`
**Validates against**: `scipy.signal.find_peaks` on synthetic chromatograms

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Peak count | Exact match | 0 | Identical algorithm, synthetic data |
| Peak indices | Position | ± 1 | Interpolation differences at edges |
| Peak heights | Float | ± 1% | Precision of prominence computation |

### 3.8 16S Amplicon Pipeline (Track 1)

**Tools**: `bio::dada2`, `bio::chimera`, `bio::taxonomy`, `bio::unifrac`
**Validates against**: QIIME2/DADA2 pipeline from Galaxy (Exp001)

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| ASV count | Count | — | Algorithm-specific (Poisson p-value) |
| Chimera count | Count | — | UCHIME scoring threshold |
| Taxonomy classification | Confidence | — | Bootstrap proportion |
| UniFrac distance | Float | `1e-10` | Branch-length weighted sums |

### 3.9 GPU Performance Benchmark

**Tool**: `benchmark_cpu_gpu` binary
**Validates**: GPU throughput advantage for batch-parallel workloads

| Workload | Expected GPU Advantage | Why |
|----------|:---------------------:|-----|
| Single-vector reduction (Shannon, Simpson) | < 1× | GPU dispatch overhead exceeds CPU time |
| Pairwise Bray-Curtis N×N | > 1× at N > 200 | O(N²) pairs amortize dispatch |
| Spectral cosine N×N | > 100× at N > 100 | GEMM dispatches all pairs at once |
| PCoA (eigendecomposition) | > 1× at N > 200 | Matrix operations benefit from parallelism |

### 3.10 Three-Tier Performance Benchmark

**Tool**: `bench.rs` (sovereign harness) + `benchmark_python_baseline.py`
**Protocol**: [`benchmarks/PROTOCOL.md`](../benchmarks/PROTOCOL.md)

The benchmark harness captures wall time, energy, and memory across all
three tiers (Python, Rust CPU, Rust GPU) in a unified JSON schema:

| Component | Measurement |
|-----------|-------------|
| `HardwareInventory` | CPU model/cores/MHz, RAM GB, GPU name/VRAM/driver/CC |
| `PowerMonitor` | RAPL CPU energy (µJ) via `/sys/class/powercap/`, nvidia-smi GPU power/temp/VRAM |
| `PhaseResult` | Wall time, N evals, µs/eval, energy report, peak RSS, notes |
| `BenchReport` | ISO 8601 timestamp + hardware + vector of phases |

**Workloads tested:**

| Workload | Sizes | Purpose |
|----------|-------|---------|
| Shannon entropy | 1K–1M | Single-vector reduction (memory-bound baseline) |
| Simpson index | 1K–1M | Single-vector reduction (arithmetic-only) |
| Variance / dot | 1K–1M | Pure arithmetic (auto-vectorization test) |
| Bray-Curtis NxN | 10–100 | Pairwise distance (GPU crossover point) |
| Spectral cosine NxN | 10–200 | GEMM-based batch (GPU-dominant workload) |
| PCoA | 10–30 | Matrix eigendecomposition (LAPACK comparison) |
| Full 16S pipeline | 10 samples | End-to-end pipeline (Galaxy vs Rust CPU vs GPU) |

**Energy methodology:**
- CPU: Intel RAPL via `energy_uj` readings at start/end; handles counter wraps
- GPU: nvidia-smi polled every 100ms; total joules via trapezoidal integration
- Cost: computed at $0.12/kWh US residential average

**Note on serde_json:** Added as a dependency for JSON model import (decision
tree inference from sklearn). Core validation and I/O remain sovereign; JSON
serialization in validation binaries is manual (`to_json()` methods).
Runtime dependencies: `flate2` (gzip), `bytemuck` (GPU casting), `serde_json`
(model import).

---

## 4. Comparison Protocol

1. Run Python / Galaxy reference with documented tool versions
2. Run Rust code on identical input data, record outputs
3. Compare on identical metrics using `validation::check` framework
4. GPU validation: CPU result is the reference, GPU must match within tolerance
5. All checks produce structured pass/fail with explicit exit codes:
   - Exit 0: all checks pass
   - Exit 1: one or more checks fail
   - Exit 2: skip (e.g., `SHADER_F64` not available)
6. Tolerances are centralized in `src/tolerances.rs` with documented rationale
7. Deterministic: same input → identical output, always

---

## 5. Hardware

All experiments run on a single consumer workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA RTX 4070 12 GB (`SHADER_F64` confirmed, Vulkan) |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Rust | stable (1.82+) |
| wgpu | v22 |

---

## 6. Acceptance Criteria

### Phase 2 (CPU): 1,241/1,241 checks pass

| Binary | Checks | Target |
|--------|:------:|--------|
| `validate_fastq` | 28 | Quality filter + merge + derep + Galaxy FastQC |
| `validate_diversity` | 27 | Analytical + simulated + evenness + rarefaction |
| `validate_16s_pipeline` | 37 | Complete 16S: FASTQ → DADA2 → chimera → taxonomy → diversity → UniFrac |
| `validate_algae_16s` | 34 | Real NCBI data (PRJNA488170) + Humphrey 2023 reference |
| `validate_voc_peaks` | 22 | Reese 2019 Table 1 (14 VOC compounds) + RI matching |
| `validate_mzml` | 7 | mzML parsing vs pyteomics |
| `validate_pfas` | 10 | Cosine + KMD + FindPFAS |
| `validate_features` | 8 | EIC + peaks + features vs asari (Exp009) |
| `validate_peaks` | 17 | Peak detection vs scipy.signal.find_peaks (Exp010) |
| `validate_public_benchmarks` | 202 | 22 samples, 4 BioProjects vs paper ground truth (Exp014) |
| `validate_extended_algae` | 35 | PRJNA382322 Nannochloropsis (Exp017) |
| `validate_pfas_library` | 26 | Jones Lab PFAS 175 compounds (Exp018) |
| `validate_newick_parse` | 30 | Newick tree parsing vs dendropy (Exp019) |
| `validate_qs_ode` | 16 | Waters 2008 QS ODE vs scipy (Exp020) |
| `validate_rf_distance` | 23 | Robinson-Foulds vs dendropy (Exp021) |
| `validate_gillespie` | 13 | Gillespie SSA vs analytical + numpy (Exp022) |
| `validate_pfas_decision_tree` | 7 | Decision tree inference vs sklearn (Exp008) |
| `validate_bistable` | 14 | Fernandez 2020 bistable switching (Exp023) |
| `validate_multi_signal` | 19 | Srivastava 2011 multi-signal QS (Exp024) |
| `validate_cooperation` | 20 | Bruger & Waters 2018 game theory (Exp025) |
| `validate_hmm` | 21 | Liu 2014 HMM primitives (Exp026) |
| `validate_capacitor` | 18 | Mhatre 2020 phenotypic capacitor (Exp027) |
| `validate_alignment` | 15 | Smith-Waterman local alignment (Exp028) |
| `validate_felsenstein` | 16 | Felsenstein pruning likelihood (Exp029) |
| `validate_barracuda_cpu` | 21 | Cross-domain CPU parity v1 (9 domains) |
| `validate_phage_defense` | 12 | Hsueh 2022 phage defense (Exp030) |
| `validate_bootstrap` | 11 | Wang 2021 RAWR bootstrap (Exp031) |
| `validate_placement` | 12 | Alamin & Liu 2024 placement (Exp032) |
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
| `validate_rare_biosphere` | 35 | Anderson 2015 rare biosphere (Exp051) |
| `validate_viral_metagenomics` | 22 | Anderson 2014 viral dN/dS (Exp052) |
| `validate_sulfur_phylogenomics` | 15 | Mateos 2023 molecular clock (Exp053) |
| `validate_phosphorus_phylogenomics` | 13 | Boden 2024 phosphorus clock (Exp054) |
| `validate_population_genomics` | 24 | Anderson 2017 ANI/SNP (Exp055) |
| `validate_pangenomics` | 24 | Moulana 2020 pangenome (Exp056) |
| `validate_barracuda_cpu_v4` | 44 | 23-domain CPU parity (Exp057) |
| `validate_barracuda_cpu_v5` | 29 | RF + GBM inference (Exp061-062) |
| **Total** | **1,241** | **All pass** |

Current status: **1,241/1,241 pass.** 63 experiments across 4 tracks.
157/157 BarraCUDA CPU parity checks across 25 algorithmic domains.
~22.5× Rust speedup over Python.

### Phase 3 (GPU): 260/260 checks pass

| Binary | Checks | Target |
|--------|:------:|--------|
| `validate_diversity_gpu` | 38 | All within `GPU_VS_CPU_*` tolerances |
| `validate_16s_pipeline_gpu` | 88 | Full pipeline: QF + DADA2 + chimera + taxonomy + diversity |
| `validate_barracuda_gpu_v3` | 14 | Extended diversity, spectral, stats GPU parity (Exp044) |
| `validate_toadstool_bio` | 14 | ToadStool bio absorption: SW, Gillespie, DT (Exp045) |
| `validate_gpu_phylo_compose` | 15 | FelsensteinGpu → bootstrap + placement (Exp046) |
| `validate_gpu_hmm_forward` | 13 | Local WGSL HMM batch forward (Exp047) |
| `benchmark_phylo_hmm_gpu` | 6 | CPU vs GPU Felsenstein + HMM timing (Exp048) |
| `validate_gpu_ode_sweep` | 12 | GPU ODE sweep (7) + bifurcation eigenvalues (5) (Exp049-050) |

| `validate_gpu_track1c` | 27 | ANI + SNP + dN/dS + pangenome shaders (Exp058) |
| `validate_gpu_23_domain_benchmark` | 20 | 23-domain GPU parity (Exp059) |
| `validate_gpu_cross_substrate` | skip | metalForge substrate proof (Exp060) |
| `validate_gpu_rf` | 13 | RF batch inference shader (Exp063) |

Current status: **260/260 pass.** 15 ToadStool primitives consumed.
9 local WGSL shaders (Write → Absorb → Lean candidates).
4 bio GPU primitives absorbed from ToadStool (Feb 20 cce8fe7c).

### Grand Total: 1,501/1,501 quantitative checks pass

---

## 7. Key Differences from hotSpring

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Nuclear physics | Life science + analytical chemistry |
| Phase A/B | Python reproduction → BarraCUDA re-execution | Galaxy/Python → Rust sovereign port |
| GPU workload | Large-matrix eigensolve, MD force evaluation | Diversity, phylogenetics, ODE sweeps, HMM |
| Validation metric | chi2/datum | Pass/fail within documented tolerance |
| Data size | Small (52–2,042 nuclei) | Large (millions of reads, thousands of spectra) |
| Local shaders | 10+ WGSL (HFB, MD, lattice QCD) | 9 WGSL (HMM, ODE, ANI, SNP, dN/dS, pangenome, RF, DADA2, quality) |
| Absorption tracking | `EVOLUTION_READINESS.md` with tiers | `EVOLUTION_READINESS.md` with tiers (adopted) |
| Key insight | GPU-resident hybrid beats CPU for matrix physics | Full GPU pipeline 2.45× faster; ODE/HMM/phylo math portable to GPU |

Both projects follow the **Write → Absorb → Lean** pattern and prove the same
thesis: sovereign compute on consumer hardware can replicate institutional
results, then exceed them via Rust + GPU.

---

## 8. Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Rust | stable (1.82+) | MSRV set in `Cargo.toml` |
| wgpu | v22 | Vulkan backend, `SHADER_F64` |
| Galaxy | 24.1 | Local Docker instance |
| QIIME2 | 2026.1.0 | Galaxy plugin (q2-amplicon-2026.1) |
| asari | 1.13.1 | LC-MS feature extraction |
| `PFΔScreen` | Latest | PFAS non-target screening |
| flate2 | 1.0 | Only runtime dependency (gzip) |
