# wetSpring Control Experiment — Status Report

**Date**: 2026-02-12 (Project initialized)
**Gate**: Eastgate (i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**Galaxy**: bgruening/galaxy-stable:latest (Docker)
**License**: AGPL-3.0-or-later

---

## Replication Protocol

Anyone can reproduce all results by:

```bash
git clone git@github.com:syntheticChemistry/wetSpring.git
cd wetSpring

# 1. Install tools (SRA Toolkit, checks Docker/Rust)
./scripts/setup_tools.sh

# 2. Download public datasets (~5-10 GB)
./scripts/download_data.sh --all

# 3. Start Galaxy (Docker, ~4 GB image)
./scripts/start_galaxy.sh

# 4. Follow experiment protocols in experiments/
```

No institutional access required. All data is from public repositories
(NCBI SRA, Zenodo, SILVA). All tools are open source.

---

## Hardware Gate

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (16C/24T, 5.2 GHz) |
| RAM | 64 GB DDR5-4800 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| OS | Pop!_OS 22.04 (Ubuntu-based) |
| Docker | 24.x with Compose v2 |

---

## Research Context

### Principal Investigators (Sandia National Laboratories)

**Chuck Smallwood, PhD** — Principal Member of Technical Staff, Bioscience.
PhD Biochemistry, University of Oklahoma. Systems biology, microbial
interactions, algal biotechnology. Leads Pond Crash Forensics program
($800K DOE Biomass grant). Pending patent on engineered probiotic for
microalgal resilience.

**Jesse Cahill, PhD** — Senior Member of Technical Staff, Bioscience.
PhD Biochemistry, Texas A&M (Center for Phage Technology, Dr. Ry Young).
Phage engineering, genome editing, phage-microbiome interactions, DNA
forensics, GenAI and Bioinformatics.

### The Problem

Open raceway ponds for algal biofuel production suffer from sudden,
catastrophic "pond crashes" caused by biological contamination — bacteria,
fungi, viruses, rotifers, zooplankton. These crashes destroy entire crops
overnight and represent ~30% loss in commercial algae cultivation. The
contaminating organisms are often unknown a priori ("unknown unknowns"),
making prevention and diagnosis difficult.

### The Computational Methods

Smallwood and Cahill's group developed and applied several computational
approaches:

1. **16S rRNA amplicon metagenomics** — Next-generation sequencing of
   microbial communities in raceway ponds, taxonomic profiling to identify
   crash agents (*Brachionus*, *Chaetonotus*, *Bacillus safensis*).
   Adapted from RapTOR (Rapid Threat Organism Recognition) homeland
   security pathogen detection.

2. **Phage genome analysis** — Sequencing, assembly, and annotation of
   bacteriophages engineered to eliminate pathogenic bacteria from algae
   cultures without disrupting the protective bacteriome.

3. **Spectroradiometric monitoring** — Hyperspectral reflectance analysis
   (350-2500 nm, every 5 min) for real-time detection of competitor
   diatoms and grazers in raceway ponds.

4. **VOC metabolomics** — GC-MS headspace profiling for early biomarkers
   of pond crashes, hours before visible symptoms.

---

## Datasets

### Available for Download

| ID | Dataset | Source | Size | BioProject |
|----|---------|--------|------|------------|
| D1 | Galaxy training 16S (mouse gut) | Zenodo 800651 | ~200 MB | — |
| D2 | Nannochloropsis microbiome | NCBI SRA | ~8 GB | PRJNA382322 |
| D3 | N. salina antibiotic treatment | Frontiers paper | TBD | TBD |
| D4 | Viral RefSeq (phage reference) | NCBI FTP | ~500 MB | — |
| D5 | SILVA 138.1 (taxonomy ref) | QIIME2 data | ~300 MB | — |

### Pending Identification

| Dataset | Source | Notes |
|---------|--------|-------|
| Pond Crash Forensics raw reads | Smallwood et al. 2016 | Check OSTI/SRA for data deposit |
| Biotic Countermeasures 16S | Smallwood/Cahill 2023 | Check OSTI 2311389 supplementary |
| Spectroradiometric time-series | Lane et al. 2021 | May be in paper supplementary |
| VOC GC-MS profiles | Lane et al. 2019 | May be in paper supplementary |

---

## Experiment Log

### Experiment 001: Galaxy Bootstrap — NOT STARTED

**Goal**: Self-host Galaxy, install tools, validate with training dataset.

- [ ] Galaxy Docker running on localhost:8080
- [ ] Admin account created
- [ ] Amplicon tools installed (FastQC, DADA2, QIIME2, Kraken2)
- [ ] Assembly tools installed (SPAdes, Prokka)
- [ ] Training dataset (D1) uploaded
- [ ] FastQC run on training data
- [ ] DADA2 denoise on paired-end 16S
- [ ] Taxonomy barplot generated

### Experiment 002: 16S Amplicon Replication — NOT STARTED

**Goal**: Run Nannochloropsis microbiome (D2) through full 16S pipeline.

- [ ] FASTQ downloaded from SRA
- [ ] Quality control (FastQC + Trimmomatic/Cutadapt)
- [ ] DADA2 denoising → ASV table
- [ ] SILVA taxonomy classification
- [ ] Alpha diversity (Shannon, Simpson, Chao1)
- [ ] Beta diversity (Bray-Curtis PCoA)
- [ ] Compare: do we see Bacteroidetes + Alphaproteobacteria dominance?
- [ ] Compare: do we see Saprospiraceae correlation with growth?

### Experiment 003: Phage Annotation — NOT STARTED

**Goal**: Annotate phage genomes from public databases, validate pipeline.

- [ ] Download reference phage genomes (D4)
- [ ] SPAdes assembly of test reads
- [ ] Prokka/Pharokka annotation
- [ ] geNomad classification
- [ ] CheckV completeness assessment

### Experiment 004: Rust FASTQ Parser — NOT STARTED

**Goal**: First Rust module — FASTQ parsing + quality filtering.

- [ ] Implement FASTQ reader (gzip-aware)
- [ ] Quality score parsing (Phred33/64)
- [ ] Adapter trimming
- [ ] Quality filtering (sliding window, min length)
- [ ] Validate: identical output to Trimmomatic on D1
- [ ] Benchmark: Rust vs Trimmomatic throughput

---

## Evolution Roadmap

```
Phase 0 (current):  Galaxy hosting + tool validation
Phase 1:            Pipeline replication with public data
Phase 2:            Rust ports of critical stages
Phase 3:            GPU acceleration via ToadStool
Phase 4:            End-to-end sovereign pipeline (no Galaxy dependency)
```

### GPU Acceleration Targets (Phase 3)

| Pipeline Stage | Current Tool | GPU Potential | Why |
|---------------|-------------|:------------:|-----|
| K-mer counting | DADA2 (R/C++) | **High** | Embarrassingly parallel, large data |
| Sequence alignment | BLAST/Bowtie2 | **High** | Smith-Waterman on GPU is well-studied |
| Taxonomic classify | Kraken2 | **High** | K-mer lookup, hash table operations |
| Distance matrices | scikit-bio | **High** | Large matrix operations → ToadStool |
| PCA/PCoA | sklearn | **High** | Eigensolve → ToadStool BatchedEigh |
| FFT (spectral) | scipy.fft | **High** | GPU FFT is mature |
| Peak detection | pyOpenMS | **Medium** | Signal processing on GPU |

---

## Relationship to hotSpring

wetSpring follows the same validation methodology as hotSpring:

| | hotSpring | wetSpring |
|--|-----------|-----------|
| Domain | Nuclear physics | Life science |
| Validation target | Binding energies (chi2) | Organism identification |
| Baseline | Python scipy/numpy | Python Galaxy/QIIME2 |
| Evolution | Rust BarraCUDA | Rust BarraCUDA |
| GPU layer | ToadStool (wgpu) | ToadStool (wgpu) |
| Success metric | chi2 match + speedup | Same taxonomy + speedup |

Both prove: sovereign compute on consumer hardware can replicate
institutional results, then exceed them via Rust+GPU.

---

*Initialized: February 12, 2026*
