# wetSpring Control Experiment — Status Report

**Date**: 2026-02-12 (Project initialized)
**Updated**: 2026-02-16 (BarraCUDA Rust validation — 36/36 PASS)
**Gate**: Eastgate (i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**Galaxy**: quay.io/bgruening/galaxy:24.1 (Docker) — upgraded from 20.09
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

### Track 1: Life Science — Principal Investigators (Sandia)

**Chuck Smallwood, PhD** — Principal Member of Technical Staff, Bioscience.
PhD Biochemistry, University of Oklahoma. Systems biology, microbial
interactions, algal biotechnology. Leads Pond Crash Forensics program
($800K DOE Biomass grant). Pending patent on engineered probiotic for
microalgal resilience.

**Jesse Cahill, PhD** — Senior Member of Technical Staff, Bioscience.
PhD Biochemistry, Texas A&M (Center for Phage Technology, Dr. Ry Young).
Phage engineering, genome editing, phage-microbiome interactions, DNA
forensics, GenAI and Bioinformatics.

#### The Problem

Open raceway ponds for algal biofuel production suffer from sudden,
catastrophic "pond crashes" caused by biological contamination — bacteria,
fungi, viruses, rotifers, zooplankton. These crashes destroy entire crops
overnight and represent ~30% loss in commercial algae cultivation. The
contaminating organisms are often unknown a priori ("unknown unknowns"),
making prevention and diagnosis difficult.

#### Computational Methods

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

### Track 2: PFAS Analytical Chemistry — Codename blueFish

**A. Daniel Jones, PhD** — Professor, Biochemistry & Molecular Biology,
Michigan State University. Associate Director, MSU Center for PFAS
Research. ~25 years in analytical chemistry. Co-developed **asari**
(Nature Communications 2023) for open-source LC-MS data processing.
Leads MSU Mass Spectrometry and Metabolomics Core facility.

#### The Problem

Per- and polyfluoroalkyl substances (PFAS, "forever chemicals") contaminate
drinking water worldwide. Current detection requires instruments costing
$500K+ (LC-MS/MS, HRMS) running proprietary vendor software (Waters
MassLynx, Thermo Xcalibur, Agilent MassHunter, AB SCIEX Analyst). This
creates a monopoly: municipalities cannot affordably monitor water, and
scientists cannot reproduce analyses across instrument platforms.

#### Computational Methods

1. **LC-MS metabolomics processing** — Feature extraction, peak detection,
   mass alignment, and quantification from liquid chromatography-mass
   spectrometry data. Current open-source: **asari** (Python).

2. **PFAS non-targeted screening** — Identifying unknown PFAS compounds
   in environmental samples via mass defect analysis, Kendrick mass defect
   (KMD), diagnostic fragment ions, and MS2 scoring. Current open-source:
   **PFΔScreen** (Python + pyOpenMS).

3. **Femtosecond laser ionization MS** — Jones's technique for detecting
   nonpolar PFAS without chromatographic separation — the key to
   eliminating the most expensive hardware component.

4. **Machine learning for water monitoring** — Region-specific ML models
   predicting PFAS contamination levels in drinking water from utility
   metadata (MSU, Water Research 2023).

5. **Molecular dynamics** — Computational modeling of PFAS-soil mineral
   interactions, PFAS-receptor binding, and toxicity prediction via
   GROMACS/AMBER + ML.

#### blueFish Vision

Replace vendor-locked analytical chemistry software with open-source
Rust+GPU alternatives. The endgame: live water monitoring for pennies
per sample using low-cost sensors + GPU-accelerated processing, instead
of $500K instruments with proprietary software.

---

## Datasets

### Track 1: Life Science — Available for Download

| ID | Dataset | Source | Size | BioProject |
|----|---------|--------|------|------------|
| D1 | Galaxy training 16S (mouse gut) | Zenodo 800651 | ~200 MB | — |
| D2 | Nannochloropsis microbiome | NCBI SRA | ~8 GB | PRJNA382322 |
| D3 | N. salina antibiotic treatment | Frontiers paper | TBD | TBD |
| D4 | Viral RefSeq (phage reference) | NCBI FTP | ~500 MB | — |
| D5 | SILVA 138.1 (taxonomy ref) | QIIME2 data | ~300 MB | — |

### Track 2: PFAS / blueFish — Available for Download

| ID | Dataset | Source | Size | Notes |
|----|---------|--------|------|-------|
| D6 | asari demo LC-MS data | shuzhao-li-lab/data (GitHub) | ~50 MB | mzML test files |
| D7 | MetaboLights PFAS studies | EMBL-EBI MetaboLights | TBD | Search MTBLS* for PFAS |
| D8 | EPA CompTox PFAS list | EPA CompTox Dashboard | ~5 MB | ~14,000 PFAS structures |
| D9 | MassBank PFAS spectra | MassBank.eu | ~10 MB | Reference MS2 spectra |
| D10 | Michigan DEQ PFAS water data | MI DEQ public reports | ~1 MB | Training data for ML models |
| D11 | NORMAN SusDat suspect list | NORMAN Network | ~20 MB | ~65,000 PFAS suspect entries |

### Pending Identification

| Dataset | Source | Notes |
|---------|--------|-------|
| Pond Crash Forensics raw reads | Smallwood et al. 2016 | Check OSTI/SRA for data deposit |
| Biotic Countermeasures 16S | Smallwood/Cahill 2023 | Check OSTI 2311389 supplementary |
| Spectroradiometric time-series | Lane et al. 2021 | May be in paper supplementary |
| VOC GC-MS profiles | Lane et al. 2019 | May be in paper supplementary |
| Jones lab PFAS HRMS data | MSU publications | Check supplementary data |
| PFΔScreen validation HRMS | Zweigle et al. 2023 | Check paper supplementary |

---

## Run Log

### 2026-02-12: Project Initialization

- Created wetSpring repository (github.com/syntheticChemistry/wetSpring)
- Scaffolded Track 1 (Life Science): Galaxy Docker setup, data download scripts,
  experiment protocols for 16S amplicon, phage annotation, spectral, and VOC pipelines
- Downloaded MiSeq SOP validation dataset from Zenodo (44 files, 372 MB)
- Downloaded SILVA 138.1 reference (QIIME2 format, 100 MB)

### 2026-02-12: Track 2 Added (blueFish)

- Added PFAS analytical chemistry as Track 2 (codename blueFish)
- Wired in A. Daniel Jones (MSU) as principal collaborator
- Added Pipelines 5-7: asari LC-MS, PFΔScreen PFAS screening, PFAS ML/MD
- Added Experiments 005-008 protocols
- Added datasets D6-D11 (asari demo, EPA CompTox, MassBank, Michigan DEQ)

### 2026-02-16: Galaxy 24.1 Operational

- **Attempted Galaxy 20.09** (bgruening/galaxy-stable:latest):
  - Tools installed via ephemeris but 10/15 failed to load
  - Error: modern tool revisions target Galaxy 22.05-24.2
  - FastQC 0.74 requires Galaxy 23.0, QIIME2 requires 22.05, Kraken2 requires 24.2
- **Upgraded to Galaxy 24.1** (quay.io/bgruening/galaxy:24.1):
  - Fresh volume, first boot 80s (DB migration, conda init)
  - 15/15 repositories installed in 50s via ephemeris
  - 32 individual tools loaded (BLAST+ expands to 12 sub-tools)
- **Conda dependency resolution**:
  - Initial FastQC run failed: `fastqc: command not found`
  - Enabled `conda_auto_install: true` and `conda_prefix: /tool_deps/_conda`
  - Container restart required for config changes
  - Subsequent runs auto-resolve conda environments
- **Fixed SPAdes owner**: YAML had `owner: iuc`, correct is `owner: nml`

### 2026-02-16: FastQC Validation

- Uploaded F3D0_R1.fastq and F3D0_R2.fastq (first paired sample from MiSeq SOP)
- **FastQC F3D0_R1** (forward reads):
  - 7,793 sequences, 1.9 Mbp, 249-251 bp
  - Encoding: Sanger/Illumina 1.9 (Phred33)
  - GC: 54%, poor quality sequences: 0
  - Per-base quality: PASS (Q32-38+ across all positions)
  - All modules: PASS
- **FastQC F3D0_R2** (reverse reads):
  - 7,793 sequences, 1.9 Mbp, 247-251 bp
  - GC: 55%, poor quality sequences: 0
  - Per-base quality: FAIL (expected — reverse reads degrade toward 3' end)
  - This is normal for paired-end Illumina and handled by DADA2's error model

### 2026-02-16: QIIME2 + DADA2 Operational

- Installed QIIME2 2026.1.0 in Galaxy's conda (`qiime2-amplicon-2026.1` env)
  - Packages: q2-dada2, q2-feature-table, q2-taxa, q2-diversity, q2galaxy
  - R 4.3.3, DADA2 1.30.0, Rcpp 1.1.0
  - Fixed `pkg_resources` (downgraded setuptools to 69.5.1)
  - Created `/usr/local/bin/q2galaxy` wrapper (unsets Galaxy venv, uses conda env)
- Uploaded all 40 FASTQ files (20 paired samples) via Galaxy API
- Created `list:paired` collection "MiSeq SOP Paired Reads"
- Imported into QIIME2 artifact: `paired-end-demux.qza` (36.9 MB)
  - `SampleData[PairedEndSequencesWithQuality]`, UUID: 86df290c

### 2026-02-16: DADA2 Denoise-Paired

- **Parameters**: trunc_len_f=240, trunc_len_r=160, n_threads=8
- **Runtime**: 43.5 seconds (i9-12900K, 8 threads)
- **Results**:
  - 232 ASVs detected across 20 samples
  - 124,249 total reads retained (mean 6,212/sample)
  - Input: 162,360 reads → Filtered: 148,641 (91.5%) → Non-chimeric: 124,249 (76.5%)
- **Per-sample retention** (input → non-chimeric):

| Sample | Input | Filtered | Denoised | Merged | Non-chimeric | % Retained |
|--------|-------|----------|----------|--------|-------------|-----------|
| F3D0 | 7,793 | 7,113 | 6,976 | 6,540 | 6,528 | 83.8% |
| F3D2 | 19,620 | 18,075 | 17,907 | 17,431 | 16,835 | 85.8% |
| F3D147 | 17,070 | 15,637 | 15,433 | 14,233 | 13,006 | 76.2% |
| Mock | 4,779 | 4,314 | 4,287 | 4,269 | 4,269 | 89.3% |

- **Mock community**: 89.3% retention with 4,269 reads — highest retention
  rate as expected (defined community, no chimeras)
- **Artifacts saved**: `dada2-table.qza`, `dada2-rep-seqs.qza`, `dada2-stats.qza`

### 2026-02-16: SILVA Taxonomy Classification

- Installed complete QIIME2 2026.1.0 amplicon distribution (fresh conda env)
  - Includes q2-feature-classifier, q2-dada2, q2-taxa, q2-diversity, qiime CLI
- Downloaded pre-trained SILVA 138 NB classifier (209 MB, sklearn 1.4.2)
- **Classification**: 232 ASVs classified in 11.2 seconds (NB, 8 jobs)
- **Phylum distribution** (9 phyla detected):

| Phylum | ASVs | Expected in Mouse Gut |
|--------|-----:|:--------------------:|
| Firmicutes | 191 | Yes (dominant) |
| Bacteroidota | 20 | Yes (major) |
| Proteobacteria | 7 | Yes (minor) |
| Actinobacteriota | 6 | Yes (minor) |
| Cyanobacteria | 3 | Yes (chloroplast) |
| Patescibacteria | 2 | Yes (rare) |
| Campylobacterota | 1 | Yes (rare) |
| Deinococcota | 1 | Rare |
| Verrucomicrobiota | 1 | Yes (Akkermansia) |

- **Key families**: Lachnospiraceae (121 ASVs), Oscillospiraceae (16),
  Muribaculaceae (15) — classic mouse gut profile
- All classifications >95% confidence (NB posterior probability)
- **Mock community**: correctly classified with highest per-read retention

### 2026-02-16: Taxonomy Barplot Generated

- Generated interactive taxonomy barplot (`taxa-barplot.qzv`, 427 KB)
- Grouped samples: Early (F3D0-F3D9), Late (F3D141-F3D150), Mock
- Viewable at https://view.qiime2.org with the QZV file

### 2026-02-16: Experiment 002 — Phytoplankton Microbiome

- **Data source**: PRJNA1195978 (80 per-sample SRA runs, 1.9 GB total)
- Downloaded 10 samples (860K reads) via SRA Toolkit `fastq-dump`
- Note: PRJNA382322 (Nannochloropsis) rejected — single multiplexed run
  requiring custom barcode demux, replaced with per-sample dataset
- **V3-V4 amplicon (2x151bp)**: too short for paired-end merge (~440bp amplicon)
  → used DADA2 `denoise-single` on forward reads (trunc=140)
- **Results**: 2,273 ASVs, 820,548 reads retained from 860K input (95.4%)
- **Taxonomy**: 41 phyla — rich marine diversity as expected:
  - Proteobacteria: 822 ASVs (36%) with Rhodobacteraceae, Colwelliaceae
  - Bacteroidota: 200 ASVs (8.8%) with Flavobacteriaceae — known algae-associated
  - Nanoarchaeota: 162 ASVs (7.1%) — marine archaea
  - Patescibacteria: 130 ASVs — ultra-small bacteria, marine
  - Verrucomicrobiota: 125 ASVs — environmental bacteria
- **Pipeline time**: 95.6s (import 13.6s, DADA2 68.0s, taxonomy 9.5s, barplot 0.1s)

### 2026-02-16: Track 2 Validation Rerun — 8/8 PASS

- Automated validation script: `scripts/validate_track2.py`
- **Exp005 (asari)**: 5/5 checks PASS (15.4s)
  - Feature detection is deterministic: 5,951 preferred, 8,659 full (exact match)
  - Unique compounds ~4,101-4,107 (±0.1%, khipu clustering stochastic, within tolerance)
- **Exp006 (FindPFAS)**: 3/3 checks PASS (0.77s)
  - 738 spectra, 62 candidates, 25 unique precursors (exact match)
- **Track 2 is deterministic and reproducible**

### 2026-02-16: Experiment 005 — asari LC-MS Metabolomics

- Installed asari 1.13.1 in dedicated Python venv
- Cloned shuzhao-li-lab/data repo, extracted MT02 demo dataset (8 mzML files)
- **Results**: 8,659 features (5,951 filtered), 4,107 unique compounds in 15.6s
- Mass accuracy: -0.6 ppm, khipu annotation with multi-charge + adducts
- asari processing pipeline validated for Track 2 LC-MS workflows

### 2026-02-16: Experiment 006 — PFAS Screening (FindPFAS)

- Installed PFAScreen + FindPFAS with pyOpenMS 3.5.0, pyteomics
- Test data: PFAS Standard Mix (ddMS2, 20 eV, 738 spectra)
- **Results**: 25 unique PFAS precursors from CF2/C2F4 fragment screening
- EPA CompTox suspect screening: 2 confirmed matches from 4,729 PFAS list
- PFAS non-targeted screening algorithm validated end-to-end

### 2026-02-16: Experiment 003 — Phage Assembly & Annotation

- Downloaded 2 Escherichia phage datasets from SRA (A4.3: 155K reads, L73: 198K reads)
- Installed SPAdes 4.2.0, Prokka 1.15.6, Pharokka 1.9.1, CheckV 1.0.3 in conda env
- Downloaded Pharokka databases (656 MB, PHROGs+CARD+VFDB+INPHARED)
- Downloaded CheckV database (checkv-db-v1.5)
- Fixed PHANOTATE `pkg_resources` (setuptools<70 in phage-tools env)
- **Assembly**: both phages assemble to single dominant scaffolds
  (A4.3: 139,846 bp, L73: 166,966 bp)
- **Annotation**: Prokka (general) and Pharokka (phage-specific) both succeed
  - Prokka: ~250 CDS + tRNAs per genome in ~2s
  - Pharokka: ~300+ CDS per genome with PHROGs functional annotation in ~115s
- **CheckV**: A4.3 = 99.67% complete (High-quality), L73 = 100% complete (DTR)
  - Zero host genes, zero contamination for both
- Full phage annotation pipeline validated end-to-end

### 2026-02-16: Validation Rerun — 8/8 PASS

- Automated validation script: `scripts/validate_exp001.py`
- Clean rerun from raw FASTQs → import → DADA2 → SILVA → barplot
- **All 8 checks passed** (exact match to original run):
  - ASV count: 232, Samples: 20, Total reads: 124,249
  - Phyla: 9, Firmicutes: 191, Bacteroidota: 20
  - Mock community: 4,269 reads, 89.3% retention
- **Pipeline is deterministic**: identical results across independent runs
- **Total time**: 71.5s (import: 14.0s, DADA2: 42.5s, taxonomy: 10.5s)
- Report: `experiments/results/001_galaxy_bootstrap/validation_report.json`

---

### 2026-02-16: BarraCUDA Rust Validation — 36/36 PASS

- **Architecture**: `wetspring-barracuda` crate, depends on `barracuda` (phase1/toadstool)
- **Pattern**: Same as hotSpring — hardcoded Python baseline values in Rust validation binaries
- **Crate**: 6 modules (3 I/O parsers, 3 bio algorithms), 4 validation binaries, 20 unit tests
- **Dependencies**: barracuda, needletail, quick-xml, base64, flate2, serde, rayon

**Track 1 — Life Science (FASTQ + Diversity):**
- `validate_fastq`: **9/9 PASS** — F3D0 7,793 seqs, 40 files 304,720 seqs, GC 54.7%, Q 35.8
- `validate_diversity`: **14/14 PASS** — Shannon/Simpson/Chao1 analytical + marine simulation + k-mers

**Track 2 — PFAS Analytical Chemistry (mzML + PFAS):**
- `validate_mzml`: **7/7 PASS** — 8 files, 6,256 spectra, 6M peaks, m/z 80-1000, base64+zlib
- `validate_pfas`: **6/6 PASS** — 738 spectra, 62 candidates, 25 unique precursors (exact match)

**Total: 36/36 checks PASS, 20/20 unit tests PASS**

| Binary | Track | Checks | Status |
|--------|-------|--------|--------|
| validate_fastq | T1 | 9/9 | PASS |
| validate_diversity | T1 | 14/14 | PASS |
| validate_mzml | T2 | 7/7 | PASS |
| validate_pfas | T2 | 6/6 | PASS |

---

## Experiment Log

### Experiment 001: Galaxy Bootstrap — COMPLETE

**Goal**: Self-host Galaxy, install tools, validate with training dataset.

- [x] Galaxy Docker running on localhost:8080 (v24.1, upgraded from v20.09)
- [x] Admin account created and activated (via psql + master API key)
- [x] Amplicon tools installed (FastQC, DADA2, QIIME2, Kraken2) — 15 repos, 32 tools
- [x] Assembly tools installed (SPAdes, Prokka, Pharokka)
- [x] Training dataset (D1) uploaded (F3D0 paired-end, 7793 sequences)
- [x] FastQC on R1 — PASS (Q32-38+, 0 poor quality, 54% GC)
- [x] FastQC on R2 — expected quality drop (normal for Illumina R2)
- [x] Uploaded all 20 paired samples (40 FASTQ files) via Galaxy API
- [x] QIIME2 import → `paired-end-demux.qza` (36.9 MB)
- [x] DADA2 denoise-paired → 232 ASVs, 124,249 reads, 43.5s runtime
- [x] SILVA 138 taxonomy → 9 phyla, 40 families, 11.2s (Firmicutes dominant)
- [x] Taxonomy barplot generated (`taxa-barplot.qzv`, 427 KB)

### Experiment 002: Phytoplankton Microbiome 16S — COMPLETE

**Goal**: Download real algae microbiome data from SRA, run full 16S pipeline.
**BioProject**: PRJNA1195978 (phytoplankton-associated bacterial communities)
**Note**: Replaced D2 (PRJNA382322, single multiplexed run) with per-sample dataset.

- [x] 10 samples downloaded from SRA (860K paired reads, 151bp MiSeq V3-V4)
- [x] DADA2 denoise-single (forward only — V3-V4 too long for 2x151 merge)
- [x] 2,273 ASVs, 820,548 reads retained, 95.6s total pipeline
- [x] SILVA 138 taxonomy → 41 phyla (marine diversity confirmed)
- [x] Proteobacteria dominant (822 ASVs, Rhodobacteraceae 67)
- [x] Bacteroidota second (200 ASVs, Flavobacteriaceae 95) — as expected
- [x] Marine-specific: Nanoarchaeota, Woesearchaeales, Bdellovibrionota
- [x] Taxonomy barplot generated (587 KB)
- [x] Alpha diversity (rarefied to 40,834 reads/sample):
  - Shannon: 2.93 ± 0.81 (range 1.78–3.85)
  - Observed features: 301 ± 222 (range 91–856)
  - Chao1: 348 ± 329 (range 91–1222)
  - Simpson: 0.86 ± 0.09 (range 0.73–0.94)
- [x] Beta diversity (Bray-Curtis PCoA):
  - Mean pairwise: 0.69 (range 0.06–0.95) — high community turnover
  - PC1: 49.7%, PC2: 30.8%, PC3: 11.1% variance explained
  - Three low-diversity samples cluster tightly (negative PC1)
- [ ] Download remaining 70 samples for full-scale analysis

### Experiment 003: Phage Annotation — COMPLETE

**Goal**: Assemble and annotate phage genomes from SRA sequencing data.

- [x] Downloaded 2 Escherichia phage datasets from SRA:
  - SRR36584166 (E. phage A4.3): 155K paired reads
  - SRR36584167 (E. phage L73): 198K paired reads
- [x] Downloaded T4 phage reference genome (NC_000866.4, 168,903 bp)
- [x] SPAdes assembly (--isolate mode, 8 threads):
  - A4.3: 20 scaffolds, 146,885 bp, largest 139,846 bp (64s)
  - L73: 44 scaffolds, 185,175 bp, largest 166,966 bp (73s)
- [x] Prokka annotation:
  - A4.3: 219 CDS, 7 tRNA (2s)
  - L73: 277 CDS, 11 tRNA (2s)
- [x] Pharokka annotation (PHROGs + CARD + VFDB + INPHARED):
  - A4.3: 289 CDS (114s)
  - L73: 345 CDS (115s)
- [x] CheckV completeness:
  - A4.3: 99.67% complete (High-quality, AAI-based), 202 viral genes, 0 contamination
  - L73: 100% complete (DTR detected), 259 viral genes, 0 contamination

### Experiment 004: Rust FASTQ + Diversity — IN PROGRESS (B2 validation)

**Goal**: Rust I/O parsers + diversity metrics — validated against Python baselines.

- [x] Implement FASTQ reader (needletail, gzip-aware, Phred33)
- [x] Validate: 7,793 seqs in F3D0, 304,720 across all 40 files — **9/9 PASS**
- [x] Implement k-mer engine (2-bit encoding, canonical k-mers, HashMap counting)
- [x] Implement alpha diversity (Shannon, Simpson, Chao1, observed features)
- [x] Implement Bray-Curtis dissimilarity + distance matrix
- [x] Validate diversity: analytical tests + simulated marine community — **14/14 PASS**
- [ ] Adapter trimming
- [ ] Quality filtering (sliding window, min length)
- [ ] Benchmark: Rust vs Trimmomatic throughput

---

### Track 2: PFAS / blueFish Experiments

### Experiment 005: asari Bootstrap — COMPLETE

**Goal**: Install asari, process demo LC-MS data, validate feature table output.

- [x] Install asari 1.13.1 (`pip install asari-metabolomics`)
- [x] Clone shuzhao-li-lab/data, extract MT02 dataset (8 mzML files, HILIC-pos)
- [x] Run asari on MT02: 15.6s runtime
  - 8,659 features detected (5,951 preferred/filtered)
  - 1,622 khipus, 6,354 empirical compounds, 4,107 unique compounds
  - Mass accuracy: -0.6 ppm
- [x] Feature table: 5,951 × 8 (features × samples)
- [x] Khipu annotation: multi-charge (1x, 2x, 3x), adducts (H+, Na/H, HCl, K/H, ACN)

### Experiment 006: PFΔScreen / FindPFAS Validation — COMPLETE

**Goal**: Install PFAS screening tools, validate on PFAS standard mix HRMS data.

- [x] Installed PFAScreen + FindPFAS (GitHub: JonZwe), pyOpenMS 3.5.0
- [x] Test data: PFAS Standard Mix, ddMS2, 20 eV (738 MS2 spectra)
- [x] FindPFAS screening: 25 unique PFAS precursors detected (8.4% of spectra)
  - CF2 (49.997 Da) and C2F4 (99.994 Da) fragment differences confirmed
  - Top hit: m/z 812.94, 13 fragment differences (classic PFAS pattern)
  - m/z range 146–912, RT range 4.8–12.5 min
- [x] EPA CompTox suspect screening: 4,729 PFAS loaded, 2 confirmed matches
  - CAS 66008-68-2 (m/z 468.97, [M-H]-)
  - CAS 375-62-2 (m/z 313.08, [M+Na]+)
- [x] Algorithm validated: mass defect + fragment differences + suspect list

### Experiment 007: Rust mzML + PFAS Screening — IN PROGRESS (B2 validation)

**Goal**: Rust mass spectrometry parsers + PFAS screening — validated against Python baselines.

- [x] Implement mzML XML parser (quick-xml, base64+zlib, 64-bit float)
- [x] Parse spectrum metadata (m/z array, intensity array, RT, MS level)
- [x] Support base64-encoded + zlib-compressed binary arrays
- [x] Validate: 8 files, 6,256 spectra, 6M+ peaks, m/z 80-1000 — **7/7 PASS**
- [x] Implement MS2 text parser (S/I/Z headers + peak lists)
- [x] Validate: 738 spectra from PFAS Standard Mix (exact match)
- [x] Implement ppm-tolerance binary search (core of suspect screening)
- [x] Implement PFAS fragment screening (CF2/C2F4/HF mass differences)
- [x] Validate: 62 candidates, 25 unique PFAS precursors (exact match) — **6/6 PASS**
- [ ] Benchmark: Rust vs pyteomics parsing throughput

### Experiment 008: PFAS ML Water Monitoring — NOT STARTED

**Goal**: Replicate MSU ML models for PFAS drinking water prediction.

- [ ] Obtain Michigan DEQ PFAS water monitoring data (D10)
- [ ] Feature engineering: water utility metadata → ML features
- [ ] Train Random Forest / GBM models (Python scikit-learn baseline)
- [ ] Validate: reproduce published prediction accuracy
- [ ] Port to Rust (smartcore or linfa)
- [ ] Benchmark: Rust vs Python training + inference time

---

## Evolution Roadmap

```
Track 1 (Life Science):
  Phase 0 [DONE]:     Galaxy hosting + tool validation (Exp001)
  Phase 1 [DONE]:     Pipeline replication with public data (Exp002, Exp003)
  Phase 2 [ACTIVE]:   Rust ports — FASTQ+kmer+diversity validated (36/36 PASS)
  Phase 3 [NEXT]:     GPU acceleration via ToadStool
  Phase 4:            End-to-end sovereign pipeline

Track 2 (PFAS / blueFish):
  Phase B0 [DONE]:    asari + PFΔScreen validation (Exp005, Exp006)
  Phase B1:           Replicate Jones/MSU LC-MS and PFAS pipelines
  Phase B2 [ACTIVE]:  Rust ports — mzML+MS2+PFAS screening validated (36/36 PASS)
  Phase B3 [NEXT]:    GPU acceleration (spectral matching, ML, MD)
  Phase B4:           Penny monitoring (real-time, low-cost sensors)
```

### GPU Acceleration Targets — Track 1 (Phase 3)

| Pipeline Stage | Current Tool | GPU Potential | Why |
|---------------|-------------|:------------:|-----|
| K-mer counting | DADA2 (R/C++) | **High** | Embarrassingly parallel, large data |
| Sequence alignment | BLAST/Bowtie2 | **High** | Smith-Waterman on GPU is well-studied |
| Taxonomic classify | Kraken2 | **High** | K-mer lookup, hash table operations |
| Distance matrices | scikit-bio | **High** | Large matrix operations → ToadStool |
| PCA/PCoA | sklearn | **High** | Eigensolve → ToadStool BatchedEigh |
| FFT (spectral) | scipy.fft | **High** | GPU FFT is mature |
| Peak detection | pyOpenMS | **Medium** | Signal processing on GPU |

### GPU Acceleration Targets — Track 2 / blueFish (Phase B3)

| Pipeline Stage | Current Tool | GPU Potential | Why |
|---------------|-------------|:------------:|-----|
| LC-MS peak detection | asari (scipy) | **High** | Parallel across scans, SIMD-like |
| Mass alignment | asari (numpy) | **High** | Large matrix operations |
| MS2 spectral matching | PFΔScreen/matchms | **High** | Cosine similarity, dot products |
| KMD analysis | PFΔScreen (numpy) | **Medium** | Mass defect arithmetic |
| PFAS suspect screening | pandas hash match | **High** | Hash table lookup on GPU |
| ML inference (RF/GBM) | scikit-learn | **Medium** | Tree traversal on GPU |
| Neural network (toxicity) | PyTorch | **High** | Standard GPU inference |
| MD force field eval | GROMACS (C++) | **High** | Particle-particle interactions |
| Trajectory analysis | MDAnalysis | **High** | Large array operations → ToadStool |

### Shared GPU Kernels (Both Tracks)

These ToadStool kernels serve **both tracks** and are useful beyond wetSpring:

| Kernel | Track 1 Use | Track 2 Use | General Use |
|--------|------------|------------|-------------|
| GPU FFT | Spectral analysis | Mass spec peak detection | Signal processing |
| GPU eigensolve | PCA/PCoA diversity | PCA of mass spec features | Linear algebra |
| GPU hash table | K-mer lookup | PFAS suspect screening | Database search |
| GPU cosine similarity | Sequence alignment scoring | MS2 spectral matching | Text/vector search |
| GPU reduction | Distance matrix summation | Feature quantification | MapReduce |
| GPU NN inference | (future) taxonomy classifier | Toxicity prediction | Any ML deployment |

---

## Evolution Analysis — What To Build

*Based on validated baseline pipelines (Exp001–006), profiled Feb 16, 2026.*

### Computational Kernels Extracted from Baselines

| Kernel | DADA2 (T1) | asari (T2) | FindPFAS (T2) | Shared |
|--------|:----------:|:----------:|:-------------:|:------:|
| FASTQ/mzML/MS2 parsing | FASTQ | mzML | MS2 | Parsers |
| Peak detection (find_peaks) | — | scipy | — | Signal |
| Gaussian fitting (curve_fit) | — | scipy | — | Fitting |
| Smoothing (uniform_filter1d) | — | scipy | — | Signal |
| LOWESS regression | Error model | RT alignment | — | Regression |
| K-mer counting / hashing | Core | — | — | Hash |
| Pairwise distance matrix | Seq comparison | m/z alignment | m/z meshgrid | MatOps |
| Tolerance search (ismembertol) | — | Centurion tree | Suspect + diag | Search |
| Hash table (dereplication) | Core | m/z buckets | Precursor dedup | Hash |
| Reduction / summation | Abundance | Peak area, SNR | Fragment counts | Reduce |
| Sort / argsort | Abundance order | m/z ordering | KMD series | Sort |
| PCA / eigensolve | PCoA diversity | Feature PCA | — | LinAlg |

### BarraCUDA Evolution — Rust Modules Needed

**Priority 1 — Parsers + Search (both tracks immediately):**

| Module | Replaces | Track 1 | Track 2 |
|--------|----------|:-------:|:-------:|
| `barracuda::io::fastq` | Python FASTQ parsing | DADA2 input | — |
| `barracuda::io::mzml` | pymzml / pyteomics | — | asari + FindPFAS |
| `barracuda::io::ms2` | pyteomics MS2 | — | FindPFAS input |
| `barracuda::search::tolerance` | ismembertolerance | Taxonomy lookup | Suspect + KMD |
| `barracuda::hash::kmer` | DADA2 dereplication | K-mer counting | — |
| `barracuda::hash::mz_bucket` | Centurion tree | — | m/z indexing |

**Priority 2 — Signal Processing:**

| Module | Replaces | Track 1 | Track 2 |
|--------|----------|:-------:|:-------:|
| `barracuda::signal::peaks` | scipy.signal.find_peaks | — | asari peak detect |
| `barracuda::signal::smooth` | uniform_filter1d | — | Mass track smoothing |
| `barracuda::signal::gaussian_fit` | scipy.optimize.curve_fit | — | Peak shape fitting |
| `barracuda::stats::lowess` | statsmodels LOWESS | DADA2 error model | asari RT align |

**Priority 3 — Linear Algebra:**

| Module | Replaces | Track 1 | Track 2 |
|--------|----------|:-------:|:-------:|
| `barracuda::linalg::distance` | skbio / numpy | Bray-Curtis, PCoA | m/z alignment |
| `barracuda::linalg::pca` | sklearn PCA | Diversity ordination | Feature PCA |

### ToadStool Inventory — What Already Exists (443 shaders, 15,700 tests)

*Reviewed Feb 16, 2026 from `phase1/toadstool/` pull.*

**Directly rewirable for wetSpring (exist in ToadStool today):**

| Existing Asset | Location | wetSpring Use |
|----------------|----------|---------------|
| `cosine_similarity.wgsl` | `shaders/math/` (f32) | MS2 spectral matching (T2) — **needs f64 variant** |
| `pairwise_distance.wgsl` | `shaders/math/` (f32, p-norm) | Bray-Curtis distance (T1), m/z meshgrid (T2) — **needs f64 + batched variant** |
| `Fft1DF64` / `Fft3DF64` | `ops/fft/` | Signal processing, peak detection assist |
| `eigh_f64.wgsl` / `BatchedEighGpu` | `shaders/linalg/` | PCA/PCoA eigensolve for diversity (T1) + feature PCA (T2) |
| `sum_reduce_f64.wgsl` | `shaders/reduce/` | Abundance sums, peak areas, fragment counts |
| `variance_reduce_f64.wgsl` | `shaders/reduce/` | Statistical spread of alpha diversity |
| `norm_reduce_f64.wgsl` | `shaders/reduce/` | L1/L2/Linf norms for distance matrices |
| `max_abs_diff_f64.wgsl` | `shaders/reduce/` | Convergence checks in iterative alignment |
| `cumsum_f64.wgsl` | `shaders/reduce/` | Cumulative distributions, rarefaction curves |
| `PipelineBuilder` | `pipeline/mod.rs` | Multi-kernel chaining (parse→peak→align→quantify) |
| `BufferPool` / `SolverBufferSet` | `device/tensor_context.rs` | Persistent buffers across LC-MS scans |
| `sparse_matvec_f64.wgsl` | `shaders/linalg/` | Sparse feature tables (most entries zero) |
| `CgGpu` / `BiCgStabGpu` | `ops/linalg/` | Large sparse linear systems (khipu network) |
| `batched_bisection_f64.wgsl` | `shaders/optimizer/` | Tolerance-based root finding (mass matching) |
| `prng_xoshiro.wgsl` | `shaders/numerical/` | Monte Carlo rarefaction, bootstrap CI |
| `weighted_dot_f64.wgsl` | `shaders/reduce/` | Weighted inner products (quadrature, integration) |

**New shaders needed (not in ToadStool yet):**

| Shader | Operation | Closest Ancestor | O(?) |
|--------|-----------|-----------------|------|
| `batched_distance_f64.wgsl` | All-pairs distance (m/z, sequence) | `pairwise_distance.wgsl` → lift to f64+batched | O(N²)→GPU |
| `hash_table_u64.wgsl` | GPU hash insert/lookup (k-mer, m/z) | New (no hash table shader exists) | O(1) amort |
| `tolerance_search_f64.wgsl` | Binary search with ± ppm tolerance | `batched_bisection_f64.wgsl` (adapt) | O(N log M) |
| `uniform_filter_f64.wgsl` | 1D moving average (mass track smooth) | New (signal processing) | O(N)/track |
| `find_peaks_f64.wgsl` | Local maxima + prominence detect | New (signal processing) | O(N)/track |
| `sort_f64.wgsl` | Bitonic sort (m/z ordering, abundance) | New (requires workgroup coordination) | O(N log²N) |
| `cosine_similarity_f64.wgsl` | f64 variant of existing shader | `cosine_similarity.wgsl` → lift to f64 | O(N)/pair |
| `bray_curtis_f64.wgsl` | Specialized ecological distance | `pairwise_distance.wgsl` (adapt formula) | O(N)/pair |

**Rewiring strategy — 3 tiers:**

```
Tier A: Pure rewire (existing shaders, new orchestrators only)
  cosine_similarity.wgsl ─→ f64 variant + MS2 spectral matcher Rust wrapper
  pairwise_distance.wgsl ─→ f64 variant + Bray-Curtis formula
  eigh_f64 + PCA ──────────→ PCoA ordination (T1 diversity, T2 feature PCA)
  sum/variance/norm reduce ─→ Alpha diversity metrics on GPU
  PipelineBuilder ──────────→ LC-MS multi-scan pipeline + 16S multi-sample pipeline
  prng_xoshiro + cumsum ───→ GPU rarefaction (random subsample + accumulate)

Tier B: Adapt existing patterns (minor shader modifications)
  batched_bisection_f64 ──→ tolerance_search_f64 (adapt objective function)
  weighted_dot_f64 ────────→ Gaussian peak fitting inner product
  sparse_matvec_f64 ──────→ Sparse feature table operations

Tier C: New shaders (fresh WGSL)
  hash_table_u64 ──────────→ K-mer hashing + m/z bucket indexing
  uniform_filter_f64 ──────→ Mass track smoothing
  find_peaks_f64 ──────────→ Peak detection (local maxima + prominence)
  sort_f64 ────────────────→ Bitonic merge sort on GPU
```

### Build Order

```
Phase 2 / B2: Rust I/O Ports (Exp004 + Exp007)
  ┌─ barracuda::io::fastq ──── validates vs Trimmomatic (Exp004)
  ├─ barracuda::io::mzml ───── validates vs pyteomics (Exp007)
  ├─ barracuda::io::ms2 ────── validates vs pyteomics
  ├─ barracuda::search::tolerance  validates vs ismembertolerance
  ├─ barracuda::hash::kmer ──── validates vs DADA2 dereplication
  └─ barracuda::hash::mz_bucket ─ validates vs Centurion tree

Phase 3A: GPU Tier A — Rewire existing ToadStool shaders
  ┌─ cosine_similarity_f64 ── f64 lift (MS2 matching)
  ├─ batched_distance_f64 ──── f64 lift + Bray-Curtis (diversity)
  ├─ GPU PCoA ─────────────── eigh_f64 on distance matrix (diversity)
  ├─ GPU rarefaction ───────── prng_xoshiro + cumsum (ecology)
  └─ GPU alpha diversity ──── reduce suite (Shannon, Simpson, Chao1)

Phase 3B: GPU Tier B — Adapt existing patterns
  ┌─ tolerance_search_f64 ──── from batched_bisection (suspect PFAS)
  ├─ sparse feature ops ────── from sparse_matvec_f64 (asari tables)
  └─ weighted peak fitting ─── from weighted_dot_f64 (Gaussian)

Phase 3C: GPU Tier C — New shaders
  ┌─ hash_table_u64 ────────── k-mer + m/z index (both tracks)
  ├─ find_peaks_f64 ────────── mass spec peak detect (T2)
  ├─ uniform_filter_f64 ────── mass track smoothing (T2)
  └─ sort_f64 ─────────────── bitonic sort (m/z ordering, abundance)
```

---

## Relationship to hotSpring

wetSpring follows the same validation methodology as hotSpring:

| | hotSpring | wetSpring T1 | wetSpring T2 (blueFish) |
|--|-----------|-------------|------------------------|
| Domain | Nuclear physics | Life science | Analytical chemistry |
| Validation target | Binding energies | Organism ID | PFAS ID + concentration |
| Baseline | Python scipy | Galaxy/QIIME2 | asari/PFΔScreen |
| Evolution | Rust BarraCUDA | Rust BarraCUDA | Rust BarraCUDA |
| GPU layer | ToadStool (wgpu) | ToadStool (wgpu) | ToadStool (wgpu) |
| Success metric | chi2 match | Same taxonomy | Same PFAS detected |
| Ultimate goal | Sovereign nuclear physics | Sovereign metagenomics | Penny water monitoring |

All three prove: sovereign compute on consumer hardware can replicate
institutional results, then exceed them via Rust+GPU.

### The ecoPrimals Thesis (Why Two Tracks)

- **hotSpring** validated that BarraCUDA/ToadStool can achieve institutional
  precision (nuclear physics, clean math, f64 throughout)
- **wetSpring Track 1** (life science) evolves shaders for messy biological
  data — sequence alignment, k-mer ops, FFT, diversity metrics
- **wetSpring Track 2** (PFAS/blueFish) evolves shaders for analytical
  chemistry — peak detection, spectral matching, mass defect, ML, MD

Each track produces GPU kernels useful **far beyond** its original domain.
Together they build a general-purpose sovereign compute platform.

---

*Initialized: February 12, 2026*
*Track 2 (blueFish) added: February 12, 2026*
*Galaxy 24.1 operational, FastQC validated: February 16, 2026*
*DADA2 denoise-paired complete (232 ASVs): February 16, 2026*
*Experiment 001 COMPLETE (SILVA taxonomy + barplot): February 16, 2026*
*Experiment 001 VALIDATED (8/8 deterministic, 71.5s): February 16, 2026*
*Experiment 002 COMPLETE (2273 ASVs, 41 phyla, real SRA data): February 16, 2026*
*Experiment 003 COMPLETE (phage assembly+annotation, 100% CheckV): February 16, 2026*
*Experiment 005 COMPLETE (asari: 5951 features, 4107 compounds, 15.6s): February 16, 2026*
*Experiment 006 COMPLETE (FindPFAS: 25 PFAS precursors, 2 suspect matches): February 16, 2026*
*Track 2 VALIDATED (8/8 deterministic, asari+FindPFAS): February 16, 2026*
*Evolution analysis written (BarraCUDA 12 modules, ToadStool 8 shaders): February 16, 2026*
*BarraCUDA Rust validation — 36/36 PASS (FASTQ, mzML, diversity, PFAS): February 16, 2026*
