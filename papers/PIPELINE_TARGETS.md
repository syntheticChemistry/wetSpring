# wetSpring — Computational Pipeline Targets

**Purpose:** Map each target paper's computational methods to specific
tools, datasets, and Rust evolution targets.

---

## Pipeline 1: 16S rRNA Amplicon Microbiome Profiling

**Source papers:** Pond Crash Forensics; Biotic Countermeasures

### What the papers compute

1. **Raw reads → QC**: Illumina paired-end 16S rRNA amplicon reads.
   Trimming of adapters and primers, quality filtering (Q>20).
2. **Denoising**: DADA2 or OTU clustering to produce Amplicon Sequence
   Variants (ASVs) or OTUs at 97% identity.
3. **Taxonomic classification**: Assign taxonomy to each ASV/OTU against
   SILVA or Greengenes reference databases. Identify crash agents:
   rotifer *Brachionus*, gastrotrich *Chaetonotus*, pathogen *B. safensis*.
4. **Diversity metrics**: Alpha diversity (Shannon, Simpson, Chao1),
   beta diversity (UniFrac, Bray-Curtis), and ordination (PCoA).
5. **Differential abundance**: Compare microbiome composition before/after
   crash events or phage treatment.

### Galaxy pipeline (Phase 0-1)

```
FASTQ → FastQC → Cutadapt/Trimmomatic → DADA2 denoise
     → Feature table → Taxonomy (SILVA classifier)
     → Alpha/beta diversity (QIIME2 core-metrics-phylogenetic)
     → Visualization (barplots, PCoA, heatmaps)
```

### Rust evolution targets (Phase 2-3)

| Stage | Python/Galaxy tool | Rust replacement | GPU potential |
|-------|-------------------|------------------|:------------:|
| FASTQ parsing | BioPython / cutadapt | `needletail` or custom | Low |
| Quality filtering | Trimmomatic | Custom Rust filter | Low |
| K-mer counting | DADA2 (R/C++) | Custom k-mer engine | **High** |
| Denoising (error model) | DADA2 core | Rust reimpl | **High** |
| Sequence alignment | vsearch / BLAST | Rust SW / minimap2-rs | **High** |
| Taxonomic classify | sklearn NBC / Kraken2 | Rust k-mer classifier | **High** |
| Diversity metrics | scikit-bio | Rust linalg (BarraCUDA) | Medium |
| Distance matrices | UniFrac (scipy) | Rust tree traversal + GPU | **High** |

### Public datasets for validation

- **Human Microbiome Project** (HMP): Well-characterized 16S dataset,
  standard benchmark for pipeline validation
- **Earth Microbiome Project** (EMP): Global survey, diverse environments
- **NCBI SRA**: Search for *Nannochloropsis* 16S amplicon studies
  (BioProject PRJNA* from Sandia if deposited)

---

## Pipeline 2: Phage Genome Annotation

**Source paper:** Biotic Countermeasures

### What the paper computes

1. **Phage isolation**: Bacteriophages isolated from algal cultures
2. **Genome sequencing**: Whole-genome sequencing of isolated phages
3. **Assembly**: De novo assembly from Illumina reads (SPAdes)
4. **Annotation**: Gene calling + functional annotation (Prokka, Pharokka)
5. **Taxonomy**: Phage classification (geNomad, CheckV for completeness)
6. **Host prediction**: Linking phages to bacterial hosts

### Galaxy pipeline (Phase 0-1)

```
FASTQ → FastQC → SPAdes assembly → Prokka/Pharokka annotation
     → geNomad classification → CheckV completeness
     → BLAST against phage databases (NCBI viral RefSeq)
```

### Rust evolution targets (Phase 2-3)

| Stage | Current tool | Rust replacement | GPU potential |
|-------|-------------|------------------|:------------:|
| Assembly | SPAdes (C++) | Rust de Bruijn graph | **High** |
| Gene calling | Prodigal (C) | Rust ORF finder | Low |
| Annotation | Prokka (Perl) | Rust HMM search | **High** |
| Classification | geNomad (Python) | Rust NN/k-mer classifier | **High** |

### Public datasets for validation

- **PhagesDB**: Curated phage genomes with annotations
- **NCBI Viral RefSeq**: Reference phage genomes
- **INPHARED**: Regularly updated phage genome database

---

## Pipeline 3: Spectroradiometric Anomaly Detection

**Source paper:** Spectroradiometric Detection

### What the paper computes

1. **Spectral acquisition**: Hyperspectral reflectance (350-2500 nm)
   every 5 minutes from raceway ponds
2. **Preprocessing**: Continuum removal, derivative spectra, SNV
3. **Feature extraction**: PCA, key absorption bands (Chl-a 680nm,
   carotenoid 500nm, water 1450nm)
4. **Classification**: PLS-DA or random forest to classify:
   healthy algae vs competitor diatom vs grazer contamination
5. **Time-series anomaly**: Detect onset of contamination from
   spectral trajectory changes

### Rust evolution targets (Phase 2-3)

| Stage | Python tool | Rust replacement | GPU potential |
|-------|------------|------------------|:------------:|
| Spectral I/O | numpy/pandas | Rust ndarray | Low |
| Derivatives | scipy.signal | Rust signal processing | Medium |
| PCA | sklearn.decomposition | BarraCUDA SVD/eigensolve | **High** |
| PLS-DA | sklearn.cross_decomposition | Rust PLS | Medium |
| Random Forest | sklearn.ensemble | Rust RF (smartcore) | Medium |
| FFT | scipy.fft | ToadStool GPU FFT | **High** |

---

## Pipeline 4: VOC Metabolomic Profiling

**Source paper:** Chemical Profiling of VOCs

### What the paper computes

1. **GC-MS data processing**: Peak detection, deconvolution, alignment
2. **Compound identification**: Match against NIST/Wiley mass spectral
   libraries
3. **Statistical analysis**: PCA, heatmaps, volcano plots comparing
   healthy vs crashing cultures
4. **Biomarker discovery**: Identify VOC signatures that predict crashes
   hours before visible symptoms

### Rust evolution targets (Phase 2-3)

| Stage | Python tool | Rust replacement | GPU potential |
|-------|------------|------------------|:------------:|
| Peak detection | pyopenms / xcms | Rust peak finder | **High** |
| Mass spec matching | matchms | Rust cosine similarity | **High** |
| Deconvolution | scipy.optimize | Rust NLS | Medium |
| Statistical modeling | statsmodels | Rust stats | Medium |

---

---

# Track 2: PFAS Analytical Chemistry (Codename: blueFish)

**Goal**: Break vendor-specific analytical instrument lock-in for water
chemistry monitoring. Replace proprietary LC-MS/HRMS data processing
pipelines with open-source Rust+GPU equivalents — ultimately enabling
live water monitoring for pennies.

**Principal Collaborator**: A. Daniel Jones, PhD — Professor, Biochemistry
& Molecular Biology, Michigan State University. Associate Director, MSU
Center for PFAS Research. ~25 years in analytical chemistry. Co-developed
**asari** (Nature Communications 2023), an open-source LC-MS metabolomics
processing tool. Leads MSU Mass Spectrometry and Metabolomics Core facility.

**Why PFAS**: Per- and polyfluoroalkyl substances ("forever chemicals") are
the defining environmental contamination crisis. Current detection requires
$500K+ instruments (LC-MS/MS) and vendor-locked software (Waters MassLynx,
Thermo Xcalibur, Agilent MassHunter, AB SCIEX Analyst). If we can replicate
and exceed these vendor pipelines in open-source Rust+GPU, we break the
analytical instrument monopoly for water safety.

---

## Pipeline 5: LC-MS Metabolomics Processing (asari)

**Source**: Li, S. et al. "Trackable and scalable LC-MS metabolomics data
processing using asari." Nature Communications 14, 4113 (2023).

### What asari computes

1. **mzML parsing**: Read vendor-neutral mass spectrometry data format
2. **Mass track extraction**: Extract ion chromatograms from 2D (m/z, RT) data
3. **Peak detection**: Find chromatographic peaks via composite scoring
4. **Mass alignment**: Align features across samples using reference mapping
5. **Feature table**: Produce sample × feature intensity matrix for statistics
6. **Quality control**: Filter features by blank ratio, CV, missingness

### Current tool chain

```
Raw vendor files → msconvert (ProteoWizard) → mzML
  → asari (Python) → Feature table
  → Statistical analysis (R/Python)
```

### Rust evolution targets

| Stage | Python tool | Rust replacement | GPU potential |
|-------|------------|------------------|:------------:|
| mzML parsing | pyteomics / asari | Rust XML/binary parser | Low |
| Mass track extraction | asari (numpy) | Rust ndarray + SIMD | Medium |
| Peak detection | asari (scipy) | Rust signal processing | **High** |
| Mass alignment | asari (custom) | Rust alignment engine | Medium |
| Feature quantification | asari (numpy) | Rust + GPU reduction | **High** |
| Statistical analysis | scipy/statsmodels | Rust stats + GPU linalg | **High** |

### Public datasets for validation

- **MetaboLights**: EMBL-EBI metabolomics repository (MTBLS studies)
- **Metabolomics Workbench**: NIH metabolomics data repository
- **asari demo data**: Test datasets from shuzhao-li-lab/data on GitHub

---

## Pipeline 6: PFAS Non-Targeted Screening (HRMS)

**Source papers**: Jones et al. (MSU Center for PFAS Research);
PFΔScreen (Zweigle et al., Analytical and Bioanalytical Chemistry 2023)

### What the pipeline computes

1. **HRMS data acquisition**: High-resolution mass spectrometry (LC-HRMS)
   of water/soil/biosolid samples, vendor-neutral mzML format
2. **Feature detection**: Extract MS1 features (m/z, RT, intensity) using
   pyOpenMS or XCMS
3. **PFAS prioritization**: Screen features for PFAS signatures:
   - Mass defect / carbon number (MD/C-m/C) analysis
   - Kendrick mass defect (KMD) for homologous series
   - Diagnostic fragment ions (MS2): CF3⁻ (m/z 68.9952), C2F5⁻ (118.9920),
     C3F7⁻ (168.9888), etc.
   - Fragment mass differences (ΔF = CF2 = 49.9968 Da)
4. **Compound annotation**: Match against PFAS spectral libraries and
   NORMAN SusDat suspect list (~65,000 PFAS entries)
5. **Quantification**: Semi-quantitative estimation using isotope-labeled
   internal standards

### Current tool chain

```
Raw vendor files → msconvert → mzML
  → pyOpenMS feature detection
  → PFΔScreen (Python) → PFAS candidate list
  → Manual verification + library matching
```

### Open-source tools to replicate

| Tool | Source | Language | Purpose |
|------|--------|----------|---------|
| PFΔScreen | github.com/JonZwe/PFAScreen | Python | PFAS feature prioritization |
| FindPFΔS | github.com/JonZwe/FindPFAS | Python | MS2 fragment mass difference mining |
| pyOpenMS | OpenMS.de | Python/C++ | Feature detection, mass spec core |
| msconvert | ProteoWizard | C++ | Vendor → mzML conversion |
| patRoon | github.com/rickhelmus/patRoon | R | Non-target screening framework |

### Rust evolution targets

| Stage | Python/C++ tool | Rust replacement | GPU potential |
|-------|----------------|------------------|:------------:|
| mzML I/O | pyteomics/pyOpenMS | Rust mzML parser | Low |
| Feature detection | pyOpenMS | Rust centroiding + peak pick | **High** |
| KMD analysis | PFΔScreen (numpy) | Rust mass defect engine | Medium |
| MS2 scoring | PFΔScreen (scipy) | Rust cosine similarity | **High** |
| Suspect screening | pandas matching | Rust hash-based lookup | **High** |
| Homologous series | PFΔScreen custom | Rust pattern matching | Medium |

### Public datasets for validation

- **NORMAN Digital Sample Freezing Platform**: Real HRMS datasets for
  non-target screening benchmarking
- **MassBank**: Open mass spectral database with PFAS reference spectra
- **EPA CompTox**: PFAS structure/property database (~14,000 PFAS)
- **MSU PFAS Center publications**: Supplementary HRMS data from
  Jones lab publications

---

## Pipeline 7: PFAS Machine Learning & Molecular Dynamics

**Source papers**: MSU Center for PFAS Research ML publications (2023-2025)

### What the papers compute

1. **Toxicity prediction**: ML screening of 260,000 PFAS structures for
   toxicity via molecular docking + MD simulations
2. **Drinking water monitoring**: Region-specific ML models predicting PFAS
   levels from water utility characteristics (Random Forest, GBM)
3. **Adsorption prediction**: GBDT models for PFAS removal efficiency on
   carbon-based materials
4. **Molecular dynamics**: PFAS-soil mineral interactions (kaolinite, clay),
   PFAS-receptor binding (PPARγ, thyroid, estrogen)
5. **Molecular probe design**: Active learning for designing selective PFAS
   capture molecules

### Current tool chain

```
PFAS SMILES/structures → RDKit molecular descriptors
  → scikit-learn / XGBoost / PyTorch models
  → GROMACS / AMBER molecular dynamics
  → Analysis (MDAnalysis, numpy)
```

### Rust evolution targets

| Stage | Python tool | Rust replacement | GPU potential |
|-------|------------|------------------|:------------:|
| Molecular descriptors | RDKit (C++/Python) | Rust cheminformatics | Medium |
| Random Forest / GBM | scikit-learn / XGBoost | Rust smartcore / LightGBM-rs | Medium |
| Neural networks | PyTorch | Rust burn / candle | **High** |
| MD simulation | GROMACS (C++) | Rust MD engine | **High** |
| Force field eval | GROMACS | Rust + ToadStool GPU | **High** |
| Trajectory analysis | MDAnalysis (Python) | Rust ndarray + GPU | **High** |

### Public datasets for validation

- **EPA CompTox PFAS**: Structures, properties, toxicity data
- **ToxCast/Tox21**: High-throughput screening assay results
- **PDBbind**: Protein-ligand binding affinity data
- **Michigan DEQ water data**: Public PFAS monitoring results

---

## Priority Order

### Track 1: Life Science (Algae / Metagenomics)
1. **Pipeline 1 (16S amplicon)** — Most mature tools, public data
   abundant, directly replicates Pond Crash Forensics. Start here.
2. **Pipeline 2 (Phage annotation)** — Builds on Pipeline 1 assembly
   skills, directly replicates Biotic Countermeasures compute.
3. **Pipeline 3 (Spectral)** — Different data modality, exercises
   signal processing → PCA path that maps to ToadStool eigensolve.
4. **Pipeline 4 (VOC)** — Mass spec domain, lower priority but
   exercises peak detection + statistical modeling.

### Track 2: PFAS Analytical Chemistry (blueFish)
5. **Pipeline 5 (LC-MS / asari)** — Open-source baseline exists
   (Nature Communications). Start here to learn mass spec data I/O.
6. **Pipeline 6 (PFAS non-target)** — PFΔScreen is open-source Python,
   directly replicable. Exercises HRMS data processing → Rust.
7. **Pipeline 7 (PFAS ML/MD)** — Heaviest compute, best GPU target.
   Molecular dynamics + neural nets → ToadStool kernels.

### Why Two Tracks

| Track | Exercises | Shader/GPU Kernel Value |
|-------|-----------|------------------------|
| Track 1 (Life Science) | Sequence alignment, k-mer ops, FFT | String matching, hash tables, signal processing |
| Track 2 (PFAS/Analytical) | Peak detection, mass defect math, ML, MD | Spectral correlation, force field eval, NN inference |

Both tracks produce GPU kernels that are useful **far beyond** their
original domain — exactly the ecoPrimals thesis.

---

## blueFish Vision: Breaking Vendor Lock-In

The ultimate goal of Track 2 is codename **blueFish**: an open-source,
GPU-accelerated analytical chemistry engine that replaces vendor-locked
instrument software.

### Current State of Analytical Chemistry Software

| Vendor | Instrument | Software | License | Cost |
|--------|-----------|----------|---------|------|
| Waters | Xevo TQ-XS | MassLynx | Proprietary | $$$$ |
| Thermo Fisher | Orbitrap | Xcalibur/FreeStyle | Proprietary | $$$$ |
| Agilent | 6546 QTOF | MassHunter | Proprietary | $$$$ |
| AB SCIEX | TripleTOF | Analyst/SCIEX OS | Proprietary | $$$$ |
| Shimadzu | LCMS-9050 | LabSolutions | Proprietary | $$$$ |

Every instrument ships with vendor-locked data processing. Scientists
cannot easily compare results across instruments, reproduce analyses
on different hardware, or modify algorithms.

### blueFish Roadmap

```
Phase B0: Replicate vendor pipelines with open-source Python (asari, PFΔScreen)
Phase B1: Port to Rust — vendor-neutral mzML → features → PFAS screen
Phase B2: GPU-accelerate critical paths (peak detection, spectral matching, ML)
Phase B3: Real-time processing — stream from instrument → Rust+GPU → results
Phase B4: Penny monitoring — low-cost sensor + Rust+GPU replaces $500K LC-MS
```

The endgame: a Raspberry Pi + cheap sensor + Rust+GPU backend that monitors
water quality in real time, for pennies per sample. The analytical intelligence
moves from the instrument to the software.

---

## Success Criteria (Matching hotSpring Pattern)

For each pipeline:
1. **Baseline**: Run in Galaxy/Python, produce identical results to published
2. **Rust port**: Implement critical stages in Rust, validate output
3. **GPU acceleration**: Profile and accelerate via ToadStool
4. **Benchmark**: Rust+GPU vs Galaxy/Python, measure speedup
5. **Sovereignty**: No institutional compute required, runs on Eastgate
