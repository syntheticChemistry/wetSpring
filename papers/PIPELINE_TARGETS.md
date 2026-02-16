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

## Priority Order

1. **Pipeline 1 (16S amplicon)** — Most mature tools, public data
   abundant, directly replicates Pond Crash Forensics. Start here.
2. **Pipeline 2 (Phage annotation)** — Builds on Pipeline 1 assembly
   skills, directly replicates Biotic Countermeasures compute.
3. **Pipeline 3 (Spectral)** — Different data modality, exercises
   signal processing → PCA path that maps to ToadStool eigensolve.
4. **Pipeline 4 (VOC)** — Mass spec domain, lower priority but
   exercises peak detection + statistical modeling.

---

## Success Criteria (Matching hotSpring Pattern)

For each pipeline:
1. **Baseline**: Run in Galaxy, produce identical results to published
2. **Rust port**: Implement critical stages in Rust, validate output
3. **GPU acceleration**: Profile and accelerate via ToadStool
4. **Benchmark**: Rust+GPU vs Galaxy/Python, measure speedup
5. **Sovereignty**: No institutional compute required, runs on Eastgate
