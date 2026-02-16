# wetSpring

**Life Science Computational Validation Study**

wetSpring is the life science counterpart to hotSpring (nuclear physics). It
re-examines and reruns computational biology pipelines from published algal
biotechnology and metagenomics research — starting with work from Sandia
National Laboratories on raceway algae pond crash forensics and phage-based
biocontrol — then progressively evolves each pipeline from Python/Galaxy
toolchains to Rust via BarraCUDA/ToadStool.

**License:** AGPL-3.0-or-later

---

## Quick Start (Full Replication)

```bash
git clone git@github.com:syntheticChemistry/wetSpring.git
cd wetSpring

# 1. Install tools (SRA Toolkit for NCBI data, checks Docker/Rust)
./scripts/setup_tools.sh

# 2. Download public datasets (~5-10 GB from NCBI SRA, Zenodo, SILVA)
./scripts/download_data.sh --all

# 3. Start Galaxy bioinformatics platform (Docker, first run pulls ~4 GB)
./scripts/start_galaxy.sh

# Galaxy is now at http://localhost:8080 — follow experiments/ for protocols
```

Requirements: Linux (Ubuntu/Pop!_OS), Docker 20.10+, ~20 GB disk.
No institutional access needed. All data is public.

---

## Origin

This work builds on published research by **Chuck Smallwood** (Principal
Member of Technical Staff, Sandia) and **Jesse Cahill** (Senior Member of
Technical Staff, Sandia) in the Bioscience group, focusing on:

- **Raceway algae pond crash forensics** — identifying microbial pathogens
  and predators that cause sudden collapse of algal biofuel cultures using
  next-generation sequencing of 16S rRNA amplicons
- **Phage-based biocontrol** — engineering bacteriophages to eliminate
  bacterial pathogens from microalgae systems without disrupting the
  protective bacteriome
- **Spectroradiometric monitoring** — real-time hyperspectral detection of
  competitor organisms in open raceway ponds

The computational methods in these studies (NGS pipelines, microbiome
profiling, phage genome annotation, spectral time-series analysis) are
excellent candidates for sovereign replication and GPU acceleration.

---

## Target Papers

### Paper 1: Pond Crash Forensics
**"Pond Crash Forensics: Presumptive identification of pond crash agents by
next generation sequencing in replicate raceway mass cultures of
*Nannochloropsis salina*"**
- DOI: OSTI 1271304
- Methods: 16S rRNA amplicon sequencing, microbiome profiling
- Compute: QIIME/DADA2-style OTU/ASV pipeline, taxonomic classification
- Data: Replicate raceway time-series before/after crash events
- Organisms identified: rotifer *Brachionus*, gastrotrich *Chaetonotus*

### Paper 2: Phage Biocontrol
**"Biotic countermeasures that rescue *Nannochloropsis gaditana* from a
*Bacillus safensis* infection"**
- DOI: OSTI 2311389
- Methods: 16S rRNA amplicon sequencing, phage isolation, bacteriome
  transplant experiments
- Compute: Amplicon pipeline + phage genome annotation + diversity metrics
- Key finding: Phage therapy eliminates pathogen without disrupting
  protective bacteriome; bacteriome transplant restores resilience

### Paper 3: Spectroradiometric Detection
**"Spectroradiometric detection of competitor diatoms and the grazer
*Poteriochromonas* in algal cultures"**
- DOI: OSTI 1828029
- Methods: Hyperspectral reflectance every 5 minutes, statistical
  classification of algal health states
- Compute: Time-series spectral analysis, anomaly detection, PCA/PLS
  classification

### Paper 4: VOC Early Warning
**"Chemical Profiling of Volatile Organic Compounds in the Headspace of
Algal Cultures as Early Biomarkers of Algal Pond Crashes"**
- DOI: OSTI 1570268
- Methods: GC-MS headspace analysis, metabolomic profiling
- Compute: Peak detection, compound identification, statistical modeling
  of crash precursors

---

## Roadmap

### Phase 0: Galaxy Hosting (Current)
- Self-host Galaxy Project via Docker on Eastgate
- Import standard bioinformatics tools (QIIME2, DADA2, FastQC, Trimmomatic,
  BLAST, Kraken2, MetaPhlAn, SPAdes, Prokka)
- Validate with public 16S rRNA datasets (Human Microbiome Project, etc.)
- Establish baseline: tool versions, runtimes, reproducibility

### Phase 1: Pipeline Replication
- Replicate Paper 1 (Pond Crash Forensics) pipeline end-to-end
  - Obtain or simulate comparable 16S amplicon data from public sources
    (SRA accessions, NCBI BioProject)
  - Run through Galaxy: QC → trimming → denoising → taxonomy → diversity
  - Compare: same organisms identified? same relative abundances?
- Replicate Paper 2 (Phage Biocontrol) bioinformatics
  - 16S amplicon + phage genome annotation (Pharokka, geNomad, CheckV)
  - Bacteriome diversity analysis (Shannon, Simpson, UniFrac)

### Phase 2: Rust Evolution (BarraCUDA)
- Port critical pipeline stages to Rust:
  - FASTQ parsing and quality filtering (replace Trimmomatic)
  - K-mer counting and denoising (replace DADA2 core)
  - Sequence alignment / classification (replace BLAST/Kraken2 core)
  - Spectral analysis and PCA (replace Python scipy/sklearn)
- Validate: identical results to Galaxy baseline
- Profile: identify GPU-accelerable stages (k-mer ops, alignment, FFT)

### Phase 3: GPU Acceleration (ToadStool)
- GPU-accelerated sequence alignment (Smith-Waterman / BLAST-like)
- GPU k-mer counting and de Bruijn graph construction
- GPU spectral FFT and anomaly detection
- GPU metabolomic peak detection
- Benchmark: Rust+GPU vs Python+Galaxy for each pipeline stage

### Phase 4: Integrated Sovereign Pipeline
- End-to-end Rust pipeline: FASTQ → taxonomy → diversity → report
- No Python dependencies, no Galaxy dependency
- Reproducible on consumer hardware (same Eastgate gate as hotSpring)
- Publishable: methods paper comparing sovereign vs institutional pipelines

---

## Project Structure

```
wetSpring/
  barracuda/         — Rust crate (future: BarraCUDA life science modules)
    src/
  control/
    galaxy/          — Galaxy Project Docker deployment
    pipelines/       — Galaxy workflow definitions (.ga files)
  data/              — Public datasets, SRA accessions, reference DBs
  experiments/       — Numbered experiment logs (like hotSpring)
  papers/            — Target paper analysis and replication notes
  scripts/           — Helper scripts (download, preprocess, validate)
```

---

## Relationship to hotSpring

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Nuclear physics (EOS, HFB) | Life science (metagenomics, phage) |
| Baseline tools | Python (scipy, numpy) | Python (Galaxy, QIIME2, DADA2) |
| Target runtime | Rust (BarraCUDA) | Rust (BarraCUDA) |
| GPU acceleration | ToadStool (wgpu f64) | ToadStool (wgpu f64/f32) |
| Validation method | chi2 vs experimental data | Organism ID vs published results |
| Hardware gate | Eastgate (RTX 4070) | Eastgate (RTX 4070) |

Both projects follow the same pattern: replicate published computational
results on sovereign hardware, then progressively migrate to Rust+GPU,
validating accuracy at every step.

---

## References

1. Smallwood, C. et al. "Pond Crash Forensics: Presumptive identification
   of pond crash agents by next generation sequencing in replicate raceway
   mass cultures of *Nannochloropsis salina*." Algal Research (2016).

2. Smallwood, C., Cahill, J. et al. "Biotic countermeasures that rescue
   *Nannochloropsis gaditana* from a *Bacillus safensis* infection."
   Sandia National Laboratories (2023). OSTI 2311389.

3. Lane, T. et al. "Spectroradiometric detection of competitor diatoms
   and the grazer *Poteriochromonas* in algal cultures." Sandia National
   Laboratories (2021). OSTI 1828029.

4. Lane, T. et al. "Chemical Profiling of Volatile Organic Compounds in
   the Headspace of Algal Cultures as Early Biomarkers of Algal Pond
   Crashes." Sandia National Laboratories (2019). OSTI 1570268.

5. The Galaxy Project. https://galaxyproject.org/ (accessed Feb 2026).
