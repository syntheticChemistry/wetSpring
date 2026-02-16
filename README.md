# wetSpring

**Life Science & Analytical Chemistry Computational Validation Study**

wetSpring is the life science and analytical chemistry counterpart to
hotSpring (nuclear physics). It operates on two tracks:

- **Track 1 (Life Science)**: Re-examine and rerun computational biology
  pipelines from published algal biotechnology and metagenomics research —
  work from Sandia National Laboratories on raceway algae pond crash forensics
  and phage-based biocontrol.

- **Track 2 (PFAS Analytical Chemistry — codename blueFish)**: Replicate and
  replace vendor-locked mass spectrometry data processing pipelines for PFAS
  ("forever chemical") detection in water — building toward live water
  monitoring for pennies.

Both tracks progressively evolve each pipeline from Python/Galaxy/vendor
toolchains to Rust via BarraCUDA/ToadStool, producing GPU shaders and
kernels useful far beyond their original domain.

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

### Track 1: Algae / Metagenomics (Sandia)

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

### Track 2: PFAS Analytical Chemistry (MSU — codename blueFish)

This work builds on the analytical chemistry research of **A. Daniel Jones**
(Professor, Biochemistry & Molecular Biology, Michigan State University;
Associate Director, MSU Center for PFAS Research; ~25 years in analytical
chemistry) and the broader MSU PFAS research program, focusing on:

- **LC-MS metabolomics** — open-source data processing via **asari**
  (Nature Communications 2023), replacing proprietary vendor pipelines
- **PFAS non-targeted screening** — detecting unknown PFAS in water/soil
  via high-resolution mass spectrometry without reference standards
- **Machine learning for water monitoring** — region-specific ML models
  predicting PFAS contamination in drinking water
- **Molecular dynamics** — computational modeling of PFAS-soil/receptor
  interactions for toxicity prediction

The long-term vision (**blueFish**): replace $500K vendor-locked analytical
instruments (Waters MassLynx, Thermo Xcalibur, Agilent MassHunter) with
open-source Rust+GPU software — enabling live water monitoring for pennies.

### The ecoPrimals Thesis

hotSpring proved that sovereign compute on consumer hardware can match
institutional nuclear physics results via precise Rust+GPU (BarraCUDA/
ToadStool). wetSpring uses "messy" life science and analytical chemistry
data to evolve **more shaders and GPU kernels** — peak detection, spectral
correlation, sequence alignment, force field evaluation, neural network
inference — that are useful far beyond their original domain.

---

## Target Papers

### Track 1: Algae / Metagenomics

#### Paper 1: Pond Crash Forensics
**"Pond Crash Forensics: Presumptive identification of pond crash agents by
next generation sequencing in replicate raceway mass cultures of
*Nannochloropsis salina*"**
- DOI: OSTI 1271304
- Methods: 16S rRNA amplicon sequencing, microbiome profiling
- Compute: QIIME/DADA2-style OTU/ASV pipeline, taxonomic classification
- Data: Replicate raceway time-series before/after crash events
- Organisms identified: rotifer *Brachionus*, gastrotrich *Chaetonotus*

#### Paper 2: Phage Biocontrol
**"Biotic countermeasures that rescue *Nannochloropsis gaditana* from a
*Bacillus safensis* infection"**
- DOI: OSTI 2311389
- Methods: 16S rRNA amplicon sequencing, phage isolation, bacteriome
  transplant experiments
- Compute: Amplicon pipeline + phage genome annotation + diversity metrics
- Key finding: Phage therapy eliminates pathogen without disrupting
  protective bacteriome; bacteriome transplant restores resilience

#### Paper 3: Spectroradiometric Detection
**"Spectroradiometric detection of competitor diatoms and the grazer
*Poteriochromonas* in algal cultures"**
- DOI: OSTI 1828029
- Methods: Hyperspectral reflectance every 5 minutes, statistical
  classification of algal health states
- Compute: Time-series spectral analysis, anomaly detection, PCA/PLS
  classification

#### Paper 4: VOC Early Warning
**"Chemical Profiling of Volatile Organic Compounds in the Headspace of
Algal Cultures as Early Biomarkers of Algal Pond Crashes"**
- DOI: OSTI 1570268
- Methods: GC-MS headspace analysis, metabolomic profiling
- Compute: Peak detection, compound identification, statistical modeling
  of crash precursors

### Track 2: PFAS Analytical Chemistry (blueFish)

#### Paper 5: LC-MS Metabolomics Processing (asari)
**"Trackable and scalable LC-MS metabolomics data processing using asari"**
- DOI: 10.1038/s41467-023-39889-1 (Nature Communications 2023)
- Authors: Li, S. et al. (Jones lab affiliation)
- Methods: LC-MS feature extraction, mass alignment, peak detection
- Code: github.com/shuzhao-li-lab/asari (Python, open source)
- Data: Demo datasets in shuzhao-li-lab/data repository

#### Paper 6: PFAS Non-Targeted Screening
**"PFΔScreen — an open-source tool for automated PFAS feature prioritization
in non-target HRMS data"**
- DOI: 10.1007/s00216-023-05070-2 (Anal. Bioanal. Chem. 2023)
- Authors: Zweigle, J. et al.
- Methods: Mass defect analysis, KMD, diagnostic fragments, MS2 scoring
- Code: github.com/JonZwe/PFAScreen (Python + pyOpenMS)
- Also: FindPFΔS (github.com/JonZwe/FindPFAS) — MS2 fragment mining

#### Paper 7: Femtosecond Laser Ionization MS for PFAS
**"Quantitative identification of nonpolar perfluoroalkyl substances by
mass spectrometry"** (Jones et al., MSU)
- Methods: Femtosecond laser ionization — PFAS detection WITHOUT
  chromatographic separation (key to low-cost monitoring)
- Significance: Removes the most expensive component (LC column + time)

#### Paper 8: ML for PFAS Drinking Water Monitoring
**"Using machine learning techniques to monitor PFAS in Michigan drinking
water"** (MSU, Water Research 2023)
- Methods: Region-specific Random Forest / GBM models predicting PFAS
  levels from water utility metadata
- Data: Michigan DEQ public water monitoring data

#### Paper 9: PFAS Molecular Dynamics
**"Molecular screening and toxicity estimation of 260,000 PFAS through
machine learning"** (MSU Center for PFAS Research)
- Methods: ML + molecular docking + MD simulations for toxicity screening
- Compute: GROMACS/AMBER MD, RDKit descriptors, scikit-learn/PyTorch

---

## Roadmap

### Track 1: Life Science (Phases 0-4)

#### Phase 0: Galaxy Hosting (Current)
- Self-host Galaxy Project via Docker on Eastgate
- Import standard bioinformatics tools (QIIME2, DADA2, FastQC, Trimmomatic,
  BLAST, Kraken2, MetaPhlAn, SPAdes, Prokka)
- Validate with public 16S rRNA datasets (Human Microbiome Project, etc.)
- Establish baseline: tool versions, runtimes, reproducibility

#### Phase 1: Pipeline Replication
- Replicate Paper 1 (Pond Crash Forensics) pipeline end-to-end
- Replicate Paper 2 (Phage Biocontrol) bioinformatics

#### Phase 2: Rust Evolution (BarraCUDA)
- FASTQ parsing, k-mer counting, denoising, alignment, classification
- Validate: identical results to Galaxy baseline

#### Phase 3: GPU Acceleration (ToadStool)
- GPU sequence alignment, k-mer counting, FFT, peak detection
- Benchmark: Rust+GPU vs Python+Galaxy

#### Phase 4: Sovereign Pipeline
- End-to-end Rust: FASTQ → taxonomy → diversity → report

### Track 2: PFAS Analytical Chemistry — blueFish (Phases B0-B4)

#### Phase B0: Open-Source Baseline (Start Here)
- Install and run **asari** on demo LC-MS datasets
- Install and run **PFΔScreen** on public HRMS data
- Validate: reproduce published feature tables and PFAS identifications
- Establish baseline: tool versions, runtimes, accuracy

#### Phase B1: Pipeline Replication
- Replicate Jones lab LC-MS metabolomics workflow (asari → feature table)
- Replicate PFΔScreen PFAS screening on contaminated water/soil data
- Replicate MSU ML models for PFAS drinking water prediction
- Compare: same PFAS identified? same concentrations?

#### Phase B2: Rust Evolution (BarraCUDA)
- Port mzML parsing to Rust (vendor-neutral mass spec I/O)
- Port peak detection and mass alignment to Rust
- Port PFAS mass defect / KMD analysis to Rust
- Port ML models to Rust (smartcore / burn)
- Validate: identical results to Python baseline

#### Phase B3: GPU Acceleration (ToadStool)
- GPU peak detection (embarrassingly parallel across scans)
- GPU spectral cosine similarity (MS2 library matching)
- GPU force field evaluation (molecular dynamics)
- GPU neural network inference (toxicity prediction)
- Benchmark: Rust+GPU vs Python for each stage

#### Phase B4: Penny Monitoring (blueFish Endgame)
- Real-time streaming: instrument → Rust+GPU → results
- Low-cost sensor integration (replace $500K LC-MS)
- Edge deployment (Raspberry Pi + GPU backend)
- Live water monitoring for pennies per sample

---

## Project Structure

```
wetSpring/
  barracuda/           — Rust crate (future: BarraCUDA life science + analytical modules)
    src/
  control/
    galaxy/            — Galaxy Project Docker deployment (Track 1)
    asari/             — asari LC-MS processing config (Track 2)
    pfascreen/         — PFΔScreen PFAS screening config (Track 2)
    pipelines/         — Workflow definitions (.ga, .yaml)
  data/                — Public datasets, SRA accessions, reference DBs
  experiments/         — Numbered experiment logs (like hotSpring)
    001-004            — Track 1: Life science experiments
    005-007+           — Track 2: PFAS analytical chemistry experiments
  papers/              — Target paper analysis and replication notes
  scripts/             — Helper scripts (download, preprocess, validate)
```

---

## Relationship to hotSpring

| Aspect | hotSpring | wetSpring Track 1 | wetSpring Track 2 |
|--------|-----------|-------------------|-------------------|
| Domain | Nuclear physics (HFB) | Life science (metagenomics) | Analytical chemistry (PFAS) |
| Baseline | Python scipy/numpy | Python Galaxy/QIIME2 | Python asari/PFΔScreen |
| Runtime | Rust (BarraCUDA) | Rust (BarraCUDA) | Rust (BarraCUDA) |
| GPU | ToadStool (wgpu f64) | ToadStool (f64/f32) | ToadStool (f64/f32) |
| Validation | chi2 vs AME2020 | Organism ID vs papers | PFAS ID vs standards |
| Hardware | Eastgate (RTX 4070) | Eastgate (RTX 4070) | Eastgate (RTX 4070) |

All three share: replicate published results on sovereign hardware, then
progressively migrate to Rust+GPU, validating accuracy at every step.

---

## References

### Track 1: Life Science

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

### Track 2: PFAS Analytical Chemistry (blueFish)

5. Li, S. et al. "Trackable and scalable LC-MS metabolomics data processing
   using asari." Nature Communications 14, 4113 (2023).

6. Zweigle, J. et al. "PFΔScreen — an open-source tool for automated PFAS
   feature prioritization in non-target HRMS data." Analytical and
   Bioanalytical Chemistry (2023).

7. Jones, A.D. et al. "Quantitative identification of nonpolar perfluoroalkyl
   substances by mass spectrometry." MSU Center for PFAS Research.

8. Nejadhashemi, A.P. et al. "Using machine learning techniques to monitor
   PFAS in Michigan drinking water." Water Research (2023).

9. MSU Center for PFAS Research. "Molecular screening and toxicity estimation
   of 260,000 PFAS through machine learning." (2022).

### Platforms & Tools

10. The Galaxy Project. https://galaxyproject.org/ (accessed Feb 2026).
11. asari. https://github.com/shuzhao-li-lab/asari (accessed Feb 2026).
12. PFΔScreen. https://github.com/JonZwe/PFAScreen (accessed Feb 2026).
