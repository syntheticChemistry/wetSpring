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

## Current Status

```
wetspring-barracuda v0.1.0
  284 tests, 30 modules, 0 clippy pedantic warnings
  388/388 CPU validation checks PASS (incl. real public NCBI data benchmarked against papers)
  106/106 GPU validation checks PASS (38 diversity + 68 pipeline parity)
  494/494 total quantitative checks PASS
  1 runtime dep (flate2); GPU deps feature-gated (barracuda, wgpu v22, tokio)
  11 ToadStool GPU primitives, 0 custom WGSL shaders
  100% math parity: CPU ↔ GPU (Shannon, Simpson, chimera, taxonomy — zero error)
  13.4× faster than Galaxy (Rust CPU); GPU scaling benchmark 20–52× (5–500 queries)
  Taxonomy: 188× faster via compact GEMM (HashMap → flat array → GPU GEMM)
  Chimera: 1,256× faster (O(N³) → k-mer sketch + prefix-sum)
  Sovereign parsers: FASTQ, mzML/XML, MS2, base64 — all in-tree

  16S pipeline:  FASTQ → quality → merge → derep → DADA2 → chimera → taxonomy → diversity → UniFrac → PCoA
  LC-MS pipeline: mzML → mass tracks → EIC → peaks → feature table
  PFAS pipeline:  mzML/MS2 → tolerance search → spectral match → KMD → homologue grouping

  GPU benchmark: 1,077× speedup on spectral cosine (200×200) — RTX 4070
  Pipeline benchmark: Rust CPU 7.3s vs Galaxy 95.6s (10 samples); GPU streaming 6.0s (1.21×)
  ToadStool wired: GemmCached + BufferPool (73.8% reuse) + TensorContext
  Streaming GPU: taxonomy GEMM 11ms avg + diversity FMR 0ms; scaling 20–52×
```

| Track | Phase | Status |
|-------|-------|--------|
| T1 Life Science | Phase 4 (Sovereign) | **Done** — complete 16S pipeline: DADA2 + chimera + taxonomy + UniFrac |
| T2 PFAS/blueFish | Phase B2 (Rust) | **Done** — mzML + MS2 + PFAS + feature extraction validated |
| Both | GPU (ToadStool) | **Done** — 38/38 GPU PASS, 1,077× spectral cosine speedup |
| Both | Benchmarks | **Done** — Python vs Rust CPU vs Rust GPU three-tier comparison |

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
- Carney, L.T. et al. *Algal Research* 17 (2016). DOI: [10.1016/j.algal.2016.05.011](https://doi.org/10.1016/j.algal.2016.05.011). OSTI 1271304.
- Methods: 16S rRNA amplicon sequencing, microbiome profiling
- Compute: QIIME/DADA2-style OTU/ASV pipeline, taxonomic classification
- Data: Replicate raceway time-series before/after crash events
- Organisms identified: rotifer *Brachionus*, gastrotrich *Chaetonotus*

#### Paper 2: Phage Biocontrol
**"Biotic countermeasures that rescue *Nannochloropsis gaditana* from a
*Bacillus safensis* infection"**
- Humphrey, B. et al. (incl. Smallwood, C.R. and Cahill, J.) *Frontiers in Microbiology* (2023). OSTI 2311389.
- Methods: 16S rRNA amplicon sequencing, phage isolation, bacteriome
  transplant experiments
- Compute: Amplicon pipeline + phage genome annotation + diversity metrics
- Key finding: Phage therapy eliminates pathogen without disrupting
  protective bacteriome; bacteriome transplant restores resilience

#### Paper 3: Spectroradiometric Detection
**"Spectroradiometric detection of competitor diatoms and the grazer
*Poteriochromonas* in algal cultures"**
- Reichardt, T.A. et al. *Algal Research* (2020). DOI: [10.1016/j.algal.2020.102020](https://doi.org/10.1016/j.algal.2020.102020). OSTI 1828029.
- Methods: Hyperspectral reflectance every 5 minutes, statistical
  classification of algal health states
- Compute: Time-series spectral analysis, anomaly detection, PCA/PLS
  classification

#### Paper 4: VOC Early Warning
**"Chemical Profiling of Volatile Organic Compounds in the Headspace of
Algal Cultures as Early Biomarkers of Algal Pond Crashes"**
- Reese, K.L. et al. (incl. Jones, A.D.) *Scientific Reports* 9 (2019). DOI: [10.1038/s41598-019-50125-z](https://doi.org/10.1038/s41598-019-50125-z). OSTI 1570268.
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

#### Phase 0: Galaxy Hosting — DONE
- Galaxy 24.1 Docker on Eastgate, 32 tools installed (QIIME2, DADA2, etc.)
- Validated with MiSeq SOP (Zenodo 800651), SILVA 138 taxonomy

#### Phase 1: Pipeline Replication — DONE
- Exp001: 232 ASVs, 124K reads, 9 phyla (deterministic, 8/8 PASS)
- Exp002: 2,273 ASVs from real SRA data (PRJNA1195978), 41 phyla
- Exp003: Phage assembly + annotation (100% CheckV completeness)

#### Phase 2: Rust Evolution (BarraCUDA) — DONE
- Sovereign FASTQ parser (gzip-aware, streaming, no needletail)
- K-mer counting (2-bit canonical), alpha/beta diversity (Shannon/Simpson/Chao1/BC)
- `Validator` harness: typed `check_count`/`check_count_u64` — zero precision-loss casts
- NaN-safe tolerance search, Result-based error propagation throughout

#### Phase 3: GPU Acceleration (ToadStool) — DONE
- `GpuF64` bridge: wgpu device → ToadStool `WgpuDevice`/`TensorContext`
- Shannon + Simpson rewired to ToadStool `FusedMapReduceF64`
- Observed features, Pielou evenness, full alpha diversity on GPU
- Pairwise spectral cosine similarity via `GemmF64` + `FusedMapReduceF64`
- PCoA ordination: CPU Jacobi + GPU `BatchedEighGpu` eigendecomposition
- Bray-Curtis → `BrayCurtisF64` (absorbed upstream, custom shader deleted)
- Spatial interpolation via `KrigingF64` (ordinary + simple kriging, 4 variograms)
- Statistics: variance, std dev, correlation, covariance, weighted dot via
  `VarianceF64`, `CorrelationF64`, `CovarianceF64`, `WeightedDotF64`
- wgpu 0.19 → v22 migration (matching ToadStool's deep debt evolution)
- **38/38 GPU validation PASS** (RTX 4070, SHADER_F64, wgpu v22)
- **Zero custom WGSL shaders** — all GPU through ToadStool primitives
- Found/fixed 2× coefficient bug in ToadStool `math_f64.wgsl` log_f64 (absorbed upstream)

#### Phase 4: Sovereign Pipeline — DONE
- Quality filtering + adapter trimming (Trimmomatic/Cutadapt equivalent)
- Paired-end read merging (VSEARCH/FLASH equivalent)
- Dereplication with abundance tracking (VSEARCH equivalent)
- **DADA2 denoising** (Callahan et al. 2016) — error model + divisive partitioning
- **Chimera detection** (UCHIME-style reference-free) — two-parent crossover model
- **Taxonomy classification** (naive Bayes on SILVA 138) — 8-mer features, bootstrap CI
- **UniFrac distance** (Lozupone & Knight 2005) — weighted + unweighted, Newick parser
- 1D peak detection (`scipy.signal.find_peaks` equivalent)
- EIC extraction + mass track detection
- Feature table extraction (asari pipeline equivalent)
- MS2 cosine similarity (matched + weighted)
- Kendrick mass defect analysis + PFAS homologue grouping
- GPU-promoted EIC extraction + batch peak integration
- GPU-promoted rarefaction with bootstrap confidence intervals
- **388/388 CPU + 38/38 GPU = 426/426 validation checks PASS** (public NCBI data benchmarked against papers)

### Track 2: PFAS Analytical Chemistry — blueFish (Phases B0-B4)

#### Phase B0: Open-Source Baseline — DONE
- asari 1.13.1: 5,951 features, 4,107 compounds from MT02 demo data
- FindPFAS: 25 PFAS precursors from standard mix (exact match)

#### Phase B1: Pipeline Replication — PENDING
- Replicate Jones/MSU LC-MS and PFAS pipelines on real water data

#### Phase B2: Rust Evolution (BarraCUDA) — DONE
- Sovereign mzML parser (in-tree XML pull parser, base64+zlib decoding)
- Sovereign MS2 parser (streaming)
- ppm-tolerance search + PFAS fragment screening
- MS2 cosine similarity (matched + weighted) + Stein-Scott weighting
- Kendrick mass defect analysis + PFAS homologue grouping
- 1D peak detection, EIC extraction, feature table construction
- **7/7 mzML + 10/10 PFAS validation PASS**, streaming I/O

#### Phase B3: GPU Acceleration (ToadStool) — DONE
- GPU pairwise spectral cosine similarity via `GemmF64` + `FusedMapReduceF64`
- **31/31 GPU validation PASS** including spectral match checks
- See `EVOLUTION_READINESS.md` for remaining promotion targets

#### Phase B4: Penny Monitoring (blueFish Endgame)
- Real-time: instrument → Rust+GPU → results
- Low-cost sensor integration, edge deployment

---

## Project Structure

```
wetSpring/
  barracuda/                — Rust crate: life science + analytical chemistry pipelines
    src/
      bio/                  — Bioinformatics + analytical chemistry algorithms (30 modules)
        quality.rs          —   Quality filtering + adapter trimming (Trimmomatic/Cutadapt)
        merge_pairs.rs      —   Paired-end read merging (VSEARCH/FLASH)
        derep.rs            —   Dereplication + abundance tracking (VSEARCH)
        dada2.rs            —   DADA2 ASV denoising (Callahan et al. 2016)
        chimera.rs          —   UCHIME-style reference-free chimera detection
        taxonomy.rs         —   Naive Bayes taxonomy classification (RDP/SILVA)
        kmer.rs             —   2-bit canonical k-mer counting
        diversity.rs        —   Shannon, Simpson, Chao1, Bray-Curtis, Pielou, rarefaction
        pcoa.rs             —   PCoA ordination (CPU Jacobi eigensolve)
        unifrac.rs          —   UniFrac distance (weighted + unweighted, Newick tree parser)
        signal.rs           —   1D peak detection (scipy.signal.find_peaks equivalent)
        eic.rs              —   Extracted Ion Chromatogram / mass track detection
        feature_table.rs    —   End-to-end asari-style feature extraction
        tolerance_search.rs —   ppm/Da search + PFAS fragment screening
        spectral_match.rs   —   MS2 cosine similarity (matched + weighted)
        kmd.rs              —   Kendrick mass defect + PFAS homologue grouping
        diversity_gpu.rs    —   GPU: Shannon/Simpson/observed/evenness/alpha/BC
        pcoa_gpu.rs         —   GPU: PCoA via ToadStool BatchedEighGpu
        spectral_match_gpu.rs — GPU: pairwise cosine via GemmF64
        kriging.rs          —   Spatial diversity interpolation via ToadStool KrigingF64
        stats_gpu.rs        —   GPU: variance/correlation/covariance/weighted dot
        eic_gpu.rs          —   GPU: batch EIC integration + peak processing
        rarefaction_gpu.rs  —   GPU: bootstrap rarefaction with confidence intervals
      io/                   — Streaming I/O parsers (all sovereign, zero external parsers)
        fastq.rs            —   FASTQ parser (gzip-aware via flate2)
        mzml.rs             —   mzML mass spectrometry parser
        ms2.rs              —   MS2 text format parser
        xml.rs              —   Minimal XML pull parser (for mzML)
      bin/                  — Validation + benchmark binaries (hotSpring pattern: pass/fail, exit 0/1)
        validate_fastq.rs   —   28/28 checks: quality + merge + derep + Galaxy FastQC
        validate_diversity.rs — 27/27 checks: analytical + simulated + evenness + rarefaction (expanded from 18)
        validate_16s_pipeline.rs — 37/37 checks: complete 16S FASTQ→DADA2→chimera→taxonomy→diversity→UniFrac
        validate_algae_16s.rs — 29/29 checks: real NCBI data (PRJNA488170) + Humphrey 2023 reference
        validate_voc_peaks.rs — 22/22 checks: Reese 2019 VOC baselines + synthetic GC-MS + RI matching
        validate_mzml.rs    —   7/7 checks vs asari/pyteomics baseline
        validate_pfas.rs    —   10/10 checks: cosine + KMD + FindPFAS
        validate_features.rs —  9/9 checks: EIC + peaks + features vs asari (Exp009)
        validate_peaks.rs   —   17/17 checks: peak detection vs scipy (Exp010)
        validate_diversity_gpu.rs — 38/38 GPU vs CPU (--features gpu)
        benchmark_cpu_gpu.rs — Three-tier performance benchmark (--features gpu)
      encoding.rs           — Sovereign base64 encoder/decoder (RFC 4648)
      error.rs              — Typed error chain (Error enum + std::error::Error)
      gpu.rs                — GpuF64 device wrapper (wgpu v22), ToadStool bridge
      tolerances.rs         — Centralized CPU + GPU validation tolerances
      validation.rs         — Validator struct + check/check_count/finish framework
    tests/
      io_roundtrip.rs       — 42 integration tests (round-trip, pipeline, edge cases)
  control/
    galaxy/                 — Galaxy 24.1 Docker deployment (Track 1)
    asari/                  — asari LC-MS processing config (Track 2)
    pfascreen/              — PFΔScreen PFAS screening config (Track 2)
  data/                     — Public datasets (Zenodo, SRA, EPA)
  experiments/              — Numbered experiment protocols, results, and baselines (see experiments/README.md)
  benchmarks/               — Performance comparison: Python vs Rust CPU vs Rust GPU
  papers/
    PIPELINE_TARGETS.md     — Target paper → pipeline → Rust module mapping
  scripts/                  — Download, preprocess, validate helpers
  whitePaper/
    README.md               — Study overview and key results
    METHODOLOGY.md          — Two-track validation protocol and acceptance criteria
  CONTROL_EXPERIMENT_STATUS.md — Experiment logs, validation counts, evolution roadmap
  EVOLUTION_READINESS.md    — Module-by-module GPU promotion readiness
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

1. Carney, L.T., Wilkenfeld, J.S., Lane, P.D. et al. "Pond Crash
   Forensics: Presumptive identification of pond crash agents by next
   generation sequencing in replicate raceway mass cultures of
   *Nannochloropsis salina*." *Algal Research* 17 (2016).
   DOI: [10.1016/j.algal.2016.05.011](https://doi.org/10.1016/j.algal.2016.05.011). OSTI 1271304.

2. Humphrey, B., Mackenzie, M., Lobitz, M. et al. (incl. Smallwood, C.R.
   and Cahill, J.) "Biotic countermeasures that rescue *Nannochloropsis
   gaditana* from a *Bacillus safensis* infection." *Frontiers in
   Microbiology* (2023). OSTI 2311389.

3. Reichardt, T.A., Maes, D., Jensen, T.J. et al. "Spectroradiometric
   detection of competitor diatoms and the grazer *Poteriochromonas* in
   algal cultures." *Algal Research* (2020).
   DOI: [10.1016/j.algal.2020.102020](https://doi.org/10.1016/j.algal.2020.102020). OSTI 1828029.

4. Reese, K.L., Fisher, C.L., Lane, P.D. et al. (incl. Jones, A.D.)
   "Chemical Profiling of Volatile Organic Compounds in the Headspace of
   Algal Cultures as Early Biomarkers of Algal Pond Crashes." *Scientific
   Reports* 9 (2019). DOI: [10.1038/s41598-019-50125-z](https://doi.org/10.1038/s41598-019-50125-z). OSTI 1570268.

### Track 2: PFAS Analytical Chemistry (blueFish)

5. Li, S. et al. "Trackable and scalable LC-MS metabolomics data processing
   using asari." *Nature Communications* 14, 4113 (2023).
   DOI: [10.1038/s41467-023-39889-1](https://doi.org/10.1038/s41467-023-39889-1).

6. Zweigle, J. et al. "PFΔScreen — an open-source tool for automated PFAS
   feature prioritization in non-target HRMS data." *Analytical and
   Bioanalytical Chemistry* (2023).
   DOI: [10.1007/s00216-023-05070-2](https://doi.org/10.1007/s00216-023-05070-2).

7. Jones, A.D. et al. "Quantitative identification of nonpolar perfluoroalkyl
   substances by mass spectrometry." MSU Center for PFAS Research.

8. Nejadhashemi, A.P. et al. "Using machine learning techniques to monitor
   PFAS in Michigan drinking water." *Water Research* (2023).

9. MSU Center for PFAS Research. "Molecular screening and toxicity estimation
   of 260,000 PFAS through machine learning." (2022).

### Algorithms Implemented

10. Callahan, B.J. et al. "DADA2: High-resolution sample inference from
    Illumina amplicon data." *Nature Methods* 13, 581–583 (2016).
    DOI: [10.1038/nmeth.3869](https://doi.org/10.1038/nmeth.3869).

11. Edgar, R.C. "UCHIME improves sensitivity and speed of chimera detection."
    *Bioinformatics* 27, 2194–2200 (2011).
    DOI: [10.1093/bioinformatics/btr381](https://doi.org/10.1093/bioinformatics/btr381).

12. Wang, Q. et al. "Naive Bayesian Classifier for Rapid Assignment of rRNA
    Sequences into the New Bacterial Taxonomy." *Appl. Environ. Microbiol.*
    73, 5261–5267 (2007). DOI: [10.1128/AEM.00062-07](https://doi.org/10.1128/AEM.00062-07).

13. Lozupone, C. & Knight, R. "UniFrac: a new phylogenetic method for
    comparing microbial communities." *Appl. Environ. Microbiol.* 71,
    8228–8235 (2005). DOI: [10.1128/AEM.71.12.8228-8235.2005](https://doi.org/10.1128/AEM.71.12.8228-8235.2005).

14. Virtanen, P. et al. "SciPy 1.0: Fundamental Algorithms for Scientific
    Computing in Python." *Nature Methods* 17, 261–272 (2020).
    DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

### Platforms & Tools

15. The Galaxy Project. https://galaxyproject.org/ (accessed Feb 2026).
16. asari. https://github.com/shuzhao-li-lab/asari (accessed Feb 2026).
17. PFΔScreen. https://github.com/JonZwe/PFAScreen (accessed Feb 2026).
