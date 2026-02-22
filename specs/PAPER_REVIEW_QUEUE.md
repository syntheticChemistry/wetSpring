# wetSpring — Paper Review Queue

**Last Updated**: February 22, 2026
**Purpose**: Track papers for reproduction/review across three tracks

---

## Completed Reproductions

| # | Paper / Pipeline | Phase | Checks | Track |
|---|-----------------|-------|--------|-------|
| 1 | Galaxy/QIIME2 16S pipeline (Exp 001-004) | 1-2 | 92/92 | Track 1 |
| 2 | asari LC-MS feature extraction (Exp 005-006) | 1-2 | 26/26 | Track 2 |
| 3 | FindPFAS screening pipeline (Exp 007-008) | 1-2 | 17/17 | Track 2 |
| 4 | GPU diversity + spectral matching | 3 | 38/38 | All |
| 5 | Sovereign 16S pipeline (DADA2 + chimera + taxonomy + UniFrac) | 4 | 37/37 | Track 1 |
| 6 | Algae pond 16S on real NCBI data (Exp012, PRJNA488170 proxy) | 5 | 29/29 | Track 1 |
| 7 | VOC peak validation vs Reese 2019 Table 1 (Exp013) | 5 | 22/22 | Track 1 |
| 8 | Public data benchmark — 4 BioProjects vs paper ground truth (Exp014) | 6 | 202/202 | Track 1 |
| 9 | Exp019 Phases 2-4: PhyNetPy RF (Exp036), PhyloNet-HMM (Exp037), SATe pipeline (Exp038) | — | 42/42 | Track 1b |
| 10 | Cahill proxy — algal pond time-series (Exp039) | — | 11/11 | Track 1 |
| 11 | Smallwood proxy — bloom surveillance (Exp040) | — | 15/15 | Track 1 |
| 12 | Jones F&T proxy — EPA PFAS ML (Exp041) | — | 14/14 | Track 2 |
| 13 | Jones MS proxy — MassBank spectral (Exp042) | — | 9/9 | Track 2 |

---

## Review Queue

### Track 1 — Microbial Ecology & Signaling (Waters, Cahill, Smallwood)

#### Tier 1 — High priority, fully specified models

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 5 | Waters et al. "Quorum Sensing Controls Biofilm Formation in V. cholerae Through Modulation of Cyclic Di-GMP" | J Bacteriology 190:2527-36 | 2008 | Waters | Foundational QS ↔ c-di-GMP model. ODE system for signal-dependent biofilm phenotype. Fully specified — direct reproduction target | **Exp020 DONE** |
| 6 | Massie et al. "Quantification of High Specificity Cyclic di-GMP Signaling" | PNAS 109:12746-51 | 2012 | Waters | How cells resolve signal from noise with 60+ enzymes controlling one molecule. Stochastic simulation primitives | **Exp022 DONE** |
| 7 | Hsueh, Severin et al. "A Broadly Conserved Deoxycytidine Deaminase Protects Bacteria from Phage Infection" | Nature Microbiology 7:1210-1220 | 2022 | Waters | Phage defense — evolutionary arms race. Connects to Cahill phage biocontrol work | **Exp030 DONE** |

#### Tier 2 — Strong candidates

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 8 | Fernandez et al. "V. cholerae adapts to sessile and motile lifestyles by c-di-GMP regulation of cell shape" | PNAS 117:29046-29054 | 2020 | Waters | Phenotypic switching = bistable dynamical system. Bifurcation analysis | **Exp023 DONE** |
| 9 | Mhatre et al. "One gene, multiple ecological strategies" | PNAS 117:21647-21657 | 2020 | Waters | Capacitor for diversity — single node enabling multiple phenotypes | **Exp027 DONE** |
| 10 | Bruger & Waters "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation" | AEM 84:e00402-18 | 2018 | Waters | Game-theoretic cooperation. Evolutionary strategy landscapes | **Exp025 DONE** |
| 11 | Waters "Au naturale: cyclic di-nucleotides for cancer immunotherapy" | Open Biol 11:210277 | 2021 | Waters | Translational application of fundamental c-di-GMP understanding | Reference |
| 12 | Srivastava et al. "Integration of Cyclic di-GMP and Quorum Sensing" | J Bacteriology 193:6331-41 | 2011 | Waters | Multi-input regulatory network integration | **Exp024 DONE** |

#### Tier 3 — Sandia connections (may require data access)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 13 | Cahill et al. — Phage-mediated biocontrol in algal raceway ponds | — | Cahill | Time-series anomaly detection for pond crash | **Exp039 DONE (Cahill proxy)** |
| 14 | Smallwood et al. — Raceway pond metagenomic surveillance | — | Smallwood | Real-world validation of 16S pipeline on pond data | **Exp040 DONE (Smallwood proxy)** |

### Track 1b — Comparative Genomics & Phylogenetics (Liu)

#### Tier 1

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 15 | Liu et al. "An HMM-based Comparative Genomic Framework for Detecting Introgression" | PLoS Comp Bio 10:e1003649 | 2014 | Liu | PhyloNet-HMM — HMM on genomic data. Validates sequence model primitives | **Exp026 + Exp037 DONE (HMM + PhyloNet-HMM)** |
| 16 | Alamin & Liu "Phylogenetic Placement of Aligned Genomes and Metagenomes with Non-tree-like Histories" | IEEE/ACM TCBB | 2024 | Liu | Metagenomic placement — classifying environmental samples. Direct wetSpring application | **Exp032 DONE** |

#### Tier 2

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 17 | Liu et al. "Rapid and accurate large-scale coestimation of sequence alignments and phylogenetic trees" (SATé) | Science 324:1561-1564 | 2009 | Liu | Divide-and-conquer + iterative refinement at massive scale | **Exp033 + Exp038 DONE (NJ + SATe pipeline)** |
| 18 | Zheng et al. "Impact of Species Tree Estimation Error on Cophylogenetic Reconstruction" | BCB (top 10%) | 2023 | Liu | Host-microbe coevolution methods | **Exp034 DONE** |
| 19 | Liu (working manuscript) — Fungi-bacteria coevolution (Burkholderiaceae) | — | Liu | Direct wetSpring application — fungal-bacterial symbiosis | Watch |
| 20 | Wang et al. "Build a better bootstrap and the RAWR shall beat a random path to your door" | Bioinformatics (ISMB) 37:i111-i119 | 2021 | Liu | Modern resampling for phylogenetic confidence | **Exp031 DONE** |

### Track 2 — Analytical Chemistry & PFAS (Jones)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 21 | Jones et al. — PFAS mass spectrometry detection pipelines | — | Jones | Extends current PFAS screening (Exp 007-008) | **Exp042 DONE (Jones MS proxy)** |
| 22 | Jones et al. — Environmental PFAS fate-and-transport modeling | — | Jones | From detection to prediction | **Exp041 DONE (Jones F&T proxy)** |

### Track 1c — Deep-Sea Metagenomics & Microbial Evolution (R. Anderson)

Rika Anderson (Carleton College) studies microbial and viral evolution in
deep-sea hydrothermal vents using metagenomics, pangenomics, and population
genomics — the same computational methods wetSpring validates in its sovereign
pipeline. Her work is the empirical corollary to the Taq polymerase argument
in `gen3/CONSTRAINED_EVOLUTION_FORMAL.md`.

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 24 | Anderson et al. "Genomic variation in microbial populations inhabiting the marine subseafloor at deep-sea hydrothermal vents" | Nature Communications 8:1114 | 2017 | R. Anderson | Population-level genomic variation shaped by geochemistry. SNP/dN/dS analysis on metagenomic data — directly exercises wetSpring's sovereign alignment and diversity pipelines | **Exp055 DONE** |
| 25 | Moulana, Anderson et al. "Selection is a significant driver of gene gain and loss in the pangenome of Sulfurovum" | mSystems 5:e00673-19 | 2020 | R. Anderson | Pangenomics under environmental constraint. Gene gain/loss driven by geochemistry at vents — constrained evolution of microbial functional repertoires. Direct computational reproduction target | **Exp056 DONE** |
| 26 | Mateos, Anderson et al. "The evolution and spread of sulfur-cycling enzymes reflect the redox state of the early Earth" | Science Advances 9:eade4847 | 2023 | R. Anderson | Phylogenomics across 3+ billion years of enzyme evolution. Tree reconciliation, molecular clock analysis, co-evolution of enzymes with geochemistry. Undergraduate co-authors — reproducible pipeline | **Exp053 DONE** |
| 27 | Boden, Anderson et al. "Timing the evolution of phosphorus-cycling enzymes through geological time" | Nature Communications 15:3703 | 2024 | R. Anderson | Same tree reconciliation methodology as Paper 26, applied to phosphorus cycle. Geobiology + bioinformatics | **Exp054 DONE** |
| 28 | Anderson et al. "Evolutionary strategies of viruses and cells in hydrothermal systems revealed through metagenomics" | PLoS ONE 9:e109696 | 2014 | R. Anderson | Viral metagenomics — phage-host interactions in vent ecosystems. Connects to Cahill (algae pond phage) and Waters (phage defense) | **Exp052 DONE** |
| 29 | Anderson, Sogin, Baross "Biogeography and ecology of the rare and abundant microbial lineages in deep-sea hydrothermal vents" | FEMS Microbiol Ecol 91:fiu016 | 2015 | R. Anderson | Rare biosphere — when does a microbial lineage constitute signal vs sampling noise? Directly connects to groundSpring Exp 004 (sequencing depth saturation) | **Exp051 DONE** |

**Why Anderson matters for wetSpring**: Her lab's computational pipelines (MAGs,
pangenomics, SNP analysis, tree reconciliation, viral metagenomics) are exactly
what wetSpring's sovereign 16S/metagenomics pipeline validates. Reproducing her
work would demonstrate that wetSpring's Rust pipeline produces the same biological
conclusions as the Galaxy/QIIME2/Python stack her lab uses. Furthermore, her
co-authors include undergraduate students, suggesting the methods are well-documented
and reproducible — ideal reproduction targets.

### Cross-Spring — Spectral Theory (Kachkovskiy, via groundSpring/hotSpring)

Kachkovskiy's spectral theory has an indirect but real connection to wetSpring
through signal processing in mass spectrometry (Track 2) and through the
mathematical framework for understanding quorum sensing as signal propagation
in a noisy medium (Track 1).

| # | Paper | Faculty | Why | Status |
|---|-------|---------|-----|--------|
| 23 | Bourgain & Kachkovskiy (2018) "Anderson localization for two interacting quasiperiodic particles." GAFA | Kachkovskiy | Quorum sensing = signal propagation through a "disordered" bacterial population. Anderson localization theory describes when signals (autoinducers) reach distant cells vs. when they're absorbed by local noise. Mathematical bridge to Waters' c-di-GMP specificity problem | Reference |

---

## Open Data Provenance Audit

All 29 reproductions use publicly accessible data or published model parameters.
No proprietary data dependencies.

| Category | Papers | Data Source | Access |
|----------|:------:|------------|--------|
| **ODE/Stochastic models** | 5-12 | Model parameters from published paper equations | Open (journal) |
| **Real 16S amplicon** | Exp001,012,014,017 | NCBI SRA (PRJNA488170, PRJNA382322, PRJNA1114688, etc.) | Open (NCBI) |
| **VOC biomarkers** | Exp013 | Table 1 from Reese 2019 (PMC6761164) | Open (PMC) |
| **PFAS screening** | Exp005-006 | asari test data (MT02) | Open (asari package) |
| **PFAS library** | Exp018 | Jones Lab (Zenodo 14341321) | Open (Zenodo) |
| **PFAS monitoring** | Exp008 | Michigan EGLE (ArcGIS REST API) | Open (state gov) |
| **PFAS ML** | Exp041 | EPA public PFAS data | Open (EPA) |
| **Spectral library** | Exp042 | MassBank | Open (MassBank) |
| **Phylogenetics** | 15-20 | Algorithms from published papers; PhyNetPy/SATé public datasets | Open (journal + repo) |
| **Deep-sea genomics** | Exp051-056 | NCBI SRA (PRJNA283159, PRJEB5293), MBL darchive, MG-RAST, Figshare, OSF | Open (public repos) |
| **Sandia proxy** | Exp039-040 | Synthetic proxy data (not original Sandia datasets) | Open (generated) |
| **Cross-spring** | Paper 23 | Mathematical framework reference (no reproduction data) | Reference only |

### Validation Tiers by Hardware

| Tier | Description | Experiments | Checks |
|------|-------------|:-----------:|:------:|
| **BarraCuda CPU** | Rust math matches Python baselines | Exp035,043,057,070,079,085 | 205/205 |
| **BarraCuda GPU** | GPU math matches CPU reference | Exp064,071,087,092 | 609 |
| **metalForge** | Substrate-independent output (CPU/GPU/NPU) | Exp060,065,080,084,086,088,093 | 172+ |
| **Streaming** | Pure GPU pipeline, zero CPU round-trips | Exp072,073,075,089,090,091 | 180+ |
| **Cross-spring** | neuralSpring primitives validated | Exp094,095 | 46 |

---

## Notes

- wetSpring has the largest paper queue because it spans three scientific domains
- Track 1 papers (Waters) are the most immediately reproducible — fully specified ODE/stochastic models
- Track 1b papers (Liu) bridge to neuralSpring via HMM ↔ LSTM isomorphism
- Track 2 papers (Jones) depend on accessing PFAS mass spec data
- Cahill/Smallwood papers may require Sandia data access agreements
- Paper 23 (Kachkovskiy) is a cross-spring reference — reproduction happens in groundSpring, wetSpring consumes the mathematical framework
