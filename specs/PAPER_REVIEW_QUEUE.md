# wetSpring — Paper Review Queue

**Last Updated**: February 19, 2026
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

---

## Review Queue

### Track 1 — Microbial Ecology & Signaling (Waters, Cahill, Smallwood)

#### Tier 1 — High priority, fully specified models

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 5 | Waters et al. "Quorum Sensing Controls Biofilm Formation in V. cholerae Through Modulation of Cyclic Di-GMP" | J Bacteriology 190:2527-36 | 2008 | Waters | Foundational QS ↔ c-di-GMP model. ODE system for signal-dependent biofilm phenotype. Fully specified — direct reproduction target | Queued |
| 6 | Massie et al. "Quantification of High Specificity Cyclic di-GMP Signaling" | PNAS 109:12746-51 | 2012 | Waters | How cells resolve signal from noise with 60+ enzymes controlling one molecule. Stochastic simulation primitives | Queued |
| 7 | Hsueh, Severin et al. "A Broadly Conserved Deoxycytidine Deaminase Protects Bacteria from Phage Infection" | Nature Microbiology 7:1210-1220 | 2022 | Waters | Phage defense — evolutionary arms race. Connects to Cahill phage biocontrol work | Queued |

#### Tier 2 — Strong candidates

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 8 | Fernandez et al. "V. cholerae adapts to sessile and motile lifestyles by c-di-GMP regulation of cell shape" | PNAS 117:29046-29054 | 2020 | Waters | Phenotypic switching = bistable dynamical system. Bifurcation analysis | Queued |
| 9 | Mhatre et al. "One gene, multiple ecological strategies" | PNAS 117:21647-21657 | 2020 | Waters | Capacitor for diversity — single node enabling multiple phenotypes | Queued |
| 10 | Bruger & Waters "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation" | AEM 84:e00402-18 | 2018 | Waters | Game-theoretic cooperation. Evolutionary strategy landscapes | Queued |
| 11 | Waters "Au naturale: cyclic di-nucleotides for cancer immunotherapy" | Open Biol 11:210277 | 2021 | Waters | Translational application of fundamental c-di-GMP understanding | Reference |
| 12 | Srivastava et al. "Integration of Cyclic di-GMP and Quorum Sensing" | J Bacteriology 193:6331-41 | 2011 | Waters | Multi-input regulatory network integration | Queued |

#### Tier 3 — Sandia connections (may require data access)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 13 | Cahill et al. — Phage-mediated biocontrol in algal raceway ponds | — | Cahill | Time-series anomaly detection for pond crash | Queued |
| 14 | Smallwood et al. — Raceway pond metagenomic surveillance | — | Smallwood | Real-world validation of 16S pipeline on pond data | Queued |

### Track 1b — Comparative Genomics & Phylogenetics (Liu)

#### Tier 1

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 15 | Liu et al. "An HMM-based Comparative Genomic Framework for Detecting Introgression" | PLoS Comp Bio 10:e1003649 | 2014 | Liu | PhyloNet-HMM — HMM on genomic data. Validates sequence model primitives | Queued |
| 16 | Alamin & Liu "Phylogenetic Placement of Aligned Genomes and Metagenomes with Non-tree-like Histories" | IEEE/ACM TCBB | 2024 | Liu | Metagenomic placement — classifying environmental samples. Direct wetSpring application | Queued |

#### Tier 2

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 17 | Liu et al. "Rapid and accurate large-scale coestimation of sequence alignments and phylogenetic trees" (SATé) | Science 324:1561-1564 | 2009 | Liu | Divide-and-conquer + iterative refinement at massive scale | Queued |
| 18 | Zheng et al. "Impact of Species Tree Estimation Error on Cophylogenetic Reconstruction" | BCB (top 10%) | 2023 | Liu | Host-microbe coevolution methods | Queued |
| 19 | Liu (working manuscript) — Fungi-bacteria coevolution (Burkholderiaceae) | — | Liu | Direct wetSpring application — fungal-bacterial symbiosis | Watch |
| 20 | Wang et al. "Build a better bootstrap and the RAWR shall beat a random path to your door" | Bioinformatics (ISMB) 37:i111-i119 | 2021 | Liu | Modern resampling for phylogenetic confidence | Queued |

### Track 2 — Analytical Chemistry & PFAS (Jones)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 21 | Jones et al. — PFAS mass spectrometry detection pipelines | — | Jones | Extends current PFAS screening (Exp 007-008) | Queued |
| 22 | Jones et al. — Environmental PFAS fate-and-transport modeling | — | Jones | From detection to prediction | Future |

---

## Notes

- wetSpring has the largest paper queue because it spans three scientific domains
- Track 1 papers (Waters) are the most immediately reproducible — fully specified ODE/stochastic models
- Track 1b papers (Liu) bridge to neuralSpring via HMM ↔ LSTM isomorphism
- Track 2 papers (Jones) depend on accessing PFAS mass spec data
- Cahill/Smallwood papers may require Sandia data access agreements
