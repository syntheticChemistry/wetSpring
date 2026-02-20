# wetSpring White Paper

**Date:** February 2026
**Status:** Validation study complete — 1,235/1,235 checks, 465 tests, 50 experiments
**License:** AGPL-3.0-or-later

---

## Document Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [STUDY.md](STUDY.md) | Main narrative — abstract, results, performance, references | Reviewers, collaborators |
| [METHODOLOGY.md](METHODOLOGY.md) | Validation protocol — two-track design, acceptance criteria | Technical validation |

---

## Study Questions

1. Can published life science algorithms (DADA2, UCHIME, RDP, UniFrac, scipy,
   sklearn) be faithfully reimplemented in pure Rust with documented tolerances?

2. Can those Rust CPU implementations be promoted to GPU via ToadStool/BarraCUDA
   with math parity, and at what speedup?

3. Can stochastic and dynamical systems models (ODE, Gillespie SSA) be ported
   from Python (scipy/numpy) to Rust with analytical convergence guarantees?

4. Can sovereign ML (decision tree inference) reproduce Python sklearn
   predictions with 100% parity, removing the Python runtime dependency?

---

## Key Results

| Claim | Evidence |
|-------|----------|
| Rust matches Python across 50 experiments | 1,035/1,035 CPU checks pass |
| GPU matches CPU (pipeline, diversity, bio, ODE, HMM) | 200/200 GPU checks pass |
| BarraCUDA CPU parity across 18 domains | 84/84 cross-domain checks pass |
| 926× spectral cosine GPU speedup | Exp016 benchmark |
| 2.45× full 16S pipeline GPU speedup | Exp015/016 benchmark |
| ODE (RK4) matches scipy across 6 models | Exp020/023/024/025/027/030 |
| Gillespie SSA converges to analytical | Exp022, 13/13 checks |
| HMM log-space (forward/Viterbi/posterior) | Exp026, 21/21 checks |
| Smith-Waterman matches pure Python | Exp028, 15/15 checks |
| Felsenstein pruning matches Python | Exp029, 16/16 checks |
| RAWR bootstrap resampling | Exp031, 11/11 checks |
| Phylogenetic placement matches Python | Exp032, 12/12 checks |
| PhyNetPy RF distances (1160 gene trees) | Exp036, 15/15 checks |
| PhyloNet-HMM discordance | Exp037, 10/10 checks |
| SATe pipeline alignment | Exp038, 17/17 checks |
| Algal pond time-series (Cahill proxy) | Exp039, 11/11 checks |
| Bloom surveillance (Smallwood proxy) | Exp040, 15/15 checks |
| EPA PFAS ML (Jones F&T proxy) | Exp041, 14/14 checks |
| MassBank spectral (Jones MS proxy) | Exp042, 9/9 checks |
| Phage defense dynamics match scipy | Exp030, 12/12 checks |
| Robinson-Foulds matches dendropy exactly | Exp021, 23/23 checks |
| Decision tree inference 100% Python parity | Exp008, 744/744 predictions |

---

## Experiment Coverage

### Track 1: Microbial Ecology (16S rRNA)

| Exp | Paper/Tool | What We Prove |
|-----|------------|---------------|
| 001 | Galaxy/QIIME2/DADA2 | Baseline pipeline setup |
| 004 | Exp001 Rust port | FASTQ + diversity match |
| 011 | Full DADA2+RDP+UniFrac | End-to-end 16S pipeline |
| 012 | PRJNA488170 real data | Algae pond 16S on NCBI data |
| 014 | 4 BioProjects (22 samples) | Cross-study reproducibility |
| 017 | PRJNA382322 Nannochloropsis | Extended algae validation |
| 039 | Cahill proxy — algal pond time-series | Time-series anomaly detection |
| 040 | Smallwood proxy — bloom surveillance | Metagenomic surveillance pipeline |

### Track 1: Mathematical Biology (continued)

| Exp | Paper | What We Prove |
|-----|-------|---------------|
| 023 | Fernandez 2020 | Bistable phenotypic switching, bifurcation |
| 024 | Srivastava 2011 | Multi-signal QS network integration |
| 025 | Bruger & Waters 2018 | Game-theoretic cooperation in QS |
| 027 | Mhatre 2020 | Phenotypic capacitor ODE, diversity via noise |
| 030 | Hsueh/Severin 2022 | Phage defense deaminase, arms race dynamics |

### Track 1b: Comparative Genomics & Phylogenetics

| Exp | Paper | What We Prove |
|-----|-------|---------------|
| 019 | PhyNetPy Newick trees | Parser correctness (30/30) |
| 020 | Waters 2008 (QS ODE) | RK4 matches scipy, 4 scenarios |
| 021 | dendropy RF distance | Bipartition tree distance (23/23) |
| 022 | Massie 2012 (Gillespie) | Stochastic→deterministic convergence |
| 026 | Liu 2014 (HMM) | Forward/backward/Viterbi/posterior in log-space |
| 028 | Smith-Waterman | Local alignment with affine gap penalties |
| 029 | Felsenstein pruning | Phylogenetic likelihood under JC69 |
| 031 | Wang 2021 (RAWR) | Bootstrap resampling for phylogenetic confidence |
| 032 | Alamin & Liu 2024 | Metagenomic placement by edge likelihood |
| 036 | PhyNetPy gene trees | RF distances vs 1160 PhyNetPy trees |
| 037 | PhyloNet-HMM | Introgression discordance detection |
| 038 | SATe pipeline | Divide-and-conquer alignment (Liu 2009) |

### GPU Composition & Evolution (Phase 8)

| Exp | Method | What We Prove |
|-----|--------|---------------|
| 043 | BarraCUDA CPU v3 | 45/45 across 18 domains, ~20× over Python |
| 044 | BarraCUDA GPU v3 | 14/14 SW/Gillespie/DT GPU parity |
| 045 | ToadStool bio absorption | 10/10 rewired primitives |
| 046 | GPU Phylo Composition | FelsensteinGpu → bootstrap + placement (15/15) |
| 047 | GPU HMM Forward | Local WGSL shader, batch forward log-space (13/13) |
| 048 | CPU vs GPU Benchmark | Felsenstein + Bootstrap + HMM timing (6/6) |
| 049 | GPU ODE Parameter Sweep | 64-batch QS sweep via local WGSL (7/7) |
| 050 | GPU Bifurcation Eigenvalues | Jacobian → BatchedEighGpu, bit-exact (5/5) |

### Track 2: Analytical Chemistry (LC-MS, PFAS)

| Exp | Paper/Tool | What We Prove |
|-----|------------|---------------|
| 005 | asari 1.13.1 | mzML parsing and feature extraction |
| 006 | FindPFAS/pyOpenMS | PFAS suspect screening |
| 008 | sklearn RF/GBM/DT | Sovereign ML for PFAS monitoring |
| 009 | asari MT02 | Feature pipeline parity |
| 013 | Reese 2019 VOC | VOC biomarker peak detection |
| 018 | Jones Lab PFAS library | 175-compound library matching |
| 041 | EPA PFAS ML (Jones F&T proxy) | Fate-and-transport ML validation |
| 042 | MassBank spectral (Jones MS proxy) | MS spectral library matching |

---

## R. Anderson Extension: Deep-Sea Metagenomics & Microbial Evolution

Rika Anderson (Carleton College) studies microbial and viral evolution in deep-sea
hydrothermal vents — her computational pipelines (MAGs, pangenomics, SNP analysis,
tree reconciliation, viral metagenomics) are exactly what wetSpring's sovereign
Rust pipeline validates.

Key connections:
- **Pangenomics**: Moulana, Anderson et al. (2020) — gene gain/loss under geochemical
  constraint. wetSpring's diversity/alignment primitives directly applicable.
- **Enzyme evolution**: Mateos, Anderson et al. (2023 *Science Advances*) — tracing
  sulfur-cycling enzymes across 3+ billion years using phylogenomics and tree
  reconciliation. Pipeline uses the same bioinformatics methods wetSpring validates.
- **Viral ecology**: Anderson et al. (2014) — phage-host interactions in vent systems.
  Connects to Cahill (algae pond phage) and Waters (phage defense deaminase, Exp030).
- **Rare biosphere**: Anderson et al. (2015) — when does a microbial lineage constitute
  signal vs. sequencing noise? Directly extends groundSpring Exp004 and wetSpring's
  rarefaction analysis.

**Papers queued**: See `specs/PAPER_REVIEW_QUEUE.md` — Papers 24-29 (Track 1c).
Reproduction targets use public genomes and metagenomes from NCBI — no wet lab
required. wetSpring's sovereign pipeline (DADA2 + taxonomy + diversity + alignment)
handles the bioinformatics.

---

## Relationship to ecoPrimals

wetSpring is one of several **Springs** — validation targets that prove
algorithms can be ported from interpreted languages to BarraCUDA/ToadStool:

- **hotSpring** — Nuclear physics, plasma, lattice QCD
- **wetSpring** — Life science, analytical chemistry, environmental monitoring
- **wateringHole** — Inter-primal coordination and semantic guidelines

Springs follow the **Write → Absorb → Lean** pattern (pioneered by hotSpring):
write and validate locally, hand off to ToadStool for absorption, then lean on
upstream primitives. This reduces dispatch overhead and round-trips via streaming
pipeline composition. wetSpring's `metalForge/` directory characterizes available
hardware (GPU, NPU, CPU) and guides Rust implementations for optimal absorption.

---

## References

### Published Tools Validated Against

| Tool | Version | Domain |
|------|---------|--------|
| DADA2 | 1.28.0 (via QIIME2) | 16S ASV denoising |
| QIIME2 | 2024.5 (via Galaxy) | Microbial ecology pipeline |
| asari | 1.13.1 | LC-MS feature extraction |
| FindPFAS | (pyOpenMS) | PFAS suspect screening |
| scipy | 1.11+ | Signal processing, ODE integration |
| sklearn | 1.3+ | ML classification |
| dendropy | 5.0.8 | Phylogenetic tree analysis |
| numpy | 1.24+ | Stochastic simulation |

### Public Data Sources

| Source | Accession | Experiment |
|--------|-----------|------------|
| NCBI SRA | PRJNA488170 | Exp012: Algae pond 16S |
| NCBI SRA | PRJNA382322 | Exp017: Nannochloropsis |
| NCBI SRA | PRJNA1114688 | Exp014: Lake microbiome |
| Zenodo | 14341321 | Exp018: Jones Lab PFAS |
| Michigan EGLE | ArcGIS REST | Exp008: PFAS surface water |
| PMC | PMC6761164 | Exp013: Reese 2019 VOC |
