# wetSpring White Paper

**Date:** February 2026
**Status:** Validation study complete — 668/668 checks, 430 tests
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
| Rust matches Python across 22 experiments | 542/542 CPU checks pass |
| GPU matches CPU (16S pipeline, diversity) | 126/126 GPU checks pass |
| 926× spectral cosine GPU speedup | Exp016 benchmark |
| 2.45× full 16S pipeline GPU speedup | Exp015/016 benchmark |
| ODE (RK4) matches scipy within 1e-6 | Exp020, 16/16 checks |
| Gillespie SSA converges to analytical | Exp022, 13/13 checks, mean within 0.2% |
| Robinson-Foulds matches dendropy exactly | Exp021, 23/23 checks |
| Decision tree inference 100% Python parity | Exp008, 744/744 predictions match |
| Newick parser matches dendropy | Exp019, 30/30 checks |

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

### Track 1b: Comparative Genomics

| Exp | Paper | What We Prove |
|-----|-------|---------------|
| 019 | PhyNetPy Newick trees | Parser correctness (30/30) |
| 020 | Waters 2008 (QS ODE) | RK4 matches scipy, 4 scenarios |
| 021 | dendropy RF distance | Bipartition tree distance (23/23) |
| 022 | Massie 2012 (Gillespie) | Stochastic→deterministic convergence |

### Track 2: Analytical Chemistry (LC-MS, PFAS)

| Exp | Paper/Tool | What We Prove |
|-----|------------|---------------|
| 005 | asari 1.13.1 | mzML parsing and feature extraction |
| 006 | FindPFAS/pyOpenMS | PFAS suspect screening |
| 008 | sklearn RF/GBM/DT | Sovereign ML for PFAS monitoring |
| 009 | asari MT02 | Feature pipeline parity |
| 013 | Reese 2019 VOC | VOC biomarker peak detection |
| 018 | Jones Lab PFAS library | 175-compound library matching |

---

## Relationship to ecoPrimals

wetSpring is one of several **Springs** — validation targets that prove
algorithms can be ported from interpreted languages to BarraCUDA/ToadStool:

- **hotSpring** — Nuclear physics, plasma, lattice QCD
- **wetSpring** — Life science, analytical chemistry, environmental monitoring
- **wateringHole** — Inter-primal coordination and semantic guidelines

Springs produce unidirectional handoffs to ToadStool, which absorbs validated
algorithms into shared GPU primitives. This reduces dispatch overhead and
round-trips via streaming pipeline composition.

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
