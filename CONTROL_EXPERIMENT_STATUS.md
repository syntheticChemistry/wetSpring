# wetSpring Control Experiment Status

**Date:** February 2026
**Status:** 22 experiments, 645 validation checks, all PASS

---

## Experiment Status

| Exp | Name | Track | Status | Checks |
|-----|------|-------|--------|--------|
| 001 | Galaxy Bootstrap (QIIME2/DADA2) | 1 | COMPLETE | 28 |
| 002 | Phytoplankton 16S (PRJNA1195978) | 1 | COMPLETE | — |
| 003 | Phage Assembly (SPAdes/Pharokka) | 1 | COMPLETE | — |
| 004 | Rust FASTQ + Diversity | 1 | COMPLETE | 18 |
| 005 | asari LC-MS Bootstrap | 2 | COMPLETE | 7 |
| 006 | PFAScreen Validation (FindPFAS) | 2 | COMPLETE | 10 |
| 007 | Rust mzML + PFAS | 2 | COMPLETE | — |
| 008 | PFAS ML Water Monitoring | 2 | COMPLETE (Phase 3) | 7 |
| 009 | Feature Pipeline (asari MT02) | 2 | COMPLETE | 9 |
| 010 | Peak Detection (scipy baseline) | cross | COMPLETE | 17 |
| 011 | 16S Pipeline End-to-End | 1 | COMPLETE | 37 |
| 012 | Algae Pond 16S (PRJNA488170) | 1 | COMPLETE | 29 |
| 013 | VOC Peak Validation (Reese 2019) | 1/cross | COMPLETE | 22 |
| 014 | Public Data Benchmarks (4 BioProjects) | 1 | COMPLETE | 202 |
| 015 | Pipeline Benchmark (Rust vs Galaxy) | 1 | COMPLETE | Benchmark |
| 016 | GPU Pipeline Parity (CPU↔GPU) | 1 | COMPLETE | 88 |
| 017 | Extended Algae (PRJNA382322) | 1 | COMPLETE | 29 |
| 018 | PFAS Library (Jones Lab Zenodo) | 2 | COMPLETE | 21 |
| 019 | Phylogenetic Validation (Phase 1) | 1b | COMPLETE | 30 |
| 020 | Waters 2008 QS/c-di-GMP ODE | 1 | COMPLETE | 16 |
| 021 | Robinson-Foulds Validation | 1b | COMPLETE | 23 |
| 022 | Massie 2012 Gillespie SSA | 1 | COMPLETE | 13 |

---

## Totals

| Category | Count |
|----------|-------|
| Experiments completed | 22 |
| CPU validation checks | 519 |
| GPU validation checks | 126 |
| **Total validation checks** | **645** |
| Rust tests | 430 (372 lib + 29 bio + 21 io + 8 doc) |

---

## Python Baseline Status

| Script | Tool | Date | Status |
|--------|------|------|--------|
| `benchmark_python_baseline.py` | QIIME2/DADA2-R | Feb 2026 | GREEN |
| `validate_public_16s_python.py` | BioPython + NCBI | Feb 2026 | GREEN |
| `waters2008_qs_ode.py` | scipy.integrate.odeint | Feb 2026 | GREEN (35/35) |
| `gillespie_baseline.py` | numpy SSA ensemble | Feb 2026 | GREEN (8/8) |
| `rf_distance_baseline.py` | dendropy RF distance | Feb 2026 | GREEN (10/10) |
| `newick_parse_baseline.py` | dendropy tree stats | Feb 2026 | GREEN (10/10) |
| `pfas_tree_export.py` | sklearn DecisionTree | Feb 2026 | GREEN (acc=0.989) |
| `exp008_pfas_ml_baseline.py` | sklearn RF+GBM | Feb 2026 | GREEN (RF F1=0.978) |

---

## Remaining Work

### Exp019 Phases 2-4 (Phylogenetic)
- Phase 1 (Newick parsing): COMPLETE — 30/30 checks
- Phase 2 (Gene tree RF distances): Needs PhyNetPy data download
- Phase 3 (PhyloNet-HMM introgression): Needs empirical data
- Phase 4 (SATe 16S alignment): Needs Dryad data download

### Exp008 Full ML Pipeline
- Phase 3 (Decision tree inference): COMPLETE — 7/7 checks, 100% parity
- Future: Random Forest ensemble inference in Rust (multiple trees)
- Future: GBM inference in Rust

### Data Audit
- Exp002 raw data: need to download 70 raw FASTQ pairs from SRA
- Trimmomatic/pyteomics baselines: deferred (not blocking)

### Tolerance Centralization
- Many validation binaries use inline tolerances
- Deferred: refactor to use `tolerances.rs` constants throughout

---

## Track Coverage

### Track 1: Microbial Ecology (16S rRNA)
**Status:** Comprehensive. 7 experiments (001, 004, 011, 012, 014, 017, 020)
cover the full 16S pipeline from FASTQ to diversity metrics, validated against
QIIME2, DADA2, and 4 BioProjects with 22 samples.

### Track 1b: Comparative Genomics
**Status:** Foundation laid. Newick parsing (Exp019), Robinson-Foulds distance
(Exp021), ODE models (Exp020), and stochastic simulation (Exp022) provide the
mathematical primitives needed for phylogenetic analysis.

### Track 2: Analytical Chemistry (LC-MS, PFAS)
**Status:** Comprehensive. 5 experiments (005, 006, 009, 013, 018) cover
mzML parsing, feature extraction, peak detection, PFAS screening, and library
matching. Exp008 adds sovereign ML for environmental monitoring.

---

## Linting Status

```
cargo fmt --check     → clean
cargo clippy --pedantic --nursery → 0 warnings
cargo doc --no-deps   → clean
```
