# wetSpring Control Experiment Status

**Date:** February 20, 2026
**Status:** 50 experiments, 1235 validation checks, all PASS

---

## Experiment Status

| Exp | Name | Track | Status | Checks |
|-----|------|-------|--------|--------|
| 001 | Galaxy Bootstrap (QIIME2/DADA2) | 1 | COMPLETE | 28 |
| 002 | Phytoplankton 16S (PRJNA1195978) | 1 | COMPLETE | — |
| 003 | Phage Assembly (SPAdes/Pharokka) | 1 | COMPLETE | — |
| 004 | Rust FASTQ + Diversity | 1 | COMPLETE | 55 |
| 005 | asari LC-MS Bootstrap | 2 | COMPLETE | 7 |
| 006 | PFAScreen Validation (FindPFAS) | 2 | COMPLETE | 10 |
| 007 | Rust mzML + PFAS | 2 | COMPLETE | — |
| 008 | PFAS ML Water Monitoring | 2 | COMPLETE (Phase 3) | 7 |
| 009 | Feature Pipeline (asari MT02) | 2 | COMPLETE | 8 |
| 010 | Peak Detection (scipy baseline) | cross | COMPLETE | 17 |
| 011 | 16S Pipeline End-to-End | 1 | COMPLETE | 37 |
| 012 | Algae Pond 16S (PRJNA488170) | 1 | COMPLETE | 34 |
| 013 | VOC Peak Validation (Reese 2019) | 1/cross | COMPLETE | 22 |
| 014 | Public Data Benchmarks (4 BioProjects) | 1 | COMPLETE | 202 |
| 015 | Pipeline Benchmark (Rust vs Galaxy) | 1 | COMPLETE | Benchmark |
| 016 | GPU Pipeline Parity (CPU↔GPU) | 1 | COMPLETE | 88 |
| 017 | Extended Algae (PRJNA382322) | 1 | COMPLETE | 35 |
| 018 | PFAS Library (Jones Lab Zenodo) | 2 | COMPLETE | 26 |
| 019 | Phylogenetic Validation (Phase 1) | 1b | COMPLETE | 30 |
| 020 | Waters 2008 QS/c-di-GMP ODE | 1 | COMPLETE | 16 |
| 021 | Robinson-Foulds Validation | 1b | COMPLETE | 23 |
| 022 | Massie 2012 Gillespie SSA | 1 | COMPLETE | 13 |
| 023 | Fernandez 2020 Bistable Switching | 1 | COMPLETE | 14 |
| 024 | Srivastava 2011 Multi-Signal QS | 1 | COMPLETE | 19 |
| 025 | Bruger & Waters 2018 Cooperation | 1 | COMPLETE | 20 |
| 026 | Liu 2014 HMM Primitives | 1b | COMPLETE | 21 |
| 027 | Mhatre 2020 Phenotypic Capacitor | 1 | COMPLETE | 18 |
| 028 | Smith-Waterman Alignment | 1b | COMPLETE | 15 |
| 029 | Felsenstein Pruning Likelihood | 1b/c | COMPLETE | 16 |
| 030 | Hsueh 2022 Phage Defense Deaminase | 1 | COMPLETE | 12 |
| 031 | Wang 2021 RAWR Bootstrap | 1b | COMPLETE | 11 |
| 032 | Alamin & Liu 2024 Placement | 1b | COMPLETE | 12 |
| 033 | Liu 2009 Neighbor-Joining (SATé core) | 1b | COMPLETE | 16 |
| 034 | Zheng 2023 DTL Reconciliation | 1b | COMPLETE | 14 |
| 035 | BarraCUDA CPU Parity v2 | cross | COMPLETE | 18 |
| 036 | PhyNetPy RF distances (Exp019 Phase 2) | 1b | COMPLETE | 15 |
| 037 | PhyloNet-HMM discordance (Exp019 Phase 3) | 1b | COMPLETE | 10 |
| 038 | SATe pipeline benchmark (Exp019 Phase 4) | 1b | COMPLETE | 17 |
| 039 | Algal pond time-series (Cahill proxy) | 1 | COMPLETE | 11 |
| 040 | Bloom surveillance (Smallwood proxy) | 1 | COMPLETE | 15 |
| 041 | EPA PFAS ML (Jones F&T proxy) | 2 | COMPLETE | 14 |
| 042 | MassBank spectral (Jones MS proxy) | 2 | COMPLETE | 9 |
| 043 | BarraCUDA CPU Parity v3 | cross | COMPLETE | 45 |
| 044 | BarraCUDA GPU v3 | cross | COMPLETE | 14 |
| 045 | ToadStool Bio Absorption | cross/GPU | COMPLETE | 10 |
| 046 | GPU Phylogenetic Composition | GPU | COMPLETE | 15 |
| 047 | GPU HMM Batch Forward | GPU | COMPLETE | 13 |
| 048 | CPU vs GPU Benchmark (Phylo + HMM) | GPU | COMPLETE | 6 |
| 049 | GPU ODE Parameter Sweep | GPU | COMPLETE | 7 |
| 050 | GPU Bifurcation Eigenvalue Analysis | GPU | COMPLETE | 5 |

---

## Totals

| Category | Count |
|----------|-------|
| Experiments completed | 50 |
| CPU validation checks | 1,035 |
| GPU validation checks | 200 |
| **Total validation checks** | **1,235** |
| Rust tests | 465 lib + integration + doc |
| BarraCUDA CPU parity | 84/84 (18 domains) |
| ToadStool primitives consumed | 15 (11 original + 4 bio) |

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
| `fernandez2020_bistable.py` | scipy ODE (bifurcation) | Feb 2026 | GREEN |
| `srivastava2011_multi_signal.py` | scipy ODE (multi-signal) | Feb 2026 | GREEN |
| `bruger2018_cooperation.py` | scipy ODE (game theory) | Feb 2026 | GREEN |
| `liu2014_hmm_baseline.py` | numpy HMM (sovereign) | Feb 2026 | GREEN |
| `mhatre2020_capacitor.py` | scipy ODE (capacitor) | Feb 2026 | GREEN |
| `smith_waterman_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `felsenstein_pruning_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `hsueh2022_phage_defense.py` | scipy ODE (phage defense) | Feb 2026 | GREEN |
| `wang2021_rawr_bootstrap.py` | pure Python (bootstrap) | Feb 2026 | GREEN |
| `alamin2024_placement.py` | pure Python (placement) | Feb 2026 | GREEN |
| `liu2009_neighbor_joining.py` | pure Python (NJ) | Feb 2026 | GREEN |
| `zheng2023_dtl_reconciliation.py` | pure Python (DTL) | Feb 2026 | GREEN |
| `phynetpy_rf_baseline.py` | PhyNetPy gene trees | Feb 2026 | GREEN |
| `phylohmm_introgression_baseline.py` | PhyloNet-HMM | Feb 2026 | GREEN |
| `sate_alignment_baseline.py` | SATe pipeline | Feb 2026 | GREEN |
| `algae_timeseries_baseline.py` | Cahill proxy | Feb 2026 | GREEN |
| `bloom_surveillance_baseline.py` | Smallwood proxy | Feb 2026 | GREEN |
| `epa_pfas_ml_baseline.py` | Jones F&T proxy | Feb 2026 | GREEN |
| `massbank_spectral_baseline.py` | Jones MS proxy | Feb 2026 | GREEN |
| `benchmark_rust_vs_python.py` | 18-domain timing (Exp043) | Feb 2026 | GREEN |

---

## Remaining Work

### Exp019 Phases 2-4 (Phylogenetic) — COMPLETE
- Phase 1 (Newick parsing): COMPLETE — 30/30 checks (Exp019)
- Phase 2 (Gene tree RF distances): COMPLETE — 15/15 checks (Exp036)
- Phase 3 (PhyloNet-HMM introgression): COMPLETE — 10/10 checks (Exp037)
- Phase 4 (SATe 16S alignment): COMPLETE — 17/17 checks (Exp038)

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
**Status:** Comprehensive. 9 experiments (001, 004, 011, 012, 014, 017, 020, 039, 040)
cover the full 16S pipeline from FASTQ to diversity metrics, validated against
QIIME2, DADA2, and 4 BioProjects with 22 samples.

### Track 1b: Comparative Genomics & Phylogenetics
**Status:** Comprehensive. 12 experiments covering the full phylogenetic
pipeline: Newick parsing (Exp019), Robinson-Foulds (Exp021), HMM (Exp026),
Smith-Waterman (Exp028), Felsenstein pruning (Exp029), bootstrap (Exp031),
placement (Exp032), Neighbor-Joining tree construction (Exp033), DTL
reconciliation (Exp034), PhyNetPy RF (Exp036), PhyloNet-HMM (Exp037), and
SATe pipeline (Exp038). This provides a complete
toolkit from sequence alignment through tree construction, evaluation,
statistical confidence, and cophylogenetic analysis.

### Track 2: Analytical Chemistry (LC-MS, PFAS)
**Status:** Comprehensive. 7 experiments (005, 006, 009, 013, 018, 041, 042) cover
mzML parsing, feature extraction, peak detection, PFAS screening, and library
matching. Exp008 adds sovereign ML for environmental monitoring.

---

## Linting Status

```
cargo fmt --check      → clean
cargo clippy --pedantic → 0 errors, 0 warnings
cargo doc --no-deps    → clean
cargo test             → 465 passed, 0 failed
validations            → 30/30 binaries PASS (27 CPU + 9 GPU)
barracuda_cpu          → 84/84 checks PASS (18 domains)
barracuda_gpu          → 200 GPU checks PASS (9 binaries)
toadstool_bio          → 10/10 checks PASS (4 new bio primitives)
```

## BarraCUDA CPU Parity

The `validate_barracuda_cpu`, `validate_barracuda_cpu_v2`, and
`validate_barracuda_cpu_v3` binaries prove pure Rust math matches Python
across all 18 algorithmic domains (v1: 9 domains, v2: 5 batch/Flat APIs,
v3: 9 remaining domains including multi-signal QS, phage defense, bootstrap,
placement, decision tree, spectral matching, extended diversity, k-mer, and
integrated pipeline). Combined: 84/84 CPU parity checks. This is the bridge
to pure GPU execution.

```
Total CPU time: ~84.5ms (release build, all 18 domains)
```
