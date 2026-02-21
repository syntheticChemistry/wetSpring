# wetSpring Control Experiment Status

**Date:** February 20, 2026
**Status:** 63 experiments, 1501 validation checks, all PASS

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
| 051 | Anderson 2015 Rare Biosphere | 1c | COMPLETE | 35 |
| 052 | Anderson 2014 Viral Metagenomics | 1c | COMPLETE | 22 |
| 053 | Mateos 2023 Sulfur Phylogenomics | 1c | COMPLETE | 15 |
| 054 | Boden 2024 Phosphorus Phylogenomics | 1c | COMPLETE | 13 |
| 055 | Anderson 2017 Population Genomics | 1c | COMPLETE | 24 |
| 056 | Moulana 2020 Pangenomics | 1c | COMPLETE | 24 |
| 057 | BarraCUDA CPU Parity v4 (Track 1c) | cross | COMPLETE | 44 |
| 058 | GPU Track 1c (ANI + SNP + Pangenome + dN/dS) | GPU | COMPLETE | 27 |
| 059 | 23-Domain Rust vs Python Benchmark | cross | COMPLETE | Benchmark |
| 060 | metalForge Cross-Substrate Validation | cross/GPU | COMPLETE | 20 |
| 061 | Random Forest Ensemble Inference | cross | COMPLETE | 13 |
| 062 | GBM Inference (Binary + Multi-Class) | cross | COMPLETE | 16 |
| 063 | GPU Random Forest Batch Inference | GPU | COMPLETE | 13 |

---

## Totals

| Category | Count |
|----------|-------|
| Experiments completed | 63 |
| CPU validation checks | 1,241 |
| GPU validation checks | 260 |
| **Total validation checks** | **1,501** |
| Rust tests | 524 lib + 58 integration/bin/doc |
| BarraCUDA CPU parity | 157/157 (25 domains) |
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
| `anderson2015_rare_biosphere.py` | diversity/rarefaction (Exp051) | Feb 2026 | GREEN |
| `anderson2014_viral_metagenomics.py` | dN/dS + diversity (Exp052) | Feb 2026 | GREEN |
| `mateos2023_sulfur_phylogenomics.py` | clock/reconciliation (Exp053) | Feb 2026 | GREEN |
| `boden2024_phosphorus_phylogenomics.py` | clock/reconciliation (Exp054) | Feb 2026 | GREEN |
| `anderson2017_population_genomics.py` | ANI/SNP (Exp055) | Feb 2026 | GREEN |
| `moulana2020_pangenomics.py` | pangenome/enrichment (Exp056) | Feb 2026 | GREEN |
| `barracuda_cpu_v4_baseline.py` | 5 Track 1c domain timing (Exp057) | Feb 2026 | GREEN |

---

## Remaining Work

### Exp019 Phases 2-4 (Phylogenetic) — COMPLETE
- Phase 1 (Newick parsing): COMPLETE — 30/30 checks (Exp019)
- Phase 2 (Gene tree RF distances): COMPLETE — 15/15 checks (Exp036)
- Phase 3 (PhyloNet-HMM introgression): COMPLETE — 10/10 checks (Exp037)
- Phase 4 (SATe 16S alignment): COMPLETE — 17/17 checks (Exp038)

### Exp008 Full ML Pipeline — COMPLETE
- Phase 3 (Decision tree inference): COMPLETE — 7/7 checks, 100% parity
- Phase 4 (Random Forest ensemble): COMPLETE — Exp061, 13/13 checks
- Phase 5 (GBM binary + multi-class): COMPLETE — Exp062, 16/16 checks
- GPU RF batch inference: COMPLETE — Exp063, 13/13 checks

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

### Track 1c: Deep-Sea Metagenomics (Anderson)
**Status:** Comprehensive. 6 experiments (051-056) plus CPU parity (Exp057).
Covers ANI, SNP calling, dN/dS, molecular clock, pangenome analysis,
phylogenomics, and rare biosphere diversity — all validated against Python
baselines and proved as pure Rust math via BarraCUDA CPU v4.

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
cargo test             → 582 passed, 0 failed
validations            → 36/36 binaries PASS (29 CPU + 12 GPU + 1 benchmark)
barracuda_cpu          → 157/157 checks PASS (25 domains)
barracuda_gpu          → 260 GPU checks PASS (12 binaries)
toadstool_bio          → 10/10 checks PASS (4 new bio primitives)
gpu_track1c            → 27/27 checks PASS (4 Track 1c WGSL shaders)
gpu_rf                 → 13/13 checks PASS (1 local WGSL shader)
```

## BarraCUDA CPU Parity

The `validate_barracuda_cpu` v1-v4 binaries prove pure Rust math matches
Python across all 23 algorithmic domains:
- v1 (Exp035): 9 core domains
- v2 (Exp035): +5 batch/flat APIs
- v3 (Exp043): +9 domains (QS, phage, bootstrap, placement, decision tree, spectral, diversity, k-mer, pipeline)
- v4 (Exp057): +5 Track 1c domains (ANI, SNP, dN/dS, molecular clock, pangenome)
- v5 (Exp061/062): +2 ML domains (Random Forest, GBM)

Combined: 157/157 CPU parity checks. This is the bridge to pure GPU execution.

```
Total CPU time: ~85ms (release build, all 25 domains, v4 adds ~0.4ms, v5 adds ~62µs)
```
