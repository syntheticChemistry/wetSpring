# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — validation target proving Python baselines can be
faithfully ported to BarraCUDA (Rust) and eventually promoted to ToadStool
(GPU shaders).

**Date:** February 2026
**License:** AGPL-3.0-or-later
**Status:** Phase 10 — Paper Parity + Stochastic Simulation

---

## What This Is

wetSpring validates the entire evolution path from interpreted-language
scientific computing (Python/numpy/scipy/sklearn) to sovereign Rust CPU
implementations, and then to GPU acceleration via ToadStool/BarraCUDA:

```
Python baseline → Rust CPU validation → GPU acceleration → sovereign pipeline
```

Two tracks cover the life science and environmental monitoring domains:

| Track | Domain | Key Algorithms |
|-------|--------|----------------|
| **Track 1** | Microbial Ecology (16S rRNA) | FASTQ QC, DADA2 denoising, chimera detection, taxonomy, UniFrac, diversity |
| **Track 1b** | Comparative Genomics | Newick parsing, Robinson-Foulds distance, ODE/stochastic models |
| **Track 2** | Analytical Chemistry (LC-MS, PFAS) | mzML parsing, EIC, peak detection, spectral matching, KMD, PFAS screening |

---

## Current Results

| Metric | Count |
|--------|-------|
| Validation checks (CPU) | 519 |
| Validation checks (GPU) | 126 |
| **Total validation checks** | **645** |
| Rust unit/integration tests | 430 |
| Experiments completed | 22 |
| Validation binaries | 17 CPU + 2 GPU |
| CPU bio modules | 23 |
| GPU bio modules | 12 |
| Python baselines | 8 scripts |

All 645 validation checks **PASS**. All 430 tests **PASS** (1 ignored — GPU-only).

### GPU Performance (Phase 3)

| Workload | CPU→GPU Speedup | Parity |
|----------|----------------|--------|
| Spectral cosine (2048 spectra) | 926× | ≤1e-10 |
| Full 16S pipeline (10 samples) | 2.45× | 88/88 |
| Shannon/Simpson diversity | 15–25× | ≤1e-6 |

---

## Evolution Path

### Phase 1: Python Control Baselines
Galaxy/QIIME2, DADA2-R, asari, scipy, sklearn, dendropy — establishing
ground truth from published tools on public data.

### Phase 2: Rust CPU Validation
Pure Rust implementations matching Python baselines within documented
tolerances. Zero unsafe, zero external commands, sovereign I/O.

### Phase 3: GPU Acceleration
ToadStool/BarraCUDA primitives (11 validated) for massively parallel
diversity, taxonomy, spectral matching, and streaming pipelines.

### Phase 4: Paper Parity (Current)
Reproducing published mathematical models (Waters 2008 ODE, Massie 2012
Gillespie SSA) and phylogenetic algorithms (Robinson-Foulds distance) in
pure Rust to validate the mathematical portability claim.

### Phase 5: Sovereign ML
Decision tree inference engine ported from sklearn — 100% prediction
parity on 744 PFAS water samples, proving ML portability without Python.

---

## Module Inventory

### CPU Bio Modules (23)

| Module | Algorithm | Validated Against |
|--------|-----------|-------------------|
| `chimera` | UCHIME-style chimera detection | DADA2-R removeBimeraDenovo |
| `dada2` | ASV denoising (Callahan 2016) | DADA2-R dada() |
| `decision_tree` | Decision tree inference | sklearn DecisionTreeClassifier |
| `derep` | Dereplication + abundance | VSEARCH --derep_fulllength |
| `diversity` | Shannon, Simpson, Chao1, Bray-Curtis, Pielou, rarefaction | QIIME2 diversity |
| `eic` | EIC/XIC extraction + peak integration | asari 1.13.1 |
| `feature_table` | Asari-style LC-MS feature extraction | asari 1.13.1 |
| `gillespie` | Gillespie SSA (stochastic simulation) | numpy ensemble statistics |
| `kmd` | Kendrick mass defect | pyOpenMS |
| `kmer` | K-mer counting (2-bit canonical) | QIIME2 feature-classifier |
| `merge_pairs` | Paired-end overlap merging | VSEARCH --fastq_mergepairs |
| `ode` | Generic RK4 ODE integrator | scipy.integrate.odeint |
| `pcoa` | PCoA (Jacobi eigendecomposition) | QIIME2 emperor |
| `phred` | Phred quality decode/encode | Biopython |
| `qs_biofilm` | Waters 2008 QS/c-di-GMP model | scipy ODE baseline |
| `quality` | Quality filtering (Trimmomatic-style) | Trimmomatic/Cutadapt |
| `robinson_foulds` | RF tree distance | dendropy |
| `signal` | 1D peak detection | scipy.signal.find_peaks |
| `spectral_match` | MS2 cosine similarity | pyOpenMS |
| `taxonomy` | Naive Bayes classifier (RDP-style) | QIIME2 classify-sklearn |
| `tolerance_search` | ppm/Da m/z search | FindPFAS |
| `unifrac` | Unweighted/weighted UniFrac + Newick parser | QIIME2 diversity |

### GPU Modules (12)

`chimera_gpu`, `dada2_gpu`, `diversity_gpu`, `eic_gpu`, `gemm_cached`,
`kriging`, `pcoa_gpu`, `quality_gpu`, `rarefaction_gpu`, `spectral_match_gpu`,
`stats_gpu`, `streaming_gpu`, `taxonomy_gpu`

### I/O Modules

`io::fastq` (streaming FASTQ/gzip), `io::mzml` (streaming mzML/base64),
`io::ms2` (streaming MS2)

---

## Repository Structure

```
wetSpring/
├── README.md                      ← this file
├── BENCHMARK_RESULTS.md           ← three-tier benchmark results
├── CONTROL_EXPERIMENT_STATUS.md   ← experiment status tracker
├── EVOLUTION_READINESS.md         ← GPU promotion readiness
├── barracuda/                     ← Rust crate (src/, tests/, Cargo.toml)
├── experiments/                   ← 22 experiment protocols + results
├── scripts/                       ← Python baselines (8 scripts)
├── specs/                         ← specifications and paper queue
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
# Run all tests
cd barracuda && cargo test

# Run all CPU validation binaries
for bin in validate_fastq validate_diversity validate_mzml validate_pfas \
           validate_features validate_peaks validate_16s_pipeline \
           validate_algae_16s validate_voc_peaks validate_public_benchmarks \
           validate_extended_algae validate_pfas_library \
           validate_qs_ode validate_rf_distance validate_gillespie \
           validate_newick_parse validate_pfas_decision_tree; do
    cargo run --bin $bin
done

# Run GPU validation (requires GPU + --features gpu)
cargo run --features gpu --bin validate_diversity_gpu
cargo run --features gpu --release --bin validate_16s_pipeline_gpu

# Run Python baselines
python3 scripts/gillespie_baseline.py
python3 scripts/rf_distance_baseline.py
python3 scripts/newick_parse_baseline.py
python3 scripts/pfas_tree_export.py
python3 scripts/waters2008_qs_ode.py
```

---

## Data Provenance

All validation data comes from public repositories:

| Source | Accession | Usage |
|--------|-----------|-------|
| NCBI SRA | PRJNA488170 | Algae pond 16S (Exp012) |
| NCBI SRA | PRJNA382322 | Nannochloropsis 16S (Exp017) |
| NCBI SRA | PRJNA1114688 | Lake microbiome 16S (Exp014) |
| Zenodo | 14341321 | Jones Lab PFAS library (Exp018) |
| Michigan EGLE | ArcGIS REST | PFAS surface water (Exp008) |
| Reese 2019 | PMC6761164 | VOC biomarkers (Exp013) |

---

## Related

- **hotSpring** — Nuclear/plasma physics validation (sibling Spring)
- **wateringHole** — Inter-primal coordination and semantic guidelines
- **ToadStool** — GPU compute engine (BarraCUDA crate)
- **ecoPrimals** — Parent ecosystem
