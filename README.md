# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — validation target proving Python baselines can be
faithfully ported to BarraCUDA (Rust) and eventually promoted to ToadStool
(GPU shaders).

**Date:** February 2026
**License:** AGPL-3.0-or-later
**Status:** Phase 12 — ToadStool Bio Absorption + Cross-System Evolution

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
| **Track 1** | Microbial Ecology (16S rRNA) | FASTQ QC, DADA2 denoising, chimera detection, taxonomy, UniFrac, diversity, ODE/stochastic models, game theory, phage defense |
| **Track 1b** | Comparative Genomics & Phylogenetics | Newick parsing, Robinson-Foulds, HMM, Smith-Waterman, Felsenstein pruning, bootstrap, placement, NJ tree construction, DTL reconciliation |
| **Track 2** | Analytical Chemistry (LC-MS, PFAS) | mzML parsing, EIC, peak detection, spectral matching, KMD, PFAS screening |

---

## Current Results

| Metric | Count |
|--------|-------|
| Validation checks (CPU) | 1,035 |
| Validation checks (GPU) | 200 |
| **Total validation checks** | **1,235** |
| Rust unit/integration tests | 465 |
| Experiments completed | 50 |
| Validation binaries | 41 CPU + 8 GPU |
| CPU bio modules | 34 |
| GPU bio modules | 15 (11 `_gpu` + `gemm_cached` + `kriging` + `hmm_gpu` + `ode_sweep_gpu`) |
| Python baselines | 28 scripts |
| BarraCUDA CPU parity | 84/84 (18 domains) |
| ToadStool primitives | 15 (inc. 4 bio: Felsenstein, Gillespie, SW, TreeInference) |
| Local WGSL shaders | 4 (Write → Absorb → Lean candidates) |

All 1,235 validation checks **PASS**. All 465 tests **PASS** (1 ignored — GPU-only).

### GPU Performance (Phase 3)

| Workload | CPU→GPU Speedup | Parity |
|----------|----------------|--------|
| Spectral cosine (2048 spectra) | 926× | ≤1e-10 |
| Full 16S pipeline (10 samples) | 2.45× | 88/88 |
| Shannon/Simpson diversity | 15–25× | ≤1e-6 |
| Bifurcation eigenvalues (5×5) | bit-exact | 2.67e-16 rel |
| ODE sweep (64 batches, 1000 steps) | math-portable | abs < 0.15 |

---

## Evolution Path

### Phase 1: Python Control Baselines
Galaxy/QIIME2, DADA2-R, asari, scipy, sklearn, dendropy — establishing
ground truth from published tools on public data.

### Phase 2: Rust CPU Validation
Pure Rust implementations matching Python baselines within documented
tolerances. Zero unsafe, zero external commands, sovereign I/O.

### Phase 3: GPU Acceleration
ToadStool/BarraCUDA primitives (15 validated) for massively parallel
diversity, taxonomy, spectral matching, streaming pipelines, and bio
algorithms (Felsenstein, Gillespie SSA, Smith-Waterman, tree inference).

### Phase 4: Paper Parity
Reproducing published mathematical models from 10 papers across ODE systems,
stochastic simulation, HMM, sequence alignment, phylogenetics, game theory,
and phage defense — all in pure Rust with documented Python baselines.

### Phase 5: Sovereign ML
Decision tree inference engine ported from sklearn — 100% prediction
parity on 744 PFAS water samples, proving ML portability without Python.

### Phase 6: BarraCUDA CPU Parity
84/84 cross-domain validation (v1 + v2 + v3) proving pure Rust math matches
Python across all 18 algorithmic domains. ~20x Rust speedup over Python.

### Phase 7: ToadStool Bio Absorption
ToadStool absorbed 4 GPU bio primitives from our handoff (commit `cce8fe7c`):
SmithWatermanGpu, GillespieGpu, TreeInferenceGpu, FelsensteinGpu.
wetSpring rewired to consume these, eliminated the fragile GEMM `include_str!`
path, and validated 10/10 GPU checks (Exp045). All 6 original primitive
requests are now addressed.

### Phase 8: GPU Composition + Write → Absorb → Lean (Current)
Composed ToadStool primitives for complex workflows (Exp046-050):
- **FelsensteinGpu** → bootstrap + placement (15/15, exact parity)
- **BatchedEighGpu** → bifurcation eigenvalues (5/5, bit-exact)
- **Local WGSL** → HMM batch forward (13/13), ODE parameter sweep (7/7)
- **NVVM f64 finding**: RTX 4070 cannot compile native f64 `exp/log/pow`.
  All transcendentals require `ShaderTemplate::for_driver_auto(_, true)`.
- 4 local shaders ready for ToadStool absorption.

---

## Module Inventory

### CPU Bio Modules (34)

| Module | Algorithm | Validated Against |
|--------|-----------|-------------------|
| `alignment` | Smith-Waterman local alignment (affine gaps) | Pure Python SW |
| `bistable` | Fernandez 2020 bistable phenotypic switching | scipy ODE bifurcation |
| `bootstrap` | RAWR bootstrap resampling (Wang 2021) | Pure Python resampling |
| `capacitor` | Mhatre 2020 phenotypic capacitor ODE | scipy ODE baseline |
| `chimera` | UCHIME-style chimera detection | DADA2-R removeBimeraDenovo |
| `cooperation` | Bruger & Waters 2018 QS game theory | scipy ODE baseline |
| `dada2` | ASV denoising (Callahan 2016) | DADA2-R dada() |
| `decision_tree` | Decision tree inference | sklearn DecisionTreeClassifier |
| `derep` | Dereplication + abundance | VSEARCH --derep_fulllength |
| `diversity` | Shannon, Simpson, Chao1, Bray-Curtis, Pielou, rarefaction | QIIME2 diversity |
| `eic` | EIC/XIC extraction + peak integration | asari 1.13.1 |
| `feature_table` | Asari-style LC-MS feature extraction | asari 1.13.1 |
| `felsenstein` | Felsenstein pruning phylogenetic likelihood | Pure Python JC69 |
| `gillespie` | Gillespie SSA (stochastic simulation) | numpy ensemble statistics |
| `hmm` | Hidden Markov Model (forward/backward/Viterbi/posterior) | numpy HMM (sovereign) |
| `kmd` | Kendrick mass defect | pyOpenMS |
| `kmer` | K-mer counting (2-bit canonical) | QIIME2 feature-classifier |
| `merge_pairs` | Paired-end overlap merging | VSEARCH --fastq_mergepairs |
| `multi_signal` | Srivastava 2011 multi-input QS network | scipy ODE baseline |
| `neighbor_joining` | Neighbor-Joining tree construction (Saitou & Nei 1987) | Pure Python NJ |
| `ode` | Generic RK4 ODE integrator | scipy.integrate.odeint |
| `pcoa` | PCoA (Jacobi eigendecomposition) | QIIME2 emperor |
| `phage_defense` | Hsueh 2022 phage defense deaminase | scipy ODE baseline |
| `phred` | Phred quality decode/encode | Biopython |
| `placement` | Alamin & Liu 2024 phylogenetic placement | Pure Python placement |
| `qs_biofilm` | Waters 2008 QS/c-di-GMP model | scipy ODE baseline |
| `quality` | Quality filtering (Trimmomatic-style) | Trimmomatic/Cutadapt |
| `reconciliation` | DTL reconciliation for cophylogenetics (Zheng 2023) | Pure Python DTL |
| `robinson_foulds` | RF tree distance | dendropy |
| `signal` | 1D peak detection | scipy.signal.find_peaks |
| `spectral_match` | MS2 cosine similarity | pyOpenMS |
| `taxonomy` | Naive Bayes classifier (RDP-style) | QIIME2 classify-sklearn |
| `tolerance_search` | ppm/Da m/z search | FindPFAS |
| `unifrac` | Unweighted/weighted UniFrac + Newick parser | QIIME2 diversity |

### GPU Modules (13)

`chimera_gpu`, `dada2_gpu`, `diversity_gpu`, `eic_gpu`, `gemm_cached`,
`kriging`, `pcoa_gpu`, `quality_gpu`, `rarefaction_gpu`, `spectral_match_gpu`,
`stats_gpu`, `streaming_gpu`, `taxonomy_gpu`

Plus 4 ToadStool bio primitives consumed directly: `SmithWatermanGpu`,
`GillespieGpu`, `TreeInferenceGpu`, `FelsensteinGpu`.

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
├── barracuda/                     ← Rust crate (src/, tests/, Cargo.toml)
│   ├── EVOLUTION_READINESS.md    ← absorption map (tiers, primitives, shaders)
│   └── src/shaders/              ← 4 local WGSL shaders (Write → Absorb → Lean)
├── experiments/                   ← 50 experiment protocols + results
├── metalForge/                    ← hardware characterization + primitive map
│   └── PRIMITIVE_MAP.md          ← Rust module ↔ ToadStool primitive mapping
├── scripts/                       ← Python baselines (28 scripts)
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
for bin in validate_qs_ode validate_rf_distance validate_gillespie \
           validate_newick_parse validate_bistable validate_multi_signal \
           validate_cooperation validate_hmm validate_capacitor \
           validate_alignment validate_felsenstein validate_barracuda_cpu \
           validate_barracuda_cpu_v2 validate_barracuda_cpu_v3 \
           validate_phage_defense validate_bootstrap validate_placement \
           validate_phynetpy_rf validate_phylohmm validate_sate_pipeline \
           validate_algae_timeseries validate_bloom_surveillance \
           validate_epa_pfas_ml validate_massbank_spectral; do
    cargo run --release --bin $bin
done

# Run GPU validation (requires GPU + --features gpu)
for gpu_bin in validate_diversity_gpu validate_16s_pipeline_gpu \
               validate_barracuda_gpu_v3 validate_gpu_phylo_compose \
               validate_gpu_hmm_forward benchmark_phylo_hmm_gpu \
               validate_gpu_ode_sweep; do
    cargo run --features gpu --bin $gpu_bin
done

# Rust vs Python head-to-head benchmark
bash scripts/benchmark_head_to_head.sh
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
