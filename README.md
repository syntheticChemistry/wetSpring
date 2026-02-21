# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — validation target proving Python baselines can be
faithfully ported to BarraCUDA (Rust) and eventually promoted to ToadStool
(GPU shaders), then shown substrate-independent via metalForge.

**Date:** February 2026
**License:** AGPL-3.0-or-later
**Status:** Phase 14 — 25-Domain Parity + GPU RF + metalForge Cross-Substrate

---

## What This Is

wetSpring validates the entire evolution path from interpreted-language
scientific computing (Python/numpy/scipy/sklearn) to sovereign Rust CPU
implementations, and then to GPU acceleration via ToadStool/BarraCUDA:

```
Python baseline → Rust CPU validation → GPU acceleration → metalForge cross-substrate
```

Four tracks cover the life science and environmental monitoring domains:

| Track | Domain | Key Algorithms |
|-------|--------|----------------|
| **Track 1** | Microbial Ecology (16S rRNA) | FASTQ QC, DADA2 denoising, chimera detection, taxonomy, UniFrac, diversity, ODE/stochastic models, game theory, phage defense |
| **Track 1b** | Comparative Genomics & Phylogenetics | Newick parsing, Robinson-Foulds, HMM, Smith-Waterman, Felsenstein pruning, bootstrap, placement, NJ tree construction, DTL reconciliation |
| **Track 1c** | Deep-Sea Metagenomics & Microbial Evolution | ANI, SNP calling, dN/dS, molecular clock, pangenomics, enrichment testing, rare biosphere diversity |
| **Track 2** | Analytical Chemistry (LC-MS, PFAS) | mzML parsing, EIC, peak detection, spectral matching, KMD, PFAS screening |

---

## Current Results

| Metric | Count |
|--------|-------|
| Validation checks (CPU) | 1,241 |
| Validation checks (GPU) | 260 |
| **Total validation checks** | **1,501** |
| Rust unit/integration tests | 582 |
| Experiments completed | 63 |
| Validation binaries | 29 CPU + 12 GPU + 1 benchmark |
| CPU bio modules | 41 |
| GPU bio modules | 20 (+4 ToadStool bio primitives consumed directly) |
| Python baselines | 35 scripts |
| BarraCUDA CPU parity | 157/157 (25 domains) |
| ToadStool primitives | 15 (inc. 4 bio: Felsenstein, Gillespie, SW, TreeInference) |
| Local WGSL shaders | 9 (Write → Absorb → Lean candidates) |

All 1,501 validation checks **PASS**. All 582 tests **PASS** (1 ignored — GPU-only).

### GPU Performance

| Workload | CPU→GPU Speedup | Parity |
|----------|----------------|--------|
| Spectral cosine (2048 spectra) | 926× | ≤1e-10 |
| Full 16S pipeline (10 samples) | 2.45× | 88/88 |
| Shannon/Simpson diversity | 15–25× | ≤1e-6 |
| Bifurcation eigenvalues (5×5) | bit-exact | 2.67e-16 rel |
| ODE sweep (64 batches, 1000 steps) | math-portable | abs < 0.15 |
| RF batch inference (6×5 trees) | CPU↔GPU exact | 13/13 parity |

### Rust vs Python (25 Domains)

| Metric | Value |
|--------|-------|
| Overall speedup | **22.5×** |
| Peak speedup | 625× (Smith-Waterman) |
| ODE domains | 15–28× |
| Track 1c domains | 6–56× |
| ML ensembles (RF + GBM) | ~30× |

---

## Evolution Path

### Phase 1–6: Foundation
Python control baselines → Rust CPU validation → GPU acceleration →
paper parity (29 papers, 10+ models) → sovereign ML (decision tree) →
BarraCUDA CPU parity (18 domains).

### Phase 7: ToadStool Bio Absorption
ToadStool absorbed 4 GPU bio primitives from our handoff (commit `cce8fe7c`):
SmithWatermanGpu, GillespieGpu, TreeInferenceGpu, FelsensteinGpu.
wetSpring rewired to consume these upstream (Exp045, 10/10).

### Phase 8: GPU Composition + Write → Absorb → Lean
Composed ToadStool primitives for complex workflows (Exp046-050):
- **FelsensteinGpu** → bootstrap + placement (15/15, exact parity)
- **BatchedEighGpu** → bifurcation eigenvalues (5/5, bit-exact)
- **Local WGSL** → HMM batch forward (13/13), ODE parameter sweep (7/7)
- 4 local shaders ready for ToadStool absorption.

### Phase 9: Track 1c — Deep-Sea Metagenomics
R. Anderson (Carleton) deep-sea hydrothermal vent papers (Exp051-056):
5 new sovereign Rust modules (`dnds`, `molecular_clock`, `ani`, `snp`,
`pangenome`) — 133 checks, 6 Python baselines.

### Phase 10: BarraCUDA CPU Parity v4 (Track 1c)
All 5 Track 1c domains validated as pure Rust math (Exp057, 44/44).
Combined v1-v4: 128/128 checks across 23 domains.

### Phase 11: GPU Track 1c Promotion (Exp058)
4 new local WGSL shaders: ANI, SNP, pangenome, dN/dS — 27/27 GPU checks.
Genetic code table on GPU, `log()` polyfill for Jukes-Cantor.

### Phase 12: 25-Domain Benchmark (Exp059) + metalForge (Exp060)
23-domain Rust vs Python benchmark: **22.5× overall speedup**.
metalForge cross-substrate validation: 20/20 checks proving CPU↔GPU parity
for Track 1c algorithms — math is substrate-independent.

### Phase 13: ML Ensembles (Exp061–063)
Random Forest (majority vote, 5 trees) and GBM (binary + multi-class
with sigmoid/softmax) — both proven as pure Rust math (29/29 CPU checks).
RF promoted to GPU via local WGSL shader (13/13 GPU checks, SoA layout).
Combined v1-v5: **157/157 checks across 25 domains**.

### Phase 14: Current — Evolution Readiness
Following hotSpring's patterns: shaping all validated Rust modules for
ToadStool absorption. 9 local WGSL shaders as handoff candidates.
metalForge proving substrate independence across CPU, GPU, and NPU
characterization. wetSpring writes extensions, ToadStool absorbs, we lean.

---

## Module Inventory

### CPU Bio Modules (41)

| Module | Algorithm | Validated Against |
|--------|-----------|-------------------|
| `alignment` | Smith-Waterman local alignment (affine gaps) | Pure Python SW |
| `ani` | Average Nucleotide Identity (pairwise + matrix) | Pure Python ANI |
| `bistable` | Fernandez 2020 bistable phenotypic switching | scipy ODE bifurcation |
| `bootstrap` | RAWR bootstrap resampling (Wang 2021) | Pure Python resampling |
| `capacitor` | Mhatre 2020 phenotypic capacitor ODE | scipy ODE baseline |
| `chimera` | UCHIME-style chimera detection | DADA2-R removeBimeraDenovo |
| `cooperation` | Bruger & Waters 2018 QS game theory | scipy ODE baseline |
| `dada2` | ASV denoising (Callahan 2016) | DADA2-R dada() |
| `decision_tree` | Decision tree inference | sklearn DecisionTreeClassifier |
| `derep` | Dereplication + abundance | VSEARCH --derep_fulllength |
| `diversity` | Shannon, Simpson, Chao1, Bray-Curtis, Pielou, rarefaction | QIIME2 diversity |
| `dnds` | Nei-Gojobori 1986 pairwise dN/dS | Pure Python + Jukes-Cantor |
| `eic` | EIC/XIC extraction + peak integration | asari 1.13.1 |
| `feature_table` | Asari-style LC-MS feature extraction | asari 1.13.1 |
| `felsenstein` | Felsenstein pruning phylogenetic likelihood | Pure Python JC69 |
| `gbm` | Gradient Boosting Machine inference (binary + multi-class) | sklearn GBM specification |
| `gillespie` | Gillespie SSA (stochastic simulation) | numpy ensemble statistics |
| `hmm` | Hidden Markov Model (forward/backward/Viterbi/posterior) | numpy HMM (sovereign) |
| `kmd` | Kendrick mass defect | pyOpenMS |
| `kmer` | K-mer counting (2-bit canonical) | QIIME2 feature-classifier |
| `merge_pairs` | Paired-end overlap merging | VSEARCH --fastq_mergepairs |
| `molecular_clock` | Strict/relaxed clock, calibration, CV | Pure Python clock |
| `multi_signal` | Srivastava 2011 multi-input QS network | scipy ODE baseline |
| `neighbor_joining` | Neighbor-Joining tree construction (Saitou & Nei 1987) | Pure Python NJ |
| `ode` | Generic RK4 ODE integrator | scipy.integrate.odeint |
| `pangenome` | Gene clustering, Heap's law, enrichment, BH FDR | Pure Python pangenome |
| `pcoa` | PCoA (Jacobi eigendecomposition) | QIIME2 emperor |
| `phage_defense` | Hsueh 2022 phage defense deaminase | scipy ODE baseline |
| `phred` | Phred quality decode/encode | Biopython |
| `placement` | Alamin & Liu 2024 phylogenetic placement | Pure Python placement |
| `qs_biofilm` | Waters 2008 QS/c-di-GMP model | scipy ODE baseline |
| `quality` | Quality filtering (Trimmomatic-style) | Trimmomatic/Cutadapt |
| `random_forest` | Random Forest ensemble inference (majority vote) | sklearn RandomForestClassifier specification |
| `reconciliation` | DTL reconciliation for cophylogenetics (Zheng 2023) | Pure Python DTL |
| `robinson_foulds` | RF tree distance | dendropy |
| `signal` | 1D peak detection | scipy.signal.find_peaks |
| `snp` | SNP calling (reference vs alt alleles, frequency) | Pure Python SNP |
| `spectral_match` | MS2 cosine similarity | pyOpenMS |
| `taxonomy` | Naive Bayes classifier (RDP-style) | QIIME2 classify-sklearn |
| `tolerance_search` | ppm/Da m/z search | FindPFAS |
| `unifrac` | Unweighted/weighted UniFrac + Newick parser | QIIME2 diversity |

### GPU Modules (20)

`ani_gpu`, `chimera_gpu`, `dada2_gpu`, `diversity_gpu`, `dnds_gpu`,
`eic_gpu`, `gemm_cached`, `hmm_gpu`, `kriging`, `ode_sweep_gpu`,
`pangenome_gpu`, `pcoa_gpu`, `quality_gpu`, `rarefaction_gpu`,
`random_forest_gpu`, `snp_gpu`, `spectral_match_gpu`, `stats_gpu`,
`streaming_gpu`, `taxonomy_gpu`

Plus 4 ToadStool bio primitives consumed directly: `SmithWatermanGpu`,
`GillespieGpu`, `TreeInferenceGpu`, `FelsensteinGpu`.

### Local WGSL Shaders (9)

| Shader | Domain | Absorption Target |
|--------|--------|-------------------|
| `quality_filter.wgsl` | Read quality trimming | `ParallelFilter<T>` |
| `dada2_e_step.wgsl` | DADA2 error model | `BatchPairReduce<f64>` |
| `hmm_forward_f64.wgsl` | HMM batch forward | `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | ODE parameter sweep | Fix upstream `BatchedOdeRK4F64` |
| `ani_batch_f64.wgsl` | ANI pairwise identity | `AniBatchF64` |
| `snp_calling_f64.wgsl` | SNP calling | `SnpCallingF64` |
| `dnds_batch_f64.wgsl` | dN/dS (Nei-Gojobori) | `DnDsBatchF64` |
| `pangenome_classify.wgsl` | Pangenome classification | `PangenomeClassifyGpu` |
| `rf_batch_inference.wgsl` | Random Forest batch inference | `RfBatchInferenceGpu` |

### I/O Modules

`io::fastq` (streaming FASTQ/gzip), `io::mzml` (streaming mzML/base64),
`io::ms2` (streaming MS2)

---

## Repository Structure

```
wetSpring/
├── README.md                      ← this file
├── BENCHMARK_RESULTS.md           ← three-tier benchmark results
├── CONTROL_EXPERIMENT_STATUS.md   ← experiment status tracker (63 experiments)
├── HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_20_2026.md  ← ToadStool handoff
├── barracuda/                     ← Rust crate (src/, tests/, Cargo.toml)
│   ├── EVOLUTION_READINESS.md    ← absorption map (tiers, primitives, shaders)
│   └── src/shaders/              ← 9 local WGSL shaders (Write → Absorb → Lean)
├── experiments/                   ← 63 experiment protocols + results
├── metalForge/                    ← hardware characterization + substrate routing
│   ├── PRIMITIVE_MAP.md          ← Rust module ↔ ToadStool primitive mapping
│   ├── ABSORPTION_STRATEGY.md   ← Write → Absorb → Lean methodology
│   └── benchmarks/
│       └── CROSS_SYSTEM_STATUS.md ← algorithm × substrate matrix
├── archive/
│   └── handoffs/                  ← fossil record of ToadStool handoffs (v1–v4)
├── scripts/                       ← Python baselines (35 scripts)
├── specs/                         ← specifications and paper queue
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
# Run all tests (582 tests)
cd barracuda && cargo test

# Run CPU validation binaries (1,241 checks across 29 binaries)
for bin in validate_qs_ode validate_rf_distance validate_gillespie \
           validate_newick_parse validate_bistable validate_multi_signal \
           validate_cooperation validate_hmm validate_capacitor \
           validate_alignment validate_felsenstein validate_barracuda_cpu \
           validate_barracuda_cpu_v2 validate_barracuda_cpu_v3 \
           validate_barracuda_cpu_v4 validate_barracuda_cpu_v5 \
           validate_phage_defense validate_bootstrap validate_placement \
           validate_phynetpy_rf validate_phylohmm validate_sate_pipeline \
           validate_algae_timeseries validate_bloom_surveillance \
           validate_epa_pfas_ml validate_massbank_spectral \
           validate_rare_biosphere validate_viral_metagenomics \
           validate_sulfur_phylogenomics validate_phosphorus_phylogenomics \
           validate_population_genomics validate_pangenomics; do
    cargo run --release --bin $bin
done

# Run GPU validation (requires GPU + --features gpu, 260 checks across 12 binaries)
for gpu_bin in validate_diversity_gpu validate_16s_pipeline_gpu \
               validate_barracuda_gpu_v3 validate_gpu_phylo_compose \
               validate_gpu_hmm_forward benchmark_phylo_hmm_gpu \
               validate_gpu_ode_sweep validate_toadstool_bio \
               validate_gpu_track1c validate_cross_substrate \
               validate_gpu_rf; do
    cargo run --features gpu --release --bin $gpu_bin
done

# 25-domain Rust vs Python benchmark
cargo run --release --bin benchmark_23_domain_timing
python3 scripts/benchmark_rust_vs_python.py
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
| MBL darchive | Sogin deep-sea amplicon | Rare biosphere (Exp051) |
| MG-RAST | Anderson 2014 viral | Viral metagenomics (Exp052) |
| Figshare | Mateos 2023 sulfur | Sulfur phylogenomics (Exp053) |
| OSF | Boden 2024 phosphorus | Phosphorus phylogenomics (Exp054) |
| NCBI SRA | PRJNA283159 | Population genomics (Exp055) |
| NCBI SRA | PRJEB5293 | Pangenomics (Exp056) |

---

## Related

- **hotSpring** — Nuclear/plasma physics validation (sibling Spring)
- **ToadStool** — GPU compute engine (BarraCUDA crate)
- **ecoPrimals** — Parent ecosystem
