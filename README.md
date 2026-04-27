# wetSpring — Life Science & Analytical Chemistry Validation

Pure Rust spring reproducing published results in metagenomics, analytical
chemistry (LC-MS, PFAS), and mathematical biology. Validates Python/R baselines
against Rust implementations, then promotes to GPU acceleration via
[barraCuda](https://github.com/ecoPrimals/barraCuda).

```
Tier 1: Python/R baseline  →  Rust CPU parity  →  GPU acceleration
           (58 scripts)        (1,594 tests)       (44 GPU modules)

Tier 2: Rust validation     →  NUCLEUS composition patterns
           (342 binaries)      (136/136 proto-nucleate, 7 deploy graphs)

Tier 3: Composition         →  IPC parity (Exp401) → Niche gate (Exp402)
           (42 niche caps)     (18 IPC roundtrips)    (63/63 checks)

Tier 4: Primal proof        →  Live NUCLEUS IPC (Exp403) → ecoBin harvest
           (48 consumed caps)   (5 primals, check_skip)    (plasmidBin)

Tier 5: guideStone          →  Self-validating NUCLEUS node (Level 4+)
           (wetspring_guidestone) (38/38 live NUCLEUS, v0.9.17 manifest)

Tier 6: Composition Explorer → Interactive NUCLEUS via shell composition
           (wetspring_composition.sh) (Phase 46 template, data viz lane)
```

| | |
|---|---|
| **Tests** | 1,594 (lib) + 18 IPC roundtrip + integration, 0 failed |
| **Validation checks** | 5,900+ across 364 binaries (342 barracuda + 22 forge) |
| **Experiments** | 380 completed + 3 proposed (383 indexed) |
| **Coverage** | 91.20% line / 90.30% function (llvm-cov gated at 90%) |
| **IPC capabilities** | 42 niche, 48 consumed (33 v0.9.17 canonical + 15 legacy), 37 dispatch, 21 domains |
| **Named tolerances** | 242 with machine-readable provenance trail |
| **Clippy** | 0 warnings (pedantic + nursery) |
| **Unsafe** | 0 (`forbid(unsafe_code)` workspace-level + per-crate) |
| **`#[allow()]`** | 0 in production (uses `#[expect(reason)]` exclusively) |
| **Local WGSL** | 0 — fully lean on barraCuda |
| **Duplicate math** | 0 — all NMF, stats, special delegated to barraCuda |
| **Composition** | 136/136 proto-nucleate alignment checks (Exp400, D01–D07, guard constant) |
| **Deploy graphs** | 7 (all canonical `[[graph.nodes]]` schema, bonding + fragments metadata) |
| **Primal gaps** | 15 open in `docs/PRIMAL_GAPS.md` (PG-01–PG-22, 7 resolved), 5 new from Phase 46 composition |
| **cargo-deny** | advisories ok, bans ok, licenses ok, sources ok |
| **License** | AGPL-3.0-or-later |
| **MSRV** | 1.87 (edition 2024) |

**Current release — V150:** Phase 46 composition explorer — data exploration & visualization lane. `wetspring_composition.sh` exercises interactive NUCLEUS via petalTongue scene graphs (100-node scenes accepted, ~41KB payload, <1ms), barraCuda IPC math (stats.mean/std_dev/correlation/fft all pass), rhizoCrypt/loamSpine/sweetGrass DAG/ledger/braid (connection reset — PG-18). `composition_nucleus.sh` launcher, `nucleus_composition_lib.sh` library, Python UDS shim for socat-free environments. 5 new gaps documented (PG-18..22). guideStone **Level 4** (38/38 pass, 4 skip, exit 0) unchanged. V149: `is_skip_error` adoption, upstream drift noted. V148: Level 4 initial (31/31). V147: N2 v0.9.15 surface.

---

## Tracks

| Track | Domain | Papers | Key Algorithms |
|-------|--------|:------:|----------------|
| **1** | Microbial Ecology (16S rRNA) | 10 | FASTQ QC, DADA2 denoising, chimera, taxonomy, UniFrac, diversity, ODE models, game theory, phage defense |
| **1b** | Comparative Genomics | 5 | Robinson-Foulds, HMM, Smith-Waterman, Felsenstein pruning, NJ trees, DTL reconciliation |
| **1c** | Deep-Sea Metagenomics | 6 | ANI, SNP calling, dN/dS, molecular clock, pangenomics, rare biosphere |
| **2** | Analytical Chemistry (LC-MS, PFAS) | 4 | mzML parsing, EIC, peak detection, spectral matching, KMD, PFAS screening |
| **3** | Drug Repurposing | 5 | NMF, pathway scoring, knowledge graph embedding, TransE, SparseGEMM |
| **4** | No-Till Soil & Anderson Geometry | 9 | Anderson localization, Lanczos, quorum sensing ODE, cooperation dynamics, pore geometry |
| **5** | Anderson Extensions | 4 | Hormesis, binding landscapes, disorder mapping, colonization resistance |
| **6** | Anaerobic Digestion | 5 | Gompertz kinetics, co-digestion, AFEX pretreatment, fungal fermentation |

---

## Architecture

Springs are scientific validation targets. barraCuda provides GPU math.
toadStool routes hardware. Springs never import each other.

```
Write  →  Validate  →  Hand off  →  Absorb  →  Lean
──────    ─────────    ─────────    ────────    ─────
CPU +     test vs      document     barraCuda   rewire to
WGSL      Python       in handoff   adds ops    upstream, delete local
```

### Crate Structure

| Crate | Purpose | Modules |
|-------|---------|---------|
| `wetspring-barracuda` | Library: bio algorithms + I/O + validation | 46 CPU + 44 GPU + 6 I/O |
| `wetspring-forge` | Hardware discovery, dispatch, visualization | 13 modules |
| `wetspring-barracuda-fuzz` | libFuzzer targets for parsers | 4 targets |

### GPU Evolution Status

| Phase | Count | Description |
|-------|:-----:|-------------|
| Lean | 22 | Direct upstream barraCuda primitive |
| Compose | 11 | Multiple barraCuda primitives wired |
| Write → Lean | 5 | ODE shaders via `generate_shader()` (local WGSL deleted) |
| Tier C | 0 | Full lean achieved |

---

## Performance

| Workload | Speedup | Parity |
|----------|--------:|--------|
| Spectral cosine (2048 spectra) | 926x | ≤ 1e-10 |
| Full 16S pipeline (10 samples) | 2.5x | 88/88 checks |
| Shannon/Simpson diversity | 15–25x | ≤ 1e-6 |
| Bifurcation eigenvalues | bit-exact | 2.67e-16 rel |
| Rust vs Python (25 domains) | **33x** | 51 ms vs 1,713 ms |
| Peak: Smith-Waterman | 625x | exact |

---

## Quick Start

```bash
# Run all tests
cargo test --workspace --all-features

# Code quality
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Run a validation binary
cargo run --release --bin validate_diversity

# Run all validation binaries (meta-runner)
cargo run --bin validate_all

# Coverage gate (workspace alias)
cargo coverage-check

# GPU validation (requires gpu feature + compatible hardware)
cargo run --features gpu --release --bin validate_barracuda_cpu_v27

# Coverage
cargo llvm-cov --workspace --html
```

---

## CPU Modules (46)

| Module | Algorithm | Validated Against |
|--------|-----------|-------------------|
| `adapter` | Adapter detection + 3' trimming | Trimmomatic/Cutadapt |
| `alignment` | Smith-Waterman (affine gaps) | Pure Python SW |
| `ani` | Average Nucleotide Identity | Pure Python ANI |
| `bistable` | Fernandez 2020 phenotypic switching | scipy ODE |
| `bootstrap` | RAWR bootstrap resampling | Pure Python |
| `capacitor` | Mhatre 2020 phenotypic capacitor | scipy ODE |
| `chimera` | UCHIME-style chimera detection | DADA2-R |
| `cooperation` | Bruger 2018 QS game theory | scipy ODE |
| `dada2` | ASV denoising (Callahan 2016) | DADA2-R |
| `decision_tree` | Decision tree inference | sklearn |
| `derep` | Dereplication + abundance | VSEARCH |
| `diversity` | Shannon, Simpson, Chao1, Bray-Curtis, Pielou | QIIME2 |
| `dnds` | Nei-Gojobori pairwise dN/dS | Pure Python |
| `eic` | EIC/XIC extraction + peak integration | asari 1.13.1 |
| `esn` | Echo State Network reservoir (NPU int8) | Pure Python ESN |
| `feature_table` | LC-MS feature extraction | asari 1.13.1 |
| `felsenstein` | Pruning phylogenetic likelihood | Pure Python JC69 |
| `gbm` | Gradient Boosting Machine | sklearn GBM |
| `gillespie` | Gillespie SSA (stochastic simulation) | numpy |
| `hmm` | HMM (forward/backward/Viterbi) | numpy |
| `hormesis` | Biphasic dose-response via Anderson | Calabrese & Mattson 2017 |
| `binding_landscape` | Colonization resistance, IPR | healthSpring exp098 |
| `kmd` | Kendrick mass defect | pyOpenMS |
| `kmer` | K-mer counting (2-bit canonical) | QIIME2 |
| `merge_pairs` | Paired-end overlap merging | VSEARCH |
| `molecular_clock` | Strict/relaxed clock, calibration | Pure Python |
| `multi_signal` | Srivastava 2011 multi-input QS | scipy ODE |
| `neighbor_joining` | NJ tree construction (Saitou & Nei 1987) | Pure Python |
| `nmf` | Non-negative Matrix Factorization | Yang 2020 / Gao 2020 |
| `ode` | Generic RK4 ODE integrator | scipy.integrate |
| `pangenome` | Gene clustering, Heap's law, BH FDR | Pure Python |
| `pcoa` | PCoA (Jacobi eigendecomposition) | QIIME2 emperor |
| `phage_defense` | Hsueh 2022 phage defense deaminase | scipy ODE |
| `placement` | Phylogenetic placement (Alamin & Liu 2024) | Pure Python |
| `qs_biofilm` | Waters 2008 QS/c-di-GMP model | scipy ODE |
| `quality` | Quality filtering (Trimmomatic-style) | Trimmomatic |
| `random_forest` | Random Forest ensemble inference | sklearn |
| `reconciliation` | DTL reconciliation (Zheng 2023) | Pure Python |
| `robinson_foulds` | RF tree distance | dendropy |
| `signal` | 1D peak detection | scipy.signal |
| `snp` | SNP calling | Pure Python |
| `spectral_match` | MS2 cosine similarity | pyOpenMS |
| `taxonomy` | Naive Bayes classifier (RDP-style) | QIIME2 |
| `tolerance_search` | ppm/Da m/z search | FindPFAS |
| `transe` | TransE knowledge graph embedding | ROBOKOP |
| `unifrac` | Unweighted/weighted UniFrac | QIIME2 |

## I/O Modules

All parsers stream from disk — no full-file buffering.

| Module | Format | Features |
|--------|--------|----------|
| `io::fastq` | FASTQ + gzip | Zero-copy `for_each_record`, iterator, stats |
| `io::mzml` | mzML (base64/zlib) | Streaming XML, reusable decode buffers |
| `io::ms2` | MS2 | Streaming line parser, single reused buffer |
| `io::mzxml` | mzXML | Streaming XML iterator |
| `io::jcamp` | JCAMP-DX | Streaming multi-block parser |
| `io::nanopore` | POD5/FAST5/NRS | Binary signal bridge, int16 streaming |

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt` | Clean |
| `cargo clippy` (pedantic + nursery) | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `#![forbid(unsafe_code)]` | Workspace-level (`[workspace.lints.rust]`) + all crate roots |
| `#![deny(clippy::unwrap_used)]` | Workspace `[lints.clippy]` |
| TODO/FIXME/HACK | 0 |
| Inline tolerance literals | 0 (242 named constants) |
| Max file size | All `.rs` under 1,000 lines (largest: 939 LOC) |
| C dependencies | 0 (`flate2` uses `rust_backend`) |
| SPDX headers | All `.rs` files |

---

## Data Provenance

All validation data from public repositories:

| Source | Accession | Usage |
|--------|-----------|-------|
| NCBI SRA | PRJNA488170 | Algae pond 16S |
| NCBI SRA | PRJNA382322 | Nannochloropsis 16S |
| NCBI SRA | PRJNA1114688 | Lake microbiome 16S |
| Zenodo | 14341321 | Jones Lab PFAS library |
| Michigan EGLE | ArcGIS REST | PFAS surface water |
| Reese 2019 | PMC6761164 | VOC biomarkers |
| MBL darchive | Sogin amplicon | Rare biosphere |
| MG-RAST | Anderson 2014 | Viral metagenomics |
| Figshare | Mateos 2023 | Sulfur phylogenomics |
| OSF | Boden 2024 | Phosphorus phylogenomics |
| NCBI SRA | PRJNA283159 | Population genomics |
| NCBI SRA | PRJEB5293 | Pangenomics |

---

## Related

| Repository | Role |
|------------|------|
| [barraCuda](https://github.com/ecoPrimals/barraCuda) | GPU compute library — 800+ WGSL shaders, f64-canonical |
| [toadStool](https://github.com/ecoPrimals/toadStool) | Hardware dispatch — GPU/NPU/CPU routing |
| [coralReef](https://github.com/ecoPrimals/coralReef) | Shader compiler — WGSL → native binary |
| [wateringHole](https://github.com/ecoPrimals/wateringHole) | Ecosystem standards and handoffs |
| hotSpring | Physics validation — precision math, DF64, ESN |
| neuralSpring | ML validation — eigensolvers, attention, TransE |
| airSpring | Agriculture validation — Richards PDE, seasonal pipelines |
| groundSpring | Soil and land-surface validation |
| healthSpring | Health and physiology validation |
| ludoSpring | Game-theoretic and strategic validation |
| primalSpring | Primal ecosystem coordination |

Detailed version history: [CHANGELOG.md](CHANGELOG.md)

---

## Part of ecoPrimals

This repo is a domain validation spring in the
[ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem.
Springs reproduce published scientific results using pure Rust and
[barraCuda](https://github.com/ecoPrimals/barraCuda) GPU primitives.

See [wateringHole](https://github.com/ecoPrimals/wateringHole) for ecosystem
documentation and standards.
