# wetSpring White Paper

**Date:** February 22, 2026
**Status:** Validation study complete — 2,229+/2,229+ checks, 740 tests, 97% bio+io line coverage, 97 experiments
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

4. Can sovereign ML (decision tree, Random Forest, GBM inference) reproduce
   Python sklearn predictions with 100% parity, removing the Python runtime?

5. Is the math truly substrate-independent — does the same algorithm produce
   identical results on CPU, GPU, and (eventually) NPU?

6. Does ToadStool's unidirectional streaming eliminate the GPU round-trip
   overhead that makes naive GPU dispatch slower than CPU for small workloads?

7. Can local WGSL shaders be structured for clean absorption into ToadStool,
   following the Write → Absorb → Lean cycle proven by hotSpring?

---

## Evolution Methodology: Write → Absorb → Lean

wetSpring adopts hotSpring's proven absorption cycle for evolving local
implementations into upstream ToadStool/BarraCUDA primitives:

1. **Write** — Implement on CPU with WGSL shader templates in `.wgsl` files
2. **Validate** — Test CPU ↔ GPU parity against Python baselines
3. **Hand off** — Document in `wateringHole/handoffs/` with binding layouts
4. **Absorb** — ToadStool integrates as `ops::bio::*` shaders
5. **Lean** — wetSpring rewires to upstream imports, deletes local code

**Current status:** 24 GPU modules lean on upstream (19 wetSpring + 5 neuralSpring); 4 local WGSL shaders
in Write phase (ODE sweep, kmer histogram, unifrac propagation, taxonomy FC).
Phase 23 structurally evolved all trajectory types to flat contiguous layouts,
unified the DADA2 error model, and eliminated per-step clones — all directly
GPU-buffer-compatible. The forge crate (`metalForge/forge/` v0.2.0) provides
substrate discovery and capability-based dispatch as an absorption seam for
ToadStool.

---

## Key Results

| Claim | Evidence |
|-------|----------|
| Rust matches Python across 96 experiments | 1,392/1,392 CPU checks + 80 dispatch + 35 layout + 57 transfer/streaming + 39 cross-spring + 10 local WGSL pass |
| GPU matches CPU across all promoted domains | 609/609 GPU checks pass (incl. 48 all-domain + 28 metalForge v3) |
| BarraCUDA CPU parity across 25 domains + 6 ODE flat | 205/205 cross-domain checks pass |
| 926× spectral cosine GPU speedup | Exp016 benchmark |
| 2.45× full 16S pipeline GPU speedup | Exp015/016 benchmark |
| 22.5× overall Rust vs Python (25 domains) | Exp059, peak 625× (SW) |
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
| Rare biosphere diversity (Anderson 2015) | Exp051, 35/35 checks |
| Viral metagenomics dN/dS (Anderson 2014) | Exp052, 22/22 checks |
| Sulfur phylogenomics molecular clock (Mateos 2023) | Exp053, 15/15 checks |
| Phosphorus phylogenomics (Boden 2024) | Exp054, 13/13 checks |
| Population genomics ANI/SNP (Anderson 2017) | Exp055, 24/24 checks |
| Pangenomics Heap's law + enrichment (Moulana 2020) | Exp056, 24/24 checks |
| GPU Track 1c: ANI + SNP + pangenome + dN/dS | Exp058, 27/27 GPU checks |
| Random Forest ensemble (5 trees, 3 classes) | Exp061, 13/13 CPU checks |
| GBM inference (binary + multi-class) | Exp062, 16/16 CPU checks |
| GPU RF batch inference (SoA layout) | Exp063, 13/13 GPU checks |
| metalForge cross-substrate CPU↔GPU parity | Exp060, 20/20 checks |
| GPU streaming 1.27× speedup over individual dispatch | Exp072, 17/17 checks |
| Dispatch overhead: streaming wins at all batch sizes | Exp073, 21/21 checks |
| GPU↔NPU↔CPU substrate router with PCIe topology | Exp074, 20/20 checks |
| Pure GPU 5-stage analytics: 0.1% pipeline overhead | Exp075, 31/31 checks |
| Cross-substrate GPU→NPU→CPU pipeline profiled | Exp076, 17/17 checks |
| ToadStool bio rewire: 8 modules lean, 2 bugs fixed | Exp077, 451/451 GPU re-validated |
| BarraCUDA CPU v7: Tier A layouts lossless | Exp085, 43/43 kmer/UniFrac/taxonomy round-trips |
| metalForge 5-stage pipeline: substrate-independent | Exp086, 45/45 dispatch routing + parity |
| GPU extended to 16 domains (EIC, PCoA, Kriging, Rarefaction) | Exp087, 50+ GPU checks |
| PCIe direct transfer: no CPU staging | Exp088, 32/32 buffer contract checks |
| Pure GPU streaming: 441-837× over round-trip | Exp090, 80/80 zero CPU round-trip pipeline |
| Streaming eliminates 92-94% of round-trip overhead | Exp091, formal benchmark at 1-128 batch |
| CPU vs GPU parity across all 16 domains | Exp092, 48/48 all-domain head-to-head |
| metalForge full 16-domain substrate-independence | Exp093, 28/28 cross-substrate v3 |
| Write → Absorb → Lean: 24 absorbed, 4 in Write | ABSORPTION_MANIFEST.md, hotSpring methodology |
| Cross-spring evolution: 5 neuralSpring primitives validated | Exp094, 39/39 checks; Exp095, 7 benchmarks |
| Local WGSL compile + dispatch | Exp096, 10/10 checks |
| Structural evolution: flat layouts + DRY models + zero-clone APIs | Exp097, 728 tests, 48/48 GPU, 39/39 cross-spring |

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

### GPU Composition & Evolution

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

### Track 1c: Deep-Sea Metagenomics & Microbial Evolution (R. Anderson)

| Exp | Paper | What We Prove |
|-----|-------|---------------|
| 051 | Anderson 2015 (rare biosphere) | Diversity, rarefaction, PCoA, rare lineage detection |
| 052 | Anderson 2014 (viral metagenomics) | Community diversity, dN/dS (Nei-Gojobori 1986) |
| 053 | Mateos 2023 (sulfur phylogenomics) | Molecular clock, DTL reconciliation, Robinson-Foulds |
| 054 | Boden 2024 (phosphorus phylogenomics) | Cross-validation of clock/reconciliation pipeline |
| 055 | Anderson 2017 (population genomics) | ANI, SNP calling, allele frequency, population pipeline |
| 056 | Moulana 2020 (pangenomics) | Core/accessory/unique genes, Heap's law, enrichment, BH FDR |

### GPU Track 1c + Cross-Substrate + ML Ensembles

| Exp | Method | What We Prove |
|-----|--------|---------------|
| 057 | BarraCUDA CPU v4 (Track 1c) | 44/44 across 5 new domains, pure Rust math |
| 058 | GPU Track 1c promotion | 27/27 ANI + SNP + pangenome + dN/dS GPU parity |
| 059 | 25-domain Rust vs Python | 22.5× overall speedup, peak 625× |
| 060 | metalForge cross-substrate | 20/20 CPU↔GPU parity for Track 1c |
| 061 | Random Forest inference | 13/13 ensemble majority-vote (5 trees, 3 classes) |
| 062 | GBM inference | 16/16 binary + multi-class (sigmoid + softmax) |
| 063 | GPU RF batch inference | 13/13 GPU SoA layout (one thread per sample×tree) |

### Consolidation, Streaming & Cross-Substrate Proofs

| Exp | Method | What We Prove |
|-----|--------|---------------|
| 064 | BarraCUDA GPU v1 consolidated | 26/26 all GPU-eligible domains in one binary |
| 065 | metalForge full portfolio | 35/35 CPU↔GPU parity for entire portfolio |
| 066 | CPU vs GPU scaling (all domains) | Scaling curves across batch sizes |
| 070 | BarraCUDA CPU full (25 domains) | 50/50 consolidated pure Rust math proof |
| 071 | BarraCUDA GPU full (11 domains) | 24/24 consolidated GPU portability proof |
| 072 | GPU streaming pipeline | 17/17 pre-warmed FMR, 1.27× streaming speedup |
| 073 | Dispatch overhead proof | 21/21 streaming beats individual at all batch sizes |
| 074 | metalForge substrate router | 20/20 GPU↔NPU↔CPU routing with PCIe topology |
| 075 | Pure GPU analytics pipeline | 31/31 five-stage GPU pipeline, 0.1% overhead |
| 076 | Cross-substrate pipeline | 17/17 GPU→NPU→CPU with latency profiling |

### GPU/NPU Readiness + Dispatch Validation (Phase 21)

| Exp | Method | What We Prove |
|-----|--------|---------------|
| 077 | ToadStool bio rewire | 451/451 GPU re-validated after 8-module absorption |
| 078 | ODE GPU sweep readiness | Flat param APIs (to_flat/from_flat) for 5 ODE modules |
| 079 | BarraCUDA CPU v6 (ODE flat) | 48/48 bitwise-identical ODE math through flat serialization |
| 080 | metalForge dispatch routing | 35/35 router across 5 substrate configs (live hardware) |
| 081 | K-mer GPU histogram | Dense 4^k histogram + sorted pairs for GPU dispatch |
| 082 | UniFrac CSR flat tree | FlatTree with CSR layout + sample matrix for GPU pairwise |
| 083 | Taxonomy NPU int8 | Affine int8 quantization, argmax parity with f64 |

### Pure GPU Streaming + Full Validation (Phase 22)

| Exp | Method | What We Prove |
|-----|--------|---------------|
| 085 | BarraCUDA CPU v7 (Tier A layouts) | 43/43 kmer histogram, UniFrac CSR, taxonomy int8 round-trip fidelity |
| 086 | metalForge pipeline proof | 45/45 five-stage dispatch routing + substrate-independent output |
| 087 | GPU extended domains | EIC, PCoA, Kriging, Rarefaction added to GPU suite |
| 088 | PCIe direct transfer proof | 32/32 GPU→NPU, NPU→GPU, GPU→GPU without CPU staging |
| 089 | ToadStool streaming dispatch | 25/25 streaming vs round-trip parity (3-stage, 5-stage chains) |
| 090 | Pure GPU streaming pipeline | 80/80 zero CPU round-trips, 441-837× streaming advantage |
| 091 | Streaming vs round-trip benchmark | CPU vs RT GPU vs streaming: streaming eliminates 92-94% overhead |
| 094 | Cross-spring evolution validation | 39/39 checks — 5 neuralSpring primitives (PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance) |
| 095 | Cross-spring scaling benchmark | 7 benchmarks — 6.5×–277× GPU speedup at realistic bio sizes |

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

## R. Anderson Track 1c: Deep-Sea Metagenomics — COMPLETE

Rika Anderson (Carleton College) studies microbial and viral evolution in deep-sea
hydrothermal vents. All 6 papers reproduced (Exp051-056), validating 133 checks
across 5 new sovereign Rust modules:

| Module | Algorithm | Paper |
|--------|-----------|-------|
| `bio::dnds` | Nei-Gojobori 1986 pairwise dN/dS | Anderson 2014, 2017 |
| `bio::molecular_clock` | Strict/relaxed clock, calibration, CV | Mateos 2023, Boden 2024 |
| `bio::ani` | Average Nucleotide Identity (pairwise + matrix) | Anderson 2017 |
| `bio::snp` | SNP calling (ref/alt alleles, frequency, density) | Anderson 2017 |
| `bio::pangenome` | Gene clustering, Heap's law, hypergeometric, BH FDR | Moulana 2020 |

All 5 modules GPU-promoted (Exp058: 27/27 checks, 4 Track 1c WGSL shaders),
cross-substrate validated (Exp060: 20/20 CPU↔GPU parity).

---

## Relationship to ecoPrimals

wetSpring is one of several **Springs** — validation targets that prove
algorithms can be ported from interpreted languages to BarraCUDA/ToadStool:

- **hotSpring** — Nuclear physics, plasma, lattice QCD (34 WGSL shaders, 637 tests)
- **wetSpring** — Life science, analytical chemistry, environmental monitoring (4 local WGSL shaders, 740 tests)
- **neuralSpring** — ML inference, eigensolvers, TensorSession
- **archive/handoffs/** — Fossil record of ToadStool handoffs (v1–v7)

Springs follow the **Write → Absorb → Lean** pattern (pioneered by hotSpring):
write and validate locally, hand off to ToadStool for absorption, then lean on
upstream primitives. This reduces dispatch overhead and round-trips via streaming
pipeline composition. wetSpring's `metalForge/` directory characterizes available
hardware (GPU, NPU, CPU) and guides Rust implementations for optimal absorption.

## Code Quality

| Metric | Value |
|--------|-------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --pedantic --nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| Line coverage (`cargo-llvm-cov`) | **~97% bio+io modules** |
| `#![deny(unsafe_code)]` | Enforced crate-wide (edition 2024; `allow` only in test env-var calls) |
| `#![deny(clippy::expect_used, clippy::unwrap_used)]` | Enforced crate-wide |
| Named tolerance constants | 43 (all scientifically justified) |
| External C dependencies | 0 (`flate2` uses `rust_backend`) |
| Max file size | All under 1000 LOC |
| SPDX headers | All `.rs` files |
| Provenance headers | All 87 validation/benchmark binaries |

## metalForge — Hardware Discovery

wetSpring includes a standalone Rust crate (`wetspring-forge`) for
runtime hardware discovery and capability-based workload routing,
following hotSpring's `metalForge/forge/` pattern:

| Component | Purpose |
|-----------|---------|
| `substrate.rs` | Substrate, Capability, Identity types |
| `probe.rs` | GPU (wgpu), CPU (`/proc`), NPU (`/dev/akida*`) |
| `inventory.rs` | Unified discovery |
| `dispatch.rs` | Capability-based routing (GPU > NPU > CPU) |
| `bridge.rs` | Forge substrate ↔ barracuda `WgpuDevice` bridge |

29 tests, clippy clean, `#![forbid(unsafe_code)]`.

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
| sklearn | 1.3+ | ML classification (DT, RF, GBM) |
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
