# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — life science biome evolving Rust implementations
and GPU shaders for ToadStool/BarraCUDA absorption. Follows the
**Write → Absorb → Lean** cycle adopted from hotSpring.

**Date:** February 22, 2026
**License:** AGPL-3.0-or-later
**Status:** Phase 22 — 19/20 GPU modules lean on upstream; 4 WGSL shaders in Write phase

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

## Evolution Architecture: Write → Absorb → Lean

wetSpring follows hotSpring's proven absorption cycle. Springs are biomes;
ToadStool/BarraCUDA is the fungus present in every biome. Springs don't
import each other — they lean on ToadStool independently.

```
Write → Validate → Hand off → Absorb → Lean
─────────────────────────────────────────────
Implement     Test against    Document in    ToadStool adds    Rewire to
on CPU +      Python +        wateringHole/  shaders/ops       upstream,
WGSL          known physics   handoffs/                        delete local
```

### Current Evolution Status

| Phase | Count | Description |
|-------|:-----:|-------------|
| **Lean** | 24 | GPU modules consuming upstream ToadStool primitives |
| **Write** | 4 | Local WGSL shaders for Tier A domains |
| **Tier A** | 7 | CPU modules with flat APIs, ready for GPU/NPU promotion |
| **Tier B** | 1 | Needs refactoring before promotion |
| **Tier C** | 14 | CPU-only, no GPU path planned |

### Local WGSL Shaders (Write Phase)

| Shader | Domain | Absorption Target |
|--------|--------|-------------------|
| `batched_qs_ode_rk4_f64.wgsl` | QS/c-di-GMP ODE sweep | `barracuda::ops::bio::ode_rk4` |
| `kmer_histogram_f64.wgsl` | K-mer counting | `barracuda::ops::bio::kmer_histogram` |
| `unifrac_propagate_f64.wgsl` | UniFrac tree propagation | `barracuda::ops::bio::unifrac_propagate` |
| `taxonomy_fc_f64.wgsl` | Taxonomy scoring (GPU/NPU) | `barracuda::ops::bio::taxonomy_fc` |

### metalForge Forge Crate (v0.2.0)

The `metalForge/forge/` crate discovers compute substrates at runtime and
routes workloads to the best capable device. Absorption seam: when ToadStool
absorbs forge, the bridge module becomes the integration point.

| Module | Purpose |
|--------|---------|
| `probe` | GPU (wgpu) + CPU (/proc) + NPU (/dev) discovery |
| `inventory` | Unified substrate view |
| `dispatch` | Capability-based workload routing |
| `streaming` | Multi-stage GPU pipeline analysis |
| `bridge` | forge ↔ barracuda device bridge |

---

## Current Results

| Metric | Count |
|--------|-------|
| Validation checks (CPU) | 1,392 |
| Validation checks (GPU) | 609 |
| Dispatch validation checks | 80 |
| Layout fidelity checks | 35 |
| Transfer/streaming checks | 57 |
| **Total validation checks** | **2,219+** |
| Rust library unit tests | 654 (+ 1 ignored — hardware-dependent) |
| Integration tests | 60 (23 bio + 16 determinism + 21 I/O) |
| Rust doc-tests | 14 |
| **Total Rust tests** | **728** |
| Experiments completed | 95 |
| Validation/benchmark binaries | 77 validate + 8 benchmark = 85 total |
| CPU bio modules | 41 |
| GPU bio modules | 20 (19 lean on ToadStool, 4 local WGSL shaders (ODE, kmer, unifrac, taxonomy)) |
| Tier A (GPU/NPU-ready) | 7 modules with flat layouts |
| Tier B (needs refactor) | 1 module remaining |
| Python baselines | 40 scripts |
| BarraCUDA CPU parity | 205/205 (25 domains + 6 ODE flat) |
| BarraCUDA GPU parity | 16 domains head-to-head (Exp092) |
| metalForge cross-system | 16 domains substrate-independent (Exp093) |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| Pure GPU streaming | 80 checks, 441-837× over round-trip (Exp090-091) |
| ToadStool primitives consumed | **28** (15 original + 8 bio absorbed Feb 22 + 5 neuralSpring absorbed) |
| Local WGSL shaders | **4** (ODE, kmer histogram, unifrac propagate, taxonomy FC) |

All 2,219+ validation checks **PASS**. All 728 tests **PASS**.

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
25-domain Rust vs Python benchmark: **22.5× overall speedup**.
metalForge cross-substrate validation: 20/20 checks proving CPU↔GPU parity
for Track 1c algorithms — math is substrate-independent.

### Phase 13: ML Ensembles (Exp061–063)
Random Forest (majority vote, 5 trees) and GBM (binary + multi-class
with sigmoid/softmax) — both proven as pure Rust math (29/29 CPU checks).
RF promoted to GPU via local WGSL shader (13/13 GPU checks, SoA layout).
Combined v1-v6: **205/205 checks across 25 domains + 6 ODE flat modules**.

### Phase 14: Evolution Readiness
Following hotSpring's patterns: shaping all validated Rust modules for
ToadStool absorption. 9 local WGSL shaders as handoff candidates
(8 absorbed in Phase 20; 1 ODE remains).
metalForge proving substrate independence across CPU, GPU, and NPU
characterization. wetSpring writes extensions, ToadStool absorbs, we lean.

### Phase 15: Code Quality Hardening
Comprehensive audit and evolution of the codebase:
- Crate-level `clippy::pedantic` + `clippy::nursery` lints enforced (0 warnings)
- `rustfmt.toml` with `max_width = 100` enforced across all 151 source files
- All inline tolerance literals replaced with named constants in `tolerances.rs`
- All validation/benchmark binaries carry structured `# Provenance` headers
- All data paths use `validation::data_dir()` for capability-based discovery
- `flate2` explicitly uses `rust_backend` (no C dependencies, ecoBin compliant)
- 11 new unit tests targeting coverage gaps; line coverage 97% bio+io (55% overall)
- 6 new doc-tests on key public API functions
- Zero `unsafe` in production code, zero `.unwrap()` in production code
- All I/O parsers confirmed streaming (no whole-file buffering)
- Smart refactoring: duplicated FASTQ decompression removed in favor of library

### Phase 16: BarraCUDA Evolution + Absorption Readiness
Following hotSpring's Write → Absorb → Lean pattern for ToadStool integration:
- **Handoff document** submitted to `../wateringHole/handoffs/` with all 9 Tier A
  shaders: binding layouts, dispatch geometry, CPU references, validation counts
- **CPU math evolution** identified: 4 local functions (`erf`, `ln_gamma`,
  `regularized_gamma`, `trapz`) that duplicate `barracuda::special`/`numerical`
  — blocked on proposed `barracuda::math` feature (CPU-only, no wgpu)
- **metalForge evolution**: hardware characterization updated with substrate
  routing, absorption strategy, and cross-system validation status
- **naga/NVVM driver profile fix** proposed: `needs_f64_exp_log_workaround()`
  should return `true` for Ada Lovelace (RTX 40-series, sm_89)
- **Evolution narrative** aligned with hotSpring: Springs write validated
  extensions, ToadStool absorbs as shared primitives, Springs lean on upstream

### Phase 17: metalForge Absorption Engineering + Pure GPU Parity
Evolving Rust implementations for ToadStool/BarraCUDA team absorption:
- **`bio::special` consolidated** into shared module (erf, ln_gamma,
  `regularized_gamma_lower`) — shaped for extraction to `barracuda::math`
- **metalForge local** characterization: GPU/NPU/CPU substrate routing with
  absorption-ready Rust patterns (SoA, `#[repr(C)]`, batch APIs, flat arrays)
- **Exp064: BarraCUDA GPU Parity v1** — consolidated GPU domain validation
  across 8 domains (diversity, BC, ANI, SNP, dN/dS, pangenome, RF, HMM).
  Pure GPU math matches CPU reference truth in a single binary
- **Exp065: metalForge Full Cross-System** — substrate-independence proof for
  full portfolio. CPU or GPU dispatch → same answer. Foundation for CPU/GPU/NPU
  routing in production
- **Absorption engineering**: Following hotSpring's pattern where Springs write
  extensions as proposals to ToadStool/BarraCUDA, get absorbed, then lean on
  upstream. 9 WGSL shaders + 4 CPU math functions ready for absorption
  (8 shaders absorbed in Phase 20)
- **Code quality gate**: named tolerances, `#![forbid(unsafe_code)]`, pedantic clippy,
  all binaries with provenance headers — absorption-grade quality

### Phase 18: Streaming Dispatch + Cross-Substrate Validation
Proving the full ToadStool dispatch model and multi-substrate routing:
- **Exp070/071: Consolidated proofs** — 25-domain CPU (50/50) + 11-domain GPU (24/24)
  in single binaries. Pure Rust math, fully portable.
- **Exp072: Streaming pipeline** — `GpuPipelineSession` with pre-warmed FMR delivers
  1.27× speedup over individual dispatch. First-call latency: 5µs (vs 110ms cold).
- **Exp073: Dispatch overhead** — streaming beats individual dispatch at all batch
  sizes [64, 256, 1K, 4K]. Pipeline caching is the correct default.
- **Exp074: Substrate router** — GPU↔NPU↔CPU routing with PCIe topology awareness.
  AKD1000 NPU detected via `/dev/akida0`, graceful CPU fallback. Math parity proven.
- **Exp075: Pure GPU 5-stage pipeline** — Alpha Diversity → Bray-Curtis → PCoA →
  Stats → Spectral Cosine. Single upload/readback. 0.1% pipeline overhead. 31/31 PASS.
- **Exp076: Cross-substrate pipeline** — GPU→NPU→CPU heterogeneous data flow with
  per-stage latency profiling. 17/17 PASS.
- **Handoff v6** — comprehensive ToadStool/BarraCUDA team handoff with all 9 shader
  binding layouts, dispatch geometry, NVVM driver profile bug, CPU math extraction plan,
  and streaming pipeline findings. See `HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md`.

### Phase 20: Current — ToadStool Bio Rewire + Cross-Spring Evolution

ToadStool sessions 31d/31g absorbed all 8 wetSpring bio WGSL shaders.
On Feb 22, wetSpring rewired all 8 GPU modules to delegate to
`barracuda::ops::bio::*`, deleted 8 local shaders (25 KB), and verified
633 tests pass with 0 clippy warnings:

- **8 bio modules rewired** — HMM, ANI, SNP, dN/dS, Pangenome, QF, DADA2, RF
  now delegate to ToadStool primitives (no local shaders)
- **2 ToadStool bugs found and fixed** during validation:
  1. SNP binding layout: `is_variant` marked read-only but shader writes to it
  2. AdapterInfo propagation: `from_existing_simple()` broke f64 polyfill detection
- **Cross-spring evolution documented** — ToadStool serves as convergence hub:
  hotSpring (precision/lattice QCD), wetSpring (bio/genomics), neuralSpring (ML/eigen)
- **4 local WGSL shaders in Write phase** — ODE, kmer histogram, unifrac propagate, taxonomy FC (ODE blocked: upstream `compile_shader` needs `compile_shader_f64`; others ready for absorption)
- **23 ToadStool primitives consumed** (up from 15)
- **New handoff** submitted: `wateringHole/handoffs/WETSPRING_TOADSTOOL_REWIRE_FEB22_2026.md`

### Phase 21: GPU/NPU Readiness + Dispatch Validation

Tier B → A promotion of 3 remaining non-ODE modules, ODE flat param validation,
and forge dispatch router proof:

- **Exp078: ODE GPU Sweep Readiness** — flat param APIs (`to_flat`/`from_flat`)
  for 5 ODE modules (qs_biofilm, bistable, multi_signal, phage_defense, cooperation)
- **Exp079: BarraCUDA CPU v6** — 48/48 checks proving flat serialization preserves
  bitwise-identical ODE math. Zero ULP drift across all 6 biological models.
- **Exp080: metalForge Dispatch Routing** — 35/35 checks validating forge router
  across full-system, GPU-only, NPU+CPU, CPU-only, and mixed PCIe configurations.
  Live hardware: 1 CPU, 3 GPUs, 1 NPU.
- **Exp081: K-mer GPU Histogram** — `to_histogram()` (flat 4^k buffer) and
  `to_sorted_pairs()` for GPU radix sort dispatch. **kmer: B → A**.
- **Exp082: UniFrac CSR Flat Tree** — `FlatTree` struct with CSR layout +
  `to_sample_matrix()` for GPU pairwise dispatch. **unifrac: B → A**.
- **Exp083: Taxonomy NPU Int8** — affine int8 quantization of Naive Bayes weights.
  Argmax parity with f64 confirmed. **taxonomy: B → A/NPU**.
- **Handoff v7** submitted: 3 new absorption candidates, ODE flat validation results,
  dispatch routing findings, and evolution recommendations.

### Phase 22: Current — Pure GPU Streaming + Full Validation Proof

Proving that the complete bioinformatics pipeline runs on pure GPU with
ToadStool's unidirectional streaming — zero CPU round-trips between stages:

- **Exp085: BarraCUDA CPU v7** — Tier A data layout fidelity (43/43 PASS).
  kmer histogram, UniFrac CSR flat tree, taxonomy int8 round-trips proven lossless.
- **Exp086: metalForge Pipeline Proof** — 5-stage end-to-end pipeline with
  dispatch routing (45/45 PASS). Substrate-independent output proven.
- **Exp087: GPU Extended Domains** — EIC, PCoA, Kriging, Rarefaction added to
  GPU validation suite. 16 domains now GPU-validated.
- **Exp088: PCIe Direct Transfer** — GPU→NPU, NPU→GPU, GPU→GPU data flow
  without CPU staging (32/32 PASS). Buffer layout contracts validated.
- **Exp089: ToadStool Streaming Dispatch** — Unidirectional streaming parity
  vs round-trip for 3-stage and 5-stage chains (25/25 PASS).
- **Exp090: Pure GPU Streaming Pipeline** — Full pipeline on GPU with zero
  CPU round-trips (80/80 PASS). Streaming is 441-837× faster than round-trip.
- **Exp091: Streaming vs Round-Trip Benchmark** — Formal timing comparison:
  round-trip GPU is 13-16× slower than CPU; streaming eliminates 92-94% of
  that overhead.
- **Exp094: Cross-Spring Evolution Validation** — 39/39 checks validating 5
  neuralSpring-evolved primitives (PairwiseHamming, PairwiseJaccard,
  SpatialPayoff, BatchFitness, LocusVariance) now consumed by wetSpring.
- **Exp095: Cross-Spring Scaling Benchmark** — 7 benchmarks across 5
  neuralSpring primitives at realistic problem sizes (6.5×–277× GPU speedup).

**Tier A: 7 modules** | **Tier B: 1 remaining** | **728 tests** | **2,219+ checks** | **85 binaries**

### Phase 19: Absorption Engineering + Debt Resolution

Deep codebase evolution following hotSpring's absorption patterns:

- **`crate::special` extraction** — sovereign math (`erf`, `ln_gamma`,
  `regularized_gamma_lower`, `normal_cdf`) promoted from `bio::special` to
  top-level module, first step toward upstream `barracuda::math` feature
- **GPU workgroup constants** — all 9 `*_gpu.rs` modules use named
  `WORKGROUP_SIZE` constants linked to their WGSL shader counterparts
- **Hardware abstraction** — `HardwareInventory::from_content()` makes
  `/proc` reads injectable; `parse_peak_rss_mb()` for testable RSS parsing
- **Absorption batch APIs** — `snp::call_snps_batch`,
  `quality::filter_reads_flat` + `QualityGpuParams` with `#[repr(C)]`,
  `pangenome::analyze_batch` — closing PRIMITIVE_MAP absorption gaps
- **Zero-copy I/O** — `FastqRefRecord` for borrowed iteration, `DecodeBuffer`
  reuse in mzML, streaming iterators throughout
- **Determinism suite** — 16 bitwise-exact tests across non-stochastic
  algorithms using `f64::to_bits()`
- **Fuzz harnesses** — 4 `cargo-fuzz` targets (FASTQ, mzML, MS2, XML)
- **Doc strictness** — `-D missing_docs -D rustdoc::broken_intra_doc_links`
  passes on both default and `gpu` features
- **metalForge bridge** — `bridge.rs` connecting forge discovery to barracuda
  device creation (following hotSpring's forge↔barracuda pattern)
- **ABSORPTION_MANIFEST.md** — tracking absorbed/ready/local modules
  (following hotSpring's manifest pattern)
- **Coverage**: 96.21% overall (up from 55%), 728 tests (up from 650),
  39 named tolerances (up from 32)

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --pedantic --nursery` | Clean (0 warnings) |
| `cargo doc --no-deps` | Clean (0 warnings) |
| Line coverage (`cargo-llvm-cov`) | **96.21% overall** |
| `#![forbid(unsafe_code)]` | **Enforced crate-wide** |
| `#![deny(clippy::expect_used, unwrap_used)]` | **Enforced crate-wide** |
| TODO/FIXME markers | **0** |
| Inline tolerance literals | **0** (all use `tolerances::` constants) |
| SPDX-License-Identifier | All `.rs` files |
| Max file size | All under 1000 LOC |
| External C dependencies | **0** (`flate2` uses `rust_backend`) |
| Named tolerance constants | 39 (scientifically justified, hierarchy-tested) |
| Provenance headers | All 85 validation/benchmark binaries |

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

All 20 GPU modules delegate to ToadStool primitives (24 lean including 5 neuralSpring + 4 local WGSL shaders (ODE, kmer, unifrac, taxonomy)).
8 bio primitives absorbed by ToadStool and rewired on Feb 22, 2026:
`HmmBatchForwardF64`, `AniBatchF64`, `SnpCallingF64`, `DnDsBatchF64`,
`PangenomeClassifyGpu`, `QualityFilterGpu`, `Dada2EStepGpu`, `RfBatchInferenceGpu`.

### Local WGSL Shaders (4 in Write phase)

| Shader | Domain | Blocker |
|--------|--------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | ODE parameter sweep | Upstream uses `compile_shader` not `compile_shader_f64` (session 39) |

### I/O Modules

`io::fastq` (streaming FASTQ/gzip), `io::mzml` (streaming mzML/base64),
`io::ms2` (streaming MS2)

---

## Repository Structure

```
wetSpring/
├── README.md                      ← this file
├── BENCHMARK_RESULTS.md           ← three-tier benchmark results
├── CONTROL_EXPERIMENT_STATUS.md   ← experiment status tracker (95 experiments)
├── HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md  ← ToadStool handoff v6
├── barracuda/                     ← Rust crate (src/, Cargo.toml, rustfmt.toml)
│   ├── EVOLUTION_READINESS.md    ← absorption map (tiers, primitives, shaders)
│   ├── ABSORPTION_MANIFEST.md    ← what's absorbed, local, planned (hotSpring pattern)
│   ├── src/
│   │   ├── lib.rs               ← crate root (pedantic + nursery lints enforced)
│   │   ├── special.rs           ← sovereign math (erf, ln_gamma, regularized_gamma)
│   │   ├── tolerances.rs        ← 39 named tolerance constants
│   │   ├── validation.rs        ← hotSpring validation framework
│   │   ├── encoding.rs          ← sovereign base64 (zero dependencies)
│   │   ├── error.rs             ← error types (no external crates)
│   │   ├── bio/                 ← 41 CPU + 20 GPU bio modules
│   │   ├── io/                  ← streaming parsers (FASTQ, mzML, MS2, XML)
│   │   ├── bench/               ← benchmark harness + power monitoring
│   │   ├── bin/                 ← 85 validation/benchmark binaries
│   │   └── shaders/             ← 4 local WGSL shaders (ODE, kmer, unifrac, taxonomy; 8 absorbed by ToadStool)
│   └── rustfmt.toml             ← max_width = 100, edition = 2021
├── experiments/                   ← 95 experiment protocols + results
├── metalForge/                    ← hardware characterization + substrate routing
│   ├── forge/                    ← Rust crate: wetspring-forge (discovery + dispatch)
│   │   ├── src/                  ← substrate.rs, probe.rs, inventory.rs, dispatch.rs, bridge.rs
│   │   └── examples/             ← inventory discovery demo
│   ├── PRIMITIVE_MAP.md          ← Rust module ↔ ToadStool primitive mapping
│   ├── ABSORPTION_STRATEGY.md   ← Write → Absorb → Lean methodology + CPU math evolution
│   └── benchmarks/
│       └── CROSS_SYSTEM_STATUS.md ← algorithm × substrate matrix
├── wateringHole/                   ← spring-local handoffs (following hotSpring pattern)
│   └── handoffs/                  ← ToadStool rewire + cross-spring evolution docs
├── ../wateringHole/handoffs/      ← inter-primal ToadStool handoffs (shared)
├── archive/
│   └── handoffs/                  ← fossil record of ToadStool handoffs (v1–v5)
├── scripts/                       ← Python baselines (40 scripts)
├── specs/                         ← specifications and paper queue
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
cd barracuda

# Run all tests (728: 654 lib + 60 integration + 14 doc)
cargo test

# Code quality checks
cargo fmt -- --check
cargo clippy --lib -- -W clippy::pedantic -W clippy::nursery
cargo doc --no-deps

# Line coverage (requires cargo-llvm-cov)
cargo llvm-cov --lib --summary-only

# Run all CPU validation binaries (1,392+ checks)
for bin in $(ls src/bin/validate_*.rs | grep -v gpu | sed 's|src/bin/||;s|\.rs||'); do
    cargo run --release --bin "$bin"
done

# Run GPU validation (requires --features gpu, 533+ checks)
for bin in $(ls src/bin/validate_*gpu*.rs src/bin/validate_toadstool*.rs \
    src/bin/validate_cross*.rs 2>/dev/null | sed 's|src/bin/||;s|\.rs||' | sort -u); do
    cargo run --features gpu --release --bin "$bin"
done

# 25-domain Rust vs Python benchmark
cargo run --release --bin benchmark_23_domain_timing
python3 ../scripts/benchmark_rust_vs_python.py
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

- **hotSpring** — Nuclear/plasma physics validation (sibling Spring, 34 WGSL shaders, 637 tests)
- **neuralSpring** — ML/neural inference validation (sibling Spring, eigensolvers, batch IPR)
- **ToadStool** — GPU compute engine (BarraCUDA crate, 570+ WGSL shaders, shared primitives)
- **wateringHole** — Inter-primal standards and handoff documents
  - `handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md` — original shader detail handoff
  - `handoffs/WETSPRING_TOADSTOOL_REWIRE_FEB22_2026.md` — rewire results + bugs + cross-spring evolution
  - `handoffs/WETSPRING_TOADSTOOL_V7_FEB22_2026.md` — v7 handoff (dispatch, layouts, streaming)
  - `handoffs/WETSPRING_TOADSTOOL_V8_FEB22_2026.md` — v8 handoff (pure GPU streaming, absorption learnings)
- **Root handoff** — `HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md` — comprehensive v6
- **ecoPrimals** — Parent ecosystem
