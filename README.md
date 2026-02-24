# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — life science biome evolving Rust implementations
and GPU shaders for ToadStool/BarraCuda absorption. Follows the
**Write → Absorb → Lean** cycle adopted from hotSpring.

**Date:** February 23, 2026
**License:** AGPL-3.0-or-later
**Status:** Phase 38 — Extension papers: cold seep 299K QS genes, luxR phylogeny geometry, mechanical wave Anderson, wave-localization synthesis, burst statistics reinterpretation; 759 tests, 149 experiments, 3,028+ checks, 138 binaries

---

## What This Is

wetSpring validates the entire evolution path from interpreted-language
scientific computing (Python/numpy/scipy/sklearn) to sovereign Rust CPU
implementations, and then to GPU acceleration via ToadStool/BarraCuda:

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
ToadStool/BarraCuda is the fungus present in every biome. Springs don't
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
| **Lean** | 27 | GPU modules consuming upstream ToadStool primitives |
| **Compose** | 7 | GPU wrappers wiring ToadStool primitives (kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock) |
| **Passthrough** | 3 | Accept GPU buffers, CPU kernel (gbm, feature_table, signal) |
| **Write** | 5 | Local WGSL ODE shaders (phage_defense, bistable, multi_signal, cooperation, capacitor) |
| **NPU** | 1 | ESN reservoir computing → int8 quantization → NPU deployment (esn) |
| **Tier B** | 0 | All promoted |
| **Tier C** | 0 | All promoted |

### Local WGSL Shaders (5 — Write phase active)

| Shader | Vars | Params | CPU ↔ GPU |
|--------|------|--------|-----------|
| `phage_defense_ode_rk4_f64.wgsl` | 4 | 11 | Exact parity (Exp099) |
| `bistable_ode_rk4_f64.wgsl` | 5 | 21 | Exact parity (Exp100) |
| `multi_signal_ode_rk4_f64.wgsl` | 7 | 24 | Exact parity (Exp100) |
| `cooperation_ode_rk4_f64.wgsl` | 4 | 13 | Exact parity (Exp101) |
| `capacitor_ode_rk4_f64.wgsl` | 6 | 16 | Exact parity (Exp101) |

All use `compile_shader_f64()` with `fmax`/`fclamp`/`fpow` polyfills.
Absorption target: ToadStool `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`.

### metalForge Forge Crate (v0.3.0)

The `metalForge/forge/` crate discovers compute substrates at runtime and
routes workloads to the best capable device. v0.3.0 adds `workloads` module
with `ShaderOrigin` tracking (Absorbed/Local/CpuOnly) for absorption planning.
Absorption seam: when ToadStool absorbs forge, the bridge module becomes the
integration point.

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
| Validation checks (CPU) | 1,476 |
| Validation checks (GPU) | 702 |
| ODE CPU ↔ GPU parity | 82 (5 local WGSL ODE domains) |
| Dispatch validation checks | 80 |
| Layout fidelity checks | 35 |
| Transfer/streaming checks | 57 |
| metalForge v6 three-tier | 24 (25/25 papers complete) |
| Pure GPU streaming v2 | 72 (analytics + ODE + phylogenetics) |
| Cross-spring spectral theory | 25 (Anderson 1D/2D/3D + Almost-Mathieu + QS bridge) |
| NPU reservoir checks | 59 (ESN → int8 → NPU-simulated inference) |
| Cross-spring evolution checks | 9 (Exp120 benchmark) |
| NCBI-scale hypothesis testing | 146 (GPU-confirmed: Vibrio QS, 2D Anderson, pangenome, atlas) |
| 3D Anderson dimensional QS | 50 (GPU-confirmed: 1D→2D→3D sweep, vent chimney, phase diagram, biofilm 3D) |
| Geometry verification + cross-ecosystem | 50 (finite-size scaling, geometry zoo, cave/spring/rhizosphere, 28×5 atlas) |
| Why analysis: mapping, scaling, dilution, eukaryote | 35 (mapping sensitivity, square-cubed law, planktonic fluid 3D, cross-domain QS) |
| Extension papers: cold seep, phylogeny, waves | 36 (cold seep 299K catalog, geometry overlay, luxR phylogeny, mechanical waves, wave-localization, burst statistics) |
| **Total validation checks** | **3,028+** |
| Rust library unit tests | 680 lib + 79 integration/doc |
| **Total Rust tests** | **759** |
| Experiments completed | 149 |
| Validation/benchmark binaries | 128 validate + 10 benchmark = 138 total |
| CPU bio modules | 45 |
| GPU bio modules | 42 (27 lean + 5 write + 7 compose + 3 passthrough) |
| Tier B (needs refactor) | 0 (all promoted) |
| Python baselines | 41 scripts |
| BarraCuda CPU parity | 380/380 (v1-v8: 25 domains + 6 ODE flat + 13 promoted) |
| BarraCuda GPU parity | 29 domains (16 absorbed + 5 local ODE + 7 compose + 1 passthrough) |
| metalForge cross-system | 37 domains CPU↔GPU (Exp103+104), **25/25 papers three-tier** |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| Pure GPU streaming | 152 checks — analytics (Exp105), ODE+phylo (Exp106), 441-837× vs round-trip |
| ToadStool primitives consumed | **31** (absorbed, Lean — aligned to ToadStool S42) |
| Local WGSL shaders | **5** (Write phase — ODE domains pending absorption) |
All 3,028+ validation checks **PASS**. All 759 tests **PASS**.

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
BarraCuda CPU parity (18 domains).

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

### Phase 10: BarraCuda CPU Parity v4 (Track 1c)
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

### Phase 16: BarraCuda Evolution + Absorption Readiness
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
Evolving Rust implementations for ToadStool/BarraCuda team absorption:
- **`bio::special` consolidated** into shared module (erf, ln_gamma,
  `regularized_gamma_lower`) — shaped for extraction to `barracuda::math`
- **metalForge local** characterization: GPU/NPU/CPU substrate routing with
  absorption-ready Rust patterns (SoA, `#[repr(C)]`, batch APIs, flat arrays)
- **Exp064: BarraCuda GPU Parity v1** — consolidated GPU domain validation
  across 8 domains (diversity, BC, ANI, SNP, dN/dS, pangenome, RF, HMM).
  Pure GPU math matches CPU reference truth in a single binary
- **Exp065: metalForge Full Cross-System** — substrate-independence proof for
  full portfolio. CPU or GPU dispatch → same answer. Foundation for CPU/GPU/NPU
  routing in production
- **Absorption engineering**: Following hotSpring's pattern where Springs write
  extensions as proposals to ToadStool/BarraCuda, get absorbed, then lean on
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
- **Handoff v6** — comprehensive ToadStool/BarraCuda team handoff with all 9 shader
  binding layouts, dispatch geometry, NVVM driver profile bug, CPU math extraction plan,
  and streaming pipeline findings. See `wateringHole/handoffs/` for current handoffs.

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
- **Coverage**: 96.21% overall (up from 55%), 740 tests (up from 650),
  39 named tolerances (up from 32)

### Phase 20: ToadStool Bio Rewire + Cross-Spring Evolution

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
- **0 local WGSL shaders** at end of Phase 20 (all absorbed by ToadStool S39-41)
- **32 ToadStool primitives consumed** at Phase 20 (recounted to 30 in Phase 27)
- Rewire handoff archived: `wateringHole/handoffs/archive/WETSPRING_TOADSTOOL_REWIRE_FEB22_2026.md`

### Phase 21: GPU/NPU Readiness + Dispatch Validation

Tier B → A promotion of 3 remaining non-ODE modules, ODE flat param validation,
and forge dispatch router proof:

- **Exp078: ODE GPU Sweep Readiness** — flat param APIs (`to_flat`/`from_flat`)
  for 5 ODE modules (qs_biofilm, bistable, multi_signal, phage_defense, cooperation)
- **Exp079: BarraCuda CPU v6** — 48/48 checks proving flat serialization preserves
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

### Phase 22: Pure GPU Streaming + Full Validation Proof

Proving that the complete bioinformatics pipeline runs on pure GPU with
ToadStool's unidirectional streaming — zero CPU round-trips between stages:

- **Exp085: BarraCuda CPU v7** — Tier A data layout fidelity (43/43 PASS).
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

### Phase 23: Structural Evolution — Flat Layouts, DRY Models, Zero-Clone APIs

Deep two-pass evolution of the barracuda codebase (Exp097):

- **ODE trajectory flattened** — `OdeResult.y: Vec<Vec<f64>>` → flat `Vec<f64>` with
  `n_vars`, `state_at()`, `states()`, `var_at()` accessors. Per-step `.clone()` eliminated
  via `extend_from_slice()`. Affects all 6 ODE modules + 6 validation binaries.
- **Gillespie trajectory flattened** — `Trajectory.states: Vec<Vec<i64>>` → flat `Vec<i64>`
  with `n_species`, `state_at()`, `states_iter()`. Same clone elimination pattern.
- **DADA2 error model unified** — `init_error_model`, `estimate_error_model`,
  `err_model_converged`, `base_to_idx`, and 5 constants shared from `dada2.rs` as
  `pub(crate)`. GPU module delegates instead of duplicating. Single source of truth.
- **UniFrac distance matrix condensed** — Returns `UnifracDistanceMatrix` with condensed
  upper-triangle `Vec<f64>` instead of `Vec<Vec<f64>>` N×N. Halves memory, aligns
  directly with `pcoa()` condensed input format.
- **Adapter trim zero-clone** — `trim_adapter_3prime` returns `Option<FastqRecord>` instead
  of `(FastqRecord, bool)`, eliminating the common-path clone.
- **PCoA coordinates flat** — `PcoaResult.coordinates: Vec<Vec<f64>>` → flat with accessors.
- **Capability-based ODE polyfill** — `dev.needs_f64_exp_log_workaround()` replaces
  hardcoded `true`.
- **Full audit clean** — Zero unsafe, zero TODO/FIXME, zero cross-primal coupling,
  zero `unimplemented!()`, zero production mocks.

728 tests pass. 48/48 CPU-GPU domain parity. 39/39 cross-spring evolution.

### Phase 24: Edition 2024 + Structural Audit

Deep quality audit and evolution (Rust edition 2024, MSRV 1.85):

- **Rust edition 2024** — migrated from 2021, all import/formatting rules applied
- **`forbid(unsafe_code)` → `deny(unsafe_code)`** — Rust 2024 makes `std::env::set_var`
  unsafe; `#[allow(unsafe_code)]` confined to test-only env-var manipulation with SAFETY docs
- **CI hardened** — `RUSTDOCFLAGS="-D warnings"`, `clippy -D pedantic -D nursery`,
  `cargo check --features json` added to workflow
- **`bio::special` shim removed** — migration to `crate::special` complete, zero consumers
- **New clippy lints resolved** — `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
- **Python baseline provenance** — all 34 scripts now carry `# Date:` headers (git creation date)
- **Coverage verified** — `cargo-llvm-cov` confirms bio+io modules avg ~97% line coverage;
  new tests for taxonomy classifier accessors (`taxon_priors`, `n_kmers_total`)
- **740 tests pass** (666 lib + 74 integration/doc). Zero clippy, fmt, doc warnings.

### Phase 27: Local WGSL ODE Write Phase + metalForge v4

Following hotSpring's pattern: writing local extensions for ToadStool to absorb.
Three new ODE WGSL shaders created for domains not covered by the existing
`BatchedOdeRK4F64` (4v/17p). Five new GPU wrappers bridge ToadStool and local code:

- **Exp099: CPU vs GPU Expanded** — `kmer_gpu` (wraps `KmerHistogramGpu`),
  `unifrac_gpu` (wraps `UniFracPropagateGpu`), `phage_defense_gpu` (local WGSL
  4v/11p, exact CPU ↔ GPU parity). metalForge GPU→CPU→GPU pipeline validated.
- **Exp100: metalForge Cross-Substrate v4** — 28/28 checks. All three local
  ODE shaders achieve exact CPU ↔ GPU parity. NPU-aware routing. GPU→GPU→CPU
  PCIe pipeline proven.
- **3 local WGSL shaders** pending ToadStool absorption as `BatchedOdeRK4Generic`
- **30 GPU modules total** (27 Lean + 3 Write)

**Tier B: 0 (all promoted Phase 28)** | **740 tests at time** | **2,284+ checks at time**

### Phase 28: Pure GPU Promotion Complete + CPU v8 + metalForge v5

All 13 remaining Tier B/C modules promoted to GPU-capable. Zero Tier B/C
modules remain. Pure GPU capability across all 42 bio GPU modules:

- **Exp101: Pure GPU Promotion** — 13 modules promoted (cooperation, capacitor,
  kmd, gbm, merge_pairs, signal, feature_table, robinson_foulds, derep,
  chimera, neighbor_joining, reconciliation, molecular_clock). 2 new WGSL
  shaders (cooperation 4v/13p, capacitor 6v/16p). 7 compose wrappers wiring
  ToadStool primitives. 3 passthrough wrappers accepting GPU buffers.
- **Exp102: BarraCuda CPU v8** — 175/175 checks validating pure Rust math for
  all 13 newly promoted domains. Analytical known-values, monotonicity,
  round-trip fidelity. Combined v1-v8: **380/380 across 31+ domains**.
- **Exp103: metalForge v5** — 29 domains validated substrate-independent.
  13 new GPU domains added to cross-system matrix. CPU↔GPU parity proven
  for all compose and write modules.
- **5 local WGSL shaders** — phage_defense, bistable, multi_signal, cooperation,
  capacitor (pending ToadStool absorption as `BatchedOdeRK4Generic<N,P>`)
- **42 GPU modules total** (27 Lean + 5 Write + 7 Compose + 3 Passthrough)

**Tier B: 0** | **Tier C: 0** | **740 tests at time** | **2,406+ checks at time**

### Phase 29: metalForge v6 — 25/25 Papers Three-Tier Complete

Closing the final three-tier matrix gaps. Every actionable paper now has CPU +
GPU + metalForge validation:

- **Exp104: metalForge v6** — 5 remaining gap domains exercised through metalForge
  routing (QS ODE, UniFrac, DADA2, K-mer, Felsenstein). 24/24 checks. 25 of 25
  actionable papers now carry full three-tier (CPU + GPU + metalForge) coverage.
- **3 new metalForge workloads** — `dada2`, `bootstrap`, `placement` registered
  in `workloads.rs`, bringing total to 28 workloads (22 absorbed, 5 local, 1 CPU-only).
- **37 metalForge domains** proven substrate-independent (Exp103+104).

**750 tests at time** | **2,430+ checks at time**

### Phase 30: Pure GPU Streaming v2 (Multi-Domain)

Expanding streaming coverage from taxonomy+diversity to 10+ domains across
analytics, ODE biology, and phylogenetics pipelines:

- **`GpuPipelineSession` expanded** — added pre-compiled `BrayCurtisF64`,
  `spectral_cosine_matrix` (GEMM + FMR norms), `stream_full_analytics`
  (taxonomy → diversity → Bray-Curtis chained, zero recompilation).
- **Exp105: Pure GPU Streaming v2** — 27/27 checks. Alpha diversity, Bray-Curtis,
  spectral cosine, full analytics pipeline — all through pre-warmed session.
- **Exp106: Streaming ODE + Phylogenetics** — 45/45 checks. 6 GPU primitives
  pre-warmed simultaneously (25 ms), each dispatched twice to prove zero
  shader recompilation: ODE sweep, phage defense, bistable, multi-signal,
  Felsenstein (1.3% GPU f64 relative error), UniFrac (exact leaf parity).
- Streaming now proven for: diversity, taxonomy, Bray-Curtis, spectral cosine,
  QS ODE, phage defense, bistable, multi-signal, Felsenstein, UniFrac.

**750 tests at time** | **2,502+ checks at time**

### Phase 31: PCoA Debt Resolution + Spectral Cross-Spring

Two exclusions from the metalForge/streaming coverage resolved:

- **PCoA naga bug resolved** — `BatchedEighGpu` shader compilation now passes
  with wgpu v22.1.0. `catch_unwind` guards removed from `validate_metalforge_full_v3`
  and `validate_cpu_vs_gpu_all_domains`; PCoA promoted to direct GPU validation.
  Naga "invalid function call" error was fixed upstream.
- **Exp107: Spectral Cross-Spring** — 25/25 checks. Bridges Kachkovskiy/Bourgain
  spectral theory (Anderson localization) to quorum-sensing domain. Exercises
  `barracuda::spectral` primitives from wetSpring: Anderson 1D/2D/3D Hamiltonians,
  Almost-Mathieu operator, Lanczos eigensolve, level statistics (⟨r⟩), Lyapunov
  exponents, and a QS-disorder analogy showing population heterogeneity localizes
  autoinducer signals.

**750 tests at time** | **2,527+ checks at time**

### Phase 32: NCBI-Scale GPU Extension (Exp108-113)

Six new experiments extending validated pipelines to NCBI-scale data:
- **Exp108**: Vibrio QS 1024-genome parameter landscape (GPU ODE sweep)
- **Exp109**: 128-taxon phylogenetic placement (NJ + Felsenstein)
- **Exp110**: 200-genome cross-ecosystem pangenome (ANI + dN/dS)
- **Exp111**: 2048-spectrum GPU spectral cosine (2.1M pairs in 105 ms)
- **Exp112**: Multi-ecosystem bloom surveillance (3 ecosystems, 1365 timepoints)
- **Exp113**: QS-disorder prediction from 8 ecosystem diversity profiles

**750 tests at time** | **2,605+ checks at time**

### Phase 33: NPU Reservoir Deployment (Exp114-119)

Six experiments deploying ESN reservoir computing models to BrainChip Akida
NPU via int8 quantization — closing the CPU → GPU → NPU pipeline:
- **Exp114**: QS phase classifier (biofilm/planktonic/intermediate) — 100% f64↔NPU agreement
- **Exp115**: Phylogenetic clade placement — 97.7% quantization fidelity
- **Exp116**: Genome binning (5 ecosystems) — int8 regularization effect
- **Exp117**: Spectral library pre-filter — 84% top-10 overlap, 2-stage pipeline
- **Exp118**: Bloom sentinel (coin-cell feasible, >1 year battery life)
- **Exp119**: QS-disorder regime classifier — physics ordering preserved through int8

### Phase 34: Cross-Spring Rewire + Evolution Benchmark (Exp120)

Complete rewiring to modern ToadStool S42 BarraCuda APIs:
- **16 bio imports modernized** from deep `ops::bio::module::Type` paths to crate-root re-exports
- **Cross-spring shader provenance documented**: 612 WGSL shaders traced to origin springs
- **Exp120**: Benchmarks diversity (wetSpring), QS ODE (hotSpring precision), ESN reservoir
  (hotSpring/neuralSpring → wetSpring NPU), with full provenance table and evolution timeline

**750 tests** | **2,673+ checks** | **109 binaries**

### Phase 35: NCBI-Scale Hypothesis Testing (Exp121-126)

GPU-confirmed results on real NCBI data (146 checks, all PASS):

- **Exp121** (14/14 GPU): Real Vibrio QS — all 200 assemblies converge to biofilm; real
  genomes cluster in biofilm-favoring parameter space unlike Exp108 synthetic grid
- **Exp122** (12/12 GPU): 2D Anderson — genuine extended plateau (8 points above midpoint
  for W>2) absent in 1D; bloom QS-active in 2D but suppressed in 1D; J_c ≈ 0.41
- **Exp123** (9/9): Temporal ESN bloom — stateful vs stateless comparison, coin-cell >1 year
- **Exp124** (10/10): NPU spectral triage — 100% recall at 20% pass rate, 3.7× speedup
- **Exp125** (11/11): Real Campylobacterota pangenome (158 NCBI assemblies, 4 ecosystems)
- **Exp126** (90/90 GPU): 28-biome global QS atlas — W monotonic with J, all biomes
  correctly placed in Anderson disorder-space

### Phase 36: 3D Anderson Dimensional QS Phase Diagram (Exp127-130)

GPU-confirmed 3D Anderson extension using hotSpring spectral primitives (50 checks, all PASS):

- **Exp127** (17/17 GPU): 1D→2D→3D dimensional sweep — plateau points: 1D=0, 2D=5, 3D=12;
  J_c(3D) ≈ 1.28 >> J_c(2D) ≈ 0.56; gut/vent/soil/ocean flip to QS-active in 3D
- **Exp128** (12/12 GPU): Vent chimney geometry — 3 of 4 zones QS-active in 3D but suppressed
  in 2D; 2D slab model misses 75% of chimney QS capability
- **Exp129** (12/12 GPU): 28-biome dimensional phase diagram — all 28 biomes QS-active in 3D,
  zero in 1D or 2D; 3D metal-insulator W_c ≈ 16.5 exceeds all natural biome disorder
- **Exp130** (9/9 GPU): Thick biofilm 3D extension — 3D block (8×8×6) has 4× wider plateau
  than 2D slab; J_c(3D) ≈ 1.25, just 6 layers of depth transform QS capability

**Novel contribution to hotSpring**: biological validation data for 3D Anderson spectral
theory — microbial ecology provides natural systems where dimensional phase transitions
have measurable consequences for collective behavior.

### Phase 36b: Geometry Verification + Cross-Ecosystem Atlas (Exp131-134)

Verification and ecosystem extension of Phase 36 findings (50 checks, all PASS):

- **Exp131** (11/11 GPU): Finite-size scaling — L=6,7,8,9,10 cubes confirm W_c converges
  to **16.53 at L=10**, almost exactly theoretical 16.5. L=8 results are RELIABLE.
- **Exp132** (11/11 GPU): Geometry zoo — block(12) > cube(11) > thin_film(7) > slab(5)
  = tube(5) > chain(0). Just 2 layers of depth add 40% more plateau than pure 2D.
- **Exp133** (17/17 GPU): Cave/hot spring/rhizosphere — 12 ecosystem zones modeled with
  physically appropriate geometries. Only 3D zones (sediments, soil pores) sustain QS.
  Stalactite films, cave walls, and mycorrhizal tubes are QS-suppressed.
- **Exp134** (11/11 GPU): 28-biome × 5-geometry atlas — block activates 28/28, thin film
  3/28 (lowest diversity only), all other geometries 0/28. True 3D is required.

### Phase 36c: Why Analysis — Mapping, Scaling, Dilution, Eukaryotes (Exp135-138)

Deep interrogation of the 100%/0% atlas split (35 checks, all PASS):

- **Exp135** (8/8 GPU): Mapping sensitivity — tested 9 α values (5–35). The 100%/0%
  split is NOT an artifact; it reflects Anderson's theorem (d≤2 all localize, d≥3
  genuine W_c≈16.5). Natural biomes J∈[0.73,0.99] always fall below 3D W_c.
  Low-diversity systems (monocultures, early colonizers) CAN do 2D QS.
- **Exp136** (6/6 GPU): Square-cubed law — interior fraction correlates r=0.53 with ⟨r⟩,
  but the dominant effect is TOPOLOGICAL (random walk recurrence in d≤2). A 5×5×5 cube
  (125 cells) beats a 30×30 sheet (900 cells). Qualitative, not quantitative.
- **Exp137** (10/10 GPU): Planktonic dilution — QS breaks at 75% occupancy. Free plankton
  (~0.1% occupancy) is QS-suppressed; particle-attached communities active. Matches
  marine biology literature. Biofilm temporal stages: early colonization 2D-active,
  climax community needs 3D.
- **Exp138** (11/11 GPU): Eukaryote scaling — bacteria, yeast, protists all QS-active in
  3D at W=13. Minimum colony: 64 cells (L=4). Tissue cells work via low diversity
  (W<3), not geometry. QS is cross-domain if 3D structure exists.

### Phase 38: Extension Papers — Cold Seep, Phylogeny, Mechanical Waves (Exp144-149)

Extending the Anderson-QS framework using 5 key papers from the literature review
(36 checks, all PASS):

- **Exp144** (8/8): Cold seep QS gene catalog — 299,355 QS genes across 170 metagenomes
  from Microbiome 2025. 34 QS types in 6 systems (AHL, AI-2, DSF, DPD, AIP, HAI).
  Deep-sea sediment = 3D → Anderson predicts high QS. 5,000× more data than Exp141.
- **Exp145** (5/5): Cold seep QS type vs geometry — signal molecule physics (diffusion,
  half-life, characteristic length) predicts AHL + AI-2 dominant (>50%). 34 QS types =
  frequency-division multiplexing in diverse 3D community.
- **Exp146** (5/5): luxR phylogeny × geometry overlay — 12 evolutionary clades. 3D_dense:
  100% retain luxR. 3D_dilute: 33% (inverted logic only). 2D_surface: 0%. Solo receptors
  (eavesdroppers) enriched in mixed-species habitats. Connects to cross-species signaling.
- **Exp147** (6/6): Mechanical wave Anderson — 4/6 bacterial communication modes subject to
  Anderson localization (chemical QS, mechanical, electromagnetic, membrane potential).
  Contact-dependent bypasses Anderson. Planktonic portfolio = zero channels.
- **Exp148** (6/6): QS wave × localization synthesis — combines Meyer et al. (PRE 2020)
  traveling wave model with Anderson framework. L_eff = min(L_QS, ξ). V. fischeri case:
  W=1.95, chemistry-limited. Soil biofilm: wave speed reduced to 22% of maximum.
- **Exp149** (6/6): Burst statistics reinterpretation — Jemielita et al. (SciRep 2019)
  findings ARE Anderson localization. "Localized QS" = localized state. "Synchronized QS"
  = extended state. Novel prediction: compute ⟨r⟩ from real cell coordinates.

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --pedantic --nursery` | Clean (0 warnings, lib) |
| `cargo doc --no-deps` | Clean (0 warnings) |
| Line coverage (`cargo-llvm-cov`) | **96.21% overall** |
| `#![deny(unsafe_code)]` | **Enforced crate-wide** (edition 2024; `allow` only in test env-var calls) |
| `#![deny(clippy::expect_used, unwrap_used)]` | **Enforced crate-wide** |
| TODO/FIXME markers | **0** |
| Inline tolerance literals | **0** (all use `tolerances::` constants) |
| SPDX-License-Identifier | All `.rs` files |
| Max file size | All under 1000 LOC |
| External C dependencies | **0** (`flate2` uses `rust_backend`) |
| Named tolerance constants | 53 (scientifically justified, hierarchy-tested) |
| Provenance headers | All 138 validation/benchmark binaries |

---

## Module Inventory

### CPU Bio Modules (45)

| Module | Algorithm | Validated Against |
|--------|-----------|-------------------|
| `adapter` | Adapter detection + 3' trimming | Trimmomatic/Cutadapt |
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
| `esn` | Echo State Network reservoir computing (NPU int8) | Pure Python ESN |
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
| `ncbi_data` | NCBI assembly data loading (Phase 35 experiments) | NCBI Entrez |
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
| `validation_helpers` | SILVA reference loading + streaming FASTA/TSV | SILVA 138.1 NR99 |

### GPU Modules (42)

**Lean (27)** — delegate to ToadStool primitives:
`ani_gpu`, `batch_fitness_gpu`, `dada2_gpu`, `diversity_gpu`, `dnds_gpu`,
`eic_gpu`, `gemm_cached`, `hamming_gpu`, `hmm_gpu`, `jaccard_gpu`, `kmer_gpu`, `kriging`,
`locus_variance_gpu`, `ode_sweep_gpu`, `pangenome_gpu`, `pcoa_gpu`, `quality_gpu`,
`rarefaction_gpu`, `random_forest_gpu`, `snp_gpu`, `spatial_payoff_gpu`,
`spectral_match_gpu`, `stats_gpu`, `streaming_gpu`, `taxonomy_gpu`, `unifrac_gpu`

**Compose (7)** — wire ToadStool primitives for GPU-accelerated workflows:
`kmd_gpu`, `merge_pairs_gpu`, `robinson_foulds_gpu`, `derep_gpu`,
`neighbor_joining_gpu`, `reconciliation_gpu`, `molecular_clock_gpu`

**Passthrough (3)** — accept GPU buffers, CPU kernel (pending ToadStool primitives):
`gbm_gpu`, `feature_table_gpu`, `signal_gpu`

**Write (5)** — compile local WGSL shaders (pending ToadStool absorption):
`bistable_gpu`, `multi_signal_gpu`, `phage_defense_gpu`, `cooperation_gpu`, `capacitor_gpu`

### I/O Modules

`io::fastq` (streaming FASTQ/gzip), `io::mzml` (streaming mzML/base64),
`io::ms2` (streaming MS2)

---

## Repository Structure

```
wetSpring/
├── README.md                      ← this file
├── BENCHMARK_RESULTS.md           ← three-tier benchmark results
├── CONTROL_EXPERIMENT_STATUS.md   ← experiment status tracker (149 experiments)
├── barracuda/                     ← Rust crate (src/, Cargo.toml, rustfmt.toml)
│   ├── EVOLUTION_READINESS.md    ← absorption map (tiers, primitives, shaders)
│   ├── ABSORPTION_MANIFEST.md    ← what's absorbed, local, planned (hotSpring pattern)
│   ├── src/
│   │   ├── lib.rs               ← crate root (pedantic + nursery lints enforced)
│   │   ├── special.rs           ← sovereign math (erf, ln_gamma, regularized_gamma)
│   │   ├── tolerances.rs        ← 53 named tolerance constants
│   │   ├── validation.rs        ← hotSpring validation framework
│   │   ├── ncbi.rs              ← NCBI Entrez helpers (API key, HTTP, E-search)
│   │   ├── encoding.rs          ← sovereign base64 (zero dependencies)
│   │   ├── error.rs             ← error types (no external crates)
│   │   ├── bio/                 ← 45 CPU + 42 GPU bio modules
│   │   ├── io/                  ← streaming parsers (FASTQ, mzML, MS2, XML)
│   │   ├── bench/               ← benchmark harness + power monitoring
│   │   ├── bin/                 ← 138 validation/benchmark binaries
│   │   └── shaders/             ← 5 local WGSL ODE shaders (Write phase)
│   └── rustfmt.toml             ← max_width = 100, edition = 2024
├── experiments/                   ← 149 experiment protocols + results
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
├── scripts/                       ← Python baselines (41 scripts)
├── specs/                         ← specifications and paper queue
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
cd barracuda

# Run all tests (759: 680 lib + 79 integration/doc)
cargo test

# Code quality checks
cargo fmt -- --check
cargo clippy --all-targets -- -D clippy::pedantic -D clippy::nursery
cargo doc --no-deps

# Line coverage (requires cargo-llvm-cov)
cargo llvm-cov --lib --summary-only

# Run all CPU validation binaries (1,476+ checks)
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

- **hotSpring** — Nuclear/plasma physics validation (sibling Spring, 35+ WGSL shaders absorbed)
- **neuralSpring** — ML/neural inference validation (sibling Spring, eigensolvers, batch IPR)
- **airSpring** — Precision agriculture / IoT validation (sibling Spring, Richards PDE, Kriging)
- **ToadStool** — GPU compute engine (BarraCuda crate, 612 WGSL shaders, shared primitives)
- **wateringHole** — Spring-local handoffs to ToadStool
  - `handoffs/WETSPRING_V022_EXTENSION_PAPERS_FEB23_2026.md` — **current** (Phase 38, extension papers, 149 experiments, 3,028+ checks)
  - `handoffs/WETSPRING_V021_WHY_ANALYSIS_FEB23_2026.md` — Phase 36c, why analysis, 138 experiments
  - `handoffs/WETSPRING_V020_3D_ANDERSON_DIMENSIONAL_QS_FEB23_2026.md` — Phase 36, 3D Anderson, 130 experiments
  - `handoffs/WETSPRING_V019_NCBI_HYPOTHESIS_TESTING_FEB23_2026.md` — Phase 35, NCBI-scale, 126 experiments
  - `handoffs/WETSPRING_V018_CROSS_SPRING_REWIRE_HANDOFF_FEB23_2026.md` — Phase 34, full rewire, 120 experiments
  - `handoffs/WETSPRING_TOADSTOOL_V17_NPU_RESERVOIR_FEB23_2026.md` — NPU reservoir, NCBI-scale, PCoA fix
  - `handoffs/WETSPRING_TOADSTOOL_V16_STREAMING_FEB23_2026.md` — streaming v2, metalForge v6
  - `handoffs/WETSPRING_TOADSTOOL_V15_ODE_GENERIC_FEB22_2026.md` — 5 ODE shaders → `BatchedOdeRK4Generic`
  - `handoffs/WETSPRING_TOADSTOOL_V14_FEB22_2026.md` — Write phase, ODE shaders, forge v0.3.0
  - `handoffs/WETSPRING_TOADSTOOL_V13_FEB22_2026.md` — edition 2024, structural evolution
  - `handoffs/archive/` — V7-V12, rewire (fossil record)
  - `CROSS_SPRING_SHADER_EVOLUTION.md` — 612 shader provenance (35 hot, 22 wet, 14 neural, 5 air)
- **ecoPrimals** — Parent ecosystem
