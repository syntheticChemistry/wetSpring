# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — life science biome validating Python baselines
against Rust implementations and GPU shaders via `barraCuda` (standalone math
primal). Follows the **Write → Absorb → Lean** cycle adopted from hotSpring.

**Date:** March 7, 2026
**License:** AGPL-3.0-or-later
**MSRV:** 1.87
**Status:** V98 — 1,046 lib tests + 200 forge tests + 100 doc tests, 293 experiments, 8,604+ validation checks, 296 binaries, standalone `barraCuda` v0.3.3 (`a898dee`, 694+ WGSL shaders, wgpu 28). V98 full chain: Paper Math v5 (52 papers, 32/32) → CPU v24 (67/67) → GPU v13 (25/25) → Streaming v11 (25/25) → metalForge v16 (24/24) = **173/173 PASS**. Zero local WGSL, zero unsafe code, 164 named tolerances, `cargo clippy -D warnings` **ZERO WARNINGS** (default + GPU), `cargo doc -D warnings` **ZERO WARNINGS**. Ecosystem: barraCuda `a898dee`, toadStool S130+ (`bfe7977b`), coralReef Phase 10 Iteration 10 (`d29a734`).

---

## What This Is

wetSpring validates the entire evolution path from interpreted-language
scientific computing (Python/numpy/scipy/sklearn) to sovereign Rust CPU
implementations, and then to GPU acceleration via `barraCuda`:

```
Python baseline → Rust CPU validation → GPU acceleration → metalForge cross-substrate
```

Six tracks cover the life science, environmental monitoring, and drug repurposing domains:

| Track | Domain | Papers | Key Algorithms |
|-------|--------|:------:|----------------|
| **Track 1** | Microbial Ecology (16S rRNA) | 10 | FASTQ QC, DADA2 denoising, chimera detection, taxonomy, UniFrac, diversity, ODE/stochastic models, game theory, phage defense |
| **Track 1b** | Comparative Genomics & Phylogenetics | 5 | Newick parsing, Robinson-Foulds, HMM, Smith-Waterman, Felsenstein pruning, bootstrap, placement, NJ tree construction, DTL reconciliation |
| **Track 1c** | Deep-Sea Metagenomics & Microbial Evolution | 6 | ANI, SNP calling, dN/dS, molecular clock, pangenomics, enrichment testing, rare biosphere diversity |
| **Track 2** | Analytical Chemistry (LC-MS, PFAS) | 4 | mzML parsing, EIC, peak detection, spectral matching, KMD, PFAS screening |
| **Track 3** | Drug Repurposing & Pharmacophenomics | 5 | NMF, pathway scoring, knowledge graph embedding, TransE, SparseGEMM |
| **Track 4** | No-Till Soil QS & Anderson Geometry | 9 | Anderson localization, Lanczos eigensolver, quorum sensing ODE, cooperation dynamics, pore-geometry mapping, tillage factorial design |

---

## Evolution Architecture: Write → Absorb → Lean

wetSpring follows hotSpring's proven absorption cycle. Springs are biomes;
`barraCuda` is the universal math primal and `toadStool` is the hardware
dispatch primal. Springs depend directly on `barraCuda` for math, and
`toadStool` orchestrates hardware routing. Springs don't import each other.

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
| **Lean** | 37 | GPU modules consuming upstream barraCuda primitives (V95: +tolerance_search_gpu, kmd_grouping_gpu, stats_extended_gpu) |
| **Compose** | 7 | GPU wrappers wiring barraCuda primitives (kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock) |
| **Passthrough** | 0 | All promoted — `gbm` and `feature_table` compose upstream, `signal` leans on `PeakDetectF64` (S62) |
| **Write → Lean** | 5 | ODE shaders fully lean — GPU modules use `generate_shader()` from `OdeSystem` traits (WGSL deleted) |
| **CPU Delegation** | 2 | `rk45_integrate` (adaptive ODE) + `gradient_1d` (numerical gradient) — V95 |
| **NPU** | 1 | ESN reservoir computing → int8 quantization → NPU deployment (esn) — bridge to ToadStool `esn_v2` |
| **Tier B** | 0 | All promoted |
| **Tier C** | 0 | All promoted |

### ODE Shader Lean (5 — Complete)

All 5 biological ODE systems now use `BatchedOdeRK4<S>::generate_shader()` from
ToadStool's generic framework. Local WGSL files deleted (30,424 bytes).

| System | Struct | Vars | Params | CPU Parity |
|--------|--------|:----:|:------:|:----------:|
| Phage Defense | `PhageDefenseOde` | 4 | 11 | Derivative-level exact |
| Bistable | `BistableOde` | 5 | 21 | Exact (0.00) |
| Multi-Signal | `MultiSignalOde` | 7 | 24 | Exact (4.44e-16) |
| Cooperation | `CooperationOde` | 4 | 13 | Exact (4.44e-16) |
| Capacitor | `CapacitorOde` | 6 | 16 | Exact (0.00) |

Upstream `integrate_cpu()` is 20–33% faster than local integrators.

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
| Validation checks (GPU) | 1,783 |
| ODE CPU ↔ GPU parity | 82 (5 ODE domains, lean on upstream `generate_shader()`) |
| Dispatch validation checks | 80 |
| Layout fidelity checks | 35 |
| Transfer/streaming checks | 57 |
| metalForge v6 three-tier | 24 (39/39 papers complete) |
| Pure GPU streaming v2 | 72 (analytics + ODE + phylogenetics) |
| Cross-spring spectral theory | 25 (Anderson 1D/2D/3D + Almost-Mathieu + QS bridge) |
| NPU reservoir checks | 59 (ESN → int8 → NPU-simulated inference) |
| Cross-spring evolution checks | 57 (Exp120 benchmark 9, Exp168 ~25, Exp260 validate_cross_spring_evolution_modern 23/23) |
| NCBI-scale hypothesis testing | 146 (GPU-confirmed: Vibrio QS, 2D Anderson, pangenome, atlas) |
| 3D Anderson dimensional QS | 50 (GPU-confirmed: 1D→2D→3D sweep, vent chimney, phase diagram, biofilm 3D) |
| Geometry verification + cross-ecosystem | 50 (finite-size scaling, geometry zoo, cave/spring/rhizosphere, 28×5 atlas) |
| Why analysis: mapping, scaling, dilution, eukaryote | 35 (mapping sensitivity, square-cubed law, planktonic fluid 3D, cross-domain QS) |
| Extension papers: cold seep, phylogeny, waves | 36 (cold seep 299K catalog, geometry overlay, luxR phylogeny, mechanical waves, wave-localization, burst statistics) |
| Phase 39: finite-size, correlated, comm, nitrifying, marine, myxo, dicty, drug repurposing | 104 (Exp150-161: scaling v2, correlated disorder, physical comm, nitrifying, interkingdom, Myxococcus, Dictyostelium, Fajgenbaum, MATRIX, NMF, repoDB, ROBOKOP) |
| Phase 42: Track 3 full-tier (CPU v9, GPU drug, metalForge drug) | 44 (Exp163-165: CPU v9 27, GPU drug 8, metalForge drug 9) |
| Phase 44: modern systems S62+DF64, diversity fusion extension | 37 (Exp166: 19, Exp167: 18) |
| Phase 45 V40: cross-spring S62+DF64 evolution | ~25 (Exp168: hotSpring precision → wetSpring bio → neuralSpring pop-gen → Track 3 GPU) |
| Nanopore pre-hardware (Exp196a-c) | 52 (signal bridge 28, simulated 16S 11, int8 quantization 13) |
| biomeOS IPC integration (Exp203-205) | 29 (science pipeline, capability discovery, sovereign fallback) |
| IPC dispatch CPU parity (Exp206) | 64 (7 domains, EXACT_F64 — zero numeric drift through IPC layer) |
| IPC dispatch GPU parity (Exp207) | 54 (6 domains, GPU↔CPU — lazy OnceLock + dispatch threshold) |
| metalForge v7 NUCLEUS (Exp208) | 75 (8 domains, mixed hardware — PCIe bypass, Tower/Node/Nest atomics) |
| V88 experiment buildout (Exp263-270) | 427 (CPU v20, CPU↔GPU v7, metalForge v12, NUCLEUS v3, ToadStool pure-math v3, CPU↔GPU pure-math, mixed-HW dispatch, biomeOS graph) |
| Exp271: Cross-Spring S79 (13 domains) | 73 |
| Exp272: Bio Brain (7 domains) | 64 |
| V98 full chain (Exp313-318) | 173 (Paper 32 + CPU 67 + GPU 25 + Streaming 25 + metalForge 24) |
| **Total validation checks** | **8,604+** |
| Rust library unit tests | 1,047 (barracuda) |
| metalForge forge tests | 200 |
| Doc-tests | 27 (barracuda 18 + forge 9) |
| **Total Rust tests** | **1,347** (lib + forge + integration + doc) |
| Library code coverage | **95.86% line / 93.54% fn / 94.99% branch** (cargo-llvm-cov) |
| Experiments completed | 293 |
| Validation/benchmark binaries | 296 |
| CPU bio modules | 47 |
| GPU bio modules | 45 (30 lean + 5 write→lean + 7 compose + 0 passthrough) |
| Tier B (needs refactor) | 0 (all promoted) |
| Python baselines | 57 scripts (all with reproduction headers + SHA-256 integrity verification) |
| BarraCuda CPU parity | 546/546 (v1-v11: 36+ domains, IPC fidelity proven) |
| BarraCuda GPU parity | 36+ domains (17 absorbed + 5 local ODE + 7 compose + IPC GPU-aware dispatch) |
| metalForge cross-system | 37+ domains CPU↔GPU (Exp103+104+165+182+208), **39/39 papers three-tier** |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| Pure GPU streaming | 152 checks — analytics (Exp105), ODE+phylo (Exp106), 441-837× vs round-trip |
| `barraCuda` primitives consumed | **150+** (always-on, zero fallback code — standalone `barraCuda` v0.3.3 `a898dee`, wgpu 28, PrecisionRoutingAdvice) |
| Local WGSL shaders | **0** (diversity fusion absorbed S63 — fully lean) |
All 8,604+ validation checks **PASS**. All 1,047 library + 200 forge tests **PASS** (1 ignored: hardware-dependent).

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
| Overall speedup | **33.4×** (Rust 51ms vs Python 1,713ms) |
| Peak speedup | 625× (Smith-Waterman) |
| ODE domains | 15–56× |
| Track 1c domains | 6–56× |
| ML ensembles (RF + GBM) | ~30× |

---

## Evolution Path

### Phases 1–39: Foundation → Full Validation (condensed)

| Phase | Key Milestone | Checks Added |
|-------|---------------|:------------:|
| 1–6 | Foundation: Python → Rust CPU → GPU → BarraCuda parity | ~200 |
| 7 | ToadStool Bio Absorption (4 primitives) | 10 |
| 8 | GPU Composition + Write → Absorb → Lean | ~40 |
| 9 | Track 1c Deep-Sea Metagenomics (5 modules) | 133 |
| 10 | BarraCuda CPU Parity v4 (Track 1c) | 44 |
| 11 | GPU Track 1c Promotion (4 WGSL shaders) | 27 |
| 12 | 25-Domain Benchmark + metalForge | 20 |
| 13 | ML Ensembles (RF, GBM) | 29 |
| 14 | Evolution Readiness (9 shader handoff) | — |
| 15 | Code Quality Hardening (pedantic, tolerances) | — |
| 16–17 | BarraCuda Evolution + metalForge Absorption | 74 |
| 18 | Streaming Dispatch + Cross-Substrate | 152 |
| 19 | Absorption Engineering + Debt Resolution | — |
| 20 | ToadStool Bio Rewire (8 shaders absorbed) | — |
| 21 | GPU/NPU Readiness + Dispatch Validation | 83 |
| 22 | Pure GPU Streaming + Full Validation | 274 |
| 23–24 | Structural Evolution + Edition 2024 | — |
| 27–28 | Local WGSL ODE + Pure GPU Promotion | 403 |
| 29–31 | metalForge v6 + Streaming v2 + Spectral Cross-Spring | 96 |
| 32–34 | NCBI-Scale GPU + NPU Reservoir + Cross-Spring Rewire | 245 |
| 35–36c | NCBI Hypothesis + 3D Anderson + Geometry + Why | 221 |
| 38–39 | Extension Papers + Finite-Size + Drug Repurposing | 140 |

*Detailed history in CHANGELOG.md.*

### Phase 44: Write Phase Extensions (hotSpring Pattern)
- First local WGSL extension: `diversity_fusion_f64.wgsl` — fused Shannon + Simpson + evenness
- Follows hotSpring's absorption pattern: WGSL + CPU reference + binding layout docs + validation
- `DiversityFusionGpu` module with documented binding layout and dispatch geometry
- Forge workloads: 5 ODE systems reclassified from `Local` to `Absorbed` (trait-generated WGSL)
- Forge `diversity_fusion` workload: `ShaderOrigin::Absorbed` (S63 lean, V48)
- Exp167: 18/18 GPU ↔ CPU parity checks PASS (Simpson exact, Shannon within log_f64 polyfill tol)
- metalForge docs updated: PRIMITIVE_MAP.md, ABSORPTION_STRATEGY.md aligned to S65
- Root docs, whitePaper, ABSORPTION_MANIFEST.md, EVOLUTION_READINESS.md cleaned and synchronized

### Phase 45: Deep Debt Resolution + Idiomatic Modernization (V38)

Comprehensive code quality evolution targeting tolerance centralization, I/O
performance, provenance completeness, and modern Rust idioms:

- **Tolerance centralization**: ~35 binaries migrated from ad-hoc `0.0`, `1e-10`,
  `0.001` literals to named `tolerances::` constants. 200+ individual replacements.
  3 new tolerance constants added: `GPU_LOG_POLYFILL` (1e-7), `ODE_NEAR_ZERO_RELATIVE`
  (1.5), `EXACT` used consistently across all validation binaries.
- **MS2 parser evolution**: `Ms2Iter` and `stats_from_file` migrated from
  `reader.lines()` (per-line allocation) to `read_line()` with reusable `String`
  buffer — zero per-line allocation.
- **Streaming I/O deepened**: `stream_taxonomy_tsv`, `stream_fasta_subsampled`
  (validation_helpers.rs), and `spawn_nvidia_smi_poller` (bench/power.rs) all
  migrated from `reader.lines()` to reusable buffer pattern.
- **Provenance hardened**: 25 binaries updated from placeholder commits to `1f9f80e`.
  13 binaries had `| Command |` rows added. 2 binaries reformatted to standard
  provenance table. All 172 binaries now carry complete provenance.
- **`#[must_use]` annotations**: Added to 7 public API functions across I/O and NCBI
  modules — `parse_ms2`, `stats_from_file` (×3), `parse_mzml`, `parse_fastq`,
  `http_get`, `esearch_count`.
- **`Vec::with_capacity`**: Added pre-allocation in 5 library files where sizes are
  known (xml.rs attributes, eic.rs bins, spectral_match.rs indices, mzml/decode.rs
  spectrum arrays, validation_helpers.rs FASTA refs).
- **Clippy strictness**: `too_many_lines` suppressed only on 2 long validation binary
  functions. Zero warnings with `-D warnings -W pedantic -W nursery`.
- **Upstream request**: `barracuda::math::{dot, l2_norm}` documented as extraction
  candidates in EVOLUTION_READINESS.md.

**833 lib tests** | **96.67% coverage** | **79 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 46: Deep Audit + Coverage + Idiomatic Evolution (V41)

Comprehensive codebase audit and targeted test expansion:

- **`missing_docs` escalated** from `warn` to `deny` — every public item now documented
- **24 new library tests** targeting weak-coverage modules:
  bench/power (59% → 71%), bench/hardware (77% → 81%), ncbi (78% → 80%),
  io/fastq (84% → 87%)
- **33 scattered `clippy::cast_precision_loss` annotations** consolidated to
  function-level annotations in 4 validation binaries
- **`ncbi_data.rs` refactored** — consolidated path discovery via `validation::data_dir`,
  extracted JSON parsing helpers, 12 new tests (68% → 95% coverage)
- **9 missing tolerance constants** added to `all_tolerances_are_non_negative` test
- **Provenance headers** added to 20 validation binaries previously missing them
- **Inline dot-product** replaced with `special::dot`/`special::l2_norm` in 2 binaries
- **Dependency audit**: 3 direct deps (all pure Rust), 0 C dependencies, 0 HTTP/TLS crates
- **All files under 1000 LOC** (max: 924 in a validation binary)

**833 lib tests** | **96.67% coverage** | **79 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 47: Deep Debt Evolution Round 2 (V42)

Continuing the deep debt resolution from Phase 46 with a focus on testability
refactoring, tolerance completeness, and barracuda team handoff:

- **`ncbi.rs` testability refactoring**: Extracted pure-logic functions from
  env-dependent wrappers — `api_key_from_paths()`, `select_backend()`,
  `resolve_cache_path()` — making all branches testable without `unsafe`
  env-var manipulation. 16 new tests. Coverage: **86.39% → 93.38%** (target 90% met).
- **Remaining bare tolerances eliminated**: `v.check(..., 0.0)` in
  `validate_metalforge_v5`, `validate_cpu_vs_gpu_all_domains`, and
  `benchmark_phylo_hmm_gpu` replaced with `tolerances::EXACT` (14 sites).
  Hardcoded `1e-3` for HMM batch parity replaced with new
  `tolerances::GPU_VS_CPU_HMM_BATCH`.
- **`ncbi_data.rs` smart refactoring** (from Phase 46): monolithic 724-line
  file split into `bio/ncbi_data/{mod,vibrio,campy,biome}.rs` submodule with
  shared JSON helpers. Each domain struct in its own file.
- **`validation_helpers.rs`**: Hardcoded SILVA filenames extracted to
  module-level `SILVA_FASTA` / `SILVA_TAX_TSV` constants.
- **`special::dot` / `special::l2_norm`**: Confirmed as correct local helpers —
  barracuda's `dotproduct` is a GPU Tensor op, not a CPU f64 slice operation.
- **79 named tolerance constants** (was 70 at V41, 77 at V54).
- **Overall coverage: 96.67%** (was 96.48% at V41).

**833 lib tests** | **96.67% coverage** | **79 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 61: Field Genomics — Nanopore Signal Bridge + Pre-Hardware Validation (V61)

`io::nanopore` module operational — sovereign POD5/NRS file parsing, synthetic
read generation from community profiles, and int8 quantization pipeline for
NPU-ready field classification. Three pre-hardware experiments validate the
complete software path before MinION hardware arrives:

- **Exp196a: Nanopore Signal Bridge** — POD5 signal structure parsing, NRS
  synthetic read format, streaming iterator API. 28/28 checks: header parsing,
  signal extraction, quality metrics, read-to-FASTQ conversion, batch iteration
- **Exp196b: Simulated 16S Pipeline** — synthetic nanopore reads through
  sovereign 16S pipeline (DADA2 → chimera → taxonomy → diversity). 11/11 checks:
  ASV recovery from noisy long reads, community reconstruction, Anderson regime
  detection from MinION-length sequences
- **Exp196c: NPU Quantization Pipeline** — community profile → int8 → ESN
  classifier → bloom/healthy/AMR classification. 13/13 checks: f64→int8 fidelity,
  regime classification agreement, NPU inference latency estimation
- **6 new named tolerances** — `NANOPORE_SIGNAL_SNR`, `BASECALL_ACCURACY`,
  `LONG_READ_OVERLAP`, `NPU_INT8_COMMUNITY`, `NANOPORE_DIVERSITY_VS_ILLUMINA`,
  `FIELD_ANDERSON_REGIME`
- **3 new validation binaries** — `validate_nanopore_signal_bridge`,
  `validate_nanopore_simulated_16s`, `validate_nanopore_int8_quantization`

**896 lib tests** | **96.67% coverage** | **92 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 62: biomeOS IPC Integration + Comprehensive Green Sweep (V62)

Full-stack biomeOS integration and comprehensive validation sweep proving the
complete Python → CPU → GPU → Pure GPU Streaming → metalForge pipeline:

- **biomeOS science primal** — IPC server (`wetspring_server`) with JSON-RPC 2.0 over
  Unix socket, Songbird registration, Neural API metrics, capability-routed science
  methods: `science.diversity`, `science.qs_model`, `science.anderson`, `science.ncbi_fetch`,
  `science.full_pipeline`
- **GPU-aware IPC dispatch** — lazy `OnceLock<GpuF64>` initialization, dispatch threshold
  routing (small workloads stay on CPU, large workloads route to GPU), `handle_anderson()`
  performs actual Lanczos spectral analysis when GPU feature enabled
- **Exp203-205: biomeOS integration** — 29 checks validating server lifecycle, dispatch
  correctness, capability discovery, and sovereign fallback
- **Exp206: CPU v11** — 64/64 IPC dispatch math fidelity (EXACT_F64: zero numeric drift)
- **Exp207: GPU v4** — 54/54 IPC science on GPU (GPU↔CPU parity, pre-warmed dispatch)
- **Exp208: metalForge v7** — 75/75 NUCLEUS atomics (PCIe bypass topology, cross-substrate)
- **Comprehensive sweep** — 28 validation binaries re-run: all PASS. Python→Rust 33.4× speedup.
  GPU streaming 441-837× vs round-trip. 39/39 papers three-tier. clippy CLEAN.
- **Fixes:** Exp185 cold seep (stochastic seed determinism), Exp189 S68 erf tolerance

**977 lib tests** | **1,103 total** | **5,061+ checks** | **211 experiments** | **0 clippy warnings**

### Phase 88: Full Experiment Buildout + Control Validation (V88)

Eight new experiments validating every layer of the compute stack — from pure
barracuda math through GPU parity to mixed-hardware NUCLEUS dispatch:

- **Exp263: BarraCuda CPU v20** — 37/37 checks. Vault DF64, cross-domain pure Rust math, 20th CPU parity round.
- **Exp264: CPU vs GPU v7** — 22/22 checks. 27-domain GPU parity proof (G17-G21 expansion).
- **Exp265: metalForge v12** — 63/63 checks. Extended cross-system dispatch with bandwidth-aware routing.
- **Exp266: NUCLEUS v3** — 106/106 checks. Tower→Node→Nest + Vault + biomeOS full lifecycle.
- **Exp267: ToadStool Dispatch v3** — 41/41 checks. Pure Rust math across 6 `barracuda` domains (stats, linalg, special, numerical, spectral). Proves every ToadStool abstraction preserves math from analytical formulae through CPU compute.
- **Exp268: CPU vs GPU Pure Math** — 38/38 checks. Deepest GPU parity layer: `FusedMapReduceF64`, `BrayCurtisF64`, `BatchedEighGpu`, graph Laplacian, DF64 pack/unpack, `GpuPipelineSession` streaming determinism.
- **Exp269: Mixed Hardware Dispatch** — 91/91 checks. NUCLEUS atomics + PCIe bypass: Tower discovery (3 GPUs), NPU→GPU bypass (0 CPU round-trips), GPU→CPU fallback, bandwidth-aware multi-GPU routing, 8-stage mixed pipeline, full 47-workload catalog dispatch (45 GPU + 2 CPU-only).
- **Exp270: biomeOS Graph Coordination** — 29/29 checks. Full biomeOS layer: socket discovery, primal orchestration, Nest sovereign fallback, 3 pipeline topologies (GPU-only, mixed, CPU-only), sovereign mode (zero external dependencies).
- **API deep dive**: `barracuda::stats::FitResult` params array (not named fields), `Result<f64>` return types, `spectral` chain (`anderson_3d` → `lanczos` → `lanczos_eigenvalues`), `graph_laplacian` flat-array API — all documented in handoff for upstream team.
- **CPU-only workload routing**: `dispatch::route` correctly handles `ShaderOrigin::CpuOnly` workloads (excluded from GPU routing, not failures).

**1,249 tests** | **270 experiments** | **253 binaries** | **427 new checks**

### Phase 89: ToadStool S79 Deep Rewire (V89)
- S71→S79 pin update (9 commits)
- `MultiHeadBioEsn` wrapper for ToadStool `MultiHeadEsn`
- IPC `SpectralAnalysis` rewire
- Exp271: Cross-Spring S79 Validation (73/73 checks, 13 domains)

### Phase 90: Bio Brain Cross-Spring Ingest (V90)
- hotSpring 4-layer brain + 36-head Gen2 ESN adapted to bio sentinel
- `BioNautilusBrain` from bingocube-nautilus
- `BioBrain` adapter: attention state machine, observation history
- 3 new IPC methods: brain.observe, brain.attention, brain.urgency
- Exp272: Bio Brain Validation (64/64 checks, 7 domains)

### Phase 91: Deep Debt Resolution + Idiomatic Modernization (V91)
- Capability-based discovery unified into `ipc::discover`
- Handler refactoring: monolithic 605-line file → 3 domain-focused modules
- `#[must_use]` on gillespie, pcoa
- `as` casts replaced with `From`/`TryFrom`
- 5 new brain handler dispatch tests

### Phase 92: Immunological Anderson + Gonzales Reproducibility (V92/V92B)
- Immunological Anderson extension
- Gonzales reproducibility validation

### Phase 92C: Deep Audit & Evolution
- 32 GPU modules received API test stubs
- 284 validators classified with provenance headers (70 Analytical, 59 GPU-parity, 53 Python-parity, 35 Pipeline, 20 Cross-spring, 12 Synthetic)
- 20+ binaries: inline tolerance literals → `tolerances::` constants
- 3 new diversity tolerance constants
- 14 new tests (power.rs, nrs.rs, brain/observation.rs)
- All doc_markdown clippy warnings fixed across 30+ files

### V98+: Upstream Rewire + Cross-Spring Evolution (Exp313–320) — current

Complete V98 validation chain plus upstream rewire to modern barraCuda/toadStool/coralReef
and cross-spring evolution validation exercising all 5 springs' contributions.

| Exp | Name | Type | Checks | Time |
|-----|------|------|:------:|-----:|
| 313 | Paper Math Control v5 — All 52 Papers | Paper | 32/32 | 0.4ms |
| 314 | BarraCuda CPU v24 — Comprehensive Bio Domain Parity | CPU | 67/67 | 2.8ms |
| 316 | BarraCuda GPU v13 — Full-Domain GPU Portability | GPU | 25/25 | 20.5ms |
| 317 | Pure GPU Streaming v11 — End-to-End Pipeline | Streaming | 25/25 | 14.9ms |
| 318 | metalForge v16 — Cross-System Paper Math | metalForge | 24/24 | 0.5ms |
| 319 | Cross-Spring Evolution V98+ — All 5 Springs | Validation | 52/52 | ~1s |
| 320 | Cross-Spring Evolution V98+ Benchmark | Benchmark | 24 ops | ~1.4s |

**225/225 validation checks PASS. 24 primitives benchmarked.** V98+ additions:
- Upstream rewire: barraCuda `a898dee`, toadStool S130+ `bfe7977b`, coralReef Iter 10 `d29a734`
- Cross-spring provenance: 28 shaders, 22 cross-spring, 5 wetSpring-authored
- All 5 springs exercised: hotSpring (DF64, Anderson, erf), wetSpring (diversity, HMM,
  Felsenstein, NMF), neuralSpring (graph Laplacian, Pearson), airSpring (6 ET₀),
  groundSpring (bootstrap, jackknife, regression)
- GPU: FusedMapReduceF64 Shannon/Simpson/BC on RTX 4070 Hybrid (DF64)
- Evolution tracking: shader origin→consumer flow across springs

**1,047 tests** | **295 experiments** | **298 binaries** | **8,656+ checks**

### Phase 95: Standalone barraCuda Rewire + Deep Debt Evolution

Rewired from ToadStool-embedded `barracuda` v0.2.0 to standalone `barraCuda`
v0.3.1. Comprehensive deep debt resolution across the full codebase:

- **barraCuda rewire**: Dependency path swap from `phase1/toadstool/crates/barracuda`
  to standalone `barraCuda/crates/barracuda` (v0.3.1). Zero API breakage — 1,044 tests
  pass without code changes. akida-driver (NPU) kept at toadStool path (independent).
- **MSRV bump**: 1.85 → 1.87 (matching standalone barraCuda requirement)
- **Tolerance centralization**: 4 production inline constants → named constants
  (`RIDGE_NAUTILUS_DEFAULT`, `BOX_MULLER_U1_FLOOR_SYNTHETIC`, `LOG_PROB_FLOOR`,
  `ANALYTICAL_LOOSE`). 60+ test tolerances across 18 files → `tolerances::` references.
- **Hardcoding evolution**: NCBI URLs → env-var configurable (`WETSPRING_NCBI_*`).
  Data dir → XDG-compliant (`XDG_DATA_HOME/wetspring` → `~/.local/share/wetspring`).
- **Modern Rust idioms**: 16 `is_multiple_of()` conversions (Rust 1.87+), 12 `const fn`
  promotions, 27 `mul_add` fused operations, 4 redundant clones removed.
- **Smart file refactoring**: `validation.rs` (668→297), `io/xml.rs` (612→285),
  `bio/dnds.rs` (583→313) — tests extracted to separate files.
- **Clippy nursery clean**: Zero warnings at both `pedantic` and `nursery` levels.
- **Doc updates**: ToadStool references → barraCuda (standalone math primal) in
  architectural docs (`gpu.rs`, `lib.rs`, `npu.rs`, `Cargo.toml`).

**1,054 tests** | **164 named tolerances** | **0 clippy warnings (pedantic + nursery)** | **0 doc warnings**

#### Deep Debt Round 3 (March 4, 2026)

Continued deep debt evolution targeting remaining audit items:

- **Tolerance migration completed**: Last inline literal (`1e-5` in `validate_repodb_nmf`)
  replaced with `tolerances::NMF_CONVERGENCE_RANK_SEARCH`. ~82 total inline literals
  migrated across 16 binaries. 4 new constants: `LIMIT_CONVERGENCE`, `VARIANCE_EXACT`,
  `NMF_SPARSITY_THRESHOLD`, `NMF_CONVERGENCE_RANK_SEARCH`.
- **Test extraction**: 6 large library files had `#[cfg(test)]` blocks extracted to
  separate `*_tests.rs` files: `bench/power.rs`, `bench/hardware.rs`, `bio/gbm.rs`,
  `bio/merge_pairs.rs`, `bio/felsenstein.rs`, `metalForge/forge/ncbi.rs`.
- **Provenance completed**: 30 binaries received `# Provenance` tables. All 284
  binaries now carry complete provenance documentation.
- **Validation coverage**: 10 new unit tests for `validation/` module (data directory
  discovery, timing helpers, `Validator` edge cases). 9 doc-tests added to forge public API.
- **`unreachable!()` eliminated**: `kmer.rs` match-with-unreachable → const lookup table.
- **Hardcoded paths evolved**: `/tmp/` in forge `data.rs` and test code → `std::env::temp_dir()`.
- **Sovereignty**: `pcoa.rs` CPU Jacobi absorption path documented (Write → Absorb → Lean).
- **Doc check fixed**: `bench` intra-doc link ambiguity resolved (was sole doc warning).
- **Audits completed**: `clone()` calls (all Arc or necessary), `read_to_string` (all
  on small files), external deps (all pure Rust). No action needed.

### Phase 92J: Deep Debt Resolution + Pedantic Evolution
- `panic!` in ESN bridge → `Result`-based error handling (zero panics in library code)
- IPC science handler: 8-arg function → `MetricCtx` struct pattern
- 50+ binaries: clippy pedantic fully clean (`--all-features -W clippy::pedantic`)
- Modern idioms: `mul_add`, `f64::from`, `f64::midpoint`, inlined format args, `if let`
- Shared `validation::bench()` helper extracted
- Provenance documentation: `experiments/results/README.md`, BASELINE_MANIFEST clarification
- Dependency health: CPU-only build is pure Rust; only C dep is `renderdoc-sys` via wgpu (GPU path)

**1,044 tests** | **281 experiments** | **268 binaries** | **8,300+ checks**

### Phase 87: blueFish WhitePaper + hotSpring Brain Architecture Review (V87)

Documentation and architectural evolution:

- **blueFish whitePaper launched** — 7 documents at `whitePaper/blueFish/` establishing chemistry as an irreducible research programme within ecoPrimals. Non-reducibility argument (Fodor, Lakatos, Anderson). Two-arm architecture (analytical + computational). Isomorphism proof: 29 comp-chem operations decomposed into BarraCUDA primitives (14 direct, 9 compose, 6 genuinely new). RootPulse provenance integration with 5 use cases. Community engagement document mapping Reddit pain points to ecoPrimals solutions.
- **hotSpring brain architecture reviewed** — 4-layer concurrent brain (NPU+GPU+CPU+GPU), Gen 2 36-head ESN with `HeadGroupDisagreement`, NautilusBrain evolutionary reservoir computing. Mapped to wetSpring bio workloads: `DiversityUpdate`, bio head groups, chemistry active learning.
- **Phase 2 primal requirements specified** — Concrete demands on LoamSpine (100+ entries/sec, Merkle proofs), SweetGrass (chemistry Entity/Braid types), rhizoCrypt (multi-agent DAG), BearDog (signing), NestGate (content-addressed chemistry data).
- **V87 handoff** — ToadStool/BarraCUDA team handoff with absorption targets: brain architecture generalization, ERI shader class planning, `esn_v2` shape bug.
- **Root doc cleanup** — removed references to phantom `BENCHMARK_RESULTS.md` and `CONTROL_EXPERIMENT_STATUS.md`, corrected binary counts.

**1,247 tests** | **262 experiments** | **238 binaries**

### Phase 86: Cross-Spring Evolution + Deep Debt Elimination (V86)

Three rounds of deep evolution work:

- **Exp260: Cross-Spring Evolution Validation** — `validate_cross_spring_evolution_modern` (23/23 checks). Validates all five Springs' primitives consumed by wetSpring via ToadStool S71+++.
- **Exp261: Cross-Spring Modern Benchmark** — `benchmark_cross_spring_modern` (12 primitives benchmarked). Provenance tracking across hotSpring, wetSpring, neuralSpring, airSpring, groundSpring.
- **Exp262: Deep Debt Elimination Round 3** — 4 module refactors (dada2, io/ms2, ncbi/nestgate, tolerances/bio), 11 new tests, clone audit. ESN bridge to ToadStool `esn_v2` for NPU reservoir inference.
- **New spec**: `specs/CROSS_SPRING_EVOLUTION.md` — cross-spring shader and primitive evolution documentation.

**1,247 tests** | **262 experiments** | **238 binaries**

### Phase 57: ToadStool S68 Universal Precision Rewire (V57)

ToadStool advanced 19 commits (S66→S68) with universal precision architecture
(all 291 f32-only shaders → f64 canonical, dual-layer DF64 pipeline). wetSpring
revalidated cleanly after contributing a CPU feature-gate fix upstream and rewired
all 6 GPU modules from `compile_shader_f64()` to `compile_shader_universal(source,
Precision::F64)` — preparing for future DF64 precision experiments. Exp189
cross-spring evolution benchmark documents the full provenance chain from all 5
Springs (hotSpring precision, wetSpring bio, neuralSpring pairwise, airSpring stats,
groundSpring bootstrap) through ToadStool S68's 700 shaders, zero f32-only.

### Phase 56: Science Extension Pipeline + Primal Integration (V56)

Extending validated science with real NCBI data through sovereign primal pipeline:

- **NCBI EFetch + SRA download** — `ncbi/efetch.rs` (FASTA/GenBank download with validation),
  `ncbi/sra.rs` (capability-discovered `fasterq-dump`/`fastq-dump`), `ncbi/cache.rs` extended
  with accession-based storage and pure-Rust SHA-256 integrity verification
- **NestGate JSON-RPC integration** — `ncbi/nestgate.rs` routes data requests through NestGate's
  Unix socket JSON-RPC API when `WETSPRING_DATA_PROVIDER=nestgate` is set. Sovereign fallback.
  Discovers socket via capability cascade (`NESTGATE_SOCKET` → XDG → `/tmp`).
- **GPU Anderson L=14-20** — `validate_anderson_gpu_scaling.rs` (Exp184b): finite-size scaling
  at larger lattice sizes (N up to 8000), 16 realizations per point, 3 new named tolerances
- **biomeOS science graph** — `graphs/science_pipeline.toml` orchestrates NestGate → wetSpring →
  ToadStool pipeline. `science` domain added to `capability_registry.toml`.
- **5 experiment protocols** — Exp184-188: real NCBI 16S pipeline, cold seep metagenomes,
  dynamic Anderson W(t), DF64 large lattice, NPU sentinel with real sensor stream

**882 lib tests** | **96.67% coverage** | **86 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 50: V48-V50 ToadStool S65 Rewire + ODE Derivative Lean

- Exp183: Cross-Spring Evolution Benchmark — 36/36 checks PASS. GPU ODE, DiversityFusion, CPU delegation, GEMM, Anderson spectral, NMF, ridge. Provenance S39→S65.
- `diversity_fusion_f64.wgsl` absorbed S63 → deleted local WGSL, lean on `barracuda::ops::bio::diversity_fusion`
- `bio::diversity` (11 functions) → delegated to `barracuda::stats::diversity` (S64)
- 5 ODE RHS functions replaced with `barracuda::numerical::ode_bio::*Ode::cpu_derivative` (~200 lines eliminated)
- ToadStool S65 aligned (`17932267`) — 66 primitives + 2 BGL helpers, zero local WGSL

### Phase 51-52: V51 GPU Validation + V52 ToadStool S66 Rewire

- V51: 1,578 GPU validation checks on RTX 4070. Tolerance refinement for chained transcendentals (`GPU_LOG_POLYFILL` 1e-7). f32→f64 type fix for neuralSpring BatchFitness/LocusVariance.
- V52: ToadStool S66 rewire — `hill()` → `barracuda::stats::hill`, `fit_heaps_law` → `barracuda::stats::fit_linear`, `compute_ci` → `barracuda::stats::{mean, percentile}`. Re-export `shannon_from_frequencies`.
- ToadStool S66 aligned (`045103a7`) — 79 primitives consumed, zero local WGSL/derivative/regression math

### Phase 53: Cross-Spring Evolution Benchmarks

- 7 cross-spring benchmark/validator binaries all PASS on RTX 4070
- PairwiseJaccard 122× GPU speedup, SpatialPayoff 22×, PairwiseHamming 10× (neuralSpring → wetSpring)
- ODE lean 18-24% faster via upstream `integrate_cpu` after ToadStool absorption optimization
- hotSpring → wetSpring precision provenance: DF64, Anderson spectral, ESN reservoir, RK4/RK45
- Full cross-spring evolution narrative documented in `specs/CROSS_SPRING_EVOLUTION.md`

**833 lib tests** | **96.67% coverage** | **79 named tolerances** | **0 clippy warnings (pedantic)**

### Phase 48: V44 Complete Cross-Spring Rewire

- Reviewed ToadStool ABSORPTION_TRACKER: all 46 V16-V22 wetSpring items DONE
- `normal_cdf` rewired: `special::normal_cdf` → `barracuda::stats::norm_cdf` (50th primitive)
- Evolution request score: 8/9 delivered (tolerance module confirmed)
- `ValidationHarness` decision documented: available upstream but local `Validator` kept
- V44 complete rewire: find_w_c, anderson_sweep_averaged, pearson_correlation added (66 primitives)
- ToadStool S65 aligned — 66 primitives + 2 BGL helpers consumed

### Phase 43: DF64 Evolution Lean (S62+DF64)
- Adopted `storage_bgl_entry`/`uniform_bgl_entry` from upstream `ComputeDispatch` module
  (6 files, ~258 lines of BGL boilerplate removed)
- `gemm_cached.rs` simplified: `ShaderTemplate::for_driver_auto` → `compile_shader_f64`
- Identified DF64 GEMM adoption blocker: `wgsl_shader_for_device()` is private upstream
- Reported PeakDetectF64 WGSL bug (f32 literal → f64 array, line 49)
- ToadStool S62+DF64 aligned — 53 primitives + 2 BGL helpers consumed

### Phase 40: Cross-Spring Evolution Rewire (Exp162)

Wired cross-spring primitives (S54-S62) into wetSpring bio workflows (66 checks, all PASS):

- **graph_laplacian** (neuralSpring S54): community interaction network spectral analysis
- **effective_rank** (neuralSpring S54): diversity matrix spectral diagnostics
- **numerical_hessian** (neuralSpring S54): ML model curvature / convexity analysis
- **disordered_laplacian** (neuralSpring S56): Anderson disorder on community graphs (QS-disorder coupling)
- **belief_propagation_chain** (neuralSpring S56): hierarchical taxonomy classification
- **boltzmann_sampling** (neuralSpring S56): MCMC parameter optimization for ODE models

Compound workflows demonstrate cross-spring synergy: neuralSpring graph (S54)
+ neuralSpring disorder (S56) + hotSpring spectral analysis → detects Anderson
localization in biofilm geometry. ~100 clippy pedantic+nursery errors from
Rust 1.93 fixed across 20+ validation binaries.

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy -W pedantic -W nursery` | **Zero warnings** (pedantic + nursery clean) |
| `cargo doc --no-deps` | Clean (0 warnings) |
| Line coverage (`cargo-llvm-cov`) | **95.86% line / 93.54% fn / 94.99% branch** (1,044 lib tests) |
| `#![deny(unsafe_code)]` | **Enforced crate-wide** (edition 2024; `allow` only in test env-var calls) |
| `#![deny(clippy::expect_used, unwrap_used)]` | **Enforced crate-wide** |
| TODO/FIXME markers | **0** |
| Inline tolerance literals | **0** (all use `tolerances::` constants) |
| SPDX-License-Identifier | All `.rs` files |
| Max file size | All under 1000 LOC |
| External C dependencies | **0** (`flate2` uses `rust_backend`) |
| Named tolerance constants | 164 (scientifically justified, hierarchy-tested) |
| Provenance headers | All 285 binaries |
| ESN ridge regression | **Proper Cholesky solve** (not diagonal approximation) |
| I/O streaming | Buffering APIs deprecated; `stats_from_file` + iterators preferred |
| Clone optimization | Hot-path clones eliminated (merge_pairs, derep entry API) |

---

## Module Inventory

### CPU Bio Modules (47)

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
| `nmf` | Non-negative Matrix Factorization (Lee & Seung 1999) | Yang 2020 / Gao 2020 (repoDB) |
| `transe` | TransE knowledge graph embedding | ROBOKOP / RotatE |
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

**Passthrough (0)** — all promoted to Lean or Compose

**Write → Lean (5)** — now using `BatchedOdeRK4<S>::generate_shader()` (WGSL deleted):
`bistable_gpu`, `multi_signal_gpu`, `phage_defense_gpu`, `cooperation_gpu`, `capacitor_gpu`

### I/O Modules

`io::fastq` (streaming FASTQ/gzip), `io::mzml` (streaming mzML/base64),
`io::ms2` (streaming MS2), `io::nanopore` (POD5/FAST5 signal bridge, NRS synthetic reads)

---

## Repository Structure

```
wetSpring/
├── README.md                      ← this file
├── barracuda/                     ← Rust crate (src/, Cargo.toml, rustfmt.toml)
│   ├── EVOLUTION_READINESS.md    ← absorption map (tiers, primitives, shaders)
│   ├── ABSORPTION_MANIFEST.md    ← what's absorbed, local, planned (hotSpring pattern)
│   ├── src/
│   │   ├── lib.rs               ← crate root (pedantic + nursery lints enforced)
│   │   ├── special.rs           ← sovereign math (erf, ln_gamma, regularized_gamma)
│   │   ├── tolerances/           ← 164 named tolerance constants (bio, gpu, spectral, instrument)
│   │   ├── validation/           ← hotSpring validation framework
│   │   ├── ncbi/                ← NCBI module (API key, HTTP, E-search, EFetch, SRA, NestGate, cache)
│   │   ├── encoding.rs          ← sovereign base64 (zero dependencies)
│   │   ├── error.rs             ← error types (no external crates)
│   │   ├── bio/                 ← 47 CPU + 45 GPU bio modules
│   │   ├── io/                  ← streaming parsers (FASTQ, mzML, MS2, XML, nanopore)
│   │   ├── bench/               ← benchmark harness + power monitoring
│   │   ├── bin/                 ← 285 validation/benchmark binaries
│   │   ├── ipc/                 ← JSON-RPC dispatch (biomeOS integration)
│   │   └── vault/               ← encrypted consent-gated data storage
│   └── rustfmt.toml             ← max_width = 100, edition = 2024
├── experiments/                   ← 281 experiment protocols + results
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
├── scripts/                       ← Python baselines (57 scripts)
├── specs/                         ← specifications and paper queue (CROSS_SPRING_EVOLUTION.md)
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
cd barracuda

# Run all tests (1,044 library tests)
cargo test --workspace

# Code quality checks
cargo fmt -p wetspring-barracuda -p wetspring-forge -- --check
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery
cargo doc --no-deps

# Line coverage (requires cargo-llvm-cov)
cargo llvm-cov --lib --summary-only

# Workspace coverage with HTML report (from root):
cargo llvm-cov --workspace --html
# Or: ./scripts/coverage.sh

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

- **hotSpring** — Nuclear/plasma physics validation (sibling Spring, precision shaders, f64 WGSL, 4-layer brain architecture, 36-head ESN, NautilusBrain)
- **neuralSpring** — ML/neural inference validation (sibling Spring, eigensolvers, batch IPR, TransE)
- **airSpring** — Precision agriculture / IoT validation (sibling Spring, Richards PDE, Kriging)
- **barraCuda** — Standalone math primal (767+ f64-canonical WGSL shaders, universal precision, `PrecisionRoutingAdvice`, `shaders::provenance`, `BatchedOdeRK45F64`)
- **toadStool** — Hardware dispatch primal (GPU/NPU/CPU routing, adaptive tuning, S130 — 19,140+ tests, `science.*` IPC, coralReef proxy)
- **coralReef** — Sovereign GPU shader compiler (WGSL/SPIR-V → native binary, Phase 10, SM70–SM89 + RDNA2, `shader.compile.*` IPC)
- **wateringHole** — Inter-primal handoffs and cross-spring coordination
  - `handoffs/WETSPRING_BARRACUDA_031_REWIRE_HANDOFF_MAR03_2026.md` — barraCuda v0.3.1 rewire
  - `handoffs/archive/` — V7-V92J (fossil record)
  - `CROSS_SPRING_SHADER_EVOLUTION.md` — 767+ shader provenance
- **blueFish** — Chemistry as irreducible research programme (`whitePaper/blueFish/`)
- **ecoPrimals** — Parent ecosystem
