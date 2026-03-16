# wetSpring — Life Science & Analytical Chemistry Validation

**An ecoPrimals Spring** — life science biome validating Python baselines
against Rust implementations and GPU shaders via `barraCuda` (standalone math
primal). Follows the **Write → Absorb → Lean** cycle adopted from hotSpring.

**Date:** March 16, 2026
**License:** AGPL-3.0-or-later
**MSRV:** 1.87
**Status:** V123 — **1,703 tests** (0 failures), 376 experiments, 5,707+ validation checks across 354 binaries. Ecosystem: barraCuda v0.3.5, toadStool S155, coralReef Phase 10. Zero local WGSL, zero unsafe code, `cargo clippy` **ZERO WARNINGS** (pedantic + nursery), `cargo fmt` clean, **zero `#[allow()]`, zero `#[expect(clippy::expect_used)]`, zero `.expect()` in validation code**. **Zero-panic validation** — `OrExit` trait replaces 1,039 `.expect()` and 632 `.unwrap()` across 192 binaries (groundSpring V109 pattern). **Dual-format capability discovery** — `capability.list` returns `operation_dependencies`, `cost_estimates`, and `semantic_mappings` alongside flat capability list. `extract_rpc_error()` centralized JSON-RPC error parsing (healthSpring pattern). Songbird registration enriched with niche metadata. Python deps pinned with upper bounds. `Validator::finish_with_code()` for composable exit handling. Superseded handoffs archived (V111–V122).

---

## What This Is

wetSpring validates the entire evolution path from interpreted-language
scientific computing (Python/numpy/scipy/sklearn and R/vegan/DADA2/phyloseq)
to sovereign Rust CPU implementations, and then to GPU acceleration via
`barraCuda`:

```
R/Python baseline → Rust CPU validation → GPU acceleration → metalForge cross-substrate
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
| **Lean** | 35 | GPU modules consuming upstream barraCuda primitives directly |
| **Compose** | 12 | GPU modules wiring multiple barraCuda primitives (kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock, chimera, gbm, feature_table, streaming, taxonomy) |
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
| `visualization` | petalTongue scenario builders (28 builders, 9 channel types, LivePipelineSession, streaming, Songbird) |

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
| V100 viz + local evolution (Exp327-332) | 173 (petalTongue 45 + CPU/GPU 27 + metalForge 19 + biomeOS 34 + local evo 24 + mixed HW 24) |
| V101 viz evolution (Exp333-334) | 78 (viz evolution 44 + science-to-viz 34) |
| **Total validation checks** | **5,707+** |
| Rust library unit tests | 1,353 (barracuda) |
| metalForge forge tests | 252 |
| Integration tests | 98 (72 barracuda + 26 forge) |
| **Total Rust tests** | **1,703** (lib + forge + integration) |
| Library code coverage | **94.01% line** (barracuda), **90%+ line** (forge) (cargo-llvm-cov) |
| Experiments completed | 376 |
| Validation/benchmark binaries | 354 (332 barracuda + 22 forge) |
| CPU bio modules | 47 |
| GPU bio modules | 47 (30 lean + 5 write→lean + 7 compose + 0 passthrough + 5 visualization) |
| Tier B (needs refactor) | 0 (all promoted) |
| Python baselines | 58 scripts (all with reproduction headers + SHA-256 integrity verification) |
| BarraCuda CPU parity | 546/546 (v1-v11: 36+ domains, IPC fidelity proven) |
| BarraCuda GPU parity | 36+ domains (17 absorbed + 5 local ODE + 7 compose + IPC GPU-aware dispatch) |
| metalForge cross-system | 37+ domains CPU↔GPU (Exp103+104+165+182+208), **39/39 papers three-tier** |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| Pure GPU streaming | 152 checks — analytics (Exp105), ODE+phylo (Exp106), 441-837× vs round-trip |
| `barraCuda` primitives consumed | **150+** (v0.3.5, always-on, zero fallback code — standalone `barraCuda`, wgpu 28, PrecisionRoutingAdvice) |
| Local WGSL shaders | **0** (diversity fusion absorbed S63 — fully lean) |
All 5,707+ validation checks **PASS**. All 1,353 library + 252 forge + 98 integration tests **PASS** (2 ignored: hardware-dependent). Zero clippy warnings, zero fmt diffs, zero `#[allow()]` in entire codebase.

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

### V98+: Upstream Rewire + Cross-Spring Evolution (Exp313–320)

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

### V99: biomeOS/NUCLEUS Integration + Cross-Primal Pipeline (Exp321–322)

wetSpring integrated as a biomeOS science primal via JSON-RPC IPC,
with deploy graph `wetspring_deploy.toml` and cross-primal pipeline validation.

| Exp | Name | Type | Checks |
|-----|------|------|:------:|
| 321 | biomeOS/NUCLEUS V98+ Integration | Integration | 42/42 |
| 322 | Cross-Primal Pipeline V98+ | Pipeline | 22/22 |

- IPC server: health, science.diversity, science.qs_model, full_pipeline, brain, metrics, ai.ecology_interpret
- Protocol: JSON-RPC 2.0, error codes, 10-request multiplexing on single connection
- NUCLEUS: environment probe, deploy graph, Songbird discovery
- Pipeline: airSpring ET₀ → wetSpring QS → diversity → neuralSpring graph → spectral

### V99 Chain: CPU v25 → GPU v14 → metalForge v17 (Exp323–326)

| Exp | Name | Checks |
|-----|------|:------:|
| 323 | CPU v25 — Cross-Primal Pure Rust Math | 46/46 |
| 324 | GPU v14 — GPU + ToadStool Dispatch | 27/27 |
| 326 | metalForge v17 — Mixed NUCLEUS + biomeOS Graph | 29/29 |

- V99 chain: 102/102 PASS (CPU → GPU → metalForge)
- Cross-primal: ET₀ hydrology, Anderson spectral, graph Laplacian, NMF
- ToadStool: FusedMapReduce dispatch, provenance tracking, DF64 Hybrid
- NUCLEUS: Tower/Node/Nest probes, biomeOS deploy graph + capability registry
- V98 chain regression: 173/173 PASS (still green)

**1,047 tests** | **300 experiments** | **305 binaries** | **8,886+ checks**

### V100: petalTongue Visualization + Local Evolution + Mixed Hardware (Exp327–332)

Full `petalTongue` visualization integration, CPU/GPU math parity, local evolution
for upstream absorption, and mixed hardware dispatch validation:

| Exp | Name | Checks |
|-----|------|:------:|
| 327 | petalTongue Visualization Schema Validation | 45/45 |
| 328 | CPU vs GPU Pure Math Parity (Viz Domains) | 27/27 |
| 329 | metalForge petalTongue Integration | 19/19 |
| 330 | biomeOS + NUCLEUS + petalTongue Full Chain | 34/34 |
| 331 | Local Evolution & Upstream Readiness | 24/24 |
| 332 | Mixed Hardware Dispatch Evolution | 24/24 |

- **petalTongue**: `DataChannel` schema types, `EcologyScenario` graph nodes/edges, IPC push client
- **Visualization module**: `barracuda/src/visualization/` — 5 scenario builders (diversity, KMD, PCoA, ODE, ordination)
- **metalForge visualization**: `forge/src/visualization/` — inventory, dispatch, NUCLEUS topology scenarios
- **Local evolution**: `FitResult.slope()` migration, `HmmModel` doc aliases, NMF bio re-export, quality test extraction
- **Bandwidth wiring**: `BioWorkload.data_bytes` on kmer (10 MB), smith_waterman (50 MB), pcoa (8 MB), dada2 (100 MB)
- **Mixed dispatch**: GPU→NPU→CPU priority chain, bandwidth-aware fallback, PCIe cost model

**1,277 tests** | **332 experiments** | **311 binaries** | **8,982+ checks**

### V101: petalTongue Visualization Evolution (Exp333–334)

Full visualization evolution: 7 new scenario builders, Spectrum DataChannel type,
StreamSession for progressive rendering, Songbird capability announcement, and
science-to-visualization IPC wiring:

| Exp | Name | Checks |
|-----|------|:------:|
| 333 | Visualization Evolution | 44/44 |
| 334 | Science-to-Viz Pipeline | 34/34 |

- **Spectrum DataChannel**: 7th channel type for FFT/power spectrum data (Anderson spectral, HRV, signal)
- **StreamSession**: Session lifecycle for progressive petalTongue rendering (`push_initial_render`, `push_timeseries_append`, `push_gauge_update`, `push_replace`)
- **Songbird capabilities**: `VisualizationAnnouncement` struct with 16 capabilities, 7 channel types
- **6 new scenario builders**: pangenome (heatmap+bar+gauge), HMM (timeseries+bar+heatmap), stochastic (timeseries+distribution+gauge), similarity (heatmap+distribution), rarefaction (timeseries+gauge), NMF (heatmap+bar)
- **Streaming pipeline builder**: Multi-node graph for GPU pipeline stages (QF→DADA2→taxonomy→diversity→β-diversity)
- **IPC wiring**: `science.diversity` and `science.full_pipeline` gain `visualization: bool` parameter — auto-build scenario + push to petalTongue
- **dump_wetspring_scenarios**: 13 scenarios (was 6), `--stream` flag for StreamSession demo

**1,687 tests** | **376 experiments** | **354 binaries** | **5,707+ checks**

### V119: Deep Debt Evolution Sprint (2026-03-15)

Niche architecture, typed errors, domain-organized refactoring, modern lint attributes,
capability-based discovery, property-based testing, Squirrel AI integration:

| Change | Detail |
|--------|--------|
| **Niche self-knowledge** | `niche.rs` module (capabilities, dependencies, cost estimates, ecology semantic mappings) + `wetspring-ecology.yaml` BYOB manifest |
| **Typed errors** | `VaultError` (vault), `NestError` / `SongbirdError` / `AssemblyError` (forge) — `Result<_, String>` reduced from ~25 to 8 |
| **7 large files → submodules** | `streaming_gpu` (670→688 LOC/3 files), `chimera` (531→562/3), `signal` (532→558/4), `msa` (565→606/3), `mzxml` (583→605/3), `mzml/decode` (580→650/3), `handlers/expanded` (485→444/7) — net **−3,496 lines** |
| **`#[expect(reason)]` migration** | 10 validation binaries: `#[allow(clippy::*)]` → `#[expect(clippy::*, reason = "validation binary: ...")]` |
| **Hardcoding eliminated** | `primal_names.rs` constants replace 15+ hardcoded strings across 8 IPC modules; 4 binaries evolved from local `discover_socket()` to library `ipc::discover` |
| **`proptest` adopted** | 4 property-based tests: Gillespie steady-state convergence, bootstrap CI coverage, rarefaction monotonicity + Shannon bounds, cooperation population bounds |
| **Squirrel AI** | `ai.ecology_interpret` capability domain (15th), `discover_squirrel()`, handler with graceful degradation (unavailable/timeout/error → Ok with status) |
| **Clone reduction** | `PhyloTree::into_flat_tree(self)` zero-copy consuming method; `Box<dyn Fn>` retained for Gillespie (documented: heterogeneous reactions require dynamic dispatch) |

- `cargo check --workspace` — clean
- `cargo test --workspace` — **1,687 passed**, 0 failed, 2 ignored
- 48 files changed, 463 insertions, 3,959 deletions (net −3,496)

### V116: Deep Audit Execution — Capability Discovery + Tolerance Centralization (2026-03-15)

Full execution of V115 audit findings:
- **`capability.list` JSON-RPC handler**: `dispatch.rs` routes `capability.list` → `handle_capability_list()` returning all 14 domains / 19 methods with primal metadata
- **`capability_domains.rs` expanded**: 11 ecology-only → 14 domains across 4 families (ecology, provenance, brain, metrics), 19 methods, test-gated `VALID_DOMAIN_PREFIXES`, `all_methods()` introspection
- **Inline tolerance centralization**: 15 replacements across 10 validation binaries — `1e-10`/`0.001`/`1e-12`/`1e-15`/`1e-5`/`1e-6` → `tolerances::PYTHON_PARITY`, `ODE_DEFAULT_DT`, `ANALYTICAL_F64`, `EXACT_F64`, `GEMM_GPU_MAX_ERR`, `GPU_VS_CPU_F64`
- **Capability-based primal discovery**: 3 binaries (`validate_primal_pipeline_v1`, `validate_workload_routing_v1`, `validate_petaltongue_live_v1`) refactored — `discover_socket(env_var, primal)` helper (env → XDG → `BIOMEOS_SOCKET_DIR` → temp), primal name checks → capability checks
- **metalForge forge lint parity**: `#![deny(missing_docs)]`, `#![warn(clippy::pedantic, clippy::nursery)]` to match barracuda
- **Doc correction**: `discovery.rs` socket fallback documentation now reflects actual `temp_dir()` behavior
- **Audit false-positive resolution**: All 4 reported `panic!()` and ~20 `#[expect(clippy::unwrap_used)]` confirmed test-only — zero production code violations
- **All 31 IPC tests pass** (capability_domains + dispatch)

### V115: Deep Audit + UniBin + Capability Domains + Tolerance Evolution (2026-03-15)

12-finding comprehensive audit executed:
- **UniBin compliance**: `wetspring` binary with `server|status|version` subcommands
- **Capability domain architecture**: `capability_domains.rs` + `capability_registry.toml` (19 capabilities, 4 domains)
- **Tolerance centralization**: NMF_CONVERGENCE, MATRIX_EPS, STABLE_SPECIAL_TINY → no inline literals
- **XDG path resolution**: `/tmp/` hardcodes eliminated, `$XDG_RUNTIME_DIR` with fallback
- **Python baseline provenance**: SHA-256 self-hash + git commit in JSON output
- **Bitflag refactoring**: `QsType` booleans → u8 bitflags
- **metalForge coverage**: 80% → 90% threshold, 12 new tests (output.rs, data.rs)
- **Cast safety documentation**: all `#[expect(clippy::cast_*)]` annotated
- **Automated baseline verification**: `scripts/verify_baseline_outputs.sh`

### V112: Streaming-Only I/O + Zero-Warning Pedantic + Capability Discovery (2026-03-14)

- Deprecated buffering parsers (`parse_fastq`, `parse_mzml`, `parse_ms2`) removed — streaming-only I/O
- All clippy pedantic+nursery warnings eliminated (40 → 0)
- Hardcoded primal paths evolved to `$PATH`/`$XDG_RUNTIME_DIR` capability discovery
- Build-breaking `_all_pielou` compilation errors fixed in 2 validation binaries
- Inline tolerance `1e-10` → `tolerances::ANALYTICAL_LOOSE`

### V111: Deep Debt Resolution + Idiomatic Evolution (2026-03-14)

Build health restored, comprehensive clippy/fmt/doc cleanup, and dependency evolution.
See CHANGELOG.md for full V111 entry.

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy -W pedantic -W nursery` | **Zero warnings** (pedantic + nursery clean) |
| `cargo doc --no-deps` | Clean (0 warnings) |
| Line coverage (`cargo-llvm-cov`) | **94.01% line** (barracuda), **90%+ line** (forge) (1,703 tests) |
| `#![forbid(unsafe_code)]` | **Enforced on all 356 crate roots** (2 lib + 354 bin; edition 2024) |
| `#![deny(clippy::expect_used, unwrap_used)]` | **Enforced crate-wide** |
| `#[expect(reason = "...")]` | All validation binary lint overrides documented |
| TODO/FIXME markers | **0** |
| Inline tolerance literals | **0** (all use `tolerances::` constants) |
| SPDX-License-Identifier | All `.rs` files |
| Max file size | All under 1000 LOC |
| External C dependencies | **0** (`flate2` uses `rust_backend`) |
| Named tolerance constants | **214** (scientifically justified, hierarchy-tested; zero inline literals — all `tolerances::*`) |
| Provenance headers | All 340 binaries |
| ESN ridge regression | **Proper Cholesky solve** (not diagonal approximation) |
| I/O streaming | **Streaming-only** — buffering `parse_*` functions removed; `FastqIter` / `MzmlIter` / `Ms2Iter` + `stats_from_file` |
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
│   │   ├── tolerances/           ← 214 named tolerance constants (bio, gpu, spectral, instrument)
│   │   ├── validation/           ← hotSpring validation framework
│   │   ├── ncbi/                ← NCBI module (API key, HTTP, E-search, EFetch, SRA, NestGate, cache)
│   │   ├── encoding.rs          ← sovereign base64 (zero dependencies)
│   │   ├── error.rs             ← error types (no external crates)
│   │   ├── bio/                 ← 47 CPU + 47 GPU bio modules
│   │   ├── io/                  ← streaming parsers (FASTQ, mzML, MS2, XML, nanopore)
│   │   ├── bench/               ← benchmark harness + power monitoring
│   │   ├── bin/                 ← 318 validation/benchmark binaries (+ 22 forge)
│   │   ├── niche.rs             ← self-knowledge: capabilities, dependencies, cost estimates
│   │   ├── ipc/                 ← JSON-RPC dispatch (biomeOS integration, capability_domains.rs, primal_names.rs)
│   │   ├── visualization/       ← petalTongue schema types, 13 scenario builders, StreamSession, Songbird
│   │   └── vault/               ← encrypted consent-gated data storage
│   ├── rustfmt.toml             ← max_width = 100, edition = 2024
│   └── capability_registry.toml ← machine-readable capability manifest (19 capabilities, 4 domains)
├── niches/                        ← BYOB niche manifests (wetspring-ecology.yaml)
├── experiments/                   ← 376 experiment protocols + results
├── metalForge/                    ← hardware characterization + substrate routing
│   ├── forge/                    ← Rust crate: wetspring-forge (discovery + dispatch)
│   │   ├── src/                  ← substrate.rs, probe.rs, inventory.rs, dispatch.rs, bridge.rs
│   │   └── examples/             ← inventory discovery demo
│   ├── PRIMITIVE_MAP.md          ← Rust module ↔ ToadStool primitive mapping
│   ├── ABSORPTION_STRATEGY.md   ← Write → Absorb → Lean methodology + CPU math evolution
│   └── benchmarks/
│       └── CROSS_SYSTEM_STATUS.md ← algorithm × substrate matrix
├── wateringHole/                   ← spring-local handoffs (following hotSpring pattern)
│   └── handoffs/                  ← ToadStool/barraCuda rewire + cross-spring evolution docs
├── scripts/                       ← Python baselines (58 scripts)
├── specs/                         ← specifications and paper queue (CROSS_SPRING_EVOLUTION.md)
├── whitePaper/                    ← validation study draft
└── data/                          ← local datasets (not committed)
```

---

## Quick Start

```bash
cd barracuda

# Run all tests (1,703 library + forge + integration tests)
cargo test --workspace --all-features

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
- **barraCuda** — Standalone math primal v0.3.5 (767+ f64-canonical WGSL shaders, universal precision, `PrecisionRoutingAdvice`, `shaders::provenance`, `BatchedOdeRK45F64`)
- **toadStool** — Hardware dispatch primal (GPU/NPU/CPU routing, adaptive tuning, S155 — 19,140+ tests, `science.*` IPC, coralReef proxy)
- **coralReef** — Sovereign GPU shader compiler (WGSL/SPIR-V → native binary, Phase 10, SM70–SM89 + RDNA2, `shader.compile.*` IPC)
- **wateringHole** — Inter-primal handoffs and cross-spring coordination
  - `handoffs/WETSPRING_V122_MODERN_IDIOMATIC_RUST_HANDOFF_MAR16_2026.md` — V122 modern idiomatic Rust evolution + toadStool/barraCuda absorption
  - `handoffs/archive/` — V7-V121 (fossil record)
  - `CROSS_SPRING_SHADER_EVOLUTION.md` — 767+ shader provenance
- **blueFish** — Chemistry as irreducible research programme (`whitePaper/blueFish/`)
- **ecoPrimals** — Parent ecosystem
