<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# baseCamp: Per-Faculty Research Briefings

**Date:** April 10, 2026
**Project:** wetSpring (ecoPrimals)
**Status:** V139 — **380** experiments indexed (377 completed + 3 PROPOSED), **5,800+** validation checks, ALL PASS; **1,580** library tests (0 failures); **356** workspace binaries (**334** barracuda + **22** forge); standalone `barraCuda` **v0.3.7** (**150+** primitives consumed; wgpu 28, 784+ WGSL shaders), toadStool S155+, coralReef Phase 10. **45** IPC capabilities, **37** dispatch methods, **9** niche dependencies (5 required + 4 optional). Composition validation: **97/97** proto-nucleate alignment (Exp400). plasmidBin **v0.8.0** harvest-ready. **V139:** NUCLEUS composition validation tier — Exp400, JSON shape fix, Squirrel niche wiring, proptest sync, 5 composition IPC round-trip tests. **V138:** primal composition patterns, friction/gap analysis. **V137:** provenance headers, tolerance centralization, IPC modularization. **V136:** thiserror, named casts, contract pinning. **V133–V135:** GPU evolution, math delegation, doc reconciliation. 0 local WGSL, 0 local math duplication, 0 `#[allow()]`.

---

## Overview

Each document in this directory summarizes what wetSpring reproduced, evolved,
and validated from one faculty member's published work. The evolution follows
a consistent five-stage path:

```
Python/Galaxy baseline
  -> Rust CPU (sovereign, 1 dependency: barraCuda)
    -> GPU acceleration (barraCuda WGSL, toadStool dispatch)
      -> Pure GPU streaming (zero CPU round-trips)
        -> metalForge cross-substrate (CPU = GPU = NPU output)
          -> NPU reservoir deployment (ESN → int8 → Akida AKD1000)
```

Every stage is validated with explicit numerical checks. All data is open.
All code is AGPL-3.0.

## V134–V137 — Deep audit, debt resolution, provenance completion, and IPC modularization

**V137** completes provenance coverage and tolerance centralization: `//! Provenance:`
headers on all 355 validation/benchmark binaries (was 5 in V136), 8 new named tolerance
constants (242 total), `ipc/connection.rs` extraction (connection pipeline separated from
server lifecycle), GPU buffer renames for clippy compliance, doc link fixes.

**V136** deepens debt resolution and ecosystem absorption: thiserror migration, named
cast helpers (~60 casts across 15 files), upstream contract pinning, bitwise determinism
tests, CI version pin, provenance headers on 5 core validators, hardcoding audit,
ipc/server.rs refactored, CONTRIBUTING.md + SECURITY.md.

**V135** completes documentation reconciliation: canonical metrics aligned across all
docs, V135 handoff crafted for wateringHole, ecoPrimals baseCamp updated.

**V134** was a deep audit: drug NMF → `barracuda::linalg::nmf`, 26 clippy fixes,
validation harness refactored into domain submodules, primal discovery extended to 7
primals, SPDX headers, CI feature-matrix.

**V133** was a deep evolution sprint: `validate_all`, `GpuContext`/`TensorSession`,
`performance_surface`, zero-copy I/O. Quality bar: **1,902** tests, **355** binaries
with full provenance, **242** named tolerances, **0** clippy warnings, **0** `unsafe`.
See companion handoffs under `wateringHole/handoffs/` for barraCuda/toadStool asks.

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domains |
|---------|------------|-------|:------:|:-----------:|:------:|---------|
| [Waters](waters.md) | MSU MMG | 1 | 7 | 020,022-025,027,030,108,114,**121** | 147+ | QS, ODE, Gillespie, bistability, phage defense, NPU QS classifier, NCBI Vibrio landscape |
| [Liu](liu.md) | MSU CMSE | 1b | 6 | 026,031-034,036-038,109,115 | 136 | HMM, phylogenetics, alignment, placement, NPU placement |
| [R. Anderson](anderson.md) | Carleton | 1c | 6 | 051-056,110,116,**125** | 170 | Deep-sea metagenomics, pangenomics, NPU binning, NCBI Campylobacterota |
| [Jones](jones.md) | MSU BMB | 2 | 2 | 041,042,111,117,**124** | 55 | PFAS mass spectrometry, EPA ML, NPU spectral triage |
| [Cahill](cahill.md) | Sandia | 1 | 1 | 039,112,118,**123** | 54 | Algal pond, NPU bloom sentinel, temporal ESN |
| [Smallwood](smallwood.md) | Sandia | 1 | 1 | 040,112,118,**123** | 58 | Bloom surveillance, NPU sentinel, temporal ESN |
| [Kachkovskiy](kachkovskiy.md) | MSU CMSE | cross | 1 | 107,113,119,**122,126,127-138,144-156** | 334 | Spectral theory, 2D/3D Anderson, geometry zoo, ecosystem atlas, finite-size scaling v1+v2, correlated disorder, mapping sensitivity, planktonic dilution, eukaryote scaling, extension papers, paper queue |
| **Fajgenbaum** | UPenn | 3 | 7 | 157–165 | 84 | Drug repurposing, pharmacophenomics, Track 3 completed |
| **Diversity Fusion** | — | GPU | 1 | 167 | 18 | CPU↔GPU parity extension |
| **Track 4 Soil QS** | — | 4 | 9 | 170–182 | 321 | No-till soil QS, Anderson pore geometry, Brandt farm, meta-analysis, tillage factorial, CPU/GPU/streaming/metalForge |
| **biomeOS IPC** | — | cross | — | 203-208 | 321 | IPC dispatch, GPU-aware routing, NUCLEUS atomics, Songbird, Neural API |
| **V66 Audit + Dispatch** | — | cross | — | 209,212-215 | 239+ | Streaming I/O parity, CPU v12, dispatch evolution, NUCLEUS V8, CPU vs GPU v5 |
| **V67 Experiment Buildout** | — | cross | — | 216-220 | 170+ | 47-domain CPU proof, Python-vs-Rust benchmark, 42-module GPU portability, unidirectional streaming, cross-substrate dispatch V67 + BandwidthTier, 11 extension papers → 50/50 three-tier |
| **V88 Buildout** | — | cross | — | 263-270 | 427 | CPU v20, CPU↔GPU pure-math, metalForge v12, NUCLEUS v3, ToadStool dispatch v3, mixed-HW, biomeOS graph |
| **V89 S79 Rewire** | — | cross | — | 271 | 73 | Cross-spring S79 provenance: 13 domains, 6 springs, `MultiHeadBioEsn`, `SpectralAnalysis` IPC, ToadStool S79 deep rewire |
| **V90 Bio Brain** | — | cross | — | 272 | 64 | hotSpring 4-layer brain → `BioBrain`, 36-head Gen2 → `BioHeadGroupDisagreement` + `AttentionState`, bingoCube/nautilus → `BioNautilusBrain`, 3 IPC methods, 7 domains, 7 springs |
| **V97c Fused Ops** | — | chain | — | Exp306-310 | 111 | Welford, Pearson, Spearman, CorrMatrix, streaming, metalForge |
| **V98 Full Chain** | — | chain | — | Exp313-318 | 173 | 52 papers (pre-Track 6), 33 bio modules, GPU Hybrid-aware, streaming, metalForge |
| **V98+ Cross-Spring** | — | evolution | — | Exp319-320 | 52 | All 5 springs, 28 shaders, 22 cross-spring, GPU FusedMapReduceF64, 24 benchmarks |
| **V99 biomeOS/NUCLEUS** | — | integration | — | Exp321-322 | 64 | biomeOS IPC server, JSON-RPC 2.0, NUCLEUS env probe, cross-primal pipeline, brain module |
| **V99 Chain** | — | chain | — | Exp323-326 | 102 | CPU v25 (46), GPU v14 (27), metalForge v17 (29), ToadStool dispatch, NUCLEUS probes |
| **V100 petalTongue + Mixed HW** | — | viz+dispatch | — | Exp327-332 | 173 | petalTongue schema, 5 scenario builders, CPU/GPU math parity, metalForge viz, biomeOS full chain, local evolution, mixed HW dispatch |
| **V101 Viz Evolution** | — | viz | — | Exp333-334 | 78 | 9 DataChannel types, 33 scenario builders, StreamSession, Songbird, IPC science→viz, streaming pipeline |
| **V108 Track 6 Anaerobic** | Liao (MSU BAE) | 6 | 5+6 | Exp336-346 | 183 | Gompertz, first-order, Monod, Haldane kinetics, anaerobic diversity, Anderson W mapping, full 6-tier chain |
| **V110 petalTongue + Anderson H3** | — | viz+science | — | Exp353-356 | 111 | petalTongue live dashboards, Anderson QS O₂-modulated W model (H3, r=0.851), stream_ecology, 6 JSON scenarios |
| **V130 Anderson Hormesis** | Gonzales/Lisabeth/Waters/Jones/Kachkovskiy | cross | 1 | Exp377-379 | 31 | Biphasic dose-response via Anderson, colonization resistance surface, binding IPR/localization, computation as experiment preprocessor, healthSpring joint |
| **Total** | | | **64** | | **5,707+** | |

### NCBI-Scale Extensions (Phase 32)

| Experiment | Faculty | Scale | Checks | GPU Prim |
|:---:|---------|-------|:------:|----------|
| Exp108 | Waters | 1024-genome QS parameter landscape | 8 | `BatchedOdeRK4F64` |
| Exp109 | Liu | 128-taxon placement + Felsenstein | 11 | NJ + Felsenstein |
| Exp110 | Anderson | 200-genome cross-ecosystem pangenome | 17 | ANI + dN/dS |
| Exp111 | Jones | 2048-spectrum spectral cosine (2.1M pairs) | 14 | `GemmF64` |
| Exp112 | Cahill/Smallwood | 1365-timepoint multi-ecosystem bloom | 23 | `FusedMapReduceF64` |
| Exp113 | Kachkovskiy | 8-ecosystem QS-disorder prediction | 5 | `barracuda::spectral` |

### NPU Reservoir Deployment (Phase 33)

| Experiment | Faculty | NPU Task | Checks | Key Finding |
|:---:|---------|----------|:------:|-------------|
| Exp114 | Waters | QS phase classifier (3-class) | 13 | 100% f64↔NPU agreement |
| Exp115 | Liu | Phylogenetic clade placement (8-class) | 9 | 97.7% quantization fidelity |
| Exp116 | Anderson | Genome ecosystem binning (5-class) | 9 | Int8 regularization effect (+6% acc) |
| Exp117 | Jones | Spectral library pre-filter (2048 spectra) | 8 | 84% top-10 overlap, 2-stage |
| Exp118 | Cahill/Smallwood | Bloom sentinel (4-state) | 11 | Coin-cell >1 year battery |
| Exp119 | Kachkovskiy | QS-disorder regime (3-regime) | 9 | Physics ordering preserved |

### NCBI-Scale Hypothesis Testing (Phase 35, GPU-confirmed)

| Experiment | Faculty | GPU Result | Checks | Key Finding |
|:---:|---------|-----------|:------:|-------------|
| Exp121 | Waters | 200 real Vibrio assemblies → GPU ODE sweep | 14 | All-biofilm landscape; real genomes don't reach planktonic |
| Exp122 | Kachkovskiy | 2D Anderson 20×20 lattice sweep | 12 | Extended plateau absent in 1D; bloom QS-active in 2D only; J_c≈0.41 |
| Exp123 | Cahill/Smallwood | Stateful vs stateless ESN bloom | 9 | Temporal memory preserves classification; coin-cell feasible |
| Exp124 | Jones | NPU→GPU spectral triage (5K library) | 10 | 100% recall, 20% pass rate, 3.7× speedup |
| Exp125 | Anderson | Real Campylobacterota pangenome (158 asm) | 11 | 4 ecosystems, 49–53% core, open pangenome confirmed |
| Exp126 | Kachkovskiy | 28-biome global QS atlas (136 BioProjects) | 90 | W monotonic with J; all biomes correctly ordered |

### 3D Anderson Dimensional QS (Phase 36, GPU-confirmed)

| Experiment | Faculty | GPU Result | Checks | Key Finding |
|:---:|---------|-----------|:------:|-------------|
| Exp127 | Kachkovskiy | 1D→2D→3D sweep (8³ lattice) | 17 | 3D plateau 2.4× wider; J_c(3D)≈1.28 >> J_c(2D)≈0.56; gut/vent/soil flip to active |
| Exp128 | Kachkovskiy/Anderson | Vent chimney geometry → QS | 12 | 3 of 4 zones 3D-active but 2D-suppressed; 2D misses 75% capability |
| Exp129 | Kachkovskiy | 28-biome × 3-dimension phase diagram | 12 | All 28 biomes QS-active in 3D, zero in 1D/2D; W_c(3D)≈16.5 exceeds all |
| Exp130 | Kachkovskiy | Thick biofilm 3D block (8×8×6) | 9 | 4× wider plateau than 2D; J_c(3D)≈1.25; 6 layers transform QS |

## Validation Chain

Every paper goes through the full evolution. Status across all 64 papers:

| Stage | What It Proves | Coverage |
|-------|---------------|----------|
| Paper math control | Published equations reproduced exactly | 63 papers (Exp341), 38 checks |
| Python baseline | Algorithm correctness against published tools | 71 scripts (all with reproduction headers + SHA-256 integrity verification) |
| **R industry baseline** | **Gold-standard parity against R/vegan, R/DADA2, R/phyloseq** | **3 R scripts, 53 checks (Exp335), JSON baselines with SHA-256** |
| BarraCuda CPU | Rust matches Python within machine precision | 26 domains (v1-v19), bit-identical to SciPy/NumPy (Exp253) |
| BarraCuda GPU | GPU matches CPU within 1e-6 | 21 GPU domains, 1,783+ checks |
| Pure GPU streaming | Zero CPU round-trips, data stays on-device | 8 streaming experiments, unidirectional proof (0.10ms overhead, Exp255) |
| metalForge | Same answer on CPU, GPU, NPU | 63/63 papers, 37+ domains |
| NPU reservoir | ESN → int8 → NPU preserves classification (Cholesky solve) | 59 checks, 6 domains |
| Cross-spring evolution | 767+ WGSL shaders traced to origin springs, 150+ barraCuda primitives consumed | 21 checks |
| NCBI-scale hypothesis | Real NCBI data + GPU-confirmed Anderson/QS/pangenome | 146 checks |
| 3D Anderson dimensional QS | hotSpring spectral primitives → ecological predictions | 50 checks |
| biomeOS IPC integration | JSON-RPC science primal, GPU-aware dispatch, Songbird registration | 321 checks (Exp203-208) |
| petalTongue visualization | 9 DataChannel types, 33 scenario builders, StreamSession, Songbird capabilities, IPC science→viz wiring | 78 checks (Exp333-334) |
| **V110 live viz + Anderson H3** | petalTongue live dashboards (IPC push + JSON export), all 9 DataChannel types validated with real math, stream_ecology module, Anderson QS O₂-modulated W model (H3, r=0.851 vs 10 environments), biomeOS/NUCLEUS readiness probing | 111 checks (Exp353-356) |
| Code quality audit | 91.20% line / 90.30% fn (gated at 90%), **streaming-only I/O** (buffering `parse_*` removed), 0 production mocks, standalone `barraCuda` v0.3.7 (150+ primitives consumed), `forbid(unsafe_code)` at workspace level + per-crate, clippy pedantic + nursery ZERO WARNINGS, **242** named tolerances (zero inline literals), **44** GPU modules, **49** CPU bio modules, **capability-based runtime discovery** (7 ecosystem primals), typed error enums (`VaultError`/`NestError`/`SongbirdError`/`AssemblyError`), validation harness refactored into domain submodules, `proptest` property-based testing, **zero `#[allow()]` in entire codebase** — all `#[expect(reason)]` with self-documenting justifications, **`//! Provenance:` headers on all 355 workspace binaries** | **1,902** tests |
| V66 dispatch evolution | Forge dispatch routing (29 workloads), streaming topology (PCIe bypass), NUCLEUS Tower/Node/Nest model, absorption audit (0 local WGSL) | 49 checks (Exp213) |
| V66 NUCLEUS V8 | IPC dispatch with V66 I/O evolution (byte-native FASTQ, bytemuck nanopore, streaming MS2), Nest metrics, CPU fallback parity, full pipeline chain | 49 checks (Exp214) |
| **V84 pipeline buildout** | Paper→CPU→GPU→Streaming proven end-to-end: 32 papers, 26 CPU domains, 21 GPU domains, Python parity, 0.10ms streaming overhead | 172 checks (Exp251-255) |
| **V85 NUCLEUS + Vault** | EMP 30K atlas, NUCLEUS data pipeline, Tower-Node deployment, Genomic Vault organ model (consent-gated encrypted storage, Merkle provenance) | 87 checks (Exp256-259) |
| **V87 blueFish whitePaper** | Chemistry as irreducible research programme (Fodor, Lakatos, Anderson). Isomorphism: 29 comp-chem ops → BarraCUDA (14 direct, 9 compose, 6 new). RootPulse provenance for DFT campaigns. hotSpring brain arch mapped to bio workloads. | 7 documents |
| **V88 experiment buildout** | Full control validation: CPU v20 (37), CPU↔GPU v7 (22), metalForge v12 (63), NUCLEUS v3 (106), ToadStool pure-math v3 (41), CPU↔GPU pure-math (38), mixed-HW dispatch (91), biomeOS graph (29). Barracuda API deep dive documented for absorption. | 427 checks (Exp263-270) |
| **V97c fused ops chain** | Full chain: CPU fused parity (38), Python benchmarks (13), GPU portability on Hybrid (21), pure GPU streaming (18), metalForge cross-substrate (21). Hybrid-aware graceful degradation for VarianceF64/CorrelationF64. DF64 FusedMapReduceF64 validated on consumer GPU. | 111 checks (Exp306-310) |
| **V97d deep audit** | Crate-wide deep audit: I/O buffering APIs deprecated (streaming-first), 104 unwrap→expect evolutions, doc accuracy (MSRV, wgpu version, rustdoc escaping), broken reference cleanup, full regression GREEN. | 125 items (Exp311) |
| **V97e provenance rewire** | Builder pattern migration (HMM, DADA2, Gillespie dispatch → struct args). PrecisionRoutingAdvice for shared-memory f64 safety. shaders::provenance API (28 shaders, 22 cross-spring, 17 consumed). Error propagation fixes. | 31 checks (Exp312) |
| **V98 full chain** | Paper Math v5 (52 papers) → CPU v24 (33 bio modules) → GPU v13 (Hybrid-aware, diversity+Anderson+chemistry) → Streaming v11 (zero CPU round-trips) → metalForge v16 (CPU=GPU=NPU). Strengthened Track 4 soil papers, analytical identities. | 173 checks (Exp313-318) |

## Extension Roadmap

See [`EXTENSION_PLAN.md`](EXTENSION_PLAN.md) for the phased plan to take
validated science to real-world data:

| Phase | What | Compute | Priority |
|-------|------|---------|----------|
| **P0** | EMP 30K samples + KBS LTER time series | Eastgate alone, hours | Now |
| **P1** | SRA longitudinal atlas + AMR surveillance | LAN mesh (10G cables) | After cabling |
| **P2** | MinION field genomics + coupled nutrient models | Eastgate + ~$1K MinION | Medium-term |

Primal integration: NestGate for NCBI data acquisition, biomeOS NUCLEUS for
local/LAN orchestration (Tower→Node→Nest atomics), ToadStool for GPU dispatch.
All IPC validated in Exp203-208 (321 checks), NUCLEUS pipeline in Exp256-259 (87 checks, vault organ model).

## Performance Summary

| Metric | Value | Source |
|--------|-------|--------|
| Rust vs Python (25 domains) | **33.4x** overall, 625x peak (Smith-Waterman) | Exp059 |
| GPU vs CPU (spectral cosine) | **926x** | Exp087 |
| GPU vs CPU (taxonomy, 500 queries) | **63x** | Exp016 |
| GPU vs CPU (pipeline, 10 samples) | **2.45x** | Exp016 |
| Streaming vs round-trip | 92-94% overhead eliminated | Exp091 |
| Galaxy vs Rust GPU (10 samples) | 95.6 s vs 3.0 s (**31.9x**) | Exp015/016 |
| Energy cost (10K samples) | Galaxy $0.40, Rust GPU **$0.02** | Exp016 |
| Hardware | Consumer GPU (RTX 4070), no HPC | All |

## Sub-theses (Gen3 Ch. 14)

| # | Sub-thesis | File | Key Experiments |
|:-:|-----------|------|-----------------|
| 01 | Anderson-QS null hypothesis | [sub_thesis_01](sub_thesis_01_anderson_qs.md) | 107, 126–138, 144–156 |
| 02 | LTEE constrained evolution | [sub_thesis_02](sub_thesis_02_ltee.md) | 143, 146 |
| 03 | BioAg rhizosphere QS | [sub_thesis_03](sub_thesis_03_bioag.md) | 129, 142, 146, 151, 153 |
| 04 | Sentinels + NPU deployment | [sub_thesis_04](sub_thesis_04_sentinels.md) | 114, 118, 123, 124, 147, 193–195 |
| 05 | Cross-species eavesdropping | [sub_thesis_05](sub_thesis_05_cross_species.md) | 142, 144–146, 151, 153–154 |
| 06 | Field genomics (nanopore + NPU) | [sub_thesis_06](sub_thesis_06_field_genomics.md) | 196a-c (pre-hardware done, 52 checks); planned: 197–202 |

### Cross-Spring Rewire + Modern Benchmark (Phase 48, V44)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp168 | Cross-spring S62 evolution | ~25 | hotSpring precision → wetSpring bio → neuralSpring validated |
| Exp169 | Modern cross-spring benchmark | 12 | All CPU primitives validated, 4-spring provenance, bit-exact delegation |

V44 rewired 6 validation binaries to modern upstream: `find_w_c` (4 files),
`pearson_correlation` (1 file). 85 primitives consumed (S70+++). Hill CPU ODE helpers
intentionally kept local (derivative-level math, GPU equivalents generated
by `BatchedOdeRK4::generate_shader()`).

### Track 4: No-Till Soil QS & Anderson Geometry (Phase 53, V52)

9 papers, 13 experiments, 321 validation checks. Extends the Anderson-QS framework
into soil microbiology: pore-network geometry maps to Anderson disorder, aggregate
stability predicts QS activation probability, and tillage history modulates
effective dimension.

| Experiment | Paper | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp170 | Martínez-García 2023 QS-pore geometry coupling | 26 | Non-linear connectivity → Anderson disorder W; P(QS) via norm_cdf |
| Exp171 | Feng 2024 pore-size diversity | 27 | Pore class → effective dimension → diversity (Shannon, Bray-Curtis) |
| Exp172 | Mukherjee 2024 distance colonization | 23 | Autoinducer diffusion + QS biofilm ODE + cooperation collapse |
| Exp173 | Islam 2014 Brandt farm soil health | 14 | Published metrics validated; no-till → lower disorder → higher diversity |
| Exp174 | Zuber & Villamil 2016 meta-analysis | 20 | Effect sizes, CI verification, Anderson predicts MBC increase |
| Exp175 | Liang 2015 31-year tillage | 19 | 2×2×2 factorial, Shannon/Pielou, 31-year temporal recovery |
| Exp176 | Tecon & Or 2017 biofilm-aggregate | 23 | Water film thickness → QS diffusion, aggregate surface → colonization |
| Exp177 | Rabot 2018 structure-function | 16 | Structural properties → Anderson parameters → functional outcomes |
| Exp178 | Wang 2025 tillage × compartment | 15 | Endosphere/rhizosphere compartment effects, stover return |
| Exp179 | CPU parity benchmark | 49 | 8 domains timed, pure Rust math validated |
| Exp180 | GPU validation | 23 | FMR + BrayCurtisF64 + Anderson 3D + ODE on GPU |
| Exp181 | Pure GPU streaming | 52 | Zero-CPU-roundtrip soil QS pipeline |
| Exp182 | metalForge cross-substrate | 14 | CPU = GPU for all Track 4 domains |

### Phase 59 Additions (Exp184–188)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp184 | Real NCBI Sovereign Pipeline | 25 | Real NCBI 16S sovereign pipeline validation |
| Exp185 | Cold Seep Sovereign Pipeline | 10 | Cold seep metagenomes sovereign pipeline (CPU + GPU Anderson) |
| Exp186 | Dynamic Anderson W(t) | 7 | Community evolution, time-dependent disorder |
| Exp187 | DF64 Anderson Large Lattice | 4 | DF64 L=24+ lattice validation |
| Exp188 | NPU Sentinel Real Stream | 10 | Real sensor stream NPU deployment |

### Phase 60 Additions (Exp193–195)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp193 | NPU Hardware Validation (Real AKD1000 DMA) | 7 sections | DMA 37 MB/s, device discovery, SRAM mapping |
| Exp194 | NPU Live ESN — sim↔hardware comparison | 23 | 3 classifiers, 20.7K infer/sec, 1.4 µJ/infer, weight mutation |
| Exp195 | Funky NPU Explorations (AKD1000 novelties) | 14 | PUF 6.34 bits, online evolution 136 gen/sec, streaming 12.9K Hz |

### Field Genomics: Nanopore + NPU Integration (Sub-thesis 06)

Extends the sentinel framework (Sub-thesis 04) and metalForge substrate routing
with in-field DNA sequencing via Oxford Nanopore MinION. The architectural
thesis: a sovereign Rust pipeline that sequences, classifies, and acts at the
edge without cloud connectivity, using the AKD1000 for real-time classification
and NPU-driven adaptive sampling.

| Program | Connects | Experiments |
|---------|----------|:-----------:|
| Bloom Sentinel Live (Great Lakes HAB) | ST01, ST04, Cahill/Smallwood | Exp196–198 |
| Soil Health Sentinel (no-till) | ST01, ST03, ST06-local, Track 4 | Exp199–200 |
| AMR Wastewater Sentinel | ST04 (pathogen emergence) | Exp201–202 |
| PFAS Dual-Mode Monitor | ST04 (PFAS), Jones Lab | planned |
| NPU Adaptive Sampling | cross-cutting technique | Exp197 |
| Deep-Sea Autonomous Lander | ST01, R. Anderson, cold seep | long-term |

**V61 status:** `io::nanopore` module operational (POD5/NRS parser, streaming
iterator, synthetic read generation). Exp196a-c validate the full pre-hardware
software path (52/52 checks PASS). Remaining: `bio::basecall` (signal → base,
awaiting MinION hardware) and Exp197-202 (real sequencer integration).

### Phase 61 Additions (Exp196a-c)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp196a | Nanopore Signal Bridge (POD5/NRS) | 28 | Header parsing, signal extraction, read→FASTQ, batch iteration |
| Exp196b | Simulated Long-Read 16S Pipeline | 11 | ASV recovery from noisy long reads, community reconstruction |
| Exp196c | NPU Int8 Quantization Pipeline | 13 | f64→int8 community fidelity, ESN regime classification |

See [Sub-thesis 06: Field Genomics](sub_thesis_06_field_genomics.md) for
full architecture, literature review, and experiment plan.

### Phase 64: Cross-Spring Modern Rewiring (Exp210)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp210 | Cross-spring modern benchmark (S70+++ Fp64Strategy, submit_and_poll, provenance) | 24 | All 5-spring provenance validated, optimal_precision wired, ToadStool resilient dispatch |

V64 wired `Fp64Strategy` and `optimal_precision()` in `GpuF64`, migrated 6 GPU
modules (5 ODE + GEMM) to `submit_and_poll()` for device-lost resilience, and
created a comprehensive cross-spring evolution benchmark tracing provenance
across hotSpring, wetSpring, neuralSpring, airSpring, and groundSpring.

### Phase 65: Progression Benchmark (Exp211)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp211 | Full progression: Python → CPU → GPU → streaming → metalForge | 16 | 27× Rust/CPU vs Python, chained GPU GEMM, workload-aware metalForge routing |

V65 proves the complete validation progression on a single unified workload:
Python baseline → BarraCuda CPU (27× faster, pure Rust) → BarraCuda GPU
(identical results via ToadStool) → Pure GPU streaming (chained
`execute_to_buffer`, zero round-trips) → metalForge cross-substrate routing
(CPU/GPU/NPU workload-aware dispatch, 10K element threshold).

### Phase 62 Additions (Exp203–208)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp203 | biomeOS Science Pipeline (IPC server, Songbird, Neural API) | 29 | Full IPC lifecycle validated, metrics reporting |
| Exp206 | BarraCuda CPU v11 (IPC dispatch math fidelity) | 64 | Zero numeric drift through IPC layer (EXACT_F64) |
| Exp207 | BarraCuda GPU v4 (IPC science on GPU) | 54 | GPU-aware dispatch, lazy OnceLock, threshold routing |
| Exp208 | metalForge v7 (NUCLEUS mixed hardware) | 75 | PCIe bypass topology, Tower/Node/Nest atomics, cross-substrate |

### Phase 66: Deep Audit + Dispatch Evolution (Exp209, 212–215)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp209 | Streaming I/O parity (byte-native FASTQ, bytemuck nanopore, MS2 streaming) | 37 | All evolved I/O paths bit-exact with batch originals |
| Exp212 | BarraCuda CPU v12 — post-audit math fidelity (I/O → diversity → QS → derep → merge) | 55 | End-to-end pipeline through evolved I/O layer preserves math |
| Exp213 | Compute dispatch + streaming evolution (forge dispatch, PCIe bypass, NUCLEUS model) | 49 | 29 workloads route correctly, 0 local WGSL (full lean), PCIe bypass streamable |
| Exp214 | NUCLEUS mixed hardware V8 — V66 I/O evolution via IPC dispatch | 49 | Tower/Node/Nest lifecycle validated through evolved I/O, Nest metrics, CPU fallback parity |
| Exp215 | CPU vs GPU v5 — V66 I/O evolution domains | ~40 | Built and ready, awaiting GPU hardware validation |

V66 deep audit: byte-native FASTQ I/O (eliminated UTF-8 assumptions), bytemuck
nanopore bulk read (zero per-sample I/O), streaming APIs for mzML/MS2/FASTQ
(`for_each_spectrum`, `for_each_record`), safe env handling (`temp_env` replacing
unsafe `set_var`), tolerance centralization (103 named constants with provenance),
`partial_cmp` → `total_cmp` migration (10 lib sites), 0 unsafe code, 0 production
mocks. Dispatch evolution (Exp213) proves the metalForge infrastructure correctly
handles all V66 workloads. NUCLEUS V8 (Exp214) proves the IPC layer preserves
math fidelity through the evolved I/O stack.

### Phase 67: Experiment Buildout + Evolution (Exp216–220)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp216 | BarraCuda CPU v13 — 47-domain pure Rust math proof | 47+ | 27 CPU domains including all Track 4 soil modules validated |
| Exp217 | Python-vs-Rust v2 benchmark (25 selected domains) | 25+ | Wall-clock timing across all major domain categories |
| Exp218 | BarraCuda GPU v5 — 42-module portability proof | 36+ | 12 GPU-accelerated domains: CPU == GPU within tolerances |
| Exp219 | Pure GPU streaming v3 — unidirectional pipeline | 32+ | Quality → diversity → PCoA → spectral, zero CPU round-trips |
| Exp220 | Cross-substrate dispatch V67 + BandwidthTier | 30+ | BandwidthTier detection, bandwidth-aware routing, GPU→NPU→CPU data flow |

V67 experiment buildout: 10 new Python baselines (9 Track 4 soil papers +
NPU spectral triage Exp124) with SHA-256 provenance in BASELINE_MANIFEST.md.
11 extension papers (Exp144-149, 152-156) promoted to three-tier by adding
metalForge workload definitions (`ShaderOrigin::Absorbed`). BandwidthTier
and ComputeDispatch from Barracuda wired into metalForge bridge/dispatch.
50/50 three-tier papers. 281 experiments, 8,300+ checks, 1,044 lib tests.

## Open Data

All 63 reproductions plus 5 fused ops experiments use publicly accessible data or published model parameters.
No proprietary data dependencies. Sources: NCBI SRA, Zenodo, MassBank, EPA,
Michigan EGLE, published ODE parameters, repoDB, algorithmic (no external data),
published soil metrics (Islam 2014, Zuber 2016, Liang 2015), published
model equations (Martínez-García 2023, Tecon & Or 2017, Wang 2025), and published
biogas kinetics parameters (Yang 2016, Chen 2016, Rojas-Sossa 2017/2019, Zhong 2016).

See `../specs/PAPER_REVIEW_QUEUE.md` for the full provenance audit.
