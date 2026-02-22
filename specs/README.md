# wetSpring Specifications

**Last Updated**: February 22, 2026
**Status**: Phase 24 — 1,392 CPU + 609 GPU + 80 dispatch + 35 layout + 57 transfer/streaming = 2,229+/2,229+ checks, ALL PASS (740 tests)
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| CPU validation | 1,392/1,392 PASS — 41 modules, 93 experiments, 25 domains + 6 ODE flat + 3 layout |
| GPU validation | 609/609 PASS — 32 ToadStool primitives, 0 local WGSL (Lean complete), 80 streaming + 48 head-to-head + 28 metalForge v3 |
| Dispatch validation | 35/35 PASS — 5 substrate configs (Exp080) |
| BarraCuda CPU parity | 205/205 — 22.5x Rust speedup over Python |
| BarraCuda GPU parity | 16 domains (Exp064/087) — pure GPU math proven |
| Pure GPU streaming | 80 checks, 441-837× over round-trip (Exp090/091) |
| metalForge cross-system | 12 domains CPU↔GPU (Exp084) + dispatch (Exp080) + pipeline (Exp086) + PCIe (Exp088) |
| Rust modules | 41 CPU + 25 GPU, 740 tests (~97% bio+io coverage) |
| Tier A (GPU/NPU-ready) | 7 modules with flat layouts (kmer, unifrac, taxonomy + 4 ODE) |
| Dependencies | 1 runtime (flate2), everything else sovereign |
| Paper queue | **ALL DONE** — 29/29 reproducible papers complete (Track 1c added) |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 1c) | R. Anderson (Carleton) — deep-sea metagenomics, population genomics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Thirteen delivered (v1–v6, rewire, cross-spring, v7–v13) |

---

## Validation Chain (Per-Paper Status)

Every paper in the queue goes through the full evolution path. Status:

| Stage | What It Proves | Status |
|-------|---------------|--------|
| **Python baseline** | Algorithm correctness against published tools | 40 scripts, all reproducible |
| **BarraCuda CPU** | Pure Rust math matches Python | 1,392 checks, 205/205 cross-domain parity |
| **BarraCuda GPU** | GPU produces same answer as CPU | 533 checks, 16 GPU domains |
| **Pure GPU streaming** | Zero CPU round-trips, data stays on-device | 80 checks, 441-837× over round-trip |
| **metalForge mixed** | Same answer on CPU, GPU, NPU — substrate-independent | 12 domains, 35+ checks + PCIe direct |

Papers with **no GPU path** (sequential algorithms: chimera, derep, NJ) stay CPU-only.
Papers with **ODE models** (Waters, Fernandez, Mhatre) are GPU-ready via flat params
but blocked: upstream `batched_ode_rk4.rs` uses `compile_shader` not `compile_shader_f64`.

### Three-Tier Control Matrix (Per Paper)

| # | Paper | CPU | GPU | metalForge | Gaps |
|---|-------|:---:|:---:|:----------:|------|
| 1 | Galaxy/QIIME2 16S | Y | Y | Partial | DADA2, chimera, UniFrac not in MF16 |
| 2 | asari LC-MS | Y | Y | Y | — |
| 3 | FindPFAS screening | Y | Y | Y | — |
| 4 | GPU diversity + spectral | Y | Y | Y | — |
| 5 | Waters 2008 QS ODE | Y | Y | N | ODE absorbed (ToadStool S41); not in MF16 |
| 6 | Massie 2012 Gillespie | Y | Y | Y | — |
| 7 | Hsueh 2022 Phage defense | Y | Partial | N | ODE compose; not in MF |
| 8 | Fernandez 2020 Bistable | Y | Y | N | ODE+bifurcation; not in MF |
| 9 | Mhatre 2020 Capacitor | Y | N | N | CPU-only ODE |
| 10 | Bruger 2018 Cooperation | Y | N | N | CPU-only game theory |
| 11 | Waters 2021 immuno | — | — | — | Reference only |
| 12 | Srivastava 2011 Multi-signal | Y | Partial | N | ODE compose; not in MF |
| 13 | Cahill proxy | Y | Y | Y | — |
| 14 | Smallwood proxy | Y | Y | Y | — |
| 15 | Liu 2014 HMM | Y | Y | Y | — |
| 16 | Alamin 2024 Placement | Y | Partial | N | Felsenstein GPU only; placement not in MF |
| 17 | Liu 2009 SATe | Y | Partial | Partial | NJ, Felsenstein CPU-only in MF |
| 18 | Zheng 2023 DTL | Y | N | N | Reconciliation CPU-only |
| 20 | Wang 2021 RAWR | Y | Partial | N | Bootstrap compose; not in MF |
| 21 | Jones PFAS MS | Y | Y | Y | — |
| 22 | Jones PFAS F&T | Y | Y | Y | — |
| 24 | Anderson 2017 Population | Y | Y | Y | — |
| 25 | Moulana 2020 Pangenome | Y | Y | Y | — |
| 26 | Mateos 2023 Sulfur | Y | N | Partial | DTL, clock not in MF |
| 27 | Boden 2024 Phosphorus | Y | N | Partial | DTL, clock not in MF |
| 28 | Anderson 2014 Viral | Y | Y | Partial | k-mer not in MF16 |
| 29 | Anderson 2015 Rare biosphere | Y | Y | Y | PCoA skipped (naga bug) |

**Full three-tier coverage (CPU + GPU + metalForge):** Papers 2, 3, 4, 6, 13, 14, 15, 21, 22, 24, 25, 29 — **12 of 25 actionable papers**.
**CPU + GPU (no metalForge):** Papers 1, 5, 7, 8, 12, 16, 17, 20, 28 — 9 papers.
**CPU only:** Papers 9, 10, 18, 26, 27 — 5 papers (CPU-appropriate: sequential ODE, game theory, DTL reconciliation).

### Gaps

| Gap | Papers Affected | Blocker | Priority |
|-----|--------|---------|----------|
| ODE models not in metalForge | 5, 7, 8, 9, 10, 12 | Local WGSL; upstream needs `compile_shader_f64` | Medium |
| DTL reconciliation CPU-only | 18, 26, 27 | Tree-recursive algorithm; no GPU benefit | Low |
| Molecular clock not in metalForge | 26, 27 | CPU-only; no GPU path planned | Low |
| k-mer histogram not in metalForge | 28 | GPU layout ready (Exp085); absorption pending | Medium |
| PCoA skipped in metalForge | 29 | naga WGSL compiler bug | Low (tracked upstream) |
| Waters 2021 (Paper 11) | — | Reference only — no computational reproduction target | N/A |
| Liu fungi-bacteria (Paper 19) | — | Manuscript in progress | Watch |
| Kachkovskiy 2018 (Paper 23) | — | Cross-spring reference; reproduction in groundSpring | N/A |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Complete | 29/29 papers reproduced across 3 tracks |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | 97 experiments, 2,229+ validation checks |
| EVOLUTION_READINESS.md | `../barracuda/` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| Handoff (v13) | `../wateringHole/handoffs/` | Current ToadStool handoff |
| whitePaper/STUDY.md | `../whitePaper/` | Full study narrative |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Two-track validation protocol |
| metalForge/ | `../metalForge/` | Hardware characterization + substrate routing |

---

## Scope

### wetSpring IS:
- **16S pipeline validation** — FASTQ → quality → merge → derep → DADA2 → chimera → taxonomy → diversity → UniFrac
- **LC-MS feature extraction** — mzML → EIC → peaks → features → spectral matching
- **PFAS screening** — KMD + tolerance search + MS2 fragment matching
- **Microbial ecology** — Alpha/beta diversity, PCoA, rarefaction
- **Deep-sea metagenomics** — ANI, SNP, dN/dS, molecular clock, pangenomics
- **ML inference** — Decision tree, Random Forest, GBM (all sovereign, no Python)
- **Sovereign Rust bioinformatics** — 41 CPU + 20 GPU modules, 1 runtime dependency

### wetSpring IS NOT:
- Sensor noise analysis (groundSpring)
- Neural network training (neuralSpring)
- Physics simulation (hotSpring)
- ET₀/irrigation (airSpring)

### wetSpring EXTENDS TO (via faculty):
- **Waters**: c-di-GMP signaling dynamics, quorum sensing, biofilm regulation, phage defense
- **Liu**: Comparative genomics, phylogenetic placement, introgression detection, cophylogenetics
- **R. Anderson**: Deep-sea metagenomics, population genomics, pangenomics, viral ecology, molecular clock
- **Cahill/Smallwood**: Algal pond metagenomics, phage biocontrol
- **Jones**: PFAS fate-and-transport, high-resolution mass spec methods

---

## Reading Order

**New to wetSpring** (20 min):
1. This README (5 min)
2. `../whitePaper/README.md` — overview and key results (10 min)
3. PAPER_REVIEW_QUEUE.md — what's next (5 min)

**Deep dive** (2 hours):
`../whitePaper/STUDY.md` → `../CONTROL_EXPERIMENT_STATUS.md` → `../barracuda/EVOLUTION_READINESS.md` → BARRACUDA_REQUIREMENTS.md

**Integration partner**:
`../wateringHole/handoffs/WETSPRING_TOADSTOOL_V13_FEB22_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
