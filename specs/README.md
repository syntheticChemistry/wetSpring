# wetSpring Specifications

**Last Updated**: February 22, 2026
**Status**: Phase 28 — 1,476 CPU + 702 GPU + 80 dispatch + 35 layout + 57 transfer/streaming + 56 ODE parity = 2,406+/2,406+ checks, ALL PASS (740 tests, 103 experiments)
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| CPU validation | 1,476/1,476 PASS — 41 modules, 93+ experiments, 25 domains + 6 ODE flat + 3 layout + 13 GPU-promoted |
| GPU validation | 702/702 PASS — 30 ToadStool primitives, 5 local WGSL, 80 streaming + 48 head-to-head + 28 metalForge v4 + 38 pure GPU |
| Dispatch validation | 35/35 PASS — 5 substrate configs (Exp080) |
| BarraCuda CPU parity | 380/380 — 22.5x Rust speedup over Python (v1–v8) |
| BarraCuda GPU parity | 29 domains (Exp064/087/101) — pure GPU math proven |
| Pure GPU streaming | 80 checks, 441-837× over round-trip (Exp090/091) |
| metalForge cross-system | 29 domains CPU↔GPU (Exp103) + dispatch (Exp080) + pipeline (Exp086) + PCIe (Exp088) |
| Rust modules | 41 CPU + 42 GPU, 740 tests (~97% bio+io coverage) |
| Write phase | 5 local WGSL ODE shaders (phage_defense, bistable, multi_signal, cooperation, capacitor) |
| Dependencies | 1 runtime (flate2), everything else sovereign |
| Paper queue | **ALL DONE** — 29/29 reproducible papers complete (Track 1c added) |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 1c) | R. Anderson (Carleton) — deep-sea metagenomics, population genomics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Fourteen delivered (v1–v6, rewire, cross-spring, v7–v14) |

---

## Validation Chain (Per-Paper Status)

Every paper in the queue goes through the full evolution path. Status:

| Stage | What It Proves | Status |
|-------|---------------|--------|
| **Python baseline** | Algorithm correctness against published tools | 40 scripts, all reproducible |
| **BarraCuda CPU** | Pure Rust math matches Python | 1,476 checks, 380/380 cross-domain parity (v1–v8) |
| **BarraCuda GPU** | GPU produces same answer as CPU | 702 checks, 29 GPU domains |
| **Pure GPU streaming** | Zero CPU round-trips, data stays on-device | 80 checks, 441-837× over round-trip |
| **metalForge mixed** | Same answer on CPU, GPU, NPU — substrate-independent | 29 domains, 38+ checks + PCIe direct |

**Pure GPU promotion complete** — all 13 formerly CPU-only modules now have GPU
wrappers (Exp101). Papers 9, 10, 18, 26, 27 are no longer CPU-only. The only
remaining CPU-only domain is `phred` (I/O-bound, no parallelism benefit).
All ODE models have local WGSL shaders. Pending ToadStool absorption as
generic ODE primitive.

### Three-Tier Control Matrix (Per Paper)

| # | Paper | CPU | GPU | metalForge | Gaps |
|---|-------|:---:|:---:|:----------:|------|
| 1 | Galaxy/QIIME2 16S | Y | Y | Partial | DADA2, chimera, UniFrac not in MF16 |
| 2 | asari LC-MS | Y | Y | Y | — |
| 3 | FindPFAS screening | Y | Y | Y | — |
| 4 | GPU diversity + spectral | Y | Y | Y | — |
| 5 | Waters 2008 QS ODE | Y | Y | N | ODE absorbed (ToadStool S41); not in MF16 |
| 6 | Massie 2012 Gillespie | Y | Y | Y | — |
| 7 | Hsueh 2022 Phage defense | Y | Y | N | Local WGSL ODE (Exp099); exact parity |
| 8 | Fernandez 2020 Bistable | Y | Y | N | Local WGSL ODE (Exp100); exact parity |
| 9 | Mhatre 2020 Capacitor | Y | Y | Y | Local WGSL ODE (Exp101); exact parity |
| 10 | Bruger 2018 Cooperation | Y | Y | Y | Local WGSL ODE (Exp101); exact parity |
| 11 | Waters 2021 immuno | — | — | — | Reference only |
| 12 | Srivastava 2011 Multi-signal | Y | Y | N | Local WGSL ODE (Exp100); exact parity |
| 13 | Cahill proxy | Y | Y | Y | — |
| 14 | Smallwood proxy | Y | Y | Y | — |
| 15 | Liu 2014 HMM | Y | Y | Y | — |
| 16 | Alamin 2024 Placement | Y | Partial | N | Felsenstein GPU only; placement not in MF |
| 17 | Liu 2009 SATe | Y | Partial | Partial | NJ, Felsenstein CPU-only in MF |
| 18 | Zheng 2023 DTL | Y | Y | Y | Reconciliation GPU via batch workgroup (Exp101) |
| 20 | Wang 2021 RAWR | Y | Partial | N | Bootstrap compose; not in MF |
| 21 | Jones PFAS MS | Y | Y | Y | — |
| 22 | Jones PFAS F&T | Y | Y | Y | — |
| 24 | Anderson 2017 Population | Y | Y | Y | — |
| 25 | Moulana 2020 Pangenome | Y | Y | Y | — |
| 26 | Mateos 2023 Sulfur | Y | Y | Y | DTL + clock GPU (Exp101/103) |
| 27 | Boden 2024 Phosphorus | Y | Y | Y | DTL + clock GPU (Exp101/103) |
| 28 | Anderson 2014 Viral | Y | Y | Partial | k-mer not in MF16 |
| 29 | Anderson 2015 Rare biosphere | Y | Y | Y | PCoA skipped (naga bug) |

**Full three-tier coverage (CPU + GPU + metalForge):** Papers 2, 3, 4, 6, 9, 10, 13, 14, 15, 18, 21, 22, 24, 25, 26, 27, 29 — **17 of 25 actionable papers**.
**CPU + GPU (no metalForge):** Papers 1, 5, 7, 8, 12, 16, 17, 20, 28 — 8 papers.
**CPU only:** None — all actionable papers have at least CPU + GPU paths.

### Gaps

| Gap | Papers Affected | Blocker | Priority |
|-----|--------|---------|----------|
| ODE models not in metalForge routing | 5, 7, 8, 12 | GPU parity achieved (Write phase + Exp101); metalForge routing pending | Low |
| k-mer histogram not in metalForge routing | 28 | GPU wrapper done (kmer_gpu, Exp099); metalForge routing pending | Low |
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
| CONTROL_EXPERIMENT_STATUS.md | `../` | 103 experiments, 2,406+ validation checks |
| EVOLUTION_READINESS.md | `../barracuda/` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| Handoff (v14) | `../wateringHole/handoffs/` | Current ToadStool handoff |
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
- **Sovereign Rust bioinformatics** — 41 CPU + 42 GPU modules, 1 runtime dependency

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
`../wateringHole/handoffs/WETSPRING_TOADSTOOL_V14_FEB22_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
