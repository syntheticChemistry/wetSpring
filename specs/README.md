# wetSpring Specifications

**Last Updated**: February 25, 2026
**Status**: Phase 44 — 3,279+/3,279+ checks, ALL PASS (812 tests, 167 experiments, ToadStool S62+DF64 aligned, 44 primitives + 2 BGL helpers + 1 WGSL extension, barracuda always-on)
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| CPU validation | 1,476+/1,476+ PASS — 46 modules, 167 experiments, 25 domains + 6 ODE flat + 3 layout + 13 GPU-promoted |
| GPU validation | 710+/710+ PASS — 44 ToadStool primitives + 2 BGL helpers (S62+DF64, always-on), 1 local WGSL extension |
| Dispatch validation | 35/35 PASS — 5 substrate configs (Exp080) |
| BarraCuda CPU parity | 407/407 — 22.5x Rust speedup over Python (v1–v9) |
| BarraCuda GPU parity | 29 domains (Exp064/087/101/164) — pure GPU math proven |
| Pure GPU streaming | 152 checks — analytics (Exp105), ODE+phylo (Exp106), 441-837× vs round-trip (Exp090/091) |
| metalForge cross-system | 37 domains CPU↔GPU (Exp103+104+165) + dispatch (Exp080) + pipeline (Exp086) + PCIe (Exp088) |
| Cross-spring spectral | 25 checks — Anderson localization + QS-disorder analogy (Exp107) |
| Finite-size scaling | 14 checks — W_c = 16.26, disorder-averaged L=6–12 (Exp150) |
| Correlated disorder | 8 checks — biofilm clustering shifts W_c > 28 (Exp151) |
| Rust modules | 46 CPU + 42 GPU + 1 Write-phase extension, 812 tests |
| Write phase | 1 local WGSL extension (`diversity_fusion_f64.wgsl` — Exp167, 18/18) |
| Dependencies | 2 runtime (flate2 + bytemuck), everything else sovereign |
| Paper queue | **ALL DONE** — 43/43 reproducible papers complete (Track 1c + Track 3 + Phase 37 extensions) |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 1c) | R. Anderson (Carleton) — deep-sea metagenomics, population genomics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Thirty-six delivered (v1–v6, rewire, cross-spring, v7–v36) |

---

## Validation Chain (Per-Paper Status)

Every paper in the queue goes through the full evolution path. Status:

| Stage | What It Proves | Status |
|-------|---------------|--------|
| **Python baseline** | Algorithm correctness against published tools | 42 scripts, all reproducible |
| **BarraCuda CPU** | Pure Rust math matches Python | 1,476 checks, 380/380 cross-domain parity (v1–v8) |
| **BarraCuda GPU** | GPU produces same answer as CPU | 702 checks, 29 GPU domains |
| **Pure GPU streaming** | Zero CPU round-trips, data stays on-device | 152 checks, 10+ domains, 441-837× over round-trip (Exp090/105/106) |
| **metalForge mixed** | Same answer on CPU, GPU, NPU — substrate-independent | 37 domains, 25/25 papers three-tier (Exp103/104) |

**Pure GPU promotion complete** — all 13 formerly CPU-only modules now have GPU
wrappers (Exp101). Papers 9, 10, 18, 26, 27 are no longer CPU-only. The only
remaining CPU-only domain is `phred` (I/O-bound, no parallelism benefit).
All ODE models now use `BatchedOdeRK4<S>::generate_shader()` — complete lean on
ToadStool's generic ODE framework (S51). 30,424 bytes of local WGSL deleted.

### Three-Tier Control Matrix (Per Paper)

| # | Paper | CPU | GPU | metalForge | Gaps |
|---|-------|:---:|:---:|:----------:|------|
| 1 | Galaxy/QIIME2 16S | Y | Y | Y | DADA2 (Exp104), chimera (Exp103), UniFrac (Exp104) |
| 2 | asari LC-MS | Y | Y | Y | — |
| 3 | FindPFAS screening | Y | Y | Y | — |
| 4 | GPU diversity + spectral | Y | Y | Y | — |
| 5 | Waters 2008 QS ODE | Y | Y | Y | QS ODE sweep via Exp104 |
| 6 | Massie 2012 Gillespie | Y | Y | Y | — |
| 7 | Hsueh 2022 Phage defense | Y | Y | Y | Phage ODE via Exp100 (v4) |
| 8 | Fernandez 2020 Bistable | Y | Y | Y | Bistable ODE via Exp100 (v4) |
| 9 | Mhatre 2020 Capacitor | Y | Y | Y | Capacitor ODE via Exp103 (v5) |
| 10 | Bruger 2018 Cooperation | Y | Y | Y | Cooperation ODE via Exp103 (v5) |
| 11 | Waters 2021 immuno | — | — | — | Reference only |
| 12 | Srivastava 2011 Multi-signal | Y | Y | Y | Multi-signal ODE via Exp100 (v4) |
| 13 | Cahill proxy | Y | Y | Y | — |
| 14 | Smallwood proxy | Y | Y | Y | — |
| 15 | Liu 2014 HMM | Y | Y | Y | — |
| 16 | Alamin 2024 Placement | Y | Y | Y | Felsenstein via Exp104 (placement = Felsenstein per edge) |
| 17 | Liu 2009 SATe | Y | Y | Y | NJ via Exp103, Felsenstein via Exp104 |
| 18 | Zheng 2023 DTL | Y | Y | Y | Reconciliation GPU via batch workgroup (Exp101) |
| 20 | Wang 2021 RAWR | Y | Y | Y | Felsenstein via Exp104 (bootstrap = Felsenstein per replicate) |
| 21 | Jones PFAS MS | Y | Y | Y | — |
| 22 | Jones PFAS F&T | Y | Y | Y | — |
| 24 | Anderson 2017 Population | Y | Y | Y | — |
| 25 | Moulana 2020 Pangenome | Y | Y | Y | — |
| 26 | Mateos 2023 Sulfur | Y | Y | Y | DTL + clock GPU (Exp101/103) |
| 27 | Boden 2024 Phosphorus | Y | Y | Y | DTL + clock GPU (Exp101/103) |
| 28 | Anderson 2014 Viral | Y | Y | Y | K-mer via Exp104 |
| 29 | Anderson 2015 Rare biosphere | Y | Y | Y | — (PCoA naga bug resolved in wgpu v22.1.0) |
| 39 | Fajgenbaum 2019 PI3K/AKT/mTOR | Y | Y | Y | Exp157 (CPU), Exp164 (GPU), Exp165 (metalForge) |
| 40 | Fajgenbaum 2025 MATRIX pharmacophenomics | Y | Y | Y | Exp158 (CPU), Exp164 (GPU), Exp165 (metalForge) |
| 41 | Yang 2020 NMF drug repurposing | Y | Y | Y | Exp159 (CPU), Exp164 (GPU), Exp165 (metalForge) |
| 42 | Gao 2020 repoDB NMF | Y | Y | Y | Exp160 (CPU), Exp164 (GPU), Exp165 (metalForge) |
| 43 | ROBOKOP KG embedding | Y | Y | Y | Exp161 (CPU), Exp164 (GPU), Exp165 (metalForge) |

**Full three-tier coverage (CPU + GPU + metalForge):** All **30 of 30 actionable papers** (25 Tracks 1-2 + 5 Track 3).
**CPU + GPU (no metalForge):** None — all actionable papers now have full three-tier coverage.
**CPU only:** None — all actionable papers have at least CPU + GPU paths.

### Remaining Exclusions (by design)

| Item | Papers Affected | Reason | Priority |
|------|--------|--------|----------|
| Waters 2021 (Paper 11) | — | Reference only — no computational reproduction target | N/A |
| Liu fungi-bacteria (Paper 19) | — | Manuscript in progress | Watch |

**Resolved in Phase 31:**
- PCoA naga bug — fixed in wgpu v22.1.0; `catch_unwind` guards removed; Paper 29 now has full three-tier including PCoA GPU
- Kachkovskiy 2018 (Paper 23) — validated via cross-spring spectral primitives (Exp107: 25/25 checks)

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Complete | 43/43 papers reproduced across 4 tracks + cross-spring |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | 167 experiments, 3,279+ validation checks, 812 tests |
| EVOLUTION_READINESS.md | `../barracuda/` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| Handoff (v36) | `../wateringHole/handoffs/WETSPRING_TOADSTOOL_V36_WRITE_PHASE_HANDOFF_FEB25_2026.md` | Current ToadStool handoff |
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
- **Drug repurposing** — NMF, knowledge graph embeddings, pharmacophenomics (Track 3)
- **Sovereign Rust bioinformatics** — 46 CPU + 42 GPU modules + 1 Write-phase WGSL extension, 2 runtime dependencies (flate2 + bytemuck), 44 ToadStool primitives + 2 BGL helpers (S62+DF64, always-on, zero fallback)

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
`../wateringHole/handoffs/WETSPRING_TOADSTOOL_V36_WRITE_PHASE_HANDOFF_FEB25_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
