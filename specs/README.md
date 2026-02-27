# wetSpring Specifications

**Last Updated**: February 27, 2026
**Status**: Phase 61 — 4,800+ checks (1,578 GPU on RTX 4070, 60 NPU on AKD1000), ALL PASS (1,022 tests, 203 experiments, ToadStool S68+ aligned (`e96576ee`), 79 primitives consumed, 0 local WGSL/derivative/regression, barracuda always-on, 92 named tolerances with full provenance, 0 ad-hoc magic numbers, clippy pedantic CLEAN, 0 Passthrough, V61 `io::nanopore` + Exp196a-c pre-hardware, 39/39 three-tier, 52/52 papers, 95.46% line / 93.54% fn / 94.99% branch)
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| CPU validation | 1,476+/1,476+ PASS — 47 modules, 203 experiments, 25 domains + 6 ODE flat + 3 layout + 13 GPU-promoted |
| GPU validation | 710+/710+ PASS — 79 ToadStool primitives (S68+ `e96576ee`, always-on), 0 local WGSL (fully lean) |
| Dispatch validation | 35/35 PASS — 5 substrate configs (Exp080) |
| BarraCuda CPU parity | 407/407 — 22.5x Rust speedup over Python (v1–v9) |
| BarraCuda GPU parity | 29 domains (Exp064/087/101/164) — pure GPU math proven |
| Pure GPU streaming | 152 checks — analytics (Exp105), ODE+phylo (Exp106), 441-837× vs round-trip (Exp090/091) |
| metalForge cross-system | 37 domains CPU↔GPU (Exp103+104+165) + dispatch (Exp080) + pipeline (Exp086) + PCIe (Exp088) |
| Cross-spring spectral | 25 checks — Anderson localization + QS-disorder analogy (Exp107) |
| Finite-size scaling | 14 checks — W_c = 16.26, disorder-averaged L=6–12 (Exp150) |
| Correlated disorder | 8 checks — biofilm clustering shifts W_c > 28 (Exp151) |
| Rust modules | 47 CPU + 42 GPU, 1,022 tests (896 barracuda lib + 60 integration + 19 doc + 47 forge) |
| Write phase | 0 local WGSL (fully lean) |
| Dependencies | 2 runtime (flate2 + bytemuck), everything else sovereign |
| Paper queue | **ALL DONE** — 52/52 reproducible papers complete (Tracks 1-4 + Phase 37 extensions + cross-spring) |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 1c) | R. Anderson (Carleton) — deep-sea metagenomics, population genomics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Fifty delivered (v1–v50) |

---

## Validation Chain (Per-Paper Status)

Every paper in the queue goes through the full evolution path. Status:

| Stage | What It Proves | Status |
|-------|---------------|--------|
| **Python baseline** | Algorithm correctness against published tools | 42 scripts, all reproducible |
| **BarraCuda CPU** | Pure Rust math matches Python | 1,476 checks, 380/380 cross-domain parity (v1–v8) |
| **BarraCuda GPU** | GPU produces same answer as CPU | 1,578 checks, 29 GPU domains |
| **Pure GPU streaming** | Zero CPU round-trips, data stays on-device | 152 checks, 10+ domains, 441-837× over round-trip (Exp090/105/106) |
| **metalForge mixed** | Same answer on CPU, GPU, NPU — substrate-independent | 37 domains, 39/39 papers three-tier (Exp103/104/165/182) |

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
| 44 | Martínez-García 2023 QS-pore geometry | Y | Y | Y | Exp170 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 45 | Feng 2024 pore-size diversity | Y | Y | Y | Exp171 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 46 | Mukherjee 2024 distance colonization | Y | Y | Y | Exp172 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 47 | Islam 2014 Brandt farm soil health | Y | Y | Y | Exp173 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 48 | Zuber & Villamil 2016 meta-analysis | Y | Y | Y | Exp174 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 49 | Liang 2015 long-term tillage | Y | Y | Y | Exp175 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 50 | Tecon & Or 2017 biofilm-aggregate | Y | Y | Y | Exp176 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 51 | Rabot 2018 structure-function | Y | Y | Y | Exp177 (CPU), Exp180 (GPU), Exp182 (metalForge) |
| 52 | Wang 2025 tillage microbiome | Y | Y | Y | Exp178 (CPU), Exp180 (GPU), Exp182 (metalForge) |

**Full three-tier coverage (CPU + GPU + metalForge):** All **39 of 39 actionable papers** (25 Tracks 1-2 + 5 Track 3 + 9 Track 4).
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
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Complete | 52/52 papers reproduced across 6 tracks + cross-spring |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |

### Data & Infrastructure

| Spec | Status | Description |
|------|--------|-------------|
| [DATA_TYPES.md](DATA_TYPES.md) | Active | Biological data type catalog — NestGate evolution primer. Profiles every format (FASTQ, FASTA, FAST5, POD5, mzML, Newick, etc.), biological entity type, taxonomy representation, and NestGate gap analysis. Drives data primal evolution. |
| [FIELD_GENOMICS_REQUIREMENTS.md](FIELD_GENOMICS_REQUIREMENTS.md) | Active | Systems needed for Sub-thesis 06 (field genomics). New BarraCUDA modules (`io::nanopore`, `bio::basecall`, `io::minknow`), experiment plan (Exp196-202), hardware requirements, metalForge sequencer substrate, NestGate integration points. |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | 203 experiments, 4,800+ validation checks, 1,022 tests |
| EVOLUTION_READINESS.md | `../barracuda/` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| Handoff (V61) | `../wateringHole/handoffs/WETSPRING_TOADSTOOL_V61_NANOPORE_FIELD_GENOMICS_HANDOFF_FEB27_2026.md` | Current ToadStool handoff |
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
- **NPU edge inference** — ESN reservoir on AKD1000, online evolution, adaptive sampling (V60)
- **Field genomics** (planned) — Nanopore sequencing + NPU classification + metalForge routing. See [FIELD_GENOMICS_REQUIREMENTS.md](FIELD_GENOMICS_REQUIREMENTS.md)
- **Data type profiling** — Biological data format catalog driving NestGate data primal evolution. See [DATA_TYPES.md](DATA_TYPES.md)
- **Sovereign Rust bioinformatics** — 47 CPU + 42 GPU modules + 0 local WGSL (fully lean), 2 runtime dependencies (flate2 + bytemuck), 79 ToadStool primitives (S68, always-on, zero fallback)

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

**Data primal / NestGate evolution**:
DATA_TYPES.md → FIELD_GENOMICS_REQUIREMENTS.md → `../whitePaper/baseCamp/sub_thesis_06_field_genomics.md`

**Integration partner**:
`../wateringHole/handoffs/WETSPRING_TOADSTOOL_V61_NANOPORE_FIELD_GENOMICS_HANDOFF_FEB27_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
