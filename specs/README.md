# wetSpring Specifications

**Last Updated**: February 22, 2026
**Status**: Phase 20 — 1,291/1,291 CPU + 451/451 GPU = 1,742/1,742 checks, ALL PASS
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| CPU validation | 1,291/1,291 PASS — 41 modules, 77 experiments, 25 domains |
| GPU validation | 451/451 PASS — 23 ToadStool primitives, 1 local WGSL shader, 18 GPU binaries |
| BarraCUDA CPU parity | 157/157 — 22.5x Rust speedup over Python |
| BarraCUDA GPU parity | 8 consolidated domains (Exp064) — pure GPU math proven |
| metalForge cross-system | 8 domains CPU↔GPU proven (Exp065) — substrate-independent |
| Rust modules | 41 CPU + 20 GPU, 707 tests (96.21% coverage) |
| Dependencies | 1 runtime (flate2), everything else sovereign |
| Paper queue | **ALL DONE** — 29/29 reproducible papers complete (Track 1c added) |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 1c) | R. Anderson (Carleton) — deep-sea metagenomics, population genomics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Eight delivered (Feb 16, 17, 19 v1-v3, 20 v4-v5, 21 v6, 22 rewire) |

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
| CONTROL_EXPERIMENT_STATUS.md | `../` | 77 experiments, 1,742 validation checks |
| EVOLUTION_READINESS.md | `../barracuda/` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| HANDOFF (v6) | `../` | Current consolidated ToadStool handoff |
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
`../HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
