# wetSpring Specifications

**Last Updated**: February 19, 2026
**Status**: Phase 6 complete — 388/388 CPU + 38/38 GPU = 426/426 checks
**Domain**: Life science (16S, metagenomics), analytical chemistry (LC-MS, PFAS), microbial signaling

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 2 (Rust CPU) | 388/388 PASS — FASTQ, diversity, 16S pipeline, mzML, PFAS, features, peaks, real NCBI data, VOC baselines, public benchmarks (4 BioProjects) |
| Phase 3 (GPU) | 38/38 PASS — Shannon, Simpson, BC, PCoA, spectral cosine (1,077x speedup) |
| Phase 5 (Paper Parity) | Honest data audit, VOC baselines from Reese 2019, proxy NCBI data |
| Phase 6 (Public Benchmark) | 22 samples from 4 independent BioProjects, 2.7M reads, marine + freshwater |
| Rust modules | 30 sovereign modules, 284 tests |
| Dependencies | 1 runtime (flate2), everything else sovereign |
| Faculty (Track 1) | Waters (MMG, MSU), Cahill (Sandia), Smallwood (Sandia) |
| Faculty (Track 1b) | Liu (CMSE, MSU) — comparative genomics, phylogenetics |
| Faculty (Track 2) | Jones (BMB/Chemistry, MSU) — PFAS mass spectrometry |
| Handoffs | Two delivered (Feb 16, Feb 17) |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Active | Papers to review/reproduce across 3 tracks |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements and gap analysis |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | Detailed experiment logs (995 lines) |
| EVOLUTION_READINESS.md | `../` | Module-by-module GPU promotion assessment |
| BENCHMARK_RESULTS.md | `../` | CPU vs GPU performance benchmarks |
| HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md | `../` | First ToadStool handoff |
| HANDOFF_SPRINGS_TO_TOADSTOOL_FEB_17_2026.md | `../` | Combined springs handoff |
| whitePaper/STUDY.md | `../whitePaper/` | Full study narrative |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Two-track validation protocol |
| experiments/README.md | `../experiments/` | Galaxy experiment guide |

---

## Scope

### wetSpring IS:
- **16S pipeline validation** — FASTQ → quality → merge → derep → DADA2 → chimera → taxonomy → diversity → UniFrac
- **LC-MS feature extraction** — mzML → EIC → peaks → features → spectral matching
- **PFAS screening** — KMD + tolerance search + MS2 fragment matching
- **Microbial ecology** — Alpha/beta diversity, PCoA, rarefaction
- **Sovereign Rust bioinformatics** — 30 modules, 1 runtime dependency

### wetSpring IS NOT:
- Sensor noise analysis (groundSpring)
- Neural network training (neuralSpring)
- Physics simulation (hotSpring)
- ET₀/irrigation (airSpring)

### wetSpring EXTENDS TO (via faculty):
- **Waters**: c-di-GMP signaling dynamics, quorum sensing, biofilm regulation, phage defense
- **Liu**: Comparative genomics, phylogenetic placement, introgression detection, cophylogenetics
- **Cahill/Smallwood**: Algal pond metagenomics, phage biocontrol
- **Jones**: PFAS fate-and-transport, high-resolution mass spec methods

---

## Reading Order

**New to wetSpring** (20 min):
1. This README (5 min)
2. `../whitePaper/README.md` — overview and key results (10 min)
3. PAPER_REVIEW_QUEUE.md — what's next (5 min)

**Deep dive** (2 hours):
`../whitePaper/STUDY.md` → `../CONTROL_EXPERIMENT_STATUS.md` → `../EVOLUTION_READINESS.md` → BARRACUDA_REQUIREMENTS.md

**Integration partner**:
`../HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md` → `../BENCHMARK_RESULTS.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All wetSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using wetSpring code, must publish source under the same license.
