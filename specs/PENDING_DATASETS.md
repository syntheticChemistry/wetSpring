<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring — Pending Datasets & Downloads

**Last Updated**: May 16, 2026 (V169)
**Purpose**: Consolidated tracker for datasets referenced but not yet downloaded,
hardware not yet available, or GPU shaders not yet written.

---

## Data Downloads Pending

| Dataset | Source | Experiment | Blocked By | Priority |
|---------|--------|------------|------------|----------|
| EPA UCMR 5 — national drinking water PFAS survey (2023-2025) | EPA | Exp041 | Download + parse | Medium |
| EPA PFOS surface water — GPS + concentration time-series | EPA | Exp041 | Download + parse | Medium |
| Jones Lab full PFAS library (175 compounds) | Jones Lab / Zenodo | Exp018 | Data access | Low |
| NCBI PRJNA294072 — 264 *E. coli* LTEE genomes (Tenaillon 2016) | NCBI SRA | Exp380 (B7) | **B7 Tier 2 COMPLETE** — download for sovereign pipeline | Medium |

## LTEE Queue — Datasets Needed for Queued Papers

| ID | Paper | Dataset | Source | Status |
|----|-------|---------|--------|--------|
| B1 | Barrick 2009 | LTEE mutation calls (12 lineages × 40K gen) | Barrick Lab / NCBI | Queued |
| B2 | Wiser 2013 | Fitness trajectory data (50K generations) | Lenski Lab / Dryad | Queued |
| B3 | Good 2017 | Allele frequency time-series (metagenomic) | NCBI SRA | Queued |
| B4 | Blount 2012 | Cit+ lineage genome sequences | NCBI | Queued |
| B5 | Leonard 2024 | Bee gut symbiont engineering constructs | mBio supplement | Queued |
| B6 | BioBrick burden 2024 | 301 plasmid burden measurements | Nature Comms supplement | Queued |
| B8 | Barrick & Waters 2025 | Phage contingency loci HMM data | bioRxiv supplement | Queued |
| E1 | Woldring 2017 | scFab sitewise diversity data | Biochemistry supplement | Queued |
| E5 | Woldring Lab 2023 | Single-cell scFab library sequences | Lab data / NCBI | Queued |

## Hardware Pending

| Hardware | Experiment | What It Enables | Status |
|----------|------------|-----------------|--------|
| BrainChip AKD1000 NPU | Exp083, Exp188 | Real int8 FC dispatch + sentinel stream | Hardware procurement |
| Titan V (GV100) GPU | Exp215 | CPU vs GPU v5 I/O evolution on Volta arch | Hardware access |

## GPU Shaders Not Yet Written

| Shader | Experiment | Domain | Blocked By |
|--------|------------|--------|------------|
| `hash_table_u64.wgsl` or histogram shader | Exp081 | k-mer histogram | WGSL design |
| Tree-propagation WGSL | Exp082 | UniFrac flat tree | WGSL design |
| DF64 Phase 2 shaders | Exp187 | Anderson large lattice | DF64 kernel evolution |

## Foundation Thread Data Contributions

wetSpring contributes to foundation threads 1, 3, 4, 5, 6, 7. Status:

| Thread | Sources | Targets | Validated | Notes |
|--------|---------|---------|-----------|-------|
| 1 (WCM) | Anchored | Anchored | Partial | ABG community data |
| 3 (Immuno) | wetSpring Gonzales data | 12 targets | 0 | Blocked on upstream NestGate |
| 4 (Enviro) | 23 sources seeded | 40 targets (incl. 4 LTEE B7) | 0 LTEE targets | B7 Tier 2 complete — targets await lithoSpore integration |
| 5 (LTEE) | Via groundSpring | 18 targets | 0 wetSpring-specific | B7 Tier 2 complete, feeds module 6 |
| 6 (Ag) | Via airSpring | 36 targets | 36 (airSpring) | Complete |
| 7 (Anderson) | 11 sources | 23 targets | Partial | Cross-cutting |
