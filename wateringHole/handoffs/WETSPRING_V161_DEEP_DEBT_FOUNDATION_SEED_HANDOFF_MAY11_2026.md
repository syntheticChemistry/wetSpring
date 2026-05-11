<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# wetSpring V161 — Deep Debt Audit Clean, Foundation Seeded, Zero Internal Gaps

**Date:** May 11, 2026
**Version:** V161
**From:** wetSpring
**To:** primalSpring, barraCuda, projectNUCLEUS, foundation teams

---

## Summary

V161 resolves the last wetSpring-internal primal gap (PG-12) and seeds
foundation Thread 04 with 36 validated targets. A comprehensive deep debt
audit confirms zero actionable debt across all dimensions.

**Metrics:** 1,962 tests, 0 failed. 6 gaps open (all external), 16 resolved/closed.
Zero wetSpring-internal gaps remain. 63/63 papers reproduced. guideStone Level 4.

---

## PG-12 Resolution

The 15 legacy Exp403 methods are now separated into
`niche::CONSUMED_CAPABILITIES_LEGACY`, distinct from the canonical
`CONSUMED_CAPABILITIES`. This makes the distinction machine-readable
for CI, composition tools, and primalSpring scorecard validation.

---

## Foundation Thread 04 — Environmental Genomics Seeded

Created `data/targets/thread04_enviro_targets.toml` with **36 validated
targets** from wetSpring's 63 paper reproductions:

| Category | Targets | Source Experiments |
|----------|---------|-------------------|
| 16S diversity (Shannon, Simpson, Chao1) | 4 | Exp001, Exp012, Exp015 |
| Anderson localization (3D, 2D, W_c) | 4 | Exp020, Exp122, Exp127, Exp131 |
| ODE solvers (RK4, Gillespie) | 2 | Exp020, Exp022/023 |
| Phylogenetics (RF, SW, NJ) | 3 | Exp021, Exp028, Exp033 |
| PFAS chemistry (PFOA, PFOS, peaks) | 3 | Exp018, Exp010 |
| Cold seep metagenomics | 2 | Exp144, Exp185 |
| Drug repurposing (NMF, Fajgenbaum) | 2 | Exp160, Exp157 |
| Gonzales dermatitis (IC50, PK) | 2 | Exp280, Exp281 |
| HMM (forward, Viterbi) | 2 | Exp026 |
| Soil & agriculture (no-till, pore QS) | 2 | Exp170-182 |
| R/Python tool parity | 2 | Exp335, Exp253 |
| NPU sentinel (classifier, latency) | 2 | Exp114, Exp118 |
| Spectral matching (MassBank, EPA) | 2 | Exp042, Exp041 |
| QS biofilm threshold | 1 | Exp020 |

Updated `THREAD_INDEX.toml`: Thread 04 status → "seeded", `data_targets` path set.

---

## Deep Debt Audit — Comprehensive Results

| Check | Result |
|-------|--------|
| Unsafe code | **ZERO** — `forbid(unsafe_code)` workspace-wide |
| Production mocks | **ZERO** — all in `#[cfg(test)]` |
| `todo!()` / `FIXME` / `HACK` | **ZERO** |
| `.unwrap()` in library code | **ZERO** — all test-only |
| `#[allow()]` attributes | **ZERO** — `#[expect(reason)]` only |
| Hardcoded `/home/` paths | **ZERO** |
| Library files >800L | **ZERO** (4 bins 831-923L are validation narratives) |
| C-binding dependencies | **wgpu native GPU only** (deny.toml bans 15 crates) |
| Python/R CPU benchmarks | **6 bins** (v2-v5 + 23-domain + three-tier) — DONE |
| Industry parity baselines | Galaxy/QIIME2/scipy/numpy/R — all DONE |
| Paper queue | **63/63 complete** (3 PROPOSED: Exp377-379) |

---

## Upstream Guidance

### foundation
- Thread 04 now has 36 targets ready for `foundation_validate.sh` provenance
  wrapping. Next: `fetch_sources.sh` for BLAKE3 content hashes on the 23
  source accessions, then `expressions/ENVIRONMENTAL_GENOMICS.md`.

### primalSpring
- PG-10 remains: `method_to_capability_domain()` needs `"spectral" | "linalg"`
  in the `"tensor"` match arm.
- wetSpring has zero internal gaps — all remaining 6 are external team dependencies.

### barraCuda
- PG-17 remains: `tensor.matmul` handle-based API needs inline-data convenience.
- New IPC method suggestion: `stats.hill_sweep` for batch Hill equation evaluation.

### projectNUCLEUS
- 11 wetSpring workload TOMLs gate-agnostic and ready.
- plasmidBin binary: 1.4M stripped release (`infra/plasmidBin/springs/wetspring`).

---

## Datasets for Expansion (from audit)

| Priority | Dataset | Status | Next Step |
|----------|---------|--------|-----------|
| P0 | EMP real OTU tables | Synthetic in Exp256 | NestGate BIOM fetch |
| P0 | KBS LTER 31-year tillage | Referenced in Exp359 | SRA accession pin |
| P1 | SRA longitudinal atlas | Mapped in extension plan | Source TOML entries |
| P1 | AMR gene databases | Referenced in extension plan | Thread 04 source expansion |
| P2 | UniProt functional profiles | Not yet sourced | Thread 04 source expansion |

---

*This document is maintained by wetSpring and fed back to primalSpring via
the wateringHole handoff protocol.*
