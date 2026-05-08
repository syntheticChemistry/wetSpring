# wetSpring V153 — Deep Debt Evolution Handoff

**Date:** May 8, 2026
**Phase:** 60+ Absorption — Post-Parity Audit Deep Debt
**From:** wetSpring
**To:** primalSpring, primal teams, all springs

---

## Executive Summary

wetSpring deep debt evolution pass: hardcoded configuration → env-configurable,
shared validation helpers extracted from >800-line files, doc drift corrected,
missing Kachkovskiy spectral-QS notebook created, and this comprehensive
evolution handoff documenting benchmark landscape, dataset roadmap, and
downstream alignment.

**Metrics snapshot:**

| Metric | Value |
|--------|-------|
| Lib tests | 252 pass, 1 ignored, 0 fail |
| Validation binaries | 342 |
| Experiment crates | 56+ directories |
| Paper notebooks | 11 (.ipynb in notebooks/papers/) |
| Meta notebooks | 5 (.ipynb in notebooks/) |
| Science IPC methods | 19 across 6 categories |
| Registered capabilities | 34 wetSpring domain methods |
| Papers reproduced | 63/63 (all done) |
| Unsafe code | Zero |
| TODOs/FIXMEs in source | Zero |
| Production mocks | Zero (all in #[cfg(test)]) |

---

## Phase 1: Hardcoding → Capability-Based Config

### What Changed

| File | Before | After |
|------|--------|-------|
| `wetspring_science_facade.rs` | Hardcoded `"127.0.0.1:3100"` and `"https://primals.eco"` literals | `DEFAULT_BIND` / `DEFAULT_CORS_ORIGIN` constants; env vars `FACADE_BIND`, `FACADE_CORS_ORIGIN` |
| `facade/provenance.rs` | Hardcoded `https://lab.primals.eco/api/v1/provenance/verify/{braid_id}` | `PROVENANCE_VERIFY_BASE` env var with fallback |
| `ncbi/efetch.rs` | Already had `EFETCH_BASE_DEFAULT` + env override | Verified consistent pattern ✓ |
| `ncbi/entrez.rs` | Already had `ENTREZ_BASE_DEFAULT` + env override | Verified consistent pattern ✓ |
| `ipc/handlers/data_fetch.rs` | ChEMBL/PubChem URLs as format strings | Retained — these are data source identifiers routed through NestGate, not server config |

### Pattern

All server-facing configuration (bind addresses, CORS origins, verification
URLs) reads from environment variables with sensible defaults. Data source
URLs (ChEMBL, PubChem, NCBI) are domain identifiers embedded in the fetch
logic — they identify *what* to fetch, not *how* to connect.

---

## Phase 2: Large File Refactoring

### Shared Validation Infrastructure Extracted

New types in `validation::timing`:

| Type | Purpose | Replaces |
|------|---------|----------|
| `CpuGpuRow` | Per-domain CPU vs GPU timing row | Local `Timing` in `validate_cpu_vs_gpu_all_domains.rs` |
| `CrossSpringEntry` | Per-primitive cross-spring benchmark with provenance | Local `BenchEntry` in `validate_cross_spring_s57.rs` |
| `print_cpu_gpu_table()` | Formatted CPU/GPU comparison table | Per-file println! formatting |
| `print_cross_spring_table()` | Cross-spring evolution summary table | Per-file println! formatting |

New shared function: `bench_print()` (already existed) — replaced local
`bench()` shadow in `validate_cross_spring_evolution_s87.rs`.

### File Status

| File | Lines | Change |
|------|:-----:|--------|
| `validate_cross_spring_s57.rs` | 924 (was 933) | `BenchEntry` → `CrossSpringEntry` from shared module |
| `validate_cpu_vs_gpu_all_domains.rs` | 913 (was 920) | `Timing` → `CpuGpuRow` from shared module |
| `validate_cross_spring_evolution_s87.rs` | 877 (was 878) | Local `bench()` → `bench_print()` |
| `benchmark_cross_spring_s65.rs` | 823 | Already uses `BenchRow`, `bench_print`, `print_bench_table` ✓ |
| `validate_anderson_qs_environments_v1.rs` | 817 | Domain-specific physics — no extractable infra |

These are validation binaries with sequential domain-by-domain logic. The
remaining line counts come from domain-specific validation code (each domain
has unique test data, tolerances, and assertions). Splitting further would
reduce readability without reducing complexity.

---

## Phase 3: Doc Drift Fixes

### gonzales.md (whitePaper/baseCamp/)

Updated paper status table from "Proposed" to "DONE" for all 6 papers
(53–58), with experiment references and check counts:

- Paper 53 (Gonzales 2013): Exp282, 15/15 PASS
- Paper 54 (Gonzales 2014): Exp280, 35/35 PASS
- Paper 55 (Gonzales 2016): Exp281, 19/19 PASS
- Paper 56 (Fleck/Gonzales 2021): Exp281, 19/19 PASS
- Paper 57 (Gonzales 2024): Exp280, 35/35 PASS
- Paper 58 (McCandless 2014): Exp281, 19/19 PASS

Added cross-reference to Exp273-286 (immuno-Anderson framework, CPU parity,
GPU validation, streaming, cross-substrate).

### PRIMAL_GAPS.md (docs/)

- Updated header to V153 with Phase 60+ context
- PG-04 (NestGate): Updated from "Declared but Not Wired" to "IPC Wired,
  Awaiting Live Deployment" — all fetch routing now through NestGate composition
- PG-05 (toadStool): Updated from "Discovery Helper but No Active Calls" to
  "IPC Discovery + barraCuda Optional" — barraCuda marked optional in Cargo.toml
- Summary table updated for PG-04 and PG-05

---

## Phase 4: Kachkovskiy Spectral-QS Notebook

Created `notebooks/papers/kachkovskiy-spectral-qs.ipynb` — the 11th paper
notebook, covering:

1. 1D Anderson Hamiltonian — construction, Gershgorin bounds, eigenvalue computation
2. Lyapunov exponent — transfer matrix, Kappus–Wegner approximation
3. Level spacing ratio — localized (Poisson) vs extended (GOE) diagnosis
4. Almost-Mathieu / Aubry–André — quasiperiodic potential, Herman formula
5. QS-Disorder analogy — 8 ecosystems, Pielou evenness → Anderson disorder
6. Dimensional phase diagram — 1D/2D/3D sweep with QS window comparison
7. Lanczos eigensolver — cross-spring primitive verification
8. 28-Biome QS atlas — global QS propagation potential map
9. Tier 2 IPC stubs for live barracuda validation

Covers experiments: 107, 113, 119, 122, 126-130, 135-138, 144-149 (312 total checks).

---

## Benchmark Landscape

### Rust Benchmarks (23 binaries in `barracuda/src/bin/benchmark_*.rs`)

| Category | Binaries | Focus |
|----------|:--------:|-------|
| Cross-spring evolution | 5 | S57, S65, S79, S86, S87 ToadStool primitives |
| CPU vs GPU parity | 3 | All 16 domains, per-domain timing |
| Science domain | 8 | Diversity, ODE, alignment, phylogenetics |
| Infrastructure | 4 | IPC latency, NestGate throughput |
| metalForge cross-substrate | 3 | CPU ↔ GPU ↔ cross-hardware |

### Python Benchmarks (3 scripts)

| Script | Purpose |
|--------|---------|
| `benchmark_python_baseline.py` | 23-domain Python timing reference |
| `benchmark_rust_vs_python.py` | Head-to-head Rust vs Python comparison |
| `barracuda_cpu_v4_baseline.py` | barraCuda CPU parity (v4 surface) |

### Industry Standard References

| Standard | Coverage | Notes |
|----------|----------|-------|
| Galaxy/QIIME2 | Exp001-004, Exp014 | Full 16S pipeline parity (92 checks) |
| DADA2 | `r-industry-parity.ipynb` | Error model reproduction in Python |
| R/vegan | `r-industry-parity.ipynb` | Shannon, Simpson, Bray-Curtis parity |
| phyloseq | `r-industry-parity.ipynb` | UniFrac distance computation |
| SciPy | Notebooks (Waters, Liu, Anderson) | ODE solvers, alignment, HMM |
| Kokkos | Not applicable | hotSpring domain (QCD/HPC) |

### What We Don't Have (And Why)

- **Criterion benchmark harness**: Would be nice for micro-benchmarks but the
  23 timing binaries + 3 Python scripts provide adequate cross-validation.
  Evolution target for a future phase.
- **Kokkos GPU benchmarks**: Kokkos is for HPC (hotSpring). wetSpring uses
  wgpu/compute shaders for GPU, not CUDA/HIP. Not comparable.

---

## Dataset Roadmap

### Active (In Use)

| Dataset | Source | Usage |
|---------|--------|-------|
| ChEMBL bioactivity | EBI (CHEMBL2103874) | Gonzales JAK inhibitor panel |
| PubChem assays | NCBI | Drug activity cross-validation |
| NCBI 16S amplicons | SRA (PRJNA488170 etc.) | Sovereign pipeline validation |
| Galaxy test data | usegalaxy.org | Pipeline parity reference |

### P0 Extension (Next Phase — Requires NestGate Live)

| Dataset | BioProject | Why |
|---------|-----------|-----|
| HMP gut/oral | PRJNA43021 | 16S diversity → Anderson disorder mapping |
| Tara Oceans | PRJEB1787 | Marine surface/deep QS atlas validation |
| Earth Microbiome | PRJEB11419 | 30,000+ samples, 96 environments |
| Lake Erie HAB | PRJNA649075 | Bloom surveillance time-series |
| Cold seep metagenomes | (2025 Microbiome) | 170 samples, 34 QS types |
| KBS LTER soil | PRJNA??? | Long-term tillage experiment |

### Reference Only (Published Values, No Raw Data Needed)

| Source | Used For |
|--------|----------|
| Gonzales 2014 Table 1 (IC50) | Hill model parameterization |
| Fleck/Gonzales 2021 PK data | Exponential decay modeling |
| Waters 2008 ODE parameters | QS c-di-GMP dynamics |
| Liao ADREC kinetics | Gompertz/Monod/Haldane models |

---

## Downstream Alignment

### projectNUCLEUS (sporeGarden)

wetSpring provides workloads for projectNUCLEUS's "ironGate" deployment:

| NUCLEUS Layer | wetSpring Contribution |
|---------------|----------------------|
| Tower (BearDog + Songbird) | Crypto hashing, consent-gated data access |
| Node (barraCuda + toadStool) | 16 GPU-accelerated science domains |
| Nest (NestGate + Provenance Trio) | External data fetch, content-addressed storage |
| Cross-atomic | Hash → Store → Retrieve → Science pipeline (exp400) |

exp400 validates the full NUCLEUS stack from wetSpring's niche perspective.

### foundation (sporeGarden)

wetSpring's provenance chain (Paper → Python → Rust → NUCLEUS composition)
is the canonical example of foundation's "scientific knowledge lineage" thread:

| foundation Thread | wetSpring Implementation |
|-------------------|------------------------|
| Data provenance | BLAKE3 content-addressed, NestGate stored |
| Computation lineage | Python baseline → Rust validation → IPC parity |
| Attribution | 63 papers, 6 faculty, cross-spring shader provenance |
| NFT vertices | Each validated computation = gAIa vertex candidate |

---

## What's NOT Needed

| Item | Reason |
|------|--------|
| Unsafe code elimination | Zero unsafe code in wetSpring |
| Mock isolation | All mocks in `#[cfg(test)]` — no production mocks |
| External dep replacement | All deps pure Rust (wgpu is the only non-pure-Rust, no alternative for GPU compute) |
| TODO/FIXME cleanup | Zero found in `barracuda/src/**/*.rs` |
| Kokkos benchmarks | hotSpring domain, not applicable to life science |

---

## Remaining Evolution Targets

| Target | Priority | Notes |
|--------|----------|-------|
| Criterion micro-benchmarks | Low | 23 binaries adequate; criterion adds CI polish |
| Full EMP/SRA ingestion | Blocked | Requires NestGate + Provenance Trio live |
| Level 5 guideStone | Medium | Requires full IPC evaporation (PG-09) |
| Exp403 v0.9.17 migration | Medium | Legacy surface cleanup (PG-12) |
| petalTongue server-side render | Low | Tier 3 notebook evolution |
| 2D/3D Anderson GPU benchmarks at scale | Medium | N > 10,000 lattice sites |

---

*This document is maintained by wetSpring and fed back to primalSpring via
`wateringHole/handoffs/` per the NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
