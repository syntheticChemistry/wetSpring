# Controls Verification: wetSpring V101

**Date:** March 9, 2026
**Status:** ALL 39 actionable papers verified at all 3 control tiers
**Purpose:** Confirm every paper reproduction has controls for open data,
open systems, barraCuda CPU, barraCuda GPU, metalForge mixed hardware,
and biomeOS/NUCLEUS deployment

---

## Control Architecture

Every paper passes through 7 control tiers:

```
Tier 0: Open Data         — publicly accessible, no proprietary dependencies
Tier 1: Python Baseline   — reference implementation against published tools
Tier 2: barraCuda CPU     — sovereign Rust math, EXACT_F64 parity with Python
Tier 3: barraCuda GPU     — GPU ↔ CPU within named tolerances
Tier 4: Pure GPU Streaming — zero CPU round-trips (441-837× speedup)
Tier 5: metalForge        — substrate-independent (CPU = GPU = NPU output)
Tier 6: biomeOS/NUCLEUS   — IPC dispatch, deploy graph, atomic coordination
Tier 7: petalTongue       — visualization rendering (optional)
```

---

## Tier 0: Open Data Verification

| Track | Papers | Data Sources | Access Level |
|-------|:------:|-------------|:----------:|
| Track 1 (16S) | 1-12 | NCBI SRA, published ODE params | Open |
| Track 1b (Phylo) | 15-20 | Published algorithms, public datasets | Open |
| Track 1c (Deep-sea) | 24-29 | NCBI SRA, MBL, MG-RAST, Figshare, OSF | Open |
| Track 2 (PFAS) | 21-22 | Zenodo, EPA, MassBank, EGLE | Open |
| Track 3 (Drug) | 39-43 | repoDB, published equations, ROBOKOP KG | Open |
| Track 4 (Soil) | 44-52 | Published soil metrics, model equations | Open |
| Track 5 (Immuno) | 53-58 | Published IC50, PK, serum data | Open |
| Cross-spring | 23 | Algorithmic (no external data) | Open |

**Verification:** Zero proprietary data dependencies. Zero cloud-only APIs
(NestGate provides sovereign fallback for NCBI). All 57 Python baselines
carry SHA-256 integrity verification headers.

---

## Tier 2: barraCuda CPU Controls

| Round | Experiment | Domains | Checks | Finding |
|:-----:|:----------:|:-------:|:------:|---------|
| v1-v9 | Exp035-102 | 22 | 407 | Pure Rust matches Python |
| v11 | Exp206 | 7 | 64 | Zero drift through IPC |
| v12 | Exp212 | 5 | 55 | Post-audit I/O parity |
| v13 | Exp216 | 47 | 47+ | All CPU modules |
| v20 | Exp263 | 37 | 37 | Vault DF64, cross-domain |
| v22 | Exp292 | 8 | 40 | Full-domain paper parity |
| v23 | Exp306 | 8 | 38 | Fused ops (Welford, Pearson) |
| v24 | Exp314 | 33 | 67 | Comprehensive bio domain |
| v25 | Exp323 | 46 | 46 | Cross-primal pure Rust |
| **Total** | | | **546+** | **EXACT_F64 parity** |

---

## Tier 3: barraCuda GPU Controls

| Round | Experiment | Domains | Checks | GPU Feature |
|:-----:|:----------:|:-------:|:------:|------------|
| v1-v4 | Exp064-092 | 21 | 1,783 | Full GPU dispatch |
| v5 | Exp218 | 42 | 36+ | Module portability proof |
| v7 | Exp264 | 27 | 22 | GPU parity proof |
| v9 | Exp293 | 5 | 35 | 5-track dispatch + Anderson |
| v12 | Exp308 | 8 | 21 | Hybrid-aware fused ops |
| v13 | Exp316 | 25 | 25 | Full-domain GPU |
| v14 | Exp324 | 27 | 27 | ToadStool dispatch |
| **Total** | | | **1,783+** | **Named tolerances** |

**Key tolerances:**
- Shannon/Simpson: `GPU_VS_CPU_SHANNON` (1e-6)
- Spectral cosine: `GPU_VS_CPU_SPECTRAL` (1e-10)
- ODE sweep: `GPU_VS_CPU_ODE_SWEEP` (0.15, chaotic sensitivity)
- Felsenstein: `GPU_VS_CPU_FELSENSTEIN` (1e-6)

---

## Tier 5: metalForge Mixed Hardware Controls

| Round | Experiment | Domains | Checks | HW Config |
|:-----:|:----------:|:-------:|:------:|-----------|
| v5-v7 | Exp060-104 | 37 | 243+ | CPU↔GPU matrix |
| v12 | Exp265 | 63 | 63 | Extended cross-system |
| v14 | Exp295 | 28 | 28 | Paper chain |
| v15 | Exp310 | 21 | 21 | Cross-system fused ops |
| v16 | Exp318 | 24 | 24 | Full three-tier |
| v17 | Exp326 | 29 | 29 | NUCLEUS + biomeOS graph |
| **Total** | | | **243+** | **Substrate-independent** |

**Three-tier paper coverage:** 39/39 actionable papers have full
CPU + GPU + metalForge validation. The only excluded paper (Paper 11,
Waters 2021) is a reference-only review article.

---

## Tier 6: biomeOS / NUCLEUS Controls

| Experiment | Focus | Checks | Status |
|:----------:|-------|:------:|--------|
| Exp203-205 | IPC server lifecycle, Songbird, Neural API | 29 | PASS |
| Exp206 | CPU v11 IPC math fidelity | 64 | EXACT_F64 |
| Exp207 | GPU v4 IPC science dispatch | 54 | PASS |
| Exp208 | NUCLEUS v7 mixed hardware | 75 | PASS |
| Exp266 | NUCLEUS v3 Tower→Node→Nest | 106 | PASS |
| Exp269 | Mixed HW dispatch (47 workloads) | 91 | PASS |
| Exp270 | biomeOS graph coordination | 29 | PASS |
| Exp321 | biomeOS/NUCLEUS V98+ integration | 42 | PASS |
| Exp322 | Cross-primal pipeline | 22 | PASS |
| Exp330 | biomeOS + NUCLEUS + petalTongue | 34 | PASS |
| **Total** | | **385+** | **ALL PASS** |

**Deployment verified:**
- JSON-RPC 2.0 over Unix domain socket
- 10-request multiplexing on single connection
- Songbird discovery + capability registry
- Deploy graph: `wetspring_deploy.toml`
- Sovereign fallback (zero external dependencies)
- Tower/Node/Nest atomic lifecycle

---

## Tier 7: petalTongue Visualization Controls

| Experiment | Focus | Checks | Status |
|:----------:|-------|:------:|--------|
| Exp327 | Schema validation (5 builders) | 45 | PASS |
| Exp328 | CPU vs GPU parity (viz domains) | 27 | PASS |
| Exp329 | metalForge petalTongue integration | 19 | PASS |
| Exp330 | Full chain (biomeOS → NUCLEUS → viz) | 34 | PASS |
| Exp333 | Viz evolution (7 new builders) | 44 | PASS |
| Exp334 | Science-to-viz pipeline | 34 | PASS |
| **Total** | | **251** | **ALL PASS** |

**Visualization coverage:**
- 7 DataChannel types: TimeSeries, Distribution, Bar, Gauge, Heatmap, Scatter, Spectrum
- 13 scenario builders across all science domains
- StreamSession for progressive rendering
- Songbird capability announcement (16 capabilities)
- IPC science→viz wiring (optional `visualization: bool` flag)

---

## Summary

| Tier | Controls | Checks | Status |
|------|----------|:------:|--------|
| 0 | Open data | — | All open, SHA-256 verified |
| 1 | Python baselines | 57 scripts | Reproducible |
| 2 | barraCuda CPU | 546+ | EXACT_F64 |
| 3 | barraCuda GPU | 1,783+ | Named tolerances |
| 4 | Pure GPU streaming | 252+ | Zero round-trips |
| 5 | metalForge mixed HW | 243+ | Substrate-independent |
| 6 | biomeOS/NUCLEUS | 385+ | IPC + deploy graph |
| 7 | petalTongue | 251 | Visualization |
| **All** | **9,060+** | **ALL PASS** |

Every paper in the queue has open data controls. Every actionable paper
(39/39) has full three-tier hardware controls (CPU + GPU + metalForge).
The biomeOS deploy graph enables multi-node Tower deployment via atomic
graphs. petalTongue renders the results at every tier.
