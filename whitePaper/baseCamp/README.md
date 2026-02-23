# baseCamp: Per-Faculty Research Briefings

**Date:** February 23, 2026
**Project:** wetSpring (ecoPrimals)
**Status:** 120 experiments, 2,673+ validation checks, ALL PASS

---

## Overview

Each document in this directory summarizes what wetSpring reproduced, evolved,
and validated from one faculty member's published work. The evolution follows
a consistent five-stage path:

```
Python/Galaxy baseline
  -> Rust CPU (sovereign, 1 dependency)
    -> GPU acceleration (ToadStool/BarraCuda WGSL)
      -> Pure GPU streaming (zero CPU round-trips)
        -> metalForge cross-substrate (CPU = GPU = NPU output)
          -> NPU reservoir deployment (ESN → int8 → Akida AKD1000)
```

Every stage is validated with explicit numerical checks. All data is open.
All code is AGPL-3.0.

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domains |
|---------|------------|-------|:------:|:-----------:|:------:|---------|
| [Waters](waters.md) | MSU MMG | 1 | 7 | 020,022-025,027,030,108,114 | 133+ | QS, ODE, Gillespie, bistability, phage defense, NPU QS classifier |
| [Liu](liu.md) | MSU CMSE | 1b | 6 | 026,031-034,036-038,109,115 | 136 | HMM, phylogenetics, alignment, placement, NPU placement |
| [R. Anderson](anderson.md) | Carleton | 1c | 6 | 051-056,110,116 | 159 | Deep-sea metagenomics, pangenomics, NPU binning |
| [Jones](jones.md) | MSU BMB | 2 | 2 | 041,042,111,117 | 45 | PFAS mass spectrometry, EPA ML, NPU spectral pre-filter |
| [Cahill](cahill.md) | Sandia | 1 | 1 | 039,112,118 | 45 | Algal pond phage biocontrol, NPU bloom sentinel |
| [Smallwood](smallwood.md) | Sandia | 1 | 1 | 040,112,118 | 49 | Bloom surveillance, NPU bloom sentinel |
| [Kachkovskiy](kachkovskiy.md) | MSU CMSE | cross | 1 | 107,113,119 | 39 | Spectral theory, Anderson localization, NPU regime classifier |
| **Total** | | | **24** | | **572+** | |

### NCBI-Scale Extensions (Phase 32)

| Experiment | Faculty | Scale | Checks | GPU Prim |
|:---:|---------|-------|:------:|----------|
| Exp108 | Waters | 1024-genome QS parameter landscape | 8 | `BatchedOdeRK4F64` |
| Exp109 | Liu | 128-taxon placement + Felsenstein | 11 | NJ + Felsenstein |
| Exp110 | Anderson | 200-genome cross-ecosystem pangenome | 17 | ANI + dN/dS |
| Exp111 | Jones | 2048-spectrum spectral cosine (2.1M pairs) | 14 | `GemmF64` |
| Exp112 | Cahill/Smallwood | 1365-timepoint multi-ecosystem bloom | 23 | `FusedMapReduceF64` |
| Exp113 | Kachkovskiy | 8-ecosystem QS-disorder prediction | 5 | `barracuda::spectral` |

### NPU Reservoir Deployment (Phase 33)

| Experiment | Faculty | NPU Task | Checks | Key Finding |
|:---:|---------|----------|:------:|-------------|
| Exp114 | Waters | QS phase classifier (3-class) | 13 | 100% f64↔NPU agreement |
| Exp115 | Liu | Phylogenetic clade placement (8-class) | 9 | 97.7% quantization fidelity |
| Exp116 | Anderson | Genome ecosystem binning (5-class) | 9 | Int8 regularization effect (+6% acc) |
| Exp117 | Jones | Spectral library pre-filter (2048 spectra) | 8 | 84% top-10 overlap, 2-stage |
| Exp118 | Cahill/Smallwood | Bloom sentinel (4-state) | 11 | Coin-cell >1 year battery |
| Exp119 | Kachkovskiy | QS-disorder regime (3-regime) | 9 | Physics ordering preserved |

## Validation Chain

Every paper goes through the full evolution. Status across all 25 actionable papers:

| Stage | What It Proves | Coverage |
|-------|---------------|----------|
| Python baseline | Algorithm correctness against published tools | 40 scripts |
| BarraCuda CPU | Rust matches Python within machine precision | 1,476 checks, 22.5x faster |
| BarraCuda GPU | GPU matches CPU within 1e-6 | 702 checks, 29 domains |
| Pure GPU streaming | Zero CPU round-trips, data stays on-device | 152 checks, 10+ domains |
| metalForge | Same answer on CPU, GPU, NPU | 25/25 papers, 37 domains |
| NPU reservoir | ESN → int8 → NPU preserves classification | 59 checks, 6 domains |
| Cross-spring evolution | 612 WGSL shaders traced to origin springs, rewired imports | 9 checks |

## Performance Summary

| Metric | Value | Source |
|--------|-------|--------|
| Rust vs Python (25 domains) | **22.5x** overall, 625x peak (Smith-Waterman) | Exp059 |
| GPU vs CPU (spectral cosine) | **926x** | Exp087 |
| GPU vs CPU (taxonomy, 500 queries) | **63x** | Exp016 |
| GPU vs CPU (pipeline, 10 samples) | **2.45x** | Exp016 |
| Streaming vs round-trip | 92-94% overhead eliminated | Exp091 |
| Galaxy vs Rust GPU (10 samples) | 95.6 s vs 3.0 s (**31.9x**) | Exp015/016 |
| Energy cost (10K samples) | Galaxy $0.40, Rust GPU **$0.02** | Exp016 |
| Hardware | Consumer GPU (RTX 4070), no HPC | All |

## Open Data

All 24 reproductions use publicly accessible data or published model parameters.
No proprietary data dependencies. Sources: NCBI SRA, Zenodo, MassBank, EPA,
Michigan EGLE, published ODE parameters, algorithmic (no external data).

See `../specs/PAPER_REVIEW_QUEUE.md` for the full provenance audit.
