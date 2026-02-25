# baseCamp: Per-Faculty Research Briefings

**Date:** February 25, 2026
**Project:** wetSpring (ecoPrimals)
**Status:** Phase 41 — 162 experiments, 3,198+ validation checks, ALL PASS; 806 tests, ToadStool S62 aligned, 44 primitives (barracuda always-on)

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
| [Waters](waters.md) | MSU MMG | 1 | 7 | 020,022-025,027,030,108,114,**121** | 147+ | QS, ODE, Gillespie, bistability, phage defense, NPU QS classifier, NCBI Vibrio landscape |
| [Liu](liu.md) | MSU CMSE | 1b | 6 | 026,031-034,036-038,109,115 | 136 | HMM, phylogenetics, alignment, placement, NPU placement |
| [R. Anderson](anderson.md) | Carleton | 1c | 6 | 051-056,110,116,**125** | 170 | Deep-sea metagenomics, pangenomics, NPU binning, NCBI Campylobacterota |
| [Jones](jones.md) | MSU BMB | 2 | 2 | 041,042,111,117,**124** | 55 | PFAS mass spectrometry, EPA ML, NPU spectral triage |
| [Cahill](cahill.md) | Sandia | 1 | 1 | 039,112,118,**123** | 54 | Algal pond, NPU bloom sentinel, temporal ESN |
| [Smallwood](smallwood.md) | Sandia | 1 | 1 | 040,112,118,**123** | 58 | Bloom surveillance, NPU sentinel, temporal ESN |
| [Kachkovskiy](kachkovskiy.md) | MSU CMSE | cross | 1 | 107,113,119,**122,126,127-138,144-156** | 334 | Spectral theory, 2D/3D Anderson, geometry zoo, ecosystem atlas, finite-size scaling v1+v2, correlated disorder, mapping sensitivity, planktonic dilution, eukaryote scaling, extension papers, paper queue |
| **Fajgenbaum** | UPenn | 3 | 2 | 157,158 | 17 | Drug repurposing, pharmacophenomics |
| **Total** | | | **43** | | **1,071+** | |

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

### NCBI-Scale Hypothesis Testing (Phase 35, GPU-confirmed)

| Experiment | Faculty | GPU Result | Checks | Key Finding |
|:---:|---------|-----------|:------:|-------------|
| Exp121 | Waters | 200 real Vibrio assemblies → GPU ODE sweep | 14 | All-biofilm landscape; real genomes don't reach planktonic |
| Exp122 | Kachkovskiy | 2D Anderson 20×20 lattice sweep | 12 | Extended plateau absent in 1D; bloom QS-active in 2D only; J_c≈0.41 |
| Exp123 | Cahill/Smallwood | Stateful vs stateless ESN bloom | 9 | Temporal memory preserves classification; coin-cell feasible |
| Exp124 | Jones | NPU→GPU spectral triage (5K library) | 10 | 100% recall, 20% pass rate, 3.7× speedup |
| Exp125 | Anderson | Real Campylobacterota pangenome (158 asm) | 11 | 4 ecosystems, 49–53% core, open pangenome confirmed |
| Exp126 | Kachkovskiy | 28-biome global QS atlas (136 BioProjects) | 90 | W monotonic with J; all biomes correctly ordered |

### 3D Anderson Dimensional QS (Phase 36, GPU-confirmed)

| Experiment | Faculty | GPU Result | Checks | Key Finding |
|:---:|---------|-----------|:------:|-------------|
| Exp127 | Kachkovskiy | 1D→2D→3D sweep (8³ lattice) | 17 | 3D plateau 2.4× wider; J_c(3D)≈1.28 >> J_c(2D)≈0.56; gut/vent/soil flip to active |
| Exp128 | Kachkovskiy/Anderson | Vent chimney geometry → QS | 12 | 3 of 4 zones 3D-active but 2D-suppressed; 2D misses 75% capability |
| Exp129 | Kachkovskiy | 28-biome × 3-dimension phase diagram | 12 | All 28 biomes QS-active in 3D, zero in 1D/2D; W_c(3D)≈16.5 exceeds all |
| Exp130 | Kachkovskiy | Thick biofilm 3D block (8×8×6) | 9 | 4× wider plateau than 2D; J_c(3D)≈1.25; 6 layers transform QS |

## Validation Chain

Every paper goes through the full evolution. Status across all 25 actionable papers:

| Stage | What It Proves | Coverage |
|-------|---------------|----------|
| Python baseline | Algorithm correctness against published tools | 42 scripts |
| BarraCuda CPU | Rust matches Python within machine precision | 1,476 checks, 22.5x faster |
| BarraCuda GPU | GPU matches CPU within 1e-6 | 702 checks, 29 domains |
| Pure GPU streaming | Zero CPU round-trips, data stays on-device | 152 checks, 10+ domains |
| metalForge | Same answer on CPU, GPU, NPU | 25/25 papers, 37 domains |
| NPU reservoir | ESN → int8 → NPU preserves classification (Cholesky solve) | 59 checks, 6 domains |
| Cross-spring evolution | 612 WGSL shaders traced to origin springs, rewired imports | 9 checks |
| NCBI-scale hypothesis | Real NCBI data + GPU-confirmed Anderson/QS/pangenome | 146 checks |
| 3D Anderson dimensional QS | hotSpring spectral primitives → ecological predictions | 50 checks |
| Code quality audit | 95.67% coverage, streaming I/O, 0 production mocks, ToadStool S62, barracuda always-on | 806 tests |

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

## Sub-theses (Gen3 Ch. 14)

| # | Sub-thesis | File | Key Experiments |
|:-:|-----------|------|-----------------|
| 01 | Anderson-QS null hypothesis | [sub_thesis_01](sub_thesis_01_anderson_qs.md) | 107, 126–138, 144–156 |
| 02 | LTEE constrained evolution | [sub_thesis_02](sub_thesis_02_ltee.md) | 143, 146 |
| 03 | BioAg rhizosphere QS | [sub_thesis_03](sub_thesis_03_bioag.md) | 129, 142, 146, 151, 153 |
| 04 | Sentinels + NPU deployment | [sub_thesis_04](sub_thesis_04_sentinels.md) | 114, 118, 123, 124, 147 |
| 05 | Cross-species eavesdropping | [sub_thesis_05](sub_thesis_05_cross_species.md) | 142, 144–146, 151, 153–154 |

## Open Data

All 43 reproductions use publicly accessible data or published model parameters.
No proprietary data dependencies. Sources: NCBI SRA, Zenodo, MassBank, EPA,
Michigan EGLE, published ODE parameters, repoDB, algorithmic (no external data).

See `../specs/PAPER_REVIEW_QUEUE.md` for the full provenance audit.
