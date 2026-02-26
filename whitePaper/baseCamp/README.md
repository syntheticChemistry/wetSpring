# baseCamp: Per-Faculty Research Briefings

**Date:** February 25, 2026
**Project:** wetSpring (ecoPrimals)
**Status:** Phase 50 — 183 experiments, 3,618+ validation checks, ALL PASS; 898 tests (819 barracuda + 47 forge + 32 integration/doc), 96.78% llvm-cov, ToadStool S65 aligned, 66 primitives + 2 BGL helpers + 0 local WGSL (fully lean) (barracuda always-on), 77 named tolerance constants, 0 Passthrough, V49 doc cleanup + evolution handoff, 39/39 three-tier, 52/52 papers

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
| **Fajgenbaum** | UPenn | 3 | 7 | 157–165 | 84 | Drug repurposing, pharmacophenomics, Track 3 completed |
| **Diversity Fusion** | — | GPU | 1 | 167 | 18 | CPU↔GPU parity extension |
| **Track 4 Soil QS** | — | 4 | 9 | 170–182 | 321 | No-till soil QS, Anderson pore geometry, Brandt farm, meta-analysis, tillage factorial, CPU/GPU/streaming/metalForge |
| **Total** | | | **52** | | **3,618+** | |

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

Every paper goes through the full evolution. Status across all 39 actionable papers:

| Stage | What It Proves | Coverage |
|-------|---------------|----------|
| Python baseline | Algorithm correctness against published tools | 42 scripts |
| BarraCuda CPU | Rust matches Python within machine precision | 1,476+ checks, 22.5x faster |
| BarraCuda GPU | GPU matches CPU within 1e-6 | 702+ checks, 29+ domains |
| Pure GPU streaming | Zero CPU round-trips, data stays on-device | 204+ checks, 10+ domains |
| metalForge | Same answer on CPU, GPU, NPU | 39/39 papers, 37+ domains |
| NPU reservoir | ESN → int8 → NPU preserves classification (Cholesky solve) | 59 checks, 6 domains |
| Cross-spring evolution | 660+ WGSL shaders traced to origin springs, rewired imports, Exp169 4-spring provenance | 21 checks |
| NCBI-scale hypothesis | Real NCBI data + GPU-confirmed Anderson/QS/pangenome | 146 checks |
| 3D Anderson dimensional QS | hotSpring spectral primitives → ecological predictions | 50 checks |
| Code quality audit | 96.78% coverage, streaming I/O, 0 production mocks, ToadStool S65, barracuda always-on, `deny(missing_docs)` | 898 tests |

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

### Cross-Spring Rewire + Modern Benchmark (Phase 48, V44)

| Experiment | Focus | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp168 | Cross-spring S62 evolution | ~25 | hotSpring precision → wetSpring bio → neuralSpring validated |
| Exp169 | Modern cross-spring benchmark | 12 | All CPU primitives validated, 4-spring provenance, bit-exact delegation |

V44 rewired 6 validation binaries to modern upstream: `find_w_c` (4 files),
`pearson_correlation` (1 file). 66 primitives consumed. Hill CPU ODE helpers
intentionally kept local (derivative-level math, GPU equivalents generated
by `BatchedOdeRK4::generate_shader()`).

### Track 4: No-Till Soil QS & Anderson Geometry (Phase 50, V48)

9 papers, 13 experiments, 321 validation checks. Extends the Anderson-QS framework
into soil microbiology: pore-network geometry maps to Anderson disorder, aggregate
stability predicts QS activation probability, and tillage history modulates
effective dimension.

| Experiment | Paper | Checks | Key Finding |
|:---:|-------|:------:|-------------|
| Exp170 | Martínez-García 2023 QS-pore geometry coupling | 26 | Non-linear connectivity → Anderson disorder W; P(QS) via norm_cdf |
| Exp171 | Feng 2024 pore-size diversity | 27 | Pore class → effective dimension → diversity (Shannon, Bray-Curtis) |
| Exp172 | Mukherjee 2024 distance colonization | 23 | Autoinducer diffusion + QS biofilm ODE + cooperation collapse |
| Exp173 | Islam 2014 Brandt farm soil health | 14 | Published metrics validated; no-till → lower disorder → higher diversity |
| Exp174 | Zuber & Villamil 2016 meta-analysis | 20 | Effect sizes, CI verification, Anderson predicts MBC increase |
| Exp175 | Liang 2015 31-year tillage | 19 | 2×2×2 factorial, Shannon/Pielou, 31-year temporal recovery |
| Exp176 | Tecon & Or 2017 biofilm-aggregate | 23 | Water film thickness → QS diffusion, aggregate surface → colonization |
| Exp177 | Rabot 2018 structure-function | 16 | Structural properties → Anderson parameters → functional outcomes |
| Exp178 | Wang 2025 tillage × compartment | 15 | Endosphere/rhizosphere compartment effects, stover return |
| Exp179 | CPU parity benchmark | 49 | 8 domains timed, pure Rust math validated |
| Exp180 | GPU validation | 23 | FMR + BrayCurtisF64 + Anderson 3D + ODE on GPU |
| Exp181 | Pure GPU streaming | 52 | Zero-CPU-roundtrip soil QS pipeline |
| Exp182 | metalForge cross-substrate | 14 | CPU = GPU for all Track 4 domains |

## Open Data

All 52 reproductions use publicly accessible data or published model parameters.
No proprietary data dependencies. Sources: NCBI SRA, Zenodo, MassBank, EPA,
Michigan EGLE, published ODE parameters, repoDB, algorithmic (no external data),
published soil metrics (Islam 2014, Zuber 2016, Liang 2015), and published
model equations (Martínez-García 2023, Tecon & Or 2017, Wang 2025).

See `../specs/PAPER_REVIEW_QUEUE.md` for the full provenance audit.
