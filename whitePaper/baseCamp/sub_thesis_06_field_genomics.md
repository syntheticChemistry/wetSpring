# Sub-thesis 06: Field Genomics — Sovereign Sequencing + Neuromorphic Edge Classification

**Date:** February 27, 2026
**Faculty:** Cahill (Sandia), Smallwood (Sandia), R. Anderson (Carleton), Jones (MSU BMB), Waters (MSU MMG)
**Status:** `io::nanopore` module operational — POD5/NRS parser, streaming iterator, synthetic read generation. Pre-hardware validation complete (Exp196a-c, 52/52 PASS). NPU live on AKD1000 (Exp193-195). 16S sovereign pipeline operational (Exp184-185). ESN classifiers validated on hardware. Deep audit complete: 95.46% line coverage, `partial_cmp` → `total_cmp`, dead code removed, baseline manifest refreshed. Remaining: `bio::basecall` (signal → base) and real MinION integration (Exp197-202).

---

## Core Claim

A nanopore sequencer (Oxford Nanopore MinION) paired with a neuromorphic
processor (BrainChip AKD1000) and BarraCUDA's sovereign Rust bioinformatics
pipeline creates an autonomous field genomics platform that sequences,
classifies, and acts without cloud connectivity. This closes the loop from
environmental DNA to actionable intelligence at the edge.

No other stack combines: (1) a pure Rust sequencing pipeline validated across
211 experiments, (2) a pure Rust NPU driver running live on neuromorphic
hardware at <10 mW, and (3) a metalForge substrate router that dispatches
workloads across CPU, GPU, NPU, and — now — sequencer.

## The Gap in Current Field Sequencing

The state of the art (2025-2026) in field-deployed nanopore sequencing:

| What Works | What Doesn't |
|------------|-------------|
| MinION is portable, field-proven (Lake Erie HABs, airborne eDNA, soil sentinels) | Basecalling requires a laptop or the Mk1C's limited ARM GPU |
| Adaptive sampling enables real-time read selection | Downstream classification still needs cloud or beefy local compute |
| RosHAB, HABSSED, NanoASV provide field-ready pipelines | All are Python/R, cloud-dependent, or require Guppy/Dorado on GPU |
| CiMBA and RISC-V SoC papers push edge basecalling | Nobody has closed the loop to neuromorphic edge classification |

The missing piece: real-time classification of sequenced reads at the edge,
on hardware that runs for years on a coin cell. That is what the
AKD1000 + BarraCUDA stack provides.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Field Genomics Unit                       │
│                                                         │
│  Environmental sample                                   │
│       │ DNA extraction (rapid kit, 10 min)              │
│       ▼                                                 │
│  ┌──────────┐                                           │
│  │ MinION   │ sequences eDNA in real time               │
│  │ (Mk1D)   │ 450 bp/s per pore × 512 pores            │
│  └────┬─────┘                                           │
│       │ FAST5/POD5 raw signal                           │
│       ▼                                                 │
│  ┌──────────────────┐                                   │
│  │ BarraCUDA        │ basecall + 16S + taxonomy         │
│  │ (host CPU/GPU)   │ sovereign Rust, no Python         │
│  └────┬─────────────┘                                   │
│       │ classified reads + community profile            │
│       ▼                                                 │
│  ┌──────────────────┐                                   │
│  │ AKD1000 NPU      │ ESN regime classification         │
│  │ (10 mW, DMA)     │ bloom/healthy/stressed/AMR/PFAS   │
│  └────┬─────────────┘                                   │
│       │ classification + confidence                     │
│       ▼                                                 │
│  ┌──────────────────┐                                   │
│  │ Decision engine  │ alert / adapt / log               │
│  │ + adaptive       │ feed accept/reject to MinKNOW     │
│  │   sampling       │ NPU decides which reads to keep   │
│  └──────────────────┘                                   │
│                                                         │
│  Power: 5W solar (MinION) + coin cell (NPU standby)    │
│  Connectivity: optional (nightly sync via Songbird)     │
└─────────────────────────────────────────────────────────┘
```

### metalForge Extension: Sequencer as Substrate

metalForge currently routes workloads across CPU, GPU, and NPU. The
sequencer represents a fourth substrate class — not a compute substrate
but a *sensing* substrate that generates data requiring classification:

```
metalForge Substrate Registry
├── CPU (i9-12900K)          — general math, fallback
├── GPU (RTX 4070)           — batch Anderson, streaming pipelines
├── NPU (AKD1000)            — edge classification, sentinel inference
└── SEQ (MinION Mk1D/Mk1C)  — DNA sensing, read generation, adaptive input
```

The dispatch logic extends naturally: metalForge routes raw signal to GPU
for basecalling (or to CPU if no GPU), routes community profiles to NPU
for classification, and routes accept/reject decisions back to the sequencer
for adaptive sampling. The same capability-based routing that handles
GPU→NPU→CPU handoffs handles SEQ→GPU→NPU→SEQ feedback loops.

## Research Programs

### Program 1: Bloom Sentinel Live — Great Lakes HAB Monitoring

**Connects:** Sub-thesis 01 (Anderson QS), Sub-thesis 04 (Sentinels), Cahill/Smallwood (Sandia)
**BarraCUDA math:** DADA2, chimera, taxonomy, Bray-Curtis, Anderson regime, ESN reservoir
**Primals:** ToadStool (akida-driver, GPU basecall), metalForge (dispatch), Songbird (telemetry)
**Hardware:** MinION Mk1D + AKD1000 + host (Pi or laptop)

MinION sequences water eDNA on-site (16S rapid barcoding, 10-min prep).
BarraCUDA's sovereign 16S pipeline processes reads in Rust. ESN bloom
classifier (validated Exp118, 123, 194) runs on AKD1000 at <10 mW.
Real-time classification: pre-bloom / active / post-bloom / toxic.

**Local angle:** CIGLR at UMich runs bi-weekly Saginaw Bay cyanotoxin
monitoring (July-October). NOAA GLERL has continuous buoy data in western
Lake Erie. A MinION + NPU station could provide continuous genomic
monitoring between their sampling events.

**Literature validated:** RosHAB (Frontiers in Microbiology 2023) — real-time
on-site HAB detection using MinION 16S. HABSSED — eDNA from Lake Erie
blooms on MinION. We replace their Python pipeline with sovereign Rust
and add neuromorphic edge classification.

**Experiments needed:**

| Exp | Name | What It Proves |
|-----|------|---------------|
| 196 | Nanopore Signal Bridge | BarraCUDA reads FAST5/POD5, bridges to existing 16S pipeline |
| 197 | NPU Adaptive Sampling | NPU classifies partial reads, drives MinKNOW accept/reject |
| 198 | Field Bloom Sentinel E2E | MinION → basecall → 16S → ESN bloom → NPU → alert |

### Program 2: Soil Health Sentinel — No-Till and BioAg Monitoring

**Connects:** Sub-thesis 01 (Anderson), Sub-thesis 03 (BioAg), Sub-thesis 06-local (No-Till), Track 4
**BarraCUDA math:** 16S, diversity, Anderson disorder, Pielou evenness, Bray-Curtis
**Primals:** ToadStool, airSpring (soil moisture), groundSpring (uncertainty)
**Hardware:** MinION Mk1D + AKD1000 + soil sensor array

Sterile sentinel inserts in soil (Environmental Microbiome 2024 pattern)
with periodic MinION 16S sequencing. Anderson localization analysis via
BarraCUDA classifies soil health state: diverse/healthy vs disturbed vs
recovering. NPU tracks community trajectory over growing seasons.

Track 4 (Exp170-182, 321 checks) already validates the Anderson framework
for soil — tillage collapses effective dimension, no-till preserves 3D
QS-active pore networks. Field sequencing adds the measurement layer.

**Experiments needed:**

| Exp | Name | What It Proves |
|-----|------|---------------|
| 199 | Soil 16S Field Pipeline | MinION soil eDNA → 16S → Anderson disorder tracking |
| 200 | Soil Health NPU Classifier | NPU classifies soil community state from 16S profile |

### Program 3: AMR Wastewater Sentinel

**Connects:** Sub-thesis 04 (Sentinels), pathogen emergence (Section 4.1-4.2)
**BarraCUDA math:** alignment (Smith-Waterman), phylo placement, pangenomics
**Primals:** ToadStool, NestGate (AMR gene database)
**Hardware:** MinION Mk1D + AKD1000

Nanopore metagenomics of hospital/municipal wastewater. Long reads (10 kb+)
resolve full resistance gene cassettes + mobile genetic elements that short
reads cannot. BarraCUDA alignment + phylo placement identifies resistance
gene hosts. NPU classifies threat level from community profile.

**Literature:** npj Antimicrobials and Resistance (2025) — nanopore for AMR
surveillance. Sewer monitoring at healthcare facilities (medRxiv 2025).
Transfer dynamics of last-resort resistance in hospital wastewater
(Water Research 2025).

**Experiments needed:**

| Exp | Name | What It Proves |
|-----|------|---------------|
| 201 | AMR Gene Detection | BarraCUDA long-read → resistance gene identification |
| 202 | AMR Threat NPU Classifier | NPU classifies resistance profile severity |

### Program 4: PFAS Dual-Mode Environmental Monitor

**Connects:** Sub-thesis 04 (PFAS section), Jones Lab (MSU BMB), Exp006-008, 041-042
**BarraCUDA math:** PFAS ML pipeline, spectral matching, community analysis
**Primals:** ToadStool (GPU spectral), metalForge (multi-substrate)
**Hardware:** MinION Mk1D + AKD1000 + (future: modified pores for PFAS sensing)

Nanopore 16S profiling of microbial community response to PFAS exposure,
paired with BarraCUDA's validated PFAS ML pipeline (Exp041: EPA fate-and-
transport, Exp042: MassBank spectral). Community shift detection via
Anderson framework provides early warning before chemical thresholds.

**Emerging:** Biological nanopores with cyclodextrin can detect individual
PFAS molecules at the single-molecule level (SciEngine 2025). Same pore
technology, chemical sensing mode. When this matures, one device does both
community profiling and contaminant detection.

### Program 5: NPU-Driven Adaptive Sampling

**Connects:** All programs above (cross-cutting technique)
**BarraCUDA math:** k-mer histograms, taxonomy lookup, ESN classifier
**Primals:** ToadStool (NPU inference), metalForge (feedback routing)
**Hardware:** AKD1000 in the classification feedback loop

Oxford Nanopore's adaptive sampling makes real-time decisions about whether
to keep or reject DNA reads as they pass through the pore. Currently this
uses GPU basecalling + CPU alignment (readfish). An NPU doing the
accept/reject classification at sub-millisecond latency on <10 mW is a
different paradigm entirely.

The AKD1000 classifies at 18.8K Hz (Exp194). A MinION generates ~500 reads
per second at peak. The NPU has 37x headroom for real-time classification
of every read as it emerges. This enables:

- **Target enrichment without wet-lab prep** — keep HAB-associated reads,
  reject host/environmental background
- **Threat detection** — keep reads matching known AMR genes, reject
  commensals
- **Rare biosphere capture** — keep reads from underrepresented taxa,
  reject dominant species (guided by Exp051 rare biosphere framework)

### Program 6: Deep-Sea Autonomous Lander (Long-Term Vision)

**Connects:** Sub-thesis 01 (Anderson), R. Anderson (Carleton), Exp144-145 (cold seep)
**BarraCUDA math:** metagenomics, dN/dS, pangenomics, Anderson spectral
**Primals:** Full stack (NestGate storage, Songbird acoustic uplink, BearDog attestation)
**Hardware:** MinION + AKD1000 + pressure-rated enclosure + acoustic modem

MinION on an autonomous underwater lander near hydrothermal vents. Sequences
vent community eDNA. Cold seep QS analysis (Exp144-145, 299K QS genes across
170 metagenomes) runs on NPU. Anderson disorder analysis of vent community
structure. Songbird uplinks classification results via acoustic modem.

The baseCamp README already mentions ">100M contigs/day at <10 mW for
deep-sea landers with nanopore sequencers" (anderson.md). This program
makes that vision concrete.

## The BarraCUDA Math Stack for Field Genomics

Every program above uses math that is already validated in BarraCUDA:

| Domain | Module | Validated | GPU | NPU |
|--------|--------|:---------:|:---:|:---:|
| Basecalling (new) | `bio::basecall` (to build) | — | planned | — |
| FAST5/POD5 I/O | `io::nanopore` | Exp196a (28 checks) | — | — |
| 16S ASV denoising | `bio::dada2` | 4,688+ checks | yes | — |
| Chimera detection | `bio::chimera` | validated | yes | — |
| Taxonomy classification | `bio::taxonomy` | validated | yes | int8 |
| K-mer histograms | `bio::kmer` | validated | yes | int8 |
| Diversity metrics | `bio::diversity` | validated | yes | — |
| Bray-Curtis distance | `bio::bray_curtis` | validated | yes | — |
| Anderson disorder | `bio::anderson_qs` | 3,400+ checks | yes | ESN |
| Pielou evenness | `bio::diversity` | validated | yes | — |
| Smith-Waterman alignment | `bio::alignment` | validated | yes | — |
| Phylo placement | `bio::phylo_placement` | validated | yes | int8 |
| Pangenomics | `bio::pangenome` | validated | yes | — |
| dN/dS (Nei-Gojobori) | `bio::dnds` | validated | yes | — |
| ESN reservoir | `bio::esn` | validated | — | live AKD1000 |
| PFAS ML | `bio::pfas_ml` | validated | yes | — |
| Spectral matching | `bio::spectral` | validated | yes | int8 |
| Random Forest | `ml::random_forest` | validated | yes | — |
| GBM inference | `ml::gbm` | validated | yes | — |

**Remaining module:** `bio::basecall` (signal → base conversion). `io::nanopore`
is operational (V61). Everything downstream is operational.

## Primal Integration

| Primal | Role in Field Genomics |
|--------|----------------------|
| **ToadStool** | Compute engine — GPU basecalling, NPU classification, CPU fallback. `akida-driver` provides sovereign NPU access. |
| **metalForge** | Substrate routing — dispatches raw signal to GPU, community profiles to NPU, adaptive sampling decisions to sequencer. New `Sequencer` substrate type. |
| **NestGate** | Data store — content-addressed storage for reads, classifications, provenance. Reference database hosting (AMR genes, 16S taxonomy). |
| **Songbird** | Networking — nightly weight sync, telemetry uplink, multi-station coordination. Acoustic modem driver for underwater deployment. |
| **BearDog** | Identity — PUF-based device attestation (Exp195 S1). Chain of custody for environmental samples. |
| **sweetGrass** | Attribution — PROV-O provenance tracking from sample to classification to alert. |
| **biomeOS** | Orchestration — capability registry, primal lifecycle, field unit boot sequence. |

## What Makes This Different

| Feature | Current Field Sequencing | With This Stack |
|---------|------------------------|----------------|
| Basecalling | Python (Guppy/Dorado) on GPU or cloud | BarraCUDA Rust on any substrate (planned) |
| Classification | Python scripts, cloud ML | NPU at 18.8K Hz, <10 mW, coin-cell years |
| Adaptive sampling | CPU/GPU alignment (readfish) | NPU at sub-ms latency, 37x read rate headroom |
| Pipeline | QIIME2/Galaxy, needs internet | Sovereign Rust, zero external dependencies |
| Power | Laptop (45-65W) or Mk1C ARM (60W) | MinION (60W) + NPU standby (10 mW) |
| Connectivity | Required for analysis | Optional (nightly sync) |
| Validation | Published tools (black box) | 211 experiments, 5,061+ checks, all open |
| Hardware lock-in | ONT software stack | Pure Rust driver, AGPL-3.0, no vendor SDK |

## Testable Predictions

| Prediction | Test | Expected Result |
|-----------|------|----------------|
| NPU adaptive sampling matches GPU readfish enrichment | Same flow cell, split analysis | Target read fraction within 5% |
| ESN bloom classifier on live 16S matches sim | MinION Lake Erie eDNA → NPU | Classification agreement > 80% (accounting for sequencing noise) |
| Soil community Anderson regime detectable from MinION 16S | Field soil → MinION → sovereign pipeline | W(field) within 2 units of Illumina W |
| AMR gene cassettes resolved by long reads | Wastewater MinION → alignment → gene context | >90% of resistance genes resolved to mobile element context |
| Full pipeline runs autonomously for 7 days | Field station with solar + battery | Continuous classification, no human intervention |

## Timeline

| Phase | Target | Deliverable |
|-------|--------|-------------|
| ~~Now~~ **Done** | Architecture + module design | This document, experiment outlines |
| ~~Next~~ **Done (V61)** | `io::nanopore` POD5/NRS reader | Exp196a-c: 52/52 pre-hardware checks PASS |
| **Hardware arrival** | End-to-end validation | Exp196-202: real reads through sovereign pipeline |
| **Field season 2026** | Saginaw Bay deployment | Continuous bloom sentinel with real sequencing |
| **Year 2** | Multi-site network | Soil + water + wastewater monitoring stations |
| **Year 3+** | Autonomous lander | Deep-sea deployment with acoustic uplink |

## Open Questions

1. Can BarraCUDA basecalling match Dorado accuracy, or is the NPU better
   used for post-basecall classification?
2. Does int8 quantization of community profiles preserve Anderson regime
   classification from real (noisy) nanopore reads?
3. What is the minimum sequencing depth for reliable Anderson regime
   detection? (Connects to Exp051 rare biosphere saturation curves)
4. Can adaptive sampling via NPU reduce per-sample flow cell cost enough
   for routine environmental monitoring (<$100/sample)?
5. Does the AKD1500 (800 GOPS, <300 mW, Nov 2025) change the power
   budget enough for always-on basecalling at the edge?

## Connection to Gen3 Thesis

Field genomics is where sovereign compute becomes sovereign science.
The constrained evolution thesis argues that environmental constraints
drive specialization. This sub-thesis turns the framework inside out:
instead of studying how constraints shaped historical evolution, we
deploy sovereign tools to detect environmental constraints IN REAL TIME
— contamination, bloom onset, resistance emergence — and respond
before they cause harm. The Anderson framework is the detection signal.
The NPU is the detector. The nanopore is the antenna.

## References

Oxford Nanopore Technologies (2026). Genomics for a Changing Planet (white paper).

Calderón-Franco et al. (2025). Applications of nanopore sequencing in bacterial
antimicrobial resistance surveillance. npj Antimicrobials and Resistance.

Pérez-Cataluña et al. (2023). Rapid on-site detection of harmful algal blooms:
real-time cyanobacteria identification using Oxford Nanopore sequencing.
Frontiers in Microbiology 14:1267652.

Patin et al. (2022). Environmental DNA sequencing data from algal blooms in
Lake Erie using Oxford Nanopore MinION. bioRxiv 2022.03.12.483776.

Samuel et al. (2025). Autonomous Marine Biomolecular Monitoring. Univ. Southampton.

Schüler et al. (2025). NanoASV: A snakemake workflow for reproducible field-based
Nanopore full-length 16S metabarcoding. CEA.

Wijeratne et al. (2025). MARTi: Real-time analysis and visualization tool for
nanopore metagenomics. Bioinformatics.

Arani et al. (2025). CiMBA: Accelerating Genome Sequencing through On-Device
Basecalling via Compute-in-Memory. arXiv:2504.07298.

Fan et al. (2025). Sequencing on Silicon: AI SoC Design for Mobile Genomics
at the Edge. arXiv:2510.09339.

Steele et al. (2024). Sterile sentinels and MinION sequencing capture active soil
microbial communities that differentiate crop rotations. Environmental Microbiome.

BrainChip Inc. (2025). AKD1500 Edge AI Co-Processor. Embedded World North America.
