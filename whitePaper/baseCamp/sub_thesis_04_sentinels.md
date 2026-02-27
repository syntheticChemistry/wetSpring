# Sub-thesis 04: Sentinels — Mechanical Wave Detection and NPU Deployment

**Date:** February 26, 2026
**Faculty:** Cahill (Sandia), Smallwood (Sandia), Kachkovskiy (MSU CMSE)
**Status:** **NPU deployment validated on real AKD1000 hardware** — coin-cell feasible (11 years at 1 Hz). 4/6 communication modes subject to Anderson localization (Exp147, 152). PFAS screening pipeline complete (Exp041-042). Track 4 soil monitoring extends sentinel framework. **V60 Live Hardware**: Exp193-195 validate ESN classifiers (QS/Bloom/Disorder) on real AKD1000 via pure Rust `akida-driver` — 18.8K Hz throughput, 1.4 µJ/infer, online readout switching in 86 µs, (1+1)-ES evolution at 136 gen/sec, 12.9K Hz temporal streaming, PUF fingerprint (6.34 bits entropy)

---

## Core Claim

The Anderson localization framework extends beyond chemical QS to ALL
microbial communication modes. 4 of 6 bacterial communication channels
are subject to Anderson localization. NPU-based sentinel devices can
detect community state transitions in real time with milliwatt power.

## Key Results

| Finding | Experiment | Checks |
|---------|:----------:|:------:|
| 4/6 communication modes subject to Anderson localization | Exp147 | 6 |
| Contact-dependent and nanowire bypass Anderson | Exp147 | — |
| NPU bloom sentinel: 4-state classifier, >1 yr battery | Exp118 | 11 |
| Temporal ESN preserves bloom classification | Exp123 | 9 |
| NPU spectral triage: 100% recall, 20% pass rate | Exp124 | 10 |
| NPU QS phase classifier: 100% f64↔NPU agreement | Exp114 | 13 |

## Communication Mode Analysis (Exp147)

| Mode | Anderson? | Example | QS Implication |
|------|:---------:|---------|----------------|
| Chemical (AHL) | Yes | V. fischeri | Localized in 1D/2D at high W |
| Mechanical | Yes | Surface sensing | Geometry-dependent |
| Electromagnetic | Yes | Bioluminescence | Distance-limited |
| Acoustic | Yes | Pressure waves | Localized in structured media |
| Contact-dependent | **No** | Myxococcus | Bypasses diffusion entirely |
| Nanowire | **No** | Geobacter | Direct electron transfer |

## NPU Deployment Architecture

```
Sensor → ESN reservoir → int8 quantization → Akida AKD1000 → Alert
                         (Cholesky solve)     (coin-cell power)
```

| Sensor Type | Classification | Accuracy | Power |
|-------------|---------------|:--------:|:-----:|
| Bloom state | 4-class | 100% | <5 mW |
| QS regime | 3-class | 100% | <5 mW |
| Spectral triage | binary | 100% recall | <5 mW |

## Nanopore Integration (Sub-thesis 06: Field Genomics)

The sentinel concept reaches full capability when paired with in-field DNA
sequencing. Current monitoring relies on periodic grab samples sent to a lab
for 16S sequencing (days-to-weeks turnaround). A MinION nanopore sequencer
at the sentinel station enables real-time community profiling:

```
MinION (sequences eDNA) → BarraCUDA (16S pipeline) → NPU (regime classifier)
                                                        ↓
                                                   Alert + adaptive sampling
                                                   (NPU tells MinION which reads to keep)
```

| Sentinel Application | Without Nanopore | With Nanopore |
|---------------------|-----------------|--------------|
| HAB prediction | Proxy sensors (chlorophyll, phycocyanin) | Direct cyanobacterial 16S identification |
| PFAS detection | Community shift from periodic lab samples | On-site 16S + (future) single-molecule PFAS sensing |
| AMR monitoring | Culture-based, days | Long-read resistance gene cassette resolution, hours |
| Soil health | Periodic lab 16S | Field-deployed continuous 16S monitoring |

The NPU's 18.8K Hz classification throughput provides 37x headroom over
MinION's peak read generation rate (~500 reads/sec), enabling real-time
adaptive sampling: the NPU classifies each partial read and decides whether
MinKNOW should keep or reject it, enriching for target organisms without
wet-lab preparation.

**Key literature:** RosHAB (Frontiers in Microbiology 2023) — on-site HAB
detection on MinION. HABSSED — Lake Erie bloom eDNA. CiMBA (arXiv 2025) —
edge basecalling accelerator. NanoASV — offline field 16S analysis.

See [Sub-thesis 06: Field Genomics](sub_thesis_06_field_genomics.md) for the
full architecture, research programs, and experiment plan (Exp196-202).

## Open Questions

1. Can an NPU sentinel detect Anderson phase transitions in real time?
2. Do mechanical wave localization effects create detection blind spots?
3. Can multi-modal sensing (chemical + mechanical) overcome Anderson barriers?
4. Does NPU adaptive sampling of nanopore reads match GPU-based readfish enrichment?
5. What is the minimum sequencing depth for reliable Anderson regime detection from field eDNA?
6. Can the full pipeline (MinION → BarraCUDA → NPU → alert) run autonomously for 7+ days?

## Connection to Gen3 Thesis

Practical deployment: spectral theory → NPU inference → field device.
Edge compute makes the Anderson framework operationally useful.
Field genomics (Sub-thesis 06) extends this from inference-only to
sequence-classify-act: the sentinel does not just classify pre-processed
data, it generates its own data via nanopore sequencing and adapts its
sampling strategy in real time via NPU-driven adaptive sampling.
