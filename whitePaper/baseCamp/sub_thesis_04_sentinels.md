# Sub-thesis 04: Sentinels — Mechanical Wave Detection and NPU Deployment

**Date:** February 24, 2026
**Faculty:** Cahill (Sandia), Smallwood (Sandia), Kachkovskiy (MSU CMSE)
**Status:** NPU deployment validated — coin-cell feasible

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

## Open Questions

1. Can an NPU sentinel detect Anderson phase transitions in real time?
2. Do mechanical wave localization effects create detection blind spots?
3. Can multi-modal sensing (chemical + mechanical) overcome Anderson barriers?

## Connection to Gen3 Thesis

Practical deployment: spectral theory → NPU inference → field device.
Edge compute makes the Anderson framework operationally useful.
