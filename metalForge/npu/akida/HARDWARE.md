# AKD1000 NPU — Life Science Applications

**Device**: BrainChip AKD1000 (inherited characterization from hotSpring)
**Purpose**: Explore ultra-low-power inference for field-deployed life science

---

## Hardware Summary (from hotSpring metalForge)

| Property | Value |
|----------|-------|
| Architecture | 80 NPs (78 enumerated), event-based digital |
| Memory | 8MB SRAM on-chip, 256Mbit LPDDR4 |
| Power | Board: ~918mW floor; chip inference: below measurement threshold |
| Interface | PCIe 2.0 x1, `/dev/akida0` |
| Inference latency | ~650μs round-trip (PCIe-dominated) |
| Key discovery | Direct weight injection via `set_variable()` (bypasses Keras pipeline) |

See `../../hotSpring/metalForge/npu/akida/HARDWARE.md` for the full deep-dive
and `BEYOND_SDK.md` for the 10 overturned SDK assumptions.

---

## wetSpring NPU Opportunities

### 1. Taxonomy Classification (Naive Bayes → FC)

Current Rust implementation (`bio::taxonomy`) uses a Naive Bayes classifier
with k-mer features. This maps naturally to a fully-connected network:

```
Input: k-mer frequency vector (e.g., 256 features for k=4)
Hidden: 1-2 FC layers (ReLU activation)
Output: taxonomy class (genus-level)

NPU mapping:
  - InputConv → FC(256, 128) → FC(128, N_genera)
  - int8 quantization sufficient (classification, not regression)
  - Expected: <1ms inference, <1mW per classification
```

**Why it matters**: Battery-powered 16S classification at field sites.
No cloud connectivity needed. Process reads locally on embedded hardware.

### 2. Anomaly Detection (Pond Crash Early Warning)

Time-series classification for algal pond crash forensics (Track 1):

```
Input: rolling window of diversity metrics (Shannon, Simpson over 7 days)
ESN reservoir: 50 neurons, spectral radius 0.9
Readout: FC(50, 2) → {normal, crash_imminent}

NPU mapping:
  - ESN reservoir computed on CPU/GPU (sparse, event-based)
  - Readout FC on NPU: 668 inference clocks (from hotSpring benchmark)
  - Continuous monitoring at ~30mW
```

### 3. PFAS Presence/Absence Screening

Binary classification from mass spectral features:

```
Input: top-20 spectral peaks (m/z, intensity pairs → 40 features)
FC network: FC(40, 32) → FC(32, 2) → {PFAS_present, clean}

NPU mapping:
  - Lightweight model for field screening
  - Trained on Jones Lab PFAS library (175 compounds, Exp018)
  - Decision tree already validated (Exp008) — retrain as FC
```

---

## Integration with ToadStool Dispatch

```
┌───────────────────────────────────┐
│ CPU: FASTQ parse → DADA2 → ASVs  │  (sequential, branching)
└────────────┬──────────────────────┘
             │ abundance vectors
             ▼
┌───────────────────────────────────┐
│ GPU: Diversity + spectral match   │  (batch-parallel, f64)
│ via ToadStool FusedMapReduceF64   │
└────────────┬──────────────────────┘
             │ feature vector
             ▼
┌───────────────────────────────────┐
│ NPU: Taxonomy + anomaly classify  │  (low-power inference)
│ via AKD1000 direct weight inject  │
└───────────────────────────────────┘
```

This three-substrate pipeline mirrors hotSpring's GPU→NPU composition
for lattice QCD phase classification.

---

## Remaining Work

1. Train taxonomy FC model from existing Naive Bayes weights
2. Quantize to int8 for AKD1000 deployment
3. Validate classification accuracy: NPU int8 vs CPU f64
4. Measure end-to-end latency: FASTQ → taxonomy on heterogeneous pipeline
5. Explore on-chip learning for adaptive taxonomy (AKD1000 supports this)
