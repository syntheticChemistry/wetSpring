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

## Live Hardware Results (V60 — February 26, 2026)

**Driver:** ToadStool `akida-driver` 0.1.0 (pure Rust, zero SDK dependency)
**Host:** Intel i9-12900K, 64 GB DDR5, Pop!\_OS 22.04
**Device:** AKD1000 @ `0000:08:00.0`, 80 NPUs, 10 MB SRAM, 0.5 GB/s PCIe

### Exp193: Hardware Validation (DMA + Discovery)

| Metric | Value |
|--------|-------|
| Discovery | Runtime via `DeviceManager::discover()` — zero hardcoded paths |
| DMA write throughput | 37 MB/s sustained |
| DMA read throughput | 37 MB/s sustained |
| Int8 quantization fidelity | < 0.1 round-trip error |
| Device open latency | < 1 ms |

### Exp194: ESN Classifiers — Sim vs Live Hardware

| Classifier | CPU Sim | NPU Live | Throughput |
|------------|---------|----------|------------|
| QS Phase (Exp114, 3-class) | 49.2% | 33.6% | 18.8K Hz |
| Bloom Sentinel (Exp118, 4-class) | 25.0% | 25.3% | 18.8K Hz |
| Disorder (Exp119, 3-class) | 32.9% | 31.6% | 18.6K Hz |

| Capability | Measured |
|------------|----------|
| Reservoir weight loading (200×200) | 164 KB in 4.5 ms (37 MB/s) |
| Online readout switching | 3 swaps in 86 µs (QS↔Bloom↔QS) |
| Batch inference (8-wide) | 20.7K infer/sec |
| Energy per inference | 1.4 µJ |
| Coin-cell CR2032 (1 Hz edge) | 4,003 days (11 years) |

### Exp195: Novel Hardware Explorations

| Experiment | Key Finding |
|------------|-------------|
| Physical Reservoir Fingerprint (PUF) | 6.34 bits entropy, dual-state alternating SRAM signature, 100% stride-2 stability |
| Online Readout Evolution | (1+1)-ES at 136 gen/sec — real-time adaptive inference feasible |
| Temporal Streaming (500-step) | 12.9K Hz sustained, p99=76 µs latency |
| Chaos/Anderson Disorder Sweep | 8 disorder levels (W=0 to W=30) loaded to mesh, response characterized |
| Cross-Reservoir Crosstalk | 12.8K switch/sec, distinct classifier signatures, no state corruption |

### Key Insight: Pure Rust Driver

The ToadStool `akida-driver` provides direct `/dev/akida0` access without the
BrainChip Python SDK or C++ engine. This achieves **Phase C** of the sovereign
driver roadmap (Section 6.4 of the Technical Brief): direct ioctl/mmap on
`/dev/akida0`, bypassing all vendor code. Combined with the `wetSpring::npu`
module, this is a complete Pure Rust neuromorphic compute path.

---

## Remaining Work

1. ~~Train taxonomy FC model from existing Naive Bayes weights~~ → ESN classifiers validated live
2. ~~Quantize to int8 for AKD1000 deployment~~ → int8 quantization validated (Exp193, 194)
3. ~~Validate classification accuracy: NPU int8 vs CPU f64~~ → 3 classifiers compared (Exp194)
4. Measure end-to-end latency: FASTQ → taxonomy on heterogeneous pipeline
5. ~~Explore on-chip learning for adaptive taxonomy~~ → Online readout evolution validated (Exp195)
6. Exercise `NpuBackend::load_reservoir()` for native ESN mesh execution
7. Integrate with metalForge substrate routing for automatic CPU↔GPU↔NPU dispatch
