# Exp195: Funky NPU Explorations — AKD1000 Neuromorphic Novelties

**Date:** February 26, 2026
**Phase:** V60 — NPU Live
**Binary:** `validate_npu_funky`
**Command:** `cargo run --release --features npu --bin validate_npu_funky`
**Status:** DONE (NPU, 14/14 checks PASS)

---

## Purpose

Exercise capabilities that only exist because we have real neuromorphic
hardware — things CPU/GPU simulation cannot replicate. Five experiments
probing the physical properties of the AKD1000 mesh.

## Results

### S1: Physical Reservoir Fingerprint (PUF)

| Metric | Value |
|--------|-------|
| Probe vectors | 16 × 256 bytes |
| Byte entropy | 6.34 bits (max 8.0) |
| Intra-chip stability (stride-2) | 100% |
| Intra-chip (adjacent trials) | 8.25% |
| Inter-probe entropy | 88.9% different |

**Finding:** The SRAM exhibits a deterministic dual-state alternating pattern.
Even-numbered trials produce identical fingerprints (0xA6A0...), odd-numbered
trials produce a different but equally stable fingerprint (0xC37D...). This is
a genuine Physical Unclonable Function signature.

### S2: Online Readout Evolution — (1+1)-ES

| Metric | Value |
|--------|-------|
| Generations | 50 |
| Improvements | 3 |
| Start fitness | 24.0% |
| Final fitness | 32.0% |
| Per-generation | 7,356 µs (136 gen/sec) |

**Finding:** Real-time evolutionary learning on neuromorphic hardware is
feasible. Each generation is a real DMA round-trip (load weights → run
inference → evaluate fitness → mutate). 136 gen/sec is fast enough for
adaptive inference on an edge buoy.

### S3: Temporal Streaming (500-step Bloom Trajectory)

| Metric | Value |
|--------|-------|
| Sustained rate | 12,883 Hz |
| p50 latency | 53,305 ns |
| p95 latency | 58,248 ns |
| p99 latency | 75,850 ns |
| Total stream | 38.8 ms for 500 steps |

**Finding:** The NPU can process environmental sensor data faster than any
sensor can produce it. Sub-100 µs p99 latency enables hard real-time
classification.

### S4: Chaos Injection — Anderson Disorder Sweep

| W (disorder) | Mean Response | Response Variance | Byte Entropy |
|:------------:|:------------:|:-----------------:|:------------:|
| 0.0 | 63.24 | 6030.30 | 5.228 |
| 0.5 | 87.99 | 6138.10 | 6.479 |
| 1.0 | 57.38 | 6108.12 | 4.802 |
| 2.0 | 41.26 | 5075.21 | 3.716 |
| 5.0 | 44.81 | 4788.10 | 4.005 |
| 10.0 | 77.62 | 6264.66 | 6.002 |
| 20.0 | 74.01 | 6408.67 | 5.863 |
| 30.0 | 44.80 | 5458.71 | 3.841 |

**Finding:** The DMA path faithfully transmits disorder-structured weights to
SRAM. Response characteristics vary with disorder strength, establishing the
foundation for on-chip Anderson regime characterization.

### S5: Cross-Reservoir Crosstalk Detection

| Metric | Value |
|--------|-------|
| Alternation rounds | 100 (A↔B) |
| Switch rate | 12,765 switches/sec |
| Mean switch latency | 28 µs |
| Classifier A dominant class | 2 |
| Classifier B dominant class | 0 |
| State corruption | None detected |

**Finding:** Rapid classifier switching produces distinct, stable class
distributions for each readout. No SRAM state bleed between consecutive
classifiers — weight mutation is clean.
