# Exp188: NPU Sentinel with Real Sensor Stream

**Date:** February 26, 2026
**Phase:** V55 — Science extensions
**Binary:** `validate_npu_sentinel_stream` (planned)
**Command:** `cargo run --release --bin validate_npu_sentinel_stream`
**Status:** Protocol defined, implementation pending
**Depends on:** Exp160 (NPU sentinel validation), Exp184 (real data pipeline)

## Purpose

Deploy the validated NPU sentinel pipeline on Eastgate's Akida AKD1000
with simulated environmental sensor input, testing real-time microbial
community health monitoring. This bridges the gap between validated
analytics (Exp160: 95% top-1, 87% recall) and deployment-ready
environmental monitoring.

## Hardware

| Component | Specification | Role |
|-----------|--------------|------|
| Akida AKD1000 | Eastgate NPU, 1.2M neurons | Spiking neural inference |
| Sensor input | Simulated pH/temp/DO stream | Environmental telemetry |
| Host CPU | Eastgate main | Feature extraction, orchestration |
| Storage | NestGate (local or Westgate) | Provenance, time series |

## Pipeline

```
Sensor stream (1 Hz) → Feature extraction (CPU)
    → Akida SNN inference (NPU, <1ms latency)
    → Classification: [healthy | stressed | critical | bloom]
    → Anderson spectral overlay (periodic, CPU/GPU)
    → NestGate storage (time series + alerts)
```

## Scenarios

### S1: Steady-State Monitoring
- 1000 simulated data points at 1 Hz (~17 minutes)
- Community parameters: stable, high-diversity soil
- Expected: >95% classified as "healthy"
- Validates throughput and latency in steady state

### S2: Stress Event Detection
- Inject gradual stress signal (diversity decline over 100 points)
- Expected: transition from "healthy" to "stressed" detected
- Latency requirement: detect within 30 data points of onset
- Anderson r should track diversity decline

### S3: Bloom Detection
- Inject sudden bloom event (single species dominance spike)
- Expected: immediate "bloom" classification
- Anderson r should drop toward Poisson (localization)
- Alert generated within 5 data points

### S4: Recovery Tracking
- After stress event, inject recovery signal
- Expected: "stressed" → "healthy" transition detected
- Anderson r should recover toward GOE
- Recovery time should match dynamic Anderson (Exp186) predictions

## Validation Checks

### S1: Throughput
- [ ] Process 1000 points at ≥1 Hz sustained
- [ ] NPU inference latency < 5ms per point (p99)
- [ ] No dropped points in pipeline

### S2: Accuracy
- [ ] Classification accuracy ≥ 90% on labeled test set
- [ ] False positive rate < 5% for "critical" alerts
- [ ] Stress detection within 30-point latency budget
- [ ] Bloom detection within 5-point latency budget

### S3: Anderson Overlay
- [ ] Periodic Anderson analysis (every 100 points) completes
- [ ] r tracks diversity trend direction correctly
- [ ] r quantiles consistent with biome classification

### S4: Storage
- [ ] All time series stored in NestGate with timestamps
- [ ] Provenance chain: sensor → feature → classification → storage
- [ ] Alert events stored with metadata

## Compute Estimate

- Feature extraction: ~1ms per point (CPU)
- Akida inference: ~0.5ms per point (NPU)
- Anderson overlay: ~2s every 100 points (CPU) or ~0.2s (GPU)
- Total: real-time capable at 1 Hz with wide margin

## Provenance

| Item | Value |
|------|-------|
| NPU hardware | BrainChip Akida AKD1000, Eastgate |
| Baseline model | Exp160 SNN sentinel (95% top-1, 87% recall) |
| Sensor simulation | Gaussian noise + trend injection |
| Anderson integration | Exp186 dynamic W(t) framework |
