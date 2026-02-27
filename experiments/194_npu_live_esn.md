# Exp194: NPU Live — ESN Reservoir on Real AKD1000

**Date:** February 26, 2026
**Phase:** V60 — NPU Live
**Binary:** `validate_npu_live`
**Command:** `cargo run --release --features npu --bin validate_npu_live`
**Status:** DONE (NPU, 23/23 checks PASS)

---

## Purpose

Run all 3 ESN classifiers (QS Phase, Bloom Sentinel, Disorder) on real AKD1000
hardware via DMA, comparing CPU int8 simulation to live NPU results. Exercises
reservoir weight loading, online readout switching, batch inference, and power
profiling.

## Results

### ESN Classifier Comparison (Sim vs Hardware)

| Classifier | Config | CPU Sim | NPU Live | Throughput |
|------------|--------|---------|----------|------------|
| QS Phase (Exp114) | 5→200→3, seed=42 | 49.2% | 33.6% | 18,837 Hz |
| Bloom Sentinel (Exp118) | 6→200→4, seed=2025 | 25.0% | 25.3% | 18,773 Hz |
| Disorder (Exp119) | 5→180→3, seed=314 | 32.9% | 31.6% | 18,626 Hz |

### Hardware Capabilities

| Capability | Measured |
|------------|----------|
| Reservoir weight loading (200×200) | 164 KB in 4.5 ms (37 MB/s) |
| Online readout switching | 3 swaps in 86 µs (QS→Bloom→QS) |
| Batch inference (8-wide) | 20,754 infer/sec |
| Energy per inference | 1.4 µJ |
| Coin-cell CR2032 (1 Hz edge) | 4,003 days (11 years) |
| NPU vs GPU energy ratio | 50× more efficient |

## Sections

| Section | What |
|---------|------|
| S1 | QS Phase Classifier — sim vs live |
| S2 | Bloom Sentinel — sim vs live |
| S3 | Disorder Classifier — sim vs live |
| S4 | Reservoir Weight Loading (W_in + W_res to SRAM) |
| S5 | Online Readout Switching (weight mutation) |
| S6 | Batch Inference (8-wide) |
| S7 | Power & Energy Profile |
