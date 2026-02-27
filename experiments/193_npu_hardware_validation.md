# Exp193: NPU Hardware Validation — Real AKD1000 DMA + Discovery

**Date:** February 26, 2026
**Phase:** V60 — NPU Live
**Binary:** `validate_npu_hardware`
**Command:** `cargo run --release --features npu --bin validate_npu_hardware`
**Status:** DONE (NPU, all checks PASS)

---

## Purpose

First wetSpring binary to interact with real AKD1000 silicon. Validates the
ToadStool `akida-driver` integration through wetSpring's `npu` module:
runtime discovery, device open, capability query, DMA read/write, and
int8 quantization fidelity.

## Hardware

| Property | Value |
|----------|-------|
| Device | BrainChip AKD1000 |
| PCIe address | `0000:08:00.0` |
| NPUs | 80 |
| SRAM | 10 MB |
| PCIe bandwidth | 0.5 GB/s (Gen2 x1) |
| Driver | ToadStool `akida-driver` 0.1.0 (pure Rust) |
| Host | Intel i9-12900K, 64 GB DDR5 |

## Sections

| Section | What | Key Result |
|---------|------|------------|
| S1 | Runtime Device Discovery | `npu_available()`, `npu_summary()` — zero hardcoded paths |
| S2 | Device Open + Capability Query | Chip version, PCIe gen/lanes/bandwidth confirmed |
| S3 | DMA Write (Host → NPU SRAM) | 37 MB/s sustained |
| S4 | DMA Read (NPU SRAM → Host) | 37 MB/s sustained |
| S5 | Int8 Quantization Fidelity | Round-trip error < 0.1 |
| S6 | Sentinel Feature DMA Round-Trip | 8-byte vector, latency + throughput |
| S7 | Bulk DMA Throughput | 100 iterations, sustained measurement |

## Provenance

- Driver: `phase1/toadstool/crates/neuromorphic/akida-driver`
- Module: `wetspring_barracuda::npu`
- Error: `Error::Npu(String)`
