# Exp269: Mixed Hardware Dispatch — NUCLEUS Atomics + PCIe Bypass

| Field | Value |
|-------|-------|
| Binary | `validate_mixed_hw_dispatch` |
| Date | 2026-03-01 |
| Crate | `wetspring-forge` (metalForge) |
| Command | `cargo run -p wetspring-forge --bin validate_mixed_hw_dispatch` |
| Checks | 91/91 PASS |

## Purpose

Validates mixed-hardware compute dispatch through NUCLEUS atomics (Tower, Node, Nest).
Proves PCIe bypass pipelines, bandwidth-aware routing, CPU fallback paths, and the
full 47-workload catalog routing through the metalForge dispatch system.

## Sections

| Section | Focus | Checks |
|---------|-------|--------|
| S1 | Tower discovery + bandwidth tiers (3 GPUs detected) | 9 |
| S2 | NPU→GPU PCIe bypass (zero CPU round-trips) | 4 |
| S3 | GPU→CPU fallback (f64 molecular clock) | 3 |
| S4 | Multi-GPU bandwidth-aware routing (6 workloads) | 12 |
| S5 | Full 8-stage mixed pipeline (GPU+NPU+CPU) | 4 |
| S6 | Write→Absorb→Lean tracking (47 workloads) | 51 |
| S7 | Full catalog Node dispatch (45 GPU + 2 CPU-only) | 4 |
| S8 | biomeOS graph: Tower+Node+Nest coordination | 5 |

## Hardware Detected

- 12th Gen Intel Core i9-12900K (CPU)
- NVIDIA GeForce RTX 4070 (GPU, PCIe Gen4)
- NVIDIA TITAN V / NVK GV100 (GPU)
- NVIDIA GeForce RTX 4070/PCIe/SSE2 (GPU, Vulkan)

## Key Results

- S1: 3 GPUs with bandwidth tiers; RTX 4070 at 37µs/MB, 3180µs/100MB
- S2: GPU→GPU→NPU pipeline: 2 chained, 0 CPU round-trips, fully streamable
- S5: 8-stage: 5 GPU chained → NPU bypass → 2 CPU stages
- S6: 96% absorption rate (45/47 absorbed by ToadStool)
- S7: All 45 GPU-capable workloads routed; 2 CPU-only (correct)
- S8: Sovereign mode — operates without Songbird or NestGate
