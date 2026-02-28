# Exp228: metalForge v8 — Cross-System (GPU → NPU → CPU) Dispatch

**Track:** cross (IPC / metalForge)
**Phase:** 71
**Status:** PASS — 33/33 checks
**Binary:** `validate_metalforge_v8_cross_system`
**Features:** `ipc`, `gpu`

## Purpose

Validates cross-system dispatch through the IPC layer: GPU, NPU, and CPU
substrates coordinated via metalForge workload routing. Proves PCIe bypass
topology, DF64 in dispatch, graceful fallback, and error handling.

## Model / Equations

| Domain | What it proves |
|--------|----------------|
| IPC dispatch parity | 5 communities × 2 QS scenarios, full pipeline |
| Workload routing | GPU=5, NPU=3, CPU=2 workload distribution |
| PCIe bypass topology | NPU→GPU direct transfer, no CPU roundtrip |
| DF64 in dispatch | Half-precision through IPC |
| Graceful fallback | Substrate unavailable → CPU fallback |
| Error handling | Invalid substrate, timeout, recovery |

## Validation

- 33 checks across dispatch, routing, topology, and fallback
- 5 communities, 2 QS scenarios exercised through full pipeline
- IPC reachability for all three substrates

## Status

PASS — 33/33 checks
