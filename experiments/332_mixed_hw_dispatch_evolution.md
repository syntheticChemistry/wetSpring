# Exp332: Mixed Hardware Dispatch Evolution

## Status: PASS (24/24)

**Binary**: `validate_mixed_hw_dispatch_v2`
**Crate**: `wetspring-forge`

## Scope

Validates bandwidth-aware routing, workload `data_bytes` wiring, mixed GPU+NPU+CPU
dispatch, NUCLEUS topology, and petalTongue visualization overlay.

| Domain | What's Validated | Checks |
|--------|-----------------|--------|
| D1 Bandwidth-Aware | `route_bandwidth_aware` falls back on high transfer cost | 4 |
| D2 Workload Wiring | `BioWorkload.data_bytes` propagates through dispatch | 6 |
| D3 Mixed Substrate | GPU→NPU→CPU priority chain, impossible routing | 5 |
| D4 NUCLEUS Topology | BandwidthTier detection, PCIe transfer cost estimation | 5 |
| D5 petalTongue | Inventory + dispatch scenarios serialize to JSON | 4 |

## Key Findings

- **Bandwidth fallback**: 100 MB over PCIe 4.0 x16 exceeds dispatch overhead → CPU fallback correct.
- **Small data**: 64 B transfer cost negligible → GPU routing preserved.
- **Preference override**: GPU-preferred workloads skip bandwidth fallback (user intent trumps heuristic).
- **NPU routing**: 8-bit quantized inference routes to NPU when available.
- **PCIe cost model**: 1 MB transfer on PciE4x16 ≈ 38 µs (matches barracuda's model).

## Architecture

```
BioWorkload (provenance.rs)
  → Workload.data_bytes (dispatch.rs)
    → route_bandwidth_aware()
      → BandwidthTier::detect_from_adapter_name()
        → transfer_cost().estimated_us()
          → fallback if transfer > GPU_DISPATCH_OVERHEAD_US
```

## Chain Position

```
Exp327 (viz schema) → Exp328 (CPU/GPU math) → Exp329 (metalForge viz)
    → Exp330 (full chain) → Exp331 (local evolution) → Exp332 (mixed HW)
```
