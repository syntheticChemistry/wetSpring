# Exp329: metalForge + petalTongue Visualization

**Phase:** V100 (petalTongue + metalForge integration)
**Date:** 2026-03-09
**Binary:** `validate_metalforge_petaltongue` (forge crate)

## Status: PASS (19/19)

## Scope

Integrates petalTongue visualization into metalForge's hardware discovery,
workload dispatch, and NUCLEUS topology systems.

## Domains

| Domain | Checks | Description |
|--------|--------|-------------|
| MF1 Inventory | 5 | Hardware nodes from live-probed substrates, memory gauges, capability bars, summary counts |
| MF2 Dispatch | 3 | Workload routing bar (all bio workloads × routable), coverage gauge |
| MF3 NUCLEUS | 6 | Tower→Node→Nest cyclic topology, substrate count gauge, edge validation |
| MF4 JSON | 5 | Scenario serialization round-trips, JSON parse, content verification |

## Architecture

```
metalForge/forge/src/visualization/
  mod.rs — inventory_scenario, dispatch_scenario, nucleus_scenario
           Reuses wetspring_barracuda::visualization::{DataChannel, EcologyScenario, ...}
```

## Key Design Decisions

- Reuses wetspring-barracuda's `DataChannel` types (no duplicate schema)
- Live hardware probing via `inventory::discover()` — real substrates
- NUCLEUS atomics (Tower→Node→Nest) represented as scenario nodes with cyclic edges
- Workload routability computed via `dispatch::route()` for real capability matching
- No petalTongue crate dependency — JSON schema + IPC only

## Chain Position

CPU v25 (46/46) → GPU v14 (27/27) → metalForge v17 (29/29) → Viz V1 (45/45) →
CPU↔GPU Math (27/27) → **metalForge+petalTongue (19/19)**
