# Exp330: biomeOS + NUCLEUS + petalTongue Full Chain

**Phase:** V100 (full ecosystem integration)
**Date:** 2026-03-09
**Binary:** `validate_biomeos_petaltongue_full` (forge crate)

## Status: PASS (34/34)

## Scope

Apex validation — every primal interaction exercised in one binary:
biomeOS → NUCLEUS → Science → petalTongue → metalForge → composed graph.

## Domains

| Domain | Checks | Description |
|--------|--------|-------------|
| B1 Capability Registry | 3 | 9 wetSpring capabilities (6 science + 3 brain) registered for biomeOS |
| B2 Science Pipeline | 5 | Shannon, Bray-Curtis, PCoA, ODE integration — CPU-computed live math |
| B3 Viz Export | 7 | ecology, ordination, dynamics, anderson, benchmark, full pipeline → JSON |
| B4 metalForge Overlay | 4 | Live hardware discovery, inventory/dispatch/nucleus scenarios |
| B5 Full Graph | 15 | Composed graph: biomeos→tower→node→diversity→nest, 10+ nodes, 8+ edges, JSON round-trip |

## Architecture

```
biomeOS → [capability registry: 9 capabilities]
       ↓
NUCLEUS → Tower (discover) → Node (dispatch) → Nest (persist)
       ↓
Science → diversity, ordination, dynamics, chemistry, anderson, benchmarks
       ↓
petalTongue → DataChannel JSON → EcologyScenario → composed full-chain graph
       ↓
metalForge → hardware inventory + workload dispatch + NUCLEUS topology overlay
```

## Composed Full-Chain Graph

The final composed `EcologyScenario` contains:
- **biomeOS**: orchestrator node with 9 registered capabilities
- **NUCLEUS**: Tower + Node + Nest atomic nodes with cyclic edges
- **Science**: diversity, beta diversity, QS biofilm, bistable switch nodes
- **Hardware**: GPU/NPU/CPU substrates with memory gauges and capability bars
- **Edges**: orchestration (biomeos→tower), data_flow (tower→node), compute (node→diversity), storage (diversity→nest), plus cross-domain science edges

## Chain Position

CPU v25 (46/46) → GPU v14 (27/27) → metalForge v17 (29/29) → Viz V1 (45/45) →
CPU↔GPU Math (27/27) → metalForge+petalTongue (19/19) → **biomeOS Full Chain (34/34)**

## Total V100 Integration

| Experiment | Checks | Status |
|-----------|--------|--------|
| Exp327 Viz V1 | 45/45 | PASS |
| Exp328 CPU↔GPU | 27/27 | PASS |
| Exp329 metalForge | 19/19 | PASS |
| Exp330 Full Chain | 34/34 | PASS |
| **Total** | **125/125** | **ALL PASS** |
