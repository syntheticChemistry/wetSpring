# Exp258: NUCLEUS Tower-Node Deployment — Live Primal Orchestration

**Status:** PASS (13/13 checks)
**Date:** 2026-03-01
**Binary:** `validate_nucleus_tower_node`
**Command:** `cargo run --release --features ipc --bin validate_nucleus_tower_node`
**Feature gate:** `ipc` (requires JSON-RPC dispatch module)

## Purpose

Validates the full NUCLEUS Tower→Node deployment model by probing binary
availability, measuring IPC vs direct dispatch overhead, and exercising
the complete science pipeline through JSON-RPC dispatch.

## Findings

### Primal Binary Status (Eastgate)

| Primal | Location | Status |
|--------|----------|--------|
| biomeOS | `../phase2/biomeOS/target/release/biomeos` | v0.1.0 ✓ |
| BearDog | `~/.local/bin/beardog` | ✓ |
| Songbird | `~/.local/bin/songbird` | ✓ |
| ToadStool | `~/.local/bin/toadstool` | ✓ |
| NestGate | `~/.local/bin/nestgate` | ✓ |
| Squirrel | `~/.local/bin/squirrel` | ✓ |

### NUCLEUS Mode Readiness

| Mode | Requires | Status |
|------|----------|--------|
| Tower | BearDog + Songbird | READY |
| Node | Tower + ToadStool | READY |
| Nest | Tower + NestGate + Squirrel | READY |
| Full | All primals | READY |

### IPC Dispatch Performance

| Path | Time (200 taxa Shannon, 1000×) | Overhead |
|------|---:|---:|
| Direct function call | 0.86 µs/call | 1.0× |
| JSON-RPC dispatch | 2.74 µs/call | 3.2× |

- **Math fidelity:** |direct - ipc| = 0.00e0 (bit-identical)
- **Full pipeline dispatch:** 0.87ms (diversity + QS model + Anderson)

### Key Insight

The 3.2× IPC overhead is negligible for science workloads where Anderson
spectral (500ms) dominates. JSON-RPC serialization adds ~2µs per call —
meaningless at scale. The dispatch architecture is validated for NUCLEUS.

## Deployment Roadmap

```
Step 1: biomeos nucleus start --mode tower --node-id eastgate
Step 2: biomeos nucleus start --mode node --node-id eastgate
Step 3: cargo run --release --bin wetspring_server
Step 4: cargo run --release --bin validate_nucleus_data_pipeline
Step 5: (after 10G cables) biomeos nucleus start --mode full
```

## Chain

Exp203-206 (IPC validated) → Exp256 (EMP atlas) → Exp257 (data pipeline) →
**Exp258 (Tower-Node)** → NUCLEUS deployment
