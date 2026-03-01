# Exp257: NUCLEUS Data Acquisition Pipeline — Three-Tier Primal Routing

**Status:** PASS (9/9 checks)
**Date:** 2026-03-01
**Binary:** `validate_nucleus_data_pipeline`
**Command:** `cargo run --release --bin validate_nucleus_data_pipeline`
**Feature gate:** none (standalone-capable; full IPC when ecosystem running)

## Purpose

Validates the full NUCLEUS data acquisition chain by probing the live
ecosystem state and exercising whichever routing tier is available:

| Tier | Strategy | When |
|------|----------|------|
| 1 | biomeOS Neural API `capability.call` | biomeOS orchestrator running |
| 2 | Direct NestGate socket | Standalone + NestGate available |
| 3 | Sovereign HTTP | No ecosystem services |

## Probes

1. **Socket Discovery** — scans for biomeOS, Songbird, NestGate, wetSpring sockets
2. **Songbird Registration** — tests `discovery.list` if Songbird available
3. **NestGate NCBI** — tests `ncbi.capabilities` and `health.check`
4. **Neural API** — tests `capability.list` and `capability.call("science.diversity")`
5. **Direct IPC** — tests `health.check` and `science.diversity` on wetSpring socket
6. **Standalone Math** — Shannon, Simpson, Chao1 always run as baseline

## NUCLEUS Evolution Scoreboard

The experiment reports a live status table:

```
╔══════════════════════════════════════════════════╗
║ Primal     │ Status  │ Evolution Need            ║
╠══════════════════════════════════════════════════╣
║ biomeOS    │ LIVE/offline │ batch dispatch       ║
║ Songbird   │ LIVE/offline │ multi-gate federation║
║ NestGate   │ LIVE/offline │ BIOM parser + SRA    ║
║ wetSpring  │ LIVE/offline │ batch pipeline       ║
║ BearDog    │ proven  │ lineage verification      ║
║ ToadStool  │ absorbing│ S70+++ shader evolution  ║
╚══════════════════════════════════════════════════╝
```

## Activation Path

```bash
biomeos nucleus start --mode node --node-id eastgate
cargo run --release --bin wetspring_server
WETSPRING_DATA_PROVIDER=nestgate cargo run --release --bin validate_nucleus_data_pipeline
```

## Chain

Exp203 (IPC pipeline) → Exp204 (Songbird) → Exp205 (sovereign fallback) →
Exp206 (IPC dispatch) → Exp256 (EMP atlas) → **Exp257 (NUCLEUS pipeline)** →
Exp258 (Tower-Node)
