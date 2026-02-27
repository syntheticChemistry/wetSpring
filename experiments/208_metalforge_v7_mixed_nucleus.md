# Exp208: metalForge v7 — Mixed Hardware NUCLEUS Atomics

**Date:** 2026-02-27
**Status:** PASS (74/74 checks)
**Track:** cross/IPC/metalForge
**Binary:** `cargo run --features ipc --release --bin validate_metalforge_v7_mixed`

## Hypothesis

Cross-substrate dispatch through the IPC layer + metalForge routing produces
correct results regardless of substrate. NUCLEUS atomic coordination
(Tower→Node→Nest) enables mixed-hardware pipelines where NPU→GPU data flows
bypass the CPU via PCIe direct transfer.

## Method

Eight validation domains:

| Domain | Checks | What it proves |
|--------|:------:|----------------|
| MF01: Cross-Substrate Diversity | 20 | 5 communities × 4 metrics — all `EXACT_F64` |
| MF02: Cross-Substrate Bray-Curtis | 5 | 3 pairs + large pair (300 species) + range |
| MF03: Cross-Substrate QS ODE | 12 | 4 scenarios × 3 checks — all `EXACT_F64` |
| MF04: PCIe Bypass Topology | 4 | 5-stage pipeline, 2 GPU-chained, NPU→GPU direct |
| MF05: GPU→CPU Fallback | 4 | Anderson graceful -32001, diversity CPU fallback, pipeline completes |
| MF06: NUCLEUS Tower→Node→Nest | 12 | Tower health, 6 caps, Node diversity parity, Nest QS storage |
| MF07: biomeOS Graph E2E | 10 | Full pipeline: diversity + QS + Anderson, all parity checked |
| MF08: Workload Routing | 7 | GPU=5, NPU=3, CPU=2 workloads, IPC reachability |

## Results

- 74/74 checks passed
- PCIe bypass: 2 GPU-chained stages (diversity→QS→Anderson) with zero CPU roundtrip
- NPU→GPU direct buffer transfer validated (taxonomy→diversity→Anderson chain)
- GPU→CPU fallback graceful (anderson -32001, diversity continues on CPU)
- NUCLEUS atomics: Tower announces 6 capabilities, Node executes with zero drift, Nest stores metrics
- Total wall-clock: 34ms (debug)

## Architecture

```
Tower (wetspring-server IPC)
  ├── Node: science.diversity    → GPU (FusedMapReduceF64)
  ├── Node: science.anderson     → GPU (Lanczos spectral)
  ├── Node: science.qs_model     → GPU (BatchedOdeRK4 sweep)
  ├── Node: science.ncbi_fetch   → CPU (three-tier: biomeOS→NestGate→sovereign)
  ├── Node: science.full_pipeline → GPU streaming (chained dispatch)
  └── Nest: metrics.snapshot     → JSON (Neural API pathway learning)

PCIe Bypass (no CPU roundtrip):
  NPU(taxonomy) →[GPU buffer]→ GPU(diversity) →[GPU buffer]→ GPU(anderson)

biomeOS Graph: science_pipeline.toml
  NestGate(fetch) → wetSpring(science) → ToadStool(GPU)
```

## Key Finding

The NUCLEUS atomic model works: Tower coordinates, Node executes on optimal
hardware, Nest captures metrics. The IPC dispatch layer + metalForge routing
enables mixed-hardware pipelines where PCIe bypass eliminates CPU roundtrips
between GPU stages. GPU→CPU fallback is graceful — the pipeline completes
with reduced capability rather than failing.

## Modules Validated

`ipc::dispatch`, `bio::diversity`, `bio::qs_biofilm`, `ncbi::nestgate`,
metalForge substrate routing model, streaming pipeline topology analysis,
NUCLEUS Tower→Node→Nest coordination
