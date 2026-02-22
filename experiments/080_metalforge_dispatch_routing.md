# Exp080: metalForge Dispatch Router Validation

**Status**: COMPLETE
**Date**: 2026-02-22
**Binary**: `validate_dispatch_routing` (in `metalForge/forge`)
**Checks**: 35 (all PASS)

## Purpose

Validates the forge dispatch router's life-science workload classification
across every substrate configuration the ecosystem will encounter. Proves that
the capability-based routing correctly directs ODE, diversity, HMM, taxonomy,
and I/O workloads to the optimal substrate (GPU/NPU/CPU) in all hardware
configurations, including mixed PCIe topologies.

## Evolution Chain Position

```
Python → Rust CPU → flat API (Exp078-079) → dispatch routing [THIS] → GPU execution
```

This experiment validates the routing layer that sits between the flat-API
workloads and actual hardware execution. Once ToadStool absorbs the f64 shader
support, these dispatch decisions become live GPU dispatches.

## Substrate Configurations Tested

| Config | Substrates | ODE Route | Taxonomy Route | FASTQ Route |
|--------|-----------|-----------|----------------|-------------|
| Full system | GPU f64 + NPU + CPU | GPU | NPU | GPU |
| GPU + CPU | GPU f64 + CPU | GPU | none | GPU |
| NPU + CPU | NPU + CPU | none (GPU) / CPU (fallback) | NPU | CPU |
| CPU only | CPU | none | none | CPU |
| Mixed PCIe | f32 iGPU + f64 dGPU + NPU + CPU | f64 GPU | NPU | GPU |

## Key Validations

### ODE Workload Routing (5 modules)
- All 5 ODE modules route to GPU when f64 GPU is present
- ODE GPU workload correctly fails when no shader-capable substrate exists
- ODE CPU fallback variant (f64-only) routes to CPU when GPU unavailable
- Mixed PCIe: ODE routes to discrete f64 GPU, not integrated f32 iGPU

### Cross-Substrate Correctness
- Taxonomy classification → NPU (quantized inference, 8-bit)
- Batch anomaly detection → NPU (quant + batch capability)
- Felsenstein pruning → GPU (f64 + shader)
- FASTQ parsing → CPU fallback when no GPU

### Preference Overrides
- User can force CPU routing even when GPU is available
- Incapable preference falls back to best available (not error)

### Live Hardware Discovery
- Discovers 1 CPU, 3 GPU(s), 1 NPU(s) on Eastgate machine
- FASTQ always routable regardless of hardware
- Felsenstein routes to real GPU on this machine

## Mixed Hardware (PCIe Bypass) Design

The mixed PCIe section validates the foundation for GPU↔NPU direct dispatch:

```
            ┌──────────┐
            │  Forge   │  capability-based routing
            │ Dispatch │
            └──┬───┬───┘
    ┌──────────┘   └──────────┐
    ▼                         ▼
┌──────────┐            ┌──────────┐
│ f64 GPU  │────PCIe────│   NPU    │   direct transfer
│ (shader) │            │ (quant)  │   bypasses CPU
└──────────┘            └──────────┘
```

When forge identifies GPU workloads (ODE, diversity) and NPU workloads
(taxonomy, anomaly), it routes each to the capable substrate. The PCIe
bypass path (GPU→NPU direct) is architecturally enabled by this routing —
forge knows which substrates are on the same PCIe fabric.

## Provenance

| Field | Value |
|-------|-------|
| Baseline | forge dispatch API v0.1.0 |
| Baseline date | 2026-02-22 |
| Exact command | `cargo run --bin validate_dispatch_routing` |
| Crate | metalForge/forge |
