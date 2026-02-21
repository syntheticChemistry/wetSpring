# Experiment 074: metalForge Substrate Router — GPU↔NPU↔CPU Dispatch

**Date:** February 21, 2026
**Status:** Active
**Binary:** `validate_substrate_router`

---

## Objective

Build and validate a substrate-aware compute router that dispatches workloads
to GPU, NPU, or CPU based on batch size, workload type, and hardware
availability. Prove correct routing and math parity across all paths.

## Background

The metalForge routing rules (from PRIMITIVE_MAP.md):
```
Batch > dispatch breakeven → GPU
Classification/inference    → NPU (or CPU fallback)
f64 transcendentals on Ada → GPU + polyfill
Small batches (< 64)       → CPU
NPU unavailable             → CPU fallback
```

## Architecture

```rust
pub enum Substrate { Cpu, Gpu, Npu }
pub enum WorkloadClass { BatchParallel, Inference, Sequential }

pub struct SubstrateRouter {
    gpu_available: bool,
    npu_available: bool,
    dispatch_breakeven: usize,  // from Exp067/068 profiling
    ada_lovelace: bool,         // needs polyfill
}
```

## Protocol

### Route 1: GPU Path (batch-parallel, N > breakeven)
Shannon, Simpson, Bray-Curtis on GPU via ToadStool FMR.

### Route 2: CPU Path (small batch or GPU unavailable)
Same operations on CPU. Validates CPU fallback is always correct.

### Route 3: NPU Path (inference workloads)
Taxonomy classification routed to NPU candidate path.
Since NPU model is not yet trained, validates the routing decision
and falls back to CPU with correct result.

### Route 4: Mixed Pipeline (GPU → NPU via PCIe bypass)
Demonstrates the routing of GPU compute output directly to NPU
inference path without CPU DRAM round-trip.
GPU computes diversity metrics → output vector routes to classification.

### Validation Checks
1. Router selects GPU for batch > breakeven
2. Router selects CPU for batch < breakeven
3. Router selects NPU for inference workloads (with CPU fallback)
4. All paths produce identical results
5. Mixed pipeline (GPU→NPU) routing demonstrated
6. PCIe topology awareness: correct device selection

## Expected Outcome
- Routing decisions match the substrate decision tree
- Math parity across all substrates (within tolerances)
- NPU path gracefully falls back to CPU when model unavailable
- Foundation for ToadStool multi-substrate absorption
