# Experiment 073: Compute Dispatch Overhead — Streaming vs Individual vs CPU

**Date:** February 21, 2026
**Status:** Active
**Binary:** `validate_dispatch_overhead_proof`

---

## Objective

Quantify ToadStool's dispatch overhead reduction by measuring the same
workload across three dispatch strategies at multiple batch sizes.
This validates the unidirectional streaming architecture and guides
absorption priorities.

## Protocol

For batch sizes [64, 256, 1024, 4096]:

### Strategy A: CPU Sequential
All operations on CPU. Zero dispatch overhead. Scales linearly with N.

### Strategy B: GPU Individual Dispatch
Each operation creates its own pipeline instance, uploads buffer, dispatches,
and reads back. Maximum overhead per operation.

### Strategy C: GPU Streaming (Pre-warmed Session)
Single `GpuPipelineSession` with pre-compiled pipelines. Buffer pool reuse,
bind-group cache. Minimum overhead per operation.

### Metrics Collected
- Total wall time per strategy per batch size
- Per-operation overhead (total_time / n_operations)
- GPU dispatch overhead = Strategy B - Strategy C (pure overhead delta)
- CPU crossover point (where GPU streaming beats CPU)

### Validation Checks
1. All three strategies produce identical results (within tolerance)
2. Strategy C overhead < Strategy B overhead at every batch size
3. GPU streaming beats CPU above the dispatch crossover point
4. Buffer pool stats show reuse in Strategy C

## Expected Outcome
- At batch=64: CPU may still win (dispatch overhead > compute savings)
- At batch=256+: GPU streaming should dominate
- Overhead delta (B-C) should be ~constant (it's PCIe transfer overhead)
