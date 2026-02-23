# Exp106: Pure GPU Streaming — ODE Biology + Phylogenetics

| Field | Value |
|-------|-------|
| Status | **PASS** — 45/45 |
| Command | `cargo run --features gpu --release --bin validate_streaming_ode_phylo` |
| Phase | 30 |
| Dependencies | Exp049 (ODE sweep), Exp100 (ODE GPU), Exp103/104 (metalForge v5/v6) |
| Binary | `barracuda/src/bin/validate_streaming_ode_phylo.rs` |

## Purpose

Prove that 6 domain-specific GPU primitives can run in pre-warmed streaming mode:
each shader is compiled once at session start, then dispatched repeatedly without
recompilation. This extends streaming coverage from the 16S analytics pipeline
(Exp105) to ODE biology and phylogenetics.

## Pre-warmed Primitives

| Primitive | Domain | Shader Origin | Dispatches |
|-----------|--------|---------------|------------|
| `OdeSweepGpu` | QS biofilm ODE | ToadStool absorbed | 4-batch parameter sweep |
| `PhageDefenseGpu` | Phage defense ODE | Local WGSL | 2 dispatches (different params) |
| `BistableGpu` | Bistable switch ODE | Local WGSL | 2 dispatches (different params) |
| `MultiSignalGpu` | Multi-signal ODE | Local WGSL | 2 dispatches (different params) |
| `FelsensteinGpu` | Phylogenetic pruning | ToadStool absorbed | 2 trees (3-taxon, 2-taxon) |
| `UniFracGpu` | UniFrac propagation | ToadStool absorbed | 2 dispatches (different data) |

## Validation Sections

| Section | Domain | Checks | Notes |
|---------|--------|--------|-------|
| S1 | QS ODE (4-batch) | 5 | All batches finite, correct output size |
| S2 | Phage Defense ODE | 7 | 4 vars finite, Bd/Bu > 0, 2nd dispatch OK |
| S3 | Bistable Switch ODE | 8 | 5 vars finite, cell > 0, biofilm ∈ [0,1], 2nd dispatch |
| S4 | Multi-Signal ODE | 10 | 7 vars finite, cell > 0, biofilm ∈ [0,1], 2nd dispatch |
| S5 | Felsenstein Pruning | 6 | CPU LL finite/negative, GPU rel err < 10%, 2nd tree |
| S6 | UniFrac Propagation | 9 | Leaf parity exact, 2nd dispatch finite |
| **Total** | | **45** | **ALL PASS** |

## Key Observations

- **Warmup**: 25.5 ms for all 6 primitives (cached shaders)
- **Execution**: 541.8 ms total across 6 domains
- **Felsenstein GPU exp/log fallback**: 1.3% relative error (3-taxon tree), 6.1%
  (extreme 2-taxon tree with long branches). Error from polynomial exp/log
  approximation compounding in recursive pruning, not from f64 arithmetic
  (which is native IEEE 754 via Vulkan).
- **Zero shader recompilation**: each primitive compiled once at session start,
  reused across multiple dispatches with different parameters/data.

## Reproduction

```bash
cd barracuda && cargo run --features gpu --release --bin validate_streaming_ode_phylo
```
