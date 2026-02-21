# Exp064: BarraCUDA GPU Parity v1 — Consolidated GPU Domain Validation

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross/GPU
**Binary:** `validate_barracuda_gpu_v1`
**Command:** `cargo run --features gpu --release --bin validate_barracuda_gpu_v1`

---

## Objective

Consolidated GPU parity validation across all GPU-eligible domains. This is
the GPU analogue of `barracuda_cpu_v1-v5` — a single binary that proves pure
GPU math matches CPU reference truth for every domain that can run on GPU.

The goal: every domain that CAN run on GPU DOES run on GPU and produces
identical results to the CPU reference. This is the "pure GPU" proof.

---

## Domains Validated

| Domain | GPU Primitive | GPU Strategy |
|--------|-------------|-------------|
| Diversity (Shannon, Simpson) | `FusedMapReduceF64` | Lean (ToadStool) |
| Bray-Curtis distance | `BrayCurtisF64` | Lean (ToadStool) |
| Spectral cosine | `GemmF64` + `FMR` | Lean (ToadStool) |
| PCoA eigendecomposition | `BatchedEighGpu` | Lean (ToadStool) |
| Smith-Waterman alignment | `SmithWatermanGpu` | Lean (ToadStool absorbed) |
| Felsenstein pruning | `FelsensteinGpu` | Lean (ToadStool absorbed) |
| Decision tree inference | `TreeInferenceGpu` | Lean (ToadStool absorbed) |
| HMM forward (log-space) | `hmm_forward_f64.wgsl` | Local WGSL |
| ODE parameter sweep | `batched_qs_ode_rk4_f64.wgsl` | Local WGSL |
| ANI pairwise | `ani_batch_f64.wgsl` | Local WGSL |
| SNP calling | `snp_calling_f64.wgsl` | Local WGSL |
| dN/dS (Nei-Gojobori) | `dnds_batch_f64.wgsl` | Local WGSL |
| Pangenome classification | `pangenome_classify.wgsl` | Local WGSL |
| Random Forest batch | `rf_batch_inference.wgsl` | Local WGSL |
| Quality filtering | `quality_filter.wgsl` | Local WGSL |

---

## Protocol

1. For each domain, compute the CPU reference value (using the validated
   sovereign Rust implementation).
2. Run the same computation on GPU via the appropriate ToadStool primitive
   or local WGSL shader.
3. Compare GPU result against CPU reference within documented tolerance
   (from `tolerances.rs`).
4. Report per-domain timing (CPU µs, GPU µs) for performance comparison.

---

## Acceptance Criteria

All GPU results must match CPU reference within `GPU_VS_CPU_*` tolerances:
- Integer operations: exact match (0.0)
- f64 arithmetic: ≤ 1e-6 (`GPU_VS_CPU_F64`)
- f64 transcendentals: ≤ 1e-10 (`GPU_VS_CPU_TRANSCENDENTAL`)
- Ensemble ML: ≤ 1e-4 (`GPU_VS_CPU_ENSEMBLE`)

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline commit | `e4358c5` |
| Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
| Exact command | `cargo run --features gpu --release --bin validate_barracuda_gpu_v1` |
| Data | Synthetic test vectors (self-contained, no external data) |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04 |
