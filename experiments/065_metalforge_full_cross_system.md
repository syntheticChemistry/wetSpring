# Exp065: metalForge Full Cross-System Validation

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross
**Binary:** `validate_metalforge_full`
**Command:** `cargo run --features gpu --release --bin validate_metalforge_full`

---

## Objective

Extend Exp060's cross-substrate validation from Track 1c only (ANI, SNP,
dN/dS, pangenome) to ALL GPU-eligible domains. This is the full metalForge
proof: every domain that CAN run on GPU produces bit-identical (within
tolerance) results whether dispatched to CPU or GPU.

Where Exp064 proves "GPU works", Exp065 proves "the routing doesn't matter" —
the metalForge substrate-independence guarantee across the full portfolio.

---

## Domains Validated (CPU ↔ GPU)

| # | Domain | GPU Primitive | CPU Module |
|---|--------|-------------|-----------|
| 1 | Shannon diversity | `FusedMapReduceF64` | `bio::diversity` |
| 2 | Simpson diversity | `FusedMapReduceF64` | `bio::diversity` |
| 3 | Bray-Curtis | `BrayCurtisF64` | `bio::diversity` |
| 4 | ANI | `ani_batch_f64.wgsl` | `bio::ani` |
| 5 | SNP calling | `snp_calling_f64.wgsl` | `bio::snp` |
| 6 | dN/dS | `dnds_batch_f64.wgsl` | `bio::dnds` |
| 7 | Pangenome | `pangenome_classify.wgsl` | `bio::pangenome` |
| 8 | Random Forest | `rf_batch_inference.wgsl` | `bio::random_forest` |
| 9 | HMM forward | `hmm_forward_f64.wgsl` | `bio::hmm` |

---

## Protocol

For each domain:
1. Compute on CPU using sovereign Rust module — this is ground truth.
2. Compute the same input on GPU via ToadStool primitive or local WGSL.
3. Assert: `|GPU - CPU| ≤ tolerance`.
4. Log CPU time and GPU time for routing decisions.

**Key principle**: the metalForge substrate router should be able to dispatch
ANY of these domains to ANY available substrate (CPU, GPU, or future NPU)
and get the same answer. This experiment proves that guarantee.

---

## Relationship to Other Experiments

| Experiment | What it proves |
|-----------|---------------|
| Exp043/057/061-062 (CPU v3-v5) | CPU math is correct (vs Python/scipy) |
| Exp064 (GPU Parity v1) | GPU math is correct (vs CPU) |
| **Exp065 (metalForge Full)** | **Routing doesn't matter — any substrate works** |
| Exp060 (metalForge Track 1c) | Predecessor — Track 1c only |

---

## Acceptance Criteria

All cross-substrate comparisons must match within documented tolerances.
Every domain reports "CPU=GPU" in the substrate column.

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline commit | `e4358c5` |
| Baseline tool | BarraCUDA CPU (sovereign Rust) |
| Exact command | `cargo run --features gpu --release --bin validate_metalforge_full` |
| Data | Synthetic test vectors (self-contained) |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!_OS 22.04 |
