# Exp035 — BarraCUDA CPU Parity v2

**Date:** February 20, 2026
**Status:** COMPLETE — 18/18 checks PASS
**Track:** Cross-domain

---

## Objective

Validate that the new GPU-ready APIs (FlatTree, batch HMM, batch SW,
Neighbor-Joining, DTL Reconciliation) produce identical results to their
sequential counterparts on CPU. This proves the GPU-ready data layouts
don't sacrifice correctness and extends BarraCUDA CPU parity to 11 domains.

## Domains Validated

| Domain | Module | API Tested | Checks |
|--------|--------|-----------|--------|
| Phylogenetic likelihood | `bio::felsenstein` | `FlatTree` vs recursive | 5 |
| Hidden Markov Models | `bio::hmm` | `forward_batch` / `viterbi_batch` vs sequential | 4 |
| Sequence alignment | `bio::alignment` | `score_batch` vs `smith_waterman_score` | 2 |
| Tree construction | `bio::neighbor_joining` | `neighbor_joining` + `distance_matrix` | 3 |
| Cophylogenetics | `bio::reconciliation` | `reconcile_dtl` | 2 |
| Cross-module | NJ → Felsenstein | End-to-end pipeline | 2 |

## Key Results

### FlatTree ↔ Recursive Parity
The `FlatTree` GPU-ready layout produces bit-exact log-likelihoods compared
to the recursive `TreeNode` implementation. This validates that tree
linearization (post-order array, column-major leaf states, precomputed
transition matrices) preserves numerical results.

### Batch ↔ Sequential Parity
All batch APIs (`forward_batch`, `viterbi_batch`, `score_batch`) produce
identical results to their sequential counterparts. This confirms that
batching for GPU dispatch doesn't affect computation.

### Cross-Module Integration
A Neighbor-Joining tree scored via both recursive and FlatTree Felsenstein
produces identical log-likelihoods, validating the NJ → Felsenstein pipeline.

## Acceptance Criteria

All 18 checks PASS:
- FlatTree LL matches recursive to <1e-12
- Batch HMM forward/Viterbi match sequential exactly
- Batch SW scores match sequential exactly
- NJ produces valid Newick with correct join count
- DTL reconciliation matches Python baseline
- Cross-module pipeline produces consistent results

## BarraCUDA CPU Parity Summary (v1 + v2)

| Version | Domains | Checks |
|---------|---------|--------|
| v1 (Exp029) | ODE, SSA, HMM, SW, Felsenstein, diversity, signal, game theory, tree distance | 21 |
| **v2 (Exp035)** | FlatTree, batch HMM, batch SW, NJ, DTL, cross-module | 18 |
| **Total** | **11 domains** | **39 checks** |

This demonstrates that pure Rust math on CPU matches Python across all
validated algorithmic domains, and the GPU-ready data layouts preserve
correctness — the bridge to pure GPU execution.
