# Exp033 — Neighbor-Joining Tree Construction (Liu 2009 SATé)

**Date:** February 20, 2026
**Status:** COMPLETE — 16/16 checks PASS
**Track:** 1b (Comparative Genomics & Phylogenetics)

---

## Objective

Implement and validate the Neighbor-Joining (NJ) algorithm (Saitou & Nei
1987), the core guide-tree primitive used by SATé (Liu 2009) for iterative
alignment-tree co-estimation. NJ builds an unrooted phylogenetic tree from
a pairwise distance matrix in O(n³) time.

## Model

### Neighbor-Joining (Saitou & Nei 1987)

Given an n×n symmetric distance matrix D:
1. Compute Q-matrix: Q(i,j) = (r-2)·D(i,j) - Σ_k D(i,k) - Σ_k D(j,k)
2. Join pair (i,j) minimizing Q
3. Compute branch lengths via row-sum differences
4. Update distance matrix for new node
5. Repeat until 2 nodes remain

### Jukes-Cantor Distance

d = -3/4 · ln(1 - 4p/3) where p = proportion of different sites.
Saturated (p ≥ 0.75) → d = 10.0.

## Modules

- `bio::neighbor_joining` — NJ algorithm with flat distance matrix (GPU-ready)
- `bio::neighbor_joining::jukes_cantor_distance` — JC corrected distance
- `bio::neighbor_joining::distance_matrix` — pairwise distance matrix (batch API)

## Test Cases

| Test | Input | Checks |
|------|-------|--------|
| 3-taxon (X,Y,Z) | d(X,Y)=0.2, d(X,Z)=d(Y,Z)=0.4 | 1 join, X-Y sisters, branch ~0.1 |
| 4-taxon (A,B,C,D) | (A,B) close, (C,D) close | 2 joins, topology preserved |
| 5-taxon from sequences | 12-site DNA alignment | S1-S2 sisters, S4-S5 sisters |
| JC distances | identity, small diff, saturated | 0.0, 0.088337, 10.0 |
| Distance matrix | 5×5 from sequences | Symmetric, diagonal zero |
| Determinism | Two runs, same input | Identical Newick output |

## Acceptance Criteria

All 16 checks PASS:
- Topology: correct sister-pair groupings
- Branch lengths: non-negative, match Python baseline
- JC distances: match Python to ≤1e-6
- Distance matrix: symmetric, diagonal zero
- Deterministic

## Baseline Provenance

- Python: `scripts/liu2009_neighbor_joining.py` (pure Python, no dendropy)
- Reference: Saitou & Nei 1987, *Mol Biol Evol* 4:406-425
- SATé reference: Liu et al. 2009, *Science* 324:1561-1564

## GPU Promotion Path

**Tier A (Rewire):** Distance matrix computation is O(n²) and embarrassingly
parallel — each pairwise JC distance is independent. Q-matrix computation
similarly parallel. `distance_matrix_batch` provides the batch entry point.
For large datasets (1000+ taxa), this is a clear GPU win.

**Tier B:** NJ join step is sequential but fast (O(n³) with small constant).
GPU benefit comes from batching many NJ calls (e.g., bootstrap replicates
each producing a different NJ tree).
