# Exp109: Large-Scale Phylogenetic Placement

**Date**: February 23, 2026
**Status**: PASS — 11/11 checks
**Binary**: `validate_phylo_placement_scale`
**Faculty**: Liu (MSU CMSE)

## Purpose

Validates phylogenetic distance computation, NJ tree construction, and
Felsenstein likelihood at 128-taxon scale with 50 placement queries.
Demonstrates scaling characteristics for real-world phylogenomic workflows
(Tara Oceans rplB-scale gene families).

## Data

Synthetic: 128 taxa × 300-site alignment with 5% mutation rate from shared
ancestor, plus 50 divergent query sequences for placement.

## Results

- JC distance matrix (128×128): 1.1 ms
- NJ tree (126 joins): 9.4 ms, 3070-char Newick
- 100 subtree Felsenstein likelihoods: 1.1 ms (LL range: -792.94 to -661.05)
- 50 placement queries: 0.8 ms → 10 unique targets
- Scaling: N=16 (0.0 ms) → N=32 (0.1 ms) → N=64 (0.6 ms) → N=128 (6.9 ms)

## Key Findings

1. Distance matrix computation is O(N²) and dominates at large N. At 128
   taxa with 300-site alignment, it's still sub-millisecond in Rust.
2. NJ tree construction scales well to 128 taxa (9.4 ms). The O(N³) algorithm
   is not a bottleneck until hundreds of taxa.
3. Placement via distance-to-nearest-reference is fast but coarse. GPU
   Felsenstein per-edge likelihood (available via ToadStool) would give
   principled placement at the cost of O(E×L) per query.

## Reproduction

```bash
cargo run --release --bin validate_phylo_placement_scale
```
