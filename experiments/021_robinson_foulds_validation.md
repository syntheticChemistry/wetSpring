# Experiment 021: Robinson-Foulds Tree Distance Validation

**Date**: 2026-02-19
**Status**: COMPLETE — Python baseline validated, Rust module validated (23/23 checks PASS)
**Track**: 1b (Comparative Genomics, Liu)
**Paper Queue**: #15, #16 (Liu PhyloNet-HMM, Alamin metagenomic placement)

---

## Objective

Implement and validate Robinson-Foulds (RF) symmetric distance for
phylogenetic tree comparison. This is the core tree-comparison primitive
needed for Exp019 (phylogenetic validation), Liu's PhyloNet-HMM introgression
detection, and Alamin's metagenomic placement.

## References

- Robinson & Foulds 1981, "Comparison of phylogenetic trees."
  *Mathematical Biosciences* 53:131-147
- Liu et al. 2014, PLoS Comp Bio 10:e1003649 (PhyloNet-HMM)
- Alamin & Liu 2024, IEEE/ACM TCBB (metagenomic placement)

## Data

### Synthetic Newick Trees

Constructed trees with known RF distances for analytical validation:
- Identical trees: RF = 0
- Single NNI (nearest-neighbor interchange): RF = 2
- Fully different topologies: RF = maximum
- Trees with polytomies and unresolved nodes

### Validation Against dendropy (Python)

- Python script: `scripts/rf_distance_baseline.py`
- Library: `dendropy` (standard phylogenetics library)
- Output: `experiments/results/021_rf_baseline/rf_python_baseline.json`

## Design

### Phase 1: RF Distance Module (`bio::robinson_foulds`)

1. Extract bipartition (split) sets from rooted/unrooted trees
2. Symmetric difference of split sets = RF distance
3. Normalized RF: RF / (2*(n-3)) for unrooted binary, RF / (2*(n-2)) for rooted
4. Handle degenerate cases: single-leaf, two-leaf, polytomies

### Phase 2: Validation Binary (`validate_rf_distance`)

1. Analytical cases with known RF distances
2. Cross-validate against dendropy on synthetic trees
3. Parse Newick, compute RF, compare to baseline

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Analytical RF (identical trees) | 0 | Definition |
| Analytical RF (single NNI) | 2 | Theory |
| Rust vs dendropy | Exact match | Cross-validation |
| Newick round-trip | Topology preserved | Parser correctness |

## Evolution Path

```
Python (dendropy) → Rust CPU (bipartition sets) → GPU (batch RF matrix)
```
