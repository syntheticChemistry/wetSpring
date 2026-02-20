# Exp036 — RF Distances on Real PhyNetPy Gene Trees

**Date:** February 20, 2026
**Status:** COMPLETE — 15/15 checks PASS
**Track:** 1b (Comparative Genomics & Phylogenetics)
**Phase:** Exp019 Phase 2

---

## Objective

Validate the Robinson-Foulds distance module on real gene trees from the
PhyNetPy project (NakhlehLab/PhyNetPy). Tests pairwise RF distances on
1,160 multi-locus gene trees simulated under deep coalescence and gene
flow (DEFJ benchmark), with 25 leaves per tree.

## Data Source

- **Repository:** NakhlehLab/PhyNetPy (GitHub)
- **Directory:** DEFJ/10Genes/withOG/E/g10/n3/{t4,t20}
- **Trees:** 10 gene trees × 10 replicates × 2 scenarios = 200 trees
- **Leaf count:** 25 per tree (3 alleles × 9 species, minus 2 absent)
- **Total pairwise RF comparisons:** 900

## Python Baseline

Script: `scripts/phynetpy_rf_baseline.py`
Output: `experiments/results/036_phynetpy_rf/python_baseline.json`

Computes RF distances using a split-based algorithm on tokenized Newick
strings. Baseline includes 3 exact sample pairs for Rust validation.

## Validation Checks (15/15)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | tree_0 leaf count | 25 | PASS |
| 2 | tree_1 leaf count | 25 | PASS |
| 3 | tree_2 leaf count | 25 | PASS |
| 4 | RF(tree_0, tree_1) | 38 | PASS |
| 5 | RF(tree_0, tree_2) | 32 | PASS |
| 6 | RF(tree_1, tree_2) | 26 | PASS |
| 7 | RF symmetry (0,1) | = | PASS |
| 8 | RF symmetry (0,2) | = | PASS |
| 9 | RF symmetry (1,2) | = | PASS |
| 10 | RF self-distance = 0 (t0) | 0 | PASS |
| 11 | RF self-distance = 0 (t1) | 0 | PASS |
| 12 | Normalized RF (self) | 0.0 | PASS |
| 13 | Normalized RF (0,1) in [0,1] | ✓ | PASS |
| 14 | Determinism | = | PASS |
| 15 | Triangle inequality | holds | PASS |

## Key Findings

- RF distances between gene trees under deep coalescence range from 4 to 38
  (mean ~28 for t20 scenario with higher reticulation)
- All metric properties (symmetry, identity, triangle inequality) hold exactly
- Normalized RF reaches 0.86 for maximally discordant pairs (high gene flow)

## GPU Promotion Path

RF is split-set comparison — constant time per edge pair. For batched pairwise
RF across thousands of trees, `FlatTree` + parallel split enumeration maps
directly to workgroup-per-pair dispatch.

## Run

```bash
cargo run --bin validate_phynetpy_rf
python3 scripts/phynetpy_rf_baseline.py
```
