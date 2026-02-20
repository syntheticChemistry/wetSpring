# Experiment 029 — Felsenstein Pruning Phylogenetic Likelihood

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1c (Phylogenetics)
**Faculty:** Liu (CSE, MSU) / Felsenstein

---

## Objective

Implement and validate Felsenstein's pruning algorithm for computing the
likelihood of nucleotide alignments given a phylogenetic tree under the
Jukes-Cantor (JC69) substitution model. This is the core computational
primitive for phylogenetic inference and a prime GPU parallelization target.

## Model

- JC69 substitution model with equal base frequencies (π = 0.25)
- Post-order traversal computing partial likelihoods bottom-up
- Per-site independence (embarrassingly parallel for GPU)

## Test Trees

| Tree | Topology | Log-Likelihood |
|------|----------|----------------|
| Identical | ((A:0.1,B:0.1):0.2,C:0.3) ACGT×3 | -8.14495204 |
| Different | ((A:0.1,B:0.1):0.2,C:0.3) AAAA,CCCC,GGGG | -25.05390763 |
| 16S (20bp) | ((sp1:0.05,sp2:0.05):0.1,sp3:0.15) | -40.88116903 |

## GPU Promotion Path

Site-parallel: each site is independent → one workgroup per site.
Shared memory for the 4×4 transition matrix. Batch over many alignments
for throughput. ToadStool `GemmF64` could handle the matrix-vector products,
or a custom Felsenstein kernel could fuse the entire pruning pass.

## Validation Binary

```bash
cargo run --bin validate_felsenstein
```
