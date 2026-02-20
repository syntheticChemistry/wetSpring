# Experiment 032: Alamin & Liu 2024 — Phylogenetic Placement

**Paper**: Alamin & Liu "Phylogenetic Placement of Aligned Genomes and Metagenomes with Non-tree-like Histories" — *IEEE/ACM TCBB*, 2024

**Track**: 1b — Comparative Genomics & Phylogenetics (Liu)

---

## Objective

Implement the core placement likelihood primitive for metagenomic classification. Given a reference phylogenetic tree and a query sequence, compute the log-likelihood of inserting the query at each edge, then select the maximum-likelihood placement. This is the fundamental operation for taxonomic assignment without full tree inference.

## Model

- **Reference tree**: 3-taxon tree (sp1, sp2, sp3) with JC69 distances
- **Edge insertion**: For each edge, split the edge and attach query as sister
- **Likelihood**: Felsenstein pruning under JC69 model per placement
- **Best placement**: Edge with maximum log-likelihood
- **Confidence**: exp(best_LL - second_best_LL) ratio

## Test Cases

| Query | Expected Best Edge | Python Best LL | Status |
|-------|-------------------|---------------|--------|
| Close to sp1 (`ACGTACGTACGT`) | Edge 2 | -29.977 | PASS |
| Close to sp3 (`ACTTACGTACGT`) | Edge 4 | -29.977 | PASS |
| Divergent (`GGGGGGGGGGGG`) | Edge 0 (root) | -62.895 | PASS |

## Baseline Provenance

- **Python script**: `scripts/alamin2024_placement.py`
- **Rust module**: `barracuda::bio::placement`
- **Dependencies**: `barracuda::bio::felsenstein`
- **Validation binary**: `validate_placement` — 12/12 checks

## Acceptance Criteria

- [x] Correct edge identification for close queries
- [x] Divergent queries placed near root
- [x] Batch placement returns correct count
- [x] Log-likelihoods match Python to 1e-4
- [x] Deterministic (bit-exact reruns)

## GPU Promotion Path

**Tier A — Rewire**: Edge-parallel computation. Each candidate placement is independent — one workgroup per edge computes Felsenstein likelihood with the query inserted. For N edges and M queries, launch N×M workgroups. ToadStool `parallel_placement` primitive. This is the highest-impact GPU target for wetSpring metagenomic pipelines.

---

*Date*: 2026-02-20 | *Checks*: 12/12 PASS
