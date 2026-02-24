# Exp160: repoDB NMF Reproduction

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (9/9 checks) |
| **Binary**     | `validate_repodb_nmf` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Drug Repurposing Track |
| **Paper**      | 42 (Gao et al. 2020, PMC7153111) |

## Core Idea

Validate NMF on repoDB-proportional data (200×150, 10 clusters, ~800 entries).
CPU tier validates math correctness and block structure recovery. Full repoDB
scale (1571×1209) is the GPU target requiring weighted NMF.

## Key Findings

- Block structure recovered: within-cluster scores > cross-cluster (discrimination ratio > 2×)
- NMF error decreases monotonically across iterations
- Factor matrices are non-negative and sparse (< 80% dense)
- Rank sensitivity: higher rank → better reconstruction but less sparsity
- CPU tier: 91 ms for 200×150; GPU target: 1571×1209 in < 10ms

## GPU Tier Roadmap

Full repoDB requires weighted NMF: only penalise known entries, treat unknowns
as missing (not zero). This needs a mask-multiply shader for ToadStool.
