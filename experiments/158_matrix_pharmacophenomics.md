# Exp158: MATRIX Computational Pharmacophenomics

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (9/9 checks) |
| **Binary**     | `validate_matrix_pharmacophenomics` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Drug Repurposing Track |
| **Paper**      | 40 (Fajgenbaum et al. Lancet Haematology 2025) |

## Core Idea

Validate the MATRIX framework for systematic drug repurposing using
pathway-bridged scoring (drug-pathway × disease-pathway → drug-disease score).
NMF recovers the same structure from the score matrix with 80% top-10 overlap.

## Key Findings

- 4-stage MATRIX pipeline validated: phenotyping → profiling → matching → ranking
- Pathway-bridged scoring identifies correct drug-disease pairs
- NMF (rank=5) achieves 6.07% relative error on score matrix
- NMF top-10 overlaps 8/10 with direct scoring
- Cosine similarity on latent factors provides complementary ranking
