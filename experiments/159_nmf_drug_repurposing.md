# Exp159: NMF Drug-Disease Matrix Factorization

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (7/7 checks) |
| **Binary**     | `validate_nmf_drug_repurposing` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Drug Repurposing Track |
| **Paper**      | 41 (Yang et al. 2020) |

## Core Idea

Reproduce the NMF drug repurposing pipeline from Yang et al. 2020. Construct a
200×100 drug-disease binary matrix with 5 clusters at 5% fill, apply NMF at
ranks 5/10/20, evaluate reconstruction quality and rank sensitivity.

## Key Findings

- NMF converges for all tested ranks (5, 10, 20)
- Best rank=20 achieves 66.8% relative error (block-structured sparse matrix)
- Euclidean vs KL divergence both produce valid predictions
- W and H factors are non-negative (NMF constraint verified)
- GPU shader analysis: 2× GEMM + element-wise per iteration, all ToadStool-compatible
