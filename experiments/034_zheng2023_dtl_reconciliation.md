# Exp034 — DTL Reconciliation (Zheng et al. 2023)

**Date:** February 20, 2026
**Status:** COMPLETE — 14/14 checks PASS
**Track:** 1b (Comparative Genomics & Phylogenetics)

---

## Objective

Implement and validate DTL (Duplication-Transfer-Loss) reconciliation for
cophylogenetic analysis, as studied in Zheng et al. 2023 (ACM-BCB top 10%).
Given a host tree and a gene/parasite tree with tip-to-tip mapping, find
the optimal event assignment that explains the gene tree within the host tree.

## Model

### DTL Event Model (Bansal et al. 2012)

Events with costs:
- **Speciation (S):** cost = 0 — both lineages diverge together
- **Duplication (D):** cost = 2 — gene duplicates on same host lineage
- **Transfer (T):** cost = 3 — horizontal transfer to different host
- **Loss (L):** cost = 1 — gene lineage lost on one host child

### DP Formulation

C[p][h] = min cost of mapping parasite node p to host node h.
For internal parasite node with children p1, p2:
- Speciation: C[p1][h1] + C[p2][h2] (or swapped)
- Duplication: D + C[p1][h] + C[p2][h]
- Transfer: T + min(C[p1][h] + C[p2][h'], C[p1][h'] + C[p2][h])
- Loss: L + C[p][h_child]

## Modules

- `bio::reconciliation` — DTL reconciliation with flat tree layout (GPU-ready)
- `bio::reconciliation::reconcile_batch` — batch reconciliation API

## Test Cases

| Test | Input | Expected | Checks |
|------|-------|----------|--------|
| Congruent 2-leaf | H=(A,B), P=(A,B), exact mapping | cost=0, host=H_AB | 3 |
| Duplication 4-leaf | P_1,P_2→H_A (dup), P_3→H_C | cost=4, host=H_root | 3 |
| Loss/speciation | P_A→H_A, P_C→H_C, 3-node host | cost=1, host=H_root | 2 |
| DP dimensions | Various | correct table sizes | 2 |
| Batch | 2 congruent reconciliations | both cost=0 | 2 |
| Determinism | Two runs | identical results | 2 |

## Acceptance Criteria

All 14 checks PASS:
- Optimal costs match Python baseline exactly
- Optimal host mapping matches Python
- DP table dimensions correct
- Batch API produces same results as sequential
- Deterministic

## Baseline Provenance

- Python: `scripts/zheng2023_dtl_reconciliation.py` (pure Python, no dependencies)
- Reference: Zheng et al. 2023, *ACM-BCB* (top 10%)
- DTL framework: Bansal, Alm & Kellis 2012, *PNAS* 109:11319-11324

## GPU Promotion Path

**Tier B (Adapt):** The DP is sequential per parasite node (post-order), but:
1. **Batch reconciliation** is embarrassingly parallel — one workgroup per
   gene family when reconciling hundreds of gene trees against one host tree
2. Within each DP row, evaluating different host mappings involves scanning
   all hosts (for transfers) — this scan could be parallelized
3. `reconcile_batch` provides the batch entry point for GPU dispatch
