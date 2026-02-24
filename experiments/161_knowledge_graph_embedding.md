# Exp161: Knowledge Graph Embedding Baseline (ROBOKOP)

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (7/7 checks) |
| **Binary**     | `validate_knowledge_graph_embedding` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Drug Repurposing Track |
| **Paper**      | 43 (ROBOKOP knowledge graph) |

## Core Idea

Build a TransE-style knowledge graph embedding baseline for drug-disease
prediction. 310 entities (100 drugs, 100 diseases, 100 genes, 10 pathways),
4 relation types, trained for 50 epochs with margin-based loss.

## Key Findings

- Training loss decreases over 50 epochs (embedding converges)
- Hits@10 > 5% on link prediction (better than random)
- Novel drug-disease predictions generated via embedding similarity
- Maps to ROBOKOP pipeline: Drug → targets → Gene → associated_with → Disease
- GPU shader: TransE update is element-wise, fully parallelisable

## Cross-Paper Pipeline

Exp157-161 form a complete drug repurposing validation chain:
Pathway scoring → MATRIX methodology → NMF factorization → repoDB benchmark → KG embedding
