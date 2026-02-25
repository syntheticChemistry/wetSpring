#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
TransE Knowledge Graph Embedding — Python Control (Exp161)

Reproduces a simplified TransE embedding for drug-disease-pathway
relationships inspired by ROBOKOP (Paper 43). TransE: h + r ≈ t,
score = -‖h + r - t‖.

This is the Python baseline. Rust binary: validate_knowledge_graph_embedding.

Date: 2026-02-25
Paper: 43 (ROBOKOP KG infrastructure)
"""

import numpy as np

EMBED_DIM = 32
N_DRUGS = 100
N_DISEASES = 100
N_GENES = 100
N_PATHWAYS = 10
N_ENTITIES = N_DRUGS + N_DISEASES + N_GENES + N_PATHWAYS  # 310
N_RELATIONS = 4  # treats, targets, associated_with, part_of
LEARNING_RATE = 0.01
MARGIN = 1.0
N_EPOCHS = 50
SEED = 42


class LcgRng:
    """Matches Rust LcgRng for reproducibility."""
    def __init__(self, seed):
        self.state = (seed + 1) & 0xFFFF_FFFF_FFFF_FFFF

    def next_f64(self):
        self.state = (self.state * 6_364_136_223_846_793_005 + 1) & 0xFFFF_FFFF_FFFF_FFFF
        bits = (self.state >> 11) | 0x3FF0_0000_0000_0000
        return np.float64(np.frombuffer(bits.to_bytes(8, 'little'), dtype=np.float64)[0] - 1.0)

    def next_usize(self, n):
        return int(self.next_f64() * n) % n


def generate_triples(rng):
    """Generate synthetic biomedical KG triples (cluster-based)."""
    triples = []
    for cluster in range(10):
        d_start = cluster * 10
        dis_start = N_DRUGS + cluster * 10
        g_start = N_DRUGS + N_DISEASES + cluster * 10
        p = N_DRUGS + N_DISEASES + N_GENES + (cluster % N_PATHWAYS)

        for i in range(10):
            drug = d_start + i
            disease = dis_start + i
            gene = g_start + i
            triples.append((drug, 0, disease))      # treats
            triples.append((drug, 1, gene))          # targets
            triples.append((gene, 2, disease))       # associated_with
            triples.append((gene, 3, p))             # part_of

        for i in range(5):
            d1 = d_start + rng.next_usize(10)
            d2 = dis_start + rng.next_usize(10)
            triples.append((d1, 0, d2))

        for i in range(3):
            g1 = g_start + rng.next_usize(10)
            d2 = dis_start + rng.next_usize(10)
            triples.append((g1, 2, d2))

    seen = set()
    unique = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def l2_normalize(vec):
    norm = np.sqrt(np.sum(vec ** 2))
    if norm > 1e-12:
        vec /= norm
    return vec


def transe_score(h, r, t):
    diff = h + r - t
    return -np.sqrt(np.sum(diff ** 2))


def run():
    checks_passed = 0
    checks_total = 0
    print("=" * 60)
    print("TransE Knowledge Graph Embedding — Python Control")
    print("=" * 60)

    rng = LcgRng(SEED)

    # Generate embeddings
    entity_emb = np.zeros((N_ENTITIES, EMBED_DIM), dtype=np.float64)
    for i in range(N_ENTITIES):
        for d in range(EMBED_DIM):
            entity_emb[i, d] = rng.next_f64() * 0.2 - 0.1
        entity_emb[i] = l2_normalize(entity_emb[i])

    relation_emb = np.zeros((N_RELATIONS, EMBED_DIM), dtype=np.float64)
    for i in range(N_RELATIONS):
        for d in range(EMBED_DIM):
            relation_emb[i, d] = rng.next_f64() * 0.2 - 0.1

    triples = generate_triples(rng)
    print(f"\n  Entities: {N_ENTITIES}")
    print(f"  Triples:  {len(triples)}")

    checks_total += 1
    if N_ENTITIES == 310:
        checks_passed += 1
        print("  ✓ CHECK 1: 310 entities")
    else:
        print(f"  ✗ CHECK 1: {N_ENTITIES} entities")

    checks_total += 1
    if len(triples) > 300:
        checks_passed += 1
        print(f"  ✓ CHECK 2: >300 triples ({len(triples)})")
    else:
        print(f"  ✗ CHECK 2: {len(triples)} triples")

    # Training
    losses = []
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for h_idx, r_idx, t_idx in triples:
            h = entity_emb[h_idx].copy()
            r = relation_emb[r_idx].copy()
            t = entity_emb[t_idx].copy()
            pos_score = transe_score(h, r, t)

            neg_t_idx = rng.next_usize(N_ENTITIES)
            neg_t = entity_emb[neg_t_idx].copy()
            neg_score = transe_score(h, r, neg_t)

            loss = max(0.0, MARGIN + abs(pos_score) - abs(neg_score))
            epoch_loss += loss

            if loss > 0.0:
                pos_diff = h + r - t
                neg_diff = h + r - neg_t
                pos_norm = np.sqrt(np.sum(pos_diff ** 2)) + 1e-12
                neg_norm = np.sqrt(np.sum(neg_diff ** 2)) + 1e-12

                grad_pos = pos_diff / pos_norm
                grad_neg = neg_diff / neg_norm

                entity_emb[h_idx] -= LEARNING_RATE * grad_pos
                relation_emb[r_idx] -= LEARNING_RATE * grad_pos
                entity_emb[t_idx] += LEARNING_RATE * grad_pos
                entity_emb[neg_t_idx] -= LEARNING_RATE * (-grad_neg)

                entity_emb[h_idx] = l2_normalize(entity_emb[h_idx])
                entity_emb[t_idx] = l2_normalize(entity_emb[t_idx])
                entity_emb[neg_t_idx] = l2_normalize(entity_emb[neg_t_idx])

        losses.append(epoch_loss / len(triples))

    print(f"\n  Training: {N_EPOCHS} epochs")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    checks_total += 1
    if losses[-1] < losses[0]:
        checks_passed += 1
        print("  ✓ CHECK 3: Training loss decreased")
    else:
        print("  ✗ CHECK 3: Training loss did not decrease")

    # Link prediction
    hits_at_10 = 0
    sample = triples[:min(100, len(triples))]
    for h_idx, r_idx, t_idx in sample:
        h = entity_emb[h_idx]
        r = relation_emb[r_idx]
        scores = []
        for e in range(N_ENTITIES):
            scores.append(transe_score(h, r, entity_emb[e]))
        ranked = np.argsort(scores)[::-1][:10]
        if t_idx in ranked:
            hits_at_10 += 1

    hits_rate = hits_at_10 / len(sample)
    print(f"\n  Hits@10: {hits_at_10}/{len(sample)} = {hits_rate:.4f}")

    checks_total += 1
    if hits_rate > 0.05:
        checks_passed += 1
        print(f"  ✓ CHECK 4: Hits@10 > 5%")
    else:
        print(f"  ✗ CHECK 4: Hits@10 = {hits_rate:.4f}")

    # Score a few triples for cross-validation with Rust
    print("\n  Sample scores (for Rust parity):")
    for h_idx, r_idx, t_idx in triples[:5]:
        s = transe_score(entity_emb[h_idx], relation_emb[r_idx], entity_emb[t_idx])
        print(f"    ({h_idx}, {r_idx}, {t_idx}) → {s:.10f}")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")
    return checks_passed == checks_total


if __name__ == "__main__":
    exit(0 if run() else 1)
