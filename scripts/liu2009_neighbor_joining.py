#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""
Python baseline: Neighbor-Joining tree construction (Saitou & Nei 1987).

Core SATé primitive (Liu 2009) — builds a guide tree from a pairwise
distance matrix. This is a pure Python implementation for BarraCUDA
CPU validation (no dendropy/Biopython dependency).

Usage:
    python3 scripts/liu2009_neighbor_joining.py

Outputs JSON with tree topology and branch lengths for Rust comparison.
"""

import json
import math


def neighbor_joining(dist_matrix, labels):
    """Neighbor-Joining algorithm (Saitou & Nei 1987).

    Args:
        dist_matrix: n×n symmetric distance matrix (list of lists)
        labels: list of n taxon labels

    Returns:
        Newick string of the resulting tree
    """
    n = len(dist_matrix)
    # Working copies
    D = [row[:] for row in dist_matrix]
    active = list(range(n))
    node_labels = list(labels)
    next_node = n

    while len(active) > 2:
        r = len(active)
        # Compute row sums for active nodes
        row_sums = {}
        for i in active:
            row_sums[i] = sum(D[i][j] for j in active if j != i)

        # Find pair (i,j) minimizing Q
        best_q = float("inf")
        best_i, best_j = active[0], active[1]
        for ai, i in enumerate(active):
            for j in active[ai + 1 :]:
                q = (r - 2) * D[i][j] - row_sums[i] - row_sums[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j

        # Branch lengths from i,j to new node
        if r > 2:
            delta = (row_sums[best_i] - row_sums[best_j]) / (r - 2)
        else:
            delta = 0.0
        li = 0.5 * (D[best_i][best_j] + delta)
        lj = 0.5 * (D[best_i][best_j] - delta)

        # Ensure non-negative
        li = max(li, 0.0)
        lj = max(lj, 0.0)

        # Create new node
        new_label = f"({node_labels[best_i]}:{li:.6f},{node_labels[best_j]}:{lj:.6f})"
        node_labels.append(new_label)

        # Compute distances from new node to all remaining
        new_idx = next_node
        next_node += 1

        # Expand D if needed
        while len(D) <= new_idx:
            D.append([0.0] * len(D[0]))
        for row in D:
            while len(row) <= new_idx:
                row.append(0.0)

        for k in active:
            if k != best_i and k != best_j:
                d_new = 0.5 * (D[best_i][k] + D[best_j][k] - D[best_i][best_j])
                D[new_idx][k] = d_new
                D[k][new_idx] = d_new
        D[new_idx][new_idx] = 0.0

        # Remove old, add new
        active.remove(best_i)
        active.remove(best_j)
        active.append(new_idx)

    # Final two nodes
    i, j = active
    final_d = D[i][j]
    newick = f"({node_labels[i]}:{final_d / 2:.6f},{node_labels[j]}:{final_d / 2:.6f});"
    return newick


def jukes_cantor_distance(seq1, seq2):
    """Jukes-Cantor corrected distance between two aligned sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be same length")
    diffs = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = diffs / len(seq1)
    if p >= 0.75:
        return 10.0  # saturated
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def main():
    # Test case 1: 4-taxon UPGMA-like distances
    labels_4 = ["A", "B", "C", "D"]
    D4 = [
        [0.0, 0.3, 0.5, 0.6],
        [0.3, 0.0, 0.6, 0.5],
        [0.5, 0.6, 0.0, 0.3],
        [0.6, 0.5, 0.3, 0.0],
    ]
    newick_4 = neighbor_joining(D4, labels_4)

    # Test case 2: 5-taxon from JC distances of known sequences
    seqs = {
        "S1": "ACGTACGTACGT",
        "S2": "ACGTACGTACTT",
        "S3": "ACTTACTTACTT",
        "S4": "TGCATGCATGCA",
        "S5": "TGCATGCATGCC",
    }
    labels_5 = list(seqs.keys())
    n = len(labels_5)
    D5 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = jukes_cantor_distance(seqs[labels_5[i]], seqs[labels_5[j]])
            D5[i][j] = d
            D5[j][i] = d
    newick_5 = neighbor_joining(D5, labels_5)

    # Test case 3: 3-taxon (trivial — single join)
    labels_3 = ["X", "Y", "Z"]
    D3 = [
        [0.0, 0.2, 0.4],
        [0.2, 0.0, 0.4],
        [0.4, 0.4, 0.0],
    ]
    newick_3 = neighbor_joining(D3, labels_3)

    results = {
        "test_4taxon": {
            "labels": labels_4,
            "distances": D4,
            "newick": newick_4,
        },
        "test_5taxon": {
            "labels": labels_5,
            "distances": D5,
            "newick": newick_5,
            "jc_distances": {
                f"{labels_5[i]}-{labels_5[j]}": D5[i][j]
                for i in range(n)
                for j in range(i + 1, n)
            },
        },
        "test_3taxon": {
            "labels": labels_3,
            "distances": D3,
            "newick": newick_3,
        },
    }

    print(json.dumps(results, indent=2))

    # Validation summary
    print("\n--- Python NJ Baseline ---")
    print(f"4-taxon: {newick_4}")
    print(f"5-taxon: {newick_5}")
    print(f"3-taxon: {newick_3}")
    print(f"JC distances (5-taxon): {D5[0][1]:.6f}, {D5[0][3]:.6f}")


if __name__ == "__main__":
    main()
