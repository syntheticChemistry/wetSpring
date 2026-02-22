#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""
Alamin & Liu 2024 phylogenetic placement — Python baseline.

References:
    Alamin & Liu 2024, IEEE/ACM TCBB
"""

import json, os, math

N_STATES = 4

def jc69_prob(fr, to, branch_len, mu=1.0):
    e = math.exp(-4 * mu * branch_len / 3)
    return 0.25 + 0.75 * e if fr == to else 0.25 * (1 - e)

def transition_matrix(branch_len, mu=1.0):
    return [[jc69_prob(i, j, branch_len, mu) for j in range(N_STATES)] for i in range(N_STATES)]

def encode_dna(seq):
    m = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [m.get(c.upper(), 4) for c in seq]

def leaf_partial(state):
    p = [0.0] * N_STATES
    if state < N_STATES: p[state] = 1.0
    else: p = [0.25] * N_STATES
    return p

def pruning(tree, mu=1.0):
    if tree["type"] == "leaf":
        return [[*leaf_partial(s)] for s in tree["states"]]
    left_p = pruning(tree["left"], mu)
    right_p = pruning(tree["right"], mu)
    tl = transition_matrix(tree["left_branch"], mu)
    tr = transition_matrix(tree["right_branch"], mu)
    partials = []
    for lp, rp in zip(left_p, right_p):
        result = [0.0] * N_STATES
        for s in range(N_STATES):
            result[s] = sum(tl[s][x]*lp[x] for x in range(N_STATES)) * sum(tr[s][x]*rp[x] for x in range(N_STATES))
        partials.append(result)
    return partials

def log_likelihood(tree, mu=1.0):
    partials = pruning(tree, mu)
    return sum(math.log(sum(0.25 * p[s] for s in range(N_STATES))) for p in partials)

def make_ref_tree():
    return {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": "sp1", "states": encode_dna("ACGTACGTACGT")},
            "right": {"type": "leaf", "name": "sp2", "states": encode_dna("ACGTACTTACGT")},
            "left_branch": 0.1, "right_branch": 0.1,
        },
        "right": {"type": "leaf", "name": "sp3", "states": encode_dna("ACTTACGTACGT")},
        "left_branch": 0.2, "right_branch": 0.3,
    }

def count_edges(tree):
    if tree["type"] == "leaf": return 1
    return 1 + count_edges(tree["left"]) + count_edges(tree["right"])

def insert_at_edge(tree, query_states, target, pendant, idx_box):
    if tree["type"] == "leaf":
        my_idx = idx_box[0]; idx_box[0] += 1
        if my_idx == target:
            return {"type": "internal",
                    "left": dict(tree), "right": {"type": "leaf", "name": "query", "states": query_states},
                    "left_branch": 0.01, "right_branch": pendant}, True
        return dict(tree), False
    my_idx = idx_box[0]; idx_box[0] += 1
    new_left, fl = insert_at_edge(tree["left"], query_states, target, pendant, idx_box)
    new_right, fr = insert_at_edge(tree["right"], query_states, target, pendant, idx_box)
    if fl or fr:
        return {"type": "internal", "left": new_left, "right": new_right,
                "left_branch": tree["left_branch"], "right_branch": tree["right_branch"]}, True
    if my_idx == target:
        return {"type": "internal",
                "left": {"type": "internal", "left": new_left, "right": new_right,
                          "left_branch": tree["left_branch"], "right_branch": tree["right_branch"]},
                "right": {"type": "leaf", "name": "query", "states": query_states},
                "left_branch": 0.01, "right_branch": pendant}, True
    return {"type": "internal", "left": new_left, "right": new_right,
            "left_branch": tree["left_branch"], "right_branch": tree["right_branch"]}, False

def placement_scan(ref_tree, query_seq, pendant=0.05, mu=1.0):
    qstates = encode_dna(query_seq)
    n_edges = count_edges(ref_tree)
    results = []
    for edge in range(n_edges):
        idx_box = [0]
        aug, _ = insert_at_edge(ref_tree, qstates, edge, pendant, idx_box)
        ll = log_likelihood(aug, mu)
        results.append({"edge": edge, "ll": ll})
    best = max(results, key=lambda x: x["ll"])
    return results, best

def main():
    ref_tree = make_ref_tree()
    queries = {
        "close_to_sp1": "ACGTACGTACGT",
        "close_to_sp3": "ACTTACGTACGT",
        "divergent": "GGGGGGGGGGGG",
    }
    all_results = {}
    for name, seq in queries.items():
        placements, best = placement_scan(ref_tree, seq)
        all_results[name] = {
            "best_edge": best["edge"],
            "best_ll": best["ll"],
            "n_edges": len(placements),
            "all_lls": [p["ll"] for p in placements],
        }
        print(f"{name:20s}: best_edge={best['edge']} best_ll={best['ll']:.6f}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "032_placement")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "placement_python_baseline.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline written.")

if __name__ == "__main__":
    main()
