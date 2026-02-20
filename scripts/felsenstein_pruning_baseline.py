#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Felsenstein pruning — Python baseline for phylogenetic likelihood.

Pure-Python (sovereign) Jukes-Cantor model, post-order tree traversal.
References:
    Felsenstein 1981, J Mol Evol 17:368-376
"""

import json, os, math
import numpy as np

N_STATES = 4

def jc69_prob(fr, to, branch_len, mu=1.0):
    e = math.exp(-4 * mu * branch_len / 3)
    if fr == to:
        return 0.25 + 0.75 * e
    return 0.25 * (1 - e)

def transition_matrix(branch_len, mu=1.0):
    return [[jc69_prob(i, j, branch_len, mu) for j in range(N_STATES)] for i in range(N_STATES)]

def encode_dna(seq):
    m = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [m.get(c.upper(), 4) for c in seq]

def leaf_partial(state):
    p = [0.0] * N_STATES
    if state < N_STATES:
        p[state] = 1.0
    else:
        p = [0.25] * N_STATES
    return p

def pruning(tree, mu=1.0):
    """Post-order traversal returning per-site partial likelihoods."""
    if tree["type"] == "leaf":
        return [[*leaf_partial(s)] for s in tree["states"]]
    left_p = pruning(tree["left"], mu)
    right_p = pruning(tree["right"], mu)
    trans_l = transition_matrix(tree["left_branch"], mu)
    trans_r = transition_matrix(tree["right_branch"], mu)
    partials = []
    for lp, rp in zip(left_p, right_p):
        result = [0.0] * N_STATES
        for s in range(N_STATES):
            left_sum = sum(trans_l[s][x] * lp[x] for x in range(N_STATES))
            right_sum = sum(trans_r[s][x] * rp[x] for x in range(N_STATES))
            result[s] = left_sum * right_sum
        partials.append(result)
    return partials

def log_likelihood(tree, mu=1.0):
    partials = pruning(tree, mu)
    pi = 0.25
    ll = 0.0
    for p in partials:
        site_lik = sum(pi * p[s] for s in range(N_STATES))
        ll += math.log(site_lik)
    return ll

def site_log_likelihoods(tree, mu=1.0):
    partials = pruning(tree, mu)
    pi = 0.25
    return [math.log(sum(pi * p[s] for s in range(N_STATES))) for p in partials]

def main():
    # Tree: ((A:0.1, B:0.1):0.2, C:0.3)
    tree_identical = {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": "A", "states": encode_dna("ACGT")},
            "right": {"type": "leaf", "name": "B", "states": encode_dna("ACGT")},
            "left_branch": 0.1, "right_branch": 0.1,
        },
        "right": {"type": "leaf", "name": "C", "states": encode_dna("ACGT")},
        "left_branch": 0.2, "right_branch": 0.3,
    }

    tree_different = {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": "A", "states": encode_dna("AAAA")},
            "right": {"type": "leaf", "name": "B", "states": encode_dna("CCCC")},
            "left_branch": 0.1, "right_branch": 0.1,
        },
        "right": {"type": "leaf", "name": "C", "states": encode_dna("GGGG")},
        "left_branch": 0.2, "right_branch": 0.3,
    }

    # Longer alignment: 20bp 16S fragment
    tree_16s = {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": "sp1", "states": encode_dna("ACGTACGTACGTACGTACGT")},
            "right": {"type": "leaf", "name": "sp2", "states": encode_dna("ACGTACTTACGTACGTACGT")},
            "left_branch": 0.05, "right_branch": 0.05,
        },
        "right": {"type": "leaf", "name": "sp3", "states": encode_dna("ACGTACGTACTTACGTACGT")},
        "left_branch": 0.1, "right_branch": 0.15,
    }

    results = {}
    for name, tree in [("identical", tree_identical), ("different", tree_different), ("16s", tree_16s)]:
        ll = log_likelihood(tree, 1.0)
        sll = site_log_likelihoods(tree, 1.0)
        results[name] = {
            "log_likelihood": ll,
            "site_log_likelihoods": sll,
            "n_sites": len(sll),
        }
        print(f"{name:12s}: LL={ll:.10f} ({len(sll)} sites)")

    # Verify JC69 properties
    print(f"\nJC69 checks:")
    print(f"  P(same, t=0):  {jc69_prob(0,0,0.0):.6f}")
    print(f"  P(diff, t=0):  {jc69_prob(0,1,0.0):.6f}")
    print(f"  P(same, t=inf):{jc69_prob(0,0,1000.0):.6f}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "029_felsenstein")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "felsenstein_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")

if __name__ == "__main__":
    main()
