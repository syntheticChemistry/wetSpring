#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""
Wang 2021 RAWR bootstrap — Python baseline.

References:
    Wang et al. 2021, Bioinformatics (ISMB) 37:i111-i119
"""

import json, os, math, random

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
    if state < N_STATES:
        p[state] = 1.0
    else:
        p = [0.25] * N_STATES
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
            left_sum = sum(tl[s][x] * lp[x] for x in range(N_STATES))
            right_sum = sum(tr[s][x] * rp[x] for x in range(N_STATES))
            result[s] = left_sum * right_sum
        partials.append(result)
    return partials

def log_likelihood(tree, mu=1.0):
    partials = pruning(tree, mu)
    return sum(math.log(sum(0.25 * p[s] for s in range(N_STATES))) for p in partials)

def resample_columns(alignment, n_sites, rng):
    return [alignment[rng.randint(0, n_sites - 1)] for _ in range(n_sites)]

def make_tree(columns, taxa_names, branches):
    return {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": taxa_names[0], "states": [c[0] for c in columns]},
            "right": {"type": "leaf", "name": taxa_names[1], "states": [c[1] for c in columns]},
            "left_branch": branches[0], "right_branch": branches[1],
        },
        "right": {"type": "leaf", "name": taxa_names[2], "states": [c[2] for c in columns]},
        "left_branch": branches[2], "right_branch": branches[3],
    }

def main():
    rows = [encode_dna("ACGTACGTACGT"), encode_dna("ACGTACTTACGT"), encode_dna("ACGTACGTACTT")]
    n_sites = len(rows[0])
    columns = [[rows[t][s] for t in range(3)] for s in range(n_sites)]

    branches = [0.1, 0.1, 0.2, 0.3]
    taxa = ["A", "B", "C"]

    tree = make_tree(columns, taxa, branches)
    original_ll = log_likelihood(tree, 1.0)
    print(f"Original LL: {original_ll:.10f}")

    rng = random.Random(42)
    n_reps = 100
    lls = []
    for _ in range(n_reps):
        rep_cols = resample_columns(columns, n_sites, rng)
        rep_tree = make_tree(rep_cols, taxa, branches)
        lls.append(log_likelihood(rep_tree, 1.0))

    results = {
        "original_ll": original_ll,
        "n_replicates": n_reps,
        "mean_ll": sum(lls) / len(lls),
        "min_ll": min(lls),
        "max_ll": max(lls),
        "std_ll": (sum((x - sum(lls)/len(lls))**2 for x in lls) / len(lls))**0.5,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "031_bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rawr_python_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written.")

if __name__ == "__main__":
    main()
