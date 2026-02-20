#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
metalForge benchmark: profile GPU-candidate modules on CPU.

Measures per-call latency for the algorithms that are candidates for
GPU promotion, establishing the CPU baseline that GPU must beat.

Usage:
    python3 metalForge/benchmarks/profile_gpu_candidates.py

Produces JSON output for cross-substrate comparison.
"""

import json
import math
import os
import random
import time

N_STATES = 4


def jc69_prob(fr, to, branch_len, mu=1.0):
    e = math.exp(-4 * mu * branch_len / 3)
    return 0.25 + 0.75 * e if fr == to else 0.25 * (1 - e)


def transition_matrix(branch_len, mu=1.0):
    return [[jc69_prob(i, j, branch_len, mu) for j in range(N_STATES)] for i in range(N_STATES)]


def encode_dna(seq):
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
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
            result[s] = sum(tl[s][x] * lp[x] for x in range(N_STATES)) * sum(
                tr[s][x] * rp[x] for x in range(N_STATES)
            )
        partials.append(result)
    return partials


def log_likelihood(tree, mu=1.0):
    partials = pruning(tree, mu)
    return sum(
        math.log(sum(0.25 * p[s] for s in range(N_STATES))) for p in partials
    )


def sw_score(query, target, match_sc=2, mismatch=-1, gap_open=-3, gap_ext=-1):
    m, n = len(query), len(target)
    H = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sc = match_sc if query[i - 1] == target[j - 1] else mismatch
            diag = H[i - 1][j - 1] + sc
            up = max(H[k][j] + gap_open + gap_ext * (i - k - 1) for k in range(i))
            left = max(H[i][k] + gap_open + gap_ext * (j - k - 1) for k in range(j))
            H[i][j] = max(0, diag, up, left)
            best = max(best, H[i][j])
    return best


def benchmark(name, func, n_iters=100):
    func()  # warmup
    start = time.perf_counter()
    for _ in range(n_iters):
        func()
    elapsed = time.perf_counter() - start
    per_call_us = elapsed / n_iters * 1e6
    print(f"  {name:40s}: {per_call_us:10.1f} µs/call ({n_iters} iters)")
    return {"name": name, "us_per_call": per_call_us, "n_iters": n_iters}


def make_tree(n_sites):
    bases = "ACGT"
    seqs = ["".join(random.choice(bases) for _ in range(n_sites)) for _ in range(3)]
    return {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": "A", "states": encode_dna(seqs[0])},
            "right": {"type": "leaf", "name": "B", "states": encode_dna(seqs[1])},
            "left_branch": 0.1,
            "right_branch": 0.1,
        },
        "right": {"type": "leaf", "name": "C", "states": encode_dna(seqs[2])},
        "left_branch": 0.2,
        "right_branch": 0.3,
    }


def main():
    random.seed(42)
    results = []
    print("metalForge CPU Baseline Benchmarks")
    print("=" * 60)

    print("\n── Felsenstein Pruning (site-parallel) ──")
    for n_sites in [100, 1000, 10000]:
        tree = make_tree(n_sites)
        iters = max(10, 10000 // n_sites)
        results.append(
            benchmark(
                f"Felsenstein {n_sites} sites",
                lambda t=tree: log_likelihood(t, 1.0),
                iters,
            )
        )

    print("\n── Smith-Waterman (pair-parallel) ──")
    for length in [50, 100, 200]:
        bases = "ACGT"
        q = [random.choice(bases) for _ in range(length)]
        t = [random.choice(bases) for _ in range(length)]
        iters = max(5, 500 // length)
        results.append(
            benchmark(
                f"SW {length}×{length}",
                lambda q=q, t=t: sw_score(q, t),
                iters,
            )
        )

    print("\n── Bootstrap (replicate-parallel) ──")
    for n_reps in [10, 100]:
        tree = make_tree(100)
        columns = [[random.randint(0, 3) for _ in range(3)] for _ in range(100)]
        taxa = ["A", "B", "C"]
        branches = [0.1, 0.1, 0.2, 0.3]

        def run_bootstrap(n=n_reps, cols=columns, tx=taxa, br=branches):
            for _ in range(n):
                rep = [cols[random.randint(0, len(cols) - 1)] for _ in range(len(cols))]
                t = make_tree_from_cols(rep, tx, br)
                log_likelihood(t, 1.0)

        results.append(benchmark(f"Bootstrap {n_reps} reps × 100 sites", run_bootstrap, 10))

    print("\n── HMM Forward (sequence-parallel) ──")
    for t_len in [100, 1000]:
        obs = [random.randint(0, 2) for _ in range(t_len)]
        log_pi = [math.log(0.6), math.log(0.4)]
        log_trans = [math.log(0.7), math.log(0.3), math.log(0.4), math.log(0.6)]
        log_emit = [math.log(0.1), math.log(0.4), math.log(0.5), math.log(0.6), math.log(0.3), math.log(0.1)]

        def hmm_forward(obs=obs, pi=log_pi, tr=log_trans, em=log_emit):
            n_states = 2
            alpha = [pi[s] + em[s * 3 + obs[0]] for s in range(n_states)]
            for t in range(1, len(obs)):
                new_alpha = []
                for j in range(n_states):
                    val = max(alpha[i] + tr[i * n_states + j] for i in range(n_states))
                    val += math.log(
                        sum(math.exp(alpha[i] + tr[i * n_states + j] - val) for i in range(n_states))
                    )
                    new_alpha.append(val + em[j * 3 + obs[t]])
                alpha = new_alpha
            return alpha

        iters = max(10, 10000 // t_len)
        results.append(benchmark(f"HMM forward T={t_len}", hmm_forward, iters))

    out_path = os.path.join(os.path.dirname(__file__), "cpu_baseline.json")
    with open(out_path, "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results},
            f,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


def make_tree_from_cols(columns, taxa, branches):
    return {
        "type": "internal",
        "left": {
            "type": "internal",
            "left": {"type": "leaf", "name": taxa[0], "states": [c[0] for c in columns]},
            "right": {"type": "leaf", "name": taxa[1], "states": [c[1] for c in columns]},
            "left_branch": branches[0],
            "right_branch": branches[1],
        },
        "right": {"type": "leaf", "name": taxa[2], "states": [c[2] for c in columns]},
        "left_branch": branches[2],
        "right_branch": branches[3],
    }


if __name__ == "__main__":
    main()
