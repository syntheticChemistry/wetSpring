#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""Exp037 baseline — HMM-based gene tree discordance detection.

Uses PhyNetPy DEFJ gene trees to test HMM classification of loci as
"concordant" (topologically similar) vs "discordant" (deeply different).

The approach models a PhyloNet-HMM style analysis:
  - For each replicate, compute pairwise RF distances between consecutive
    gene trees. Low RF = topologically similar = concordant block.
  - Run a 2-state HMM where state 0 emits low RF and state 1 emits high RF.
  - Viterbi decoding identifies discordant regions.

This mirrors Liu 2014's PhyloNet-HMM for introgression detection.

Requires: Python 3.8+ (stdlib only)
"""
import json
import math
import os
import re
import sys
from collections import Counter


def parse_newick_leaves(nwk: str) -> list[str]:
    clean = re.sub(r":[0-9eE.+-]+", "", nwk).replace(";", "")
    clean = clean.replace("(", " ").replace(")", " ").replace(",", " ")
    return sorted(tok for tok in clean.split() if tok)


def tokenize_newick(nwk: str) -> list[str]:
    tokens = []
    i = 0
    s = nwk.strip().rstrip(";")
    while i < len(s):
        if s[i] in "(),":
            tokens.append(s[i])
            i += 1
        elif s[i] == ":":
            j = i + 1
            while j < len(s) and s[j] not in "(),;:":
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i] in " \t\n\r":
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in "(),;:":
                j += 1
            tokens.append(s[i:j].strip())
            i = j
    return tokens


def collect_splits(nwk: str) -> set[frozenset[str]]:
    all_leaves = parse_newick_leaves(nwk)
    n = len(all_leaves)
    all_set = frozenset(all_leaves)
    splits: set[frozenset[str]] = set()
    stack: list[set[str]] = []
    tokens = tokenize_newick(nwk)
    for tok in tokens:
        if tok == "(":
            stack.append(set())
        elif tok == ")":
            top = stack.pop()
            if stack:
                stack[-1].update(top)
            clade = frozenset(top)
            comp = all_set - clade
            if 1 < len(clade) < n - 1:
                canon = min(clade, comp, key=lambda x: (len(x), sorted(x)))
                splits.add(canon)
        elif tok == ",":
            pass
        elif tok.startswith(":"):
            pass
        else:
            if stack:
                stack[-1].add(tok)
    return splits


def rf_distance(nwk_a: str, nwk_b: str) -> int:
    sa = collect_splits(nwk_a)
    sb = collect_splits(nwk_b)
    return len(sa.symmetric_difference(sb))


def log_sum_exp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def hmm_forward(obs: list[int], log_init: list[float],
                log_trans: list[list[float]],
                log_emit_fn) -> float:
    n_states = len(log_init)
    fwd = [log_init[s] + log_emit_fn(obs[0], s) for s in range(n_states)]
    for t in range(1, len(obs)):
        new_fwd = []
        for j in range(n_states):
            val = float("-inf")
            for i in range(n_states):
                val = log_sum_exp(val, fwd[i] + log_trans[i][j])
            new_fwd.append(val + log_emit_fn(obs[t], j))
        fwd = new_fwd
    total = float("-inf")
    for v in fwd:
        total = log_sum_exp(total, v)
    return total


def hmm_viterbi(obs: list[int], log_init: list[float],
                log_trans: list[list[float]],
                log_emit_fn) -> list[int]:
    n_states = len(log_init)
    T = len(obs)
    dp = [[0.0] * n_states for _ in range(T)]
    bp = [[0] * n_states for _ in range(T)]
    for s in range(n_states):
        dp[0][s] = log_init[s] + log_emit_fn(obs[0], s)
    for t in range(1, T):
        for j in range(n_states):
            best_val = float("-inf")
            best_i = 0
            for i in range(n_states):
                v = dp[t - 1][i] + log_trans[i][j]
                if v > best_val:
                    best_val = v
                    best_i = i
            dp[t][j] = best_val + log_emit_fn(obs[t], j)
            bp[t][j] = best_i
    best_last = max(range(n_states), key=lambda s: dp[T - 1][s])
    path = [0] * T
    path[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1][path[t + 1]]
    return path


def main():
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "phynetpy_gene_trees", "repo", "DEFJ"
    )
    if not os.path.isdir(data_dir):
        print("[ERROR] PhyNetPy DEFJ data not found.", file=sys.stderr)
        sys.exit(1)

    # Collect consecutive RF distances across all replicates as observation sequence
    all_rf: list[int] = []
    all_trees: list[str] = []

    for scenario in ["t20", "t4"]:
        scenario_dir = os.path.join(
            data_dir, "10Genes", "withOG", "E", "g10", "n3", scenario
        )
        if not os.path.isdir(scenario_dir):
            continue
        for rep in range(1, 11):
            gtfile = os.path.join(
                scenario_dir, f"r{rep}",
                f"E2GTg10n3{scenario}r{rep}-g_trees.newick"
            )
            if not os.path.isfile(gtfile):
                continue
            with open(gtfile) as f:
                trees = [l.strip() for l in f if l.strip()]
            all_trees.extend(trees)

    if len(all_trees) < 2:
        print("[ERROR] Need at least 2 gene trees", file=sys.stderr)
        sys.exit(1)

    # Compute consecutive pairwise RF distances
    for i in range(len(all_trees) - 1):
        d = rf_distance(all_trees[i], all_trees[i + 1])
        all_rf.append(d)

    # Discretize: "low" RF (≤ median) vs "high" RF (> median)
    median_rf = sorted(all_rf)[len(all_rf) // 2]
    obs_binary = [0 if d <= median_rf else 1 for d in all_rf]

    # HMM: 2 states (concordant block, discordant block)
    switch_prob = 0.1
    log_init = [math.log(0.6), math.log(0.4)]
    log_trans = [
        [math.log(1 - switch_prob), math.log(switch_prob)],
        [math.log(switch_prob), math.log(1 - switch_prob)],
    ]

    def log_emit(obs_val: int, state: int) -> float:
        if state == 0:
            return math.log(0.8) if obs_val == 0 else math.log(0.2)
        else:
            return math.log(0.2) if obs_val == 0 else math.log(0.8)

    log_lik = hmm_forward(obs_binary, log_init, log_trans, log_emit)
    viterbi_path = hmm_viterbi(obs_binary, log_init, log_trans, log_emit)

    n_concordant = viterbi_path.count(0)
    n_discordant = viterbi_path.count(1)
    frac_discordant = n_discordant / len(viterbi_path)

    output = {
        "experiment": "Exp037",
        "description": "HMM discordance detection on PhyNetPy consecutive gene trees",
        "n_trees": len(all_trees),
        "n_observations": len(all_rf),
        "median_rf": median_rf,
        "rf_distribution": dict(sorted(Counter(all_rf).items())),
        "hmm_params": {
            "n_states": 2,
            "switch_prob": switch_prob,
            "log_likelihood": log_lik,
            "emission": {
                "concordant_low": 0.8,
                "concordant_high": 0.2,
                "discordant_low": 0.2,
                "discordant_high": 0.8,
            },
        },
        "viterbi_results": {
            "n_concordant": n_concordant,
            "n_discordant": n_discordant,
            "fraction_discordant": frac_discordant,
        },
        "first_20_rf": all_rf[:20],
        "first_20_binary": obs_binary[:20],
        "first_20_viterbi": viterbi_path[:20],
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "results", "037_phylohmm"
    )
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "python_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exp037 baseline: {len(all_rf)} consecutive RF observations")
    print(f"  Median RF: {median_rf}")
    print(f"  RF distribution: {dict(sorted(Counter(all_rf).items()))}")
    print(f"  HMM log-likelihood: {log_lik:.4f}")
    print(f"  Concordant: {n_concordant}, Discordant: {n_discordant} ({frac_discordant:.1%})")
    print(f"Output: {out_path}/python_baseline.json")


if __name__ == "__main__":
    main()
