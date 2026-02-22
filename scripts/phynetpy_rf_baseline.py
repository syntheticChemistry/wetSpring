#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""Exp036 baseline — Robinson-Foulds distances on real PhyNetPy gene trees.

Source: NakhlehLab/PhyNetPy DEFJ/ directory
Trees: Multi-locus gene trees simulated under deep coalescence + gene flow.
Each replicate has 10 gene trees with the same 25-leaf label set (3 alleles
per species × 9 species, minus 2 absent alleles).

This script computes pairwise RF distances between gene trees within each
replicate, producing a baseline for Rust validation of the RF module on
real phylogenetic data at scale.

Requires: Python 3.8+ (stdlib only)
"""
import json
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


def main():
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "phynetpy_gene_trees", "repo", "DEFJ"
    )
    if not os.path.isdir(data_dir):
        print(f"[ERROR] PhyNetPy DEFJ data not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_comparisons = 0

    for scenario_name, sub in [("t20", "t20"), ("t4", "t4")]:
        scenario_dir = os.path.join(
            data_dir, "10Genes", "withOG", "E", "g10", "n3", sub
        )
        if not os.path.isdir(scenario_dir):
            continue

        for rep in range(1, 11):
            rep_dir = os.path.join(scenario_dir, f"r{rep}")
            gtree_file = os.path.join(
                rep_dir, f"E2GTg10n3{sub}r{rep}-g_trees.newick"
            )
            if not os.path.isfile(gtree_file):
                continue

            with open(gtree_file) as f:
                gene_trees = [line.strip() for line in f if line.strip()]

            n_trees = len(gene_trees)
            if n_trees < 2:
                continue

            n_leaves = len(parse_newick_leaves(gene_trees[0]))

            pairwise_rf = []
            for i in range(n_trees):
                for j in range(i + 1, n_trees):
                    d = rf_distance(gene_trees[i], gene_trees[j])
                    pairwise_rf.append(d)

            n_pairs = len(pairwise_rf)
            total_comparisons += n_pairs

            results.append({
                "scenario": scenario_name,
                "replicate": rep,
                "n_gene_trees": n_trees,
                "n_leaves": n_leaves,
                "n_pairs": n_pairs,
                "rf_distances": pairwise_rf,
                "mean_rf": sum(pairwise_rf) / n_pairs if n_pairs else 0,
                "max_rf": max(pairwise_rf) if pairwise_rf else 0,
                "min_rf": min(pairwise_rf) if pairwise_rf else 0,
            })

    # Also embed the first 3 Newick pairs for exact Rust reproduction
    sample_trees = []
    scenario_dir = os.path.join(data_dir, "10Genes", "withOG", "E", "g10", "n3", "t20")
    r1_file = os.path.join(scenario_dir, "r1", "E2GTg10n3t20r1-g_trees.newick")
    if os.path.isfile(r1_file):
        with open(r1_file) as f:
            trees = [l.strip() for l in f if l.strip()]
        for i in range(min(3, len(trees))):
            for j in range(i + 1, min(3, len(trees))):
                d = rf_distance(trees[i], trees[j])
                sample_trees.append({
                    "tree_a_idx": i,
                    "tree_b_idx": j,
                    "tree_a": trees[i],
                    "tree_b": trees[j],
                    "expected_rf": d,
                })

    output = {
        "experiment": "Exp036",
        "description": "Pairwise RF distances on PhyNetPy DEFJ gene trees",
        "source": "NakhlehLab/PhyNetPy DEFJ/10Genes/withOG/E/g10/n3",
        "total_comparisons": total_comparisons,
        "n_replicates": len(results),
        "replicates": results,
        "sample_pairs": sample_trees,
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "results", "036_phynetpy_rf"
    )
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "python_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exp036 baseline: {total_comparisons} pairwise RF comparisons "
          f"across {len(results)} replicates")
    for r in results[:3]:
        print(f"  {r['scenario']}/r{r['replicate']}: {r['n_gene_trees']} trees, "
              f"{r['n_leaves']} leaves, mean_rf={r['mean_rf']:.1f}")
    if len(results) > 3:
        print(f"  ... and {len(results) - 3} more replicates")
    if sample_trees:
        print(f"  Sample pairs: {len(sample_trees)} with exact expected RF values")
    print(f"Output: {out_path}/python_baseline.json")


if __name__ == "__main__":
    main()
