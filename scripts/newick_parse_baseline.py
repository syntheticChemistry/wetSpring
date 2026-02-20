#!/usr/bin/env python3
"""Newick parsing baseline — Python/dendropy reference.

Parses a set of Newick trees and extracts leaf counts, branch length sums,
and topology features. Produces ground truth for Rust validation.

Requires: pip install dendropy
"""
import json
import sys
import time
from pathlib import Path

try:
    import dendropy
except ImportError:
    print("ERROR: dendropy not installed. Run: pip install dendropy")
    sys.exit(1)

WORKSPACE = Path(__file__).resolve().parent.parent

TEST_CASES = [
    {
        "name": "simple_4leaf",
        "newick": "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
    },
    {
        "name": "simple_5leaf",
        "newick": "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
    },
    {
        "name": "balanced_6leaf",
        "newick": "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
    },
    {
        "name": "caterpillar_6leaf",
        "newick": "(((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1);",
    },
    {
        "name": "trivial_2leaf",
        "newick": "(A:0.5,B:0.5);",
    },
    {
        "name": "trivial_3leaf",
        "newick": "((A:0.1,B:0.2):0.3,C:0.4);",
    },
    {
        "name": "unequal_branch_7leaf",
        "newick": "(((A:0.01,B:0.99):0.5,(C:0.5,D:0.5):0.01):0.1,(E:0.3,(F:0.7,G:0.2):0.4):0.6);",
    },
    {
        "name": "star_4leaf",
        "newick": "(A:0.1,B:0.2,C:0.3,D:0.4);",
    },
    {
        "name": "deep_caterpillar_8leaf",
        "newick": "((((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1):0.1,(G:0.1,H:0.1):0.1);",
    },
    {
        "name": "zero_length_branches",
        "newick": "((A:0.0,B:0.0):0.0,(C:0.0,D:0.0):0.0);",
    },
]


def analyze_tree(newick_str):
    """Parse a Newick tree with dendropy and extract statistics."""
    tree = dendropy.Tree.get(data=newick_str, schema="newick")
    leaves = [leaf.taxon.label for leaf in tree.leaf_node_iter()]
    leaves.sort()

    total_bl = sum(
        edge.length for edge in tree.preorder_edge_iter()
        if edge.length is not None
    )

    internal_nodes = sum(
        1 for node in tree.preorder_node_iter()
        if not node.is_leaf()
    )

    return {
        "n_leaves": len(leaves),
        "leaf_labels": leaves,
        "total_branch_length": round(total_bl, 10),
        "n_internal_nodes": internal_nodes,
    }


def main():
    print("=" * 70)
    print("  Exp019 Phase 1: Newick Parsing — Python Baseline")
    print("=" * 70)

    t0 = time.time()
    results = []
    checks = []

    for tc in TEST_CASES:
        name = tc["name"]
        newick = tc["newick"]
        stats = analyze_tree(newick)

        result = {
            "name": name,
            "newick": newick,
            **stats,
        }
        results.append(result)

        status = "PASS"
        checks.append({"label": f"{name}: parsed successfully", "status": status})
        print(f"  [PASS] {name}: {stats['n_leaves']} leaves, "
              f"BL={stats['total_branch_length']:.6f}, "
              f"{stats['n_internal_nodes']} internal")

    elapsed = time.time() - t0
    n_pass = sum(1 for c in checks if c["status"] == "PASS")

    out_dir = WORKSPACE / "experiments/results/019_phylogenetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "newick_parse_python_baseline.json"

    output = {
        "experiment": "019_phylogenetic_validation_phase1",
        "tool": "dendropy",
        "dendropy_version": dendropy.__version__,
        "python_version": sys.version.split()[0],
        "total_cases": len(results),
        "total_pass": n_pass,
        "elapsed_seconds": round(elapsed, 4),
        "cases": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {n_pass}/{len(checks)} PASS")
    print(f"  Results: {out_path}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print(f"{'=' * 70}")

    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
