#!/usr/bin/env python3
"""Robinson-Foulds distance baseline — Python/dendropy reference.

Generates ground-truth RF distances for synthetic Newick tree pairs.
Used by validate_rf_distance (Exp021) for Rust validation.

Requires: pip install dendropy
"""
import json
import sys
import time
from pathlib import Path

try:
    import dendropy
    from dendropy.calculate import treecompare
except ImportError:
    print("  ERROR: dendropy not installed. Run: pip install dendropy")
    sys.exit(1)

WORKSPACE = Path(__file__).resolve().parent.parent

# Synthetic test cases with known analytical RF distances.
# Each case: (name, newick_a, newick_b, expected_rf, n_leaves, description)
TEST_CASES = [
    (
        "identical_4leaf",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        0,
        4,
        "Identical 4-leaf trees → RF = 0",
    ),
    (
        "single_nni_4leaf",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        "((A:0.1,C:0.3):0.3,(B:0.2,D:0.4):0.5);",
        2,
        4,
        "Single NNI on 4-leaf tree → RF = 2",
    ),
    (
        "identical_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        0,
        5,
        "Identical 5-leaf trees → RF = 0",
    ),
    (
        "rearranged_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,C:0.3):0.2,B:0.1):0.1,(D:0.2,E:0.3):0.4);",
        2,
        5,
        "B and C swapped in 5-leaf tree → RF = 2",
    ),
    (
        "fully_different_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,D:0.2):0.2,E:0.3):0.1,(B:0.1,C:0.3):0.4);",
        4,
        5,
        "Maximally different 5-leaf topologies → RF = 4",
    ),
    (
        "identical_6leaf",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        0,
        6,
        "Identical 6-leaf balanced trees → RF = 0",
    ),
    (
        "caterpillar_vs_balanced_6leaf",
        "(((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1);",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        2,
        6,
        "Caterpillar vs balanced 6-leaf → RF = 2 (dendropy confirmed)",
    ),
    (
        "two_leaf",
        "(A:0.5,B:0.5);",
        "(A:0.5,B:0.5);",
        0,
        2,
        "Trivial 2-leaf tree → RF = 0",
    ),
    (
        "three_leaf_identical",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        0,
        3,
        "Identical 3-leaf → RF = 0",
    ),
    (
        "three_leaf_rearranged",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        "((A:0.1,C:0.4):0.3,B:0.2);",
        0,
        3,
        "3-leaf trees are star topologies when unrooted → RF = 0",
    ),
]


def compute_rf_dendropy(newick_a, newick_b):
    """Compute unweighted RF distance using dendropy."""
    tns = dendropy.TaxonNamespace()
    tree_a = dendropy.Tree.get(data=newick_a, schema="newick", taxon_namespace=tns)
    tree_b = dendropy.Tree.get(data=newick_b, schema="newick", taxon_namespace=tns)
    return treecompare.symmetric_difference(tree_a, tree_b)


def main():
    print("=" * 70)
    print("  Exp021: Robinson-Foulds Distance — Python/dendropy Baseline")
    print("=" * 70)

    t0 = time.time()
    results = []
    total_pass = 0

    for name, nwk_a, nwk_b, expected_rf, n_leaves, desc in TEST_CASES:
        rf = compute_rf_dendropy(nwk_a, nwk_b)
        passed = rf == expected_rf
        total_pass += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: RF={rf} (expected {expected_rf}) — {desc}")
        results.append({
            "name": name,
            "newick_a": nwk_a,
            "newick_b": nwk_b,
            "rf_distance": rf,
            "expected_rf": expected_rf,
            "n_leaves": n_leaves,
            "description": desc,
            "status": status,
        })

    elapsed = time.time() - t0

    out_dir = WORKSPACE / "experiments/results/021_rf_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rf_python_baseline.json"

    output = {
        "experiment": "021_robinson_foulds_validation",
        "tool": "dendropy",
        "dendropy_version": dendropy.__version__,
        "python_version": sys.version.split()[0],
        "total_cases": len(TEST_CASES),
        "total_pass": total_pass,
        "elapsed_seconds": round(elapsed, 4),
        "cases": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {total_pass}/{len(TEST_CASES)} PASS")
    print(f"  Results: {out_path}")
    print(f"  Elapsed: {elapsed:.4f}s")
    print(f"{'=' * 70}")

    return 0 if total_pass == len(TEST_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
