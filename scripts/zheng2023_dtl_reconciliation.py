#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""
Python baseline: DTL (Duplication-Transfer-Loss) reconciliation.

Core cophylogenetic primitive (Zheng et al. 2023 BCB) — reconciles a
gene/parasite tree with a species/host tree under the DTL event model.

Usage:
    python3 scripts/zheng2023_dtl_reconciliation.py

Outputs JSON with optimal costs and event counts for Rust comparison.
"""

import json

# Event costs
COST_SPEC = 0  # co-speciation (free)
COST_DUP = 2   # duplication
COST_TRANS = 3  # horizontal transfer
COST_LOSS = 1   # loss

INF = float("inf")


def reconcile_dtl(host_tree, parasite_tree, tip_mapping,
                  cost_dup=COST_DUP, cost_trans=COST_TRANS, cost_loss=COST_LOSS):
    """DTL reconciliation via DP on host × parasite node product space.

    Trees are dicts: {"name": str, "children": [left, right]} for internal,
    {"name": str} for leaves.

    tip_mapping: {parasite_leaf_name: host_leaf_name}

    Returns: (optimal_cost, event_counts, mapping)
    """
    # Collect nodes in post-order
    def postorder(node):
        nodes = []
        if "children" in node:
            for c in node["children"]:
                nodes.extend(postorder(c))
        nodes.append(node["name"])
        return nodes

    h_post = postorder(host_tree)
    p_post = postorder(parasite_tree)

    # Build lookup
    def build_lookup(tree):
        lookup = {}
        def visit(node):
            lookup[node["name"]] = node
            if "children" in node:
                for c in node["children"]:
                    visit(c)
        visit(tree)
        return lookup

    h_lookup = build_lookup(host_tree)
    p_lookup = build_lookup(parasite_tree)

    def is_leaf(node):
        return "children" not in node

    # DP table: C[p][h] = min cost of mapping parasite node p to host node h
    C = {}
    event = {}  # track which event was chosen

    for p in p_post:
        C[p] = {}
        event[p] = {}
        p_node = p_lookup[p]

        for h in h_post:
            h_node = h_lookup[h]

            if is_leaf(p_node):
                # Parasite leaf: must map to its host via tip_mapping
                if is_leaf(h_node):
                    if tip_mapping.get(p) == h:
                        C[p][h] = 0
                        event[p][h] = "tip"
                    else:
                        C[p][h] = INF
                        event[p][h] = "impossible"
                else:
                    # Parasite leaf mapped to internal host node = loss needed
                    h_children = h_node["children"]
                    best = INF
                    for hc in h_children:
                        val = C[p][hc["name"]] + cost_loss
                        if val < best:
                            best = val
                    C[p][h] = best
                    event[p][h] = "loss" if best < INF else "impossible"
            else:
                # Internal parasite node with children p1, p2
                p1 = p_node["children"][0]["name"]
                p2 = p_node["children"][1]["name"]

                best = INF
                best_event = "none"

                if is_leaf(h_node):
                    # Can only do duplication at a leaf host
                    dup_cost = cost_dup + C[p1][h] + C[p2][h]
                    if dup_cost < best:
                        best = dup_cost
                        best_event = "duplication"

                    # Or transfer: one child stays, other goes anywhere
                    for h2 in h_post:
                        trans1 = cost_trans + C[p1][h] + C[p2][h2]
                        trans2 = cost_trans + C[p1][h2] + C[p2][h]
                        if trans1 < best:
                            best = trans1
                            best_event = "transfer"
                        if trans2 < best:
                            best = trans2
                            best_event = "transfer"
                else:
                    h_children = h_node["children"]
                    h1 = h_children[0]["name"]
                    h2 = h_children[1]["name"]

                    # Co-speciation: p1→h1, p2→h2 or p1→h2, p2→h1
                    spec1 = C[p1][h1] + C[p2][h2]
                    spec2 = C[p1][h2] + C[p2][h1]
                    if spec1 < best:
                        best = spec1
                        best_event = "speciation"
                    if spec2 < best:
                        best = spec2
                        best_event = "speciation"

                    # Duplication: both children map to same host
                    dup_cost = cost_dup + C[p1][h] + C[p2][h]
                    if dup_cost < best:
                        best = dup_cost
                        best_event = "duplication"

                    # Transfer: one child stays, other goes anywhere
                    for ht in h_post:
                        trans1 = cost_trans + C[p1][h] + C[p2][ht]
                        trans2 = cost_trans + C[p1][ht] + C[p2][h]
                        if trans1 < best:
                            best = trans1
                            best_event = "transfer"
                        if trans2 < best:
                            best = trans2
                            best_event = "transfer"

                    # Loss: pass through to child host
                    for hc in h_children:
                        loss_cost = cost_loss + C[p][hc["name"]]
                        if loss_cost < best:
                            best = loss_cost
                            best_event = "loss"

                C[p][h] = best
                event[p][h] = best_event

    # Optimal: min over all host nodes for parasite root
    p_root = p_post[-1]
    opt_cost = INF
    opt_host = None
    for h in h_post:
        if C[p_root][h] < opt_cost:
            opt_cost = C[p_root][h]
            opt_host = h

    return opt_cost, opt_host, C, event


def main():
    # Test case 1: Simple 3-leaf congruent trees (no events needed)
    host1 = {
        "name": "H_AB",
        "children": [
            {"name": "H_A"},
            {"name": "H_B"},
        ],
    }
    para1 = {
        "name": "P_AB",
        "children": [
            {"name": "P_A"},
            {"name": "P_B"},
        ],
    }
    tip_map1 = {"P_A": "H_A", "P_B": "H_B"}
    cost1, host1_opt, _, _ = reconcile_dtl(host1, para1, tip_map1)

    # Test case 2: 4-leaf with one duplication
    host2 = {
        "name": "H_root",
        "children": [
            {"name": "H_AB", "children": [{"name": "H_A"}, {"name": "H_B"}]},
            {"name": "H_CD", "children": [{"name": "H_C"}, {"name": "H_D"}]},
        ],
    }
    # Parasite tree has a duplication: P1 and P2 both map to host A
    para2 = {
        "name": "P_root",
        "children": [
            {
                "name": "P_12",
                "children": [{"name": "P_1"}, {"name": "P_2"}],
            },
            {"name": "P_3"},
        ],
    }
    tip_map2 = {"P_1": "H_A", "P_2": "H_A", "P_3": "H_C"}
    cost2, host2_opt, _, _ = reconcile_dtl(host2, para2, tip_map2)

    # Test case 3: Transfer event needed
    host3 = {
        "name": "H_root",
        "children": [
            {"name": "H_AB", "children": [{"name": "H_A"}, {"name": "H_B"}]},
            {"name": "H_C"},
        ],
    }
    para3 = {
        "name": "P_root",
        "children": [
            {"name": "P_A"},
            {"name": "P_C"},
        ],
    }
    # P_A → H_A, P_C → H_C: co-speciation at root won't work perfectly
    tip_map3 = {"P_A": "H_A", "P_C": "H_C"}
    cost3, host3_opt, _, _ = reconcile_dtl(host3, para3, tip_map3)

    results = {
        "test_congruent": {
            "optimal_cost": cost1,
            "optimal_host": host1_opt,
            "description": "congruent 2-leaf trees, zero cost",
        },
        "test_duplication": {
            "optimal_cost": cost2,
            "optimal_host": host2_opt,
            "description": "3-leaf parasite on 4-leaf host, duplication needed",
        },
        "test_transfer": {
            "optimal_cost": cost3,
            "optimal_host": host3_opt,
            "description": "2-leaf parasite on 3-node host tree",
        },
        "costs": {
            "speciation": COST_SPEC,
            "duplication": COST_DUP,
            "transfer": COST_TRANS,
            "loss": COST_LOSS,
        },
    }

    print(json.dumps(results, indent=2))

    print("\n--- Python DTL Baseline ---")
    print(f"Congruent: cost={cost1}, mapped to {host1_opt}")
    print(f"Duplication: cost={cost2}, mapped to {host2_opt}")
    print(f"Transfer: cost={cost3}, mapped to {host3_opt}")


if __name__ == "__main__":
    main()
