// SPDX-License-Identifier: AGPL-3.0-or-later
//! DTL (Duplication-Transfer-Loss) reconciliation for cophylogenetics.
//!
//! Reconciles a gene/parasite tree with a species/host tree under the
//! DTL event model (speciation=0, duplication=D, transfer=T, loss=L).
//! This is the core primitive for cophylogenetic reconstruction studied
//! in Zheng et al. 2023 (BCB top 10%).
//!
//! # References
//!
//! - Zheng et al. 2023, *ACM-BCB* (top 10%)
//!   "Impact of Species Tree Estimation Error on Cophylogenetic Reconstruction"
//! - Bansal, Alm & Kellis 2012, *PNAS* 109:11319-11324 (DTL-reconciliation)
//!
//! # GPU Promotion
//!
//! The DP table is `[n_parasite × n_host]`. Within each parasite node's
//! row, evaluating different host mappings requires scanning all hosts
//! (for transfers), so each row is `O(n_host²)`. Batch reconciliation of
//! multiple gene families is embarrassingly parallel — one workgroup per
//! gene tree.

/// Event types in DTL reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlEvent {
    /// Tip-to-tip mapping (zero cost).
    Tip,
    /// Co-speciation (both lineages diverge together).
    Speciation,
    /// Gene duplication on same host lineage.
    Duplication,
    /// Horizontal transfer to a different host lineage.
    Transfer,
    /// Gene lineage lost on one host child.
    Loss,
    /// No valid mapping exists.
    Impossible,
}

/// Cost parameters for DTL events.
#[derive(Debug, Clone, Copy)]
pub struct DtlCosts {
    pub duplication: u32,
    pub transfer: u32,
    pub loss: u32,
}

impl Default for DtlCosts {
    fn default() -> Self {
        Self {
            duplication: 2,
            transfer: 3,
            loss: 1,
        }
    }
}

/// A node in a flat tree representation for reconciliation.
#[derive(Debug, Clone)]
pub struct FlatRecTree {
    /// Node names in post-order.
    pub names: Vec<String>,
    /// Left child index (`u32::MAX` for leaves).
    pub left_child: Vec<u32>,
    /// Right child index (`u32::MAX` for leaves).
    pub right_child: Vec<u32>,
}

const NO_CHILD: u32 = u32::MAX;
const INF_COST: u32 = u32::MAX / 2;

impl FlatRecTree {
    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    #[inline]
    fn is_leaf(&self, idx: usize) -> bool {
        self.left_child[idx] == NO_CHILD
    }
}

/// Result of DTL reconciliation.
#[derive(Debug, Clone)]
pub struct DtlResult {
    /// Optimal total cost.
    pub optimal_cost: u32,
    /// Host node where parasite root maps optimally.
    pub optimal_host: String,
    /// DP table: `cost[p_idx * n_host + h_idx]` (flat, GPU-ready layout).
    pub cost_table: Vec<u32>,
    /// Event table: `event[p_idx * n_host + h_idx]`.
    pub event_table: Vec<DtlEvent>,
}

/// DTL reconciliation via DP on host × parasite product space.
///
/// Both trees must be in post-order (children before parents).
/// `tip_mapping` maps parasite leaf name → host leaf name.
///
/// # Panics
///
/// Panics if trees are empty.
#[must_use]
pub fn reconcile_dtl(
    host: &FlatRecTree,
    parasite: &FlatRecTree,
    tip_mapping: &[(String, String)],
    costs: &DtlCosts,
) -> DtlResult {
    let nh = host.len();
    let np = parasite.len();
    assert!(!host.is_empty() && !parasite.is_empty());

    // Build tip map: parasite name → host name
    let tip_map: std::collections::HashMap<&str, &str> = tip_mapping
        .iter()
        .map(|(p, h)| (p.as_str(), h.as_str()))
        .collect();

    // Name→index lookup for host
    let host_idx: std::collections::HashMap<&str, usize> = host
        .names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut cost = vec![INF_COST; np * nh];
    let mut event = vec![DtlEvent::Impossible; np * nh];

    for p in 0..np {
        if parasite.is_leaf(p) {
            // Parasite leaf
            let p_name = &parasite.names[p];
            if let Some(&mapped_host) = tip_map.get(p_name.as_str()) {
                if let Some(&h_target) = host_idx.get(mapped_host) {
                    // Direct mapping
                    cost[p * nh + h_target] = 0;
                    event[p * nh + h_target] = DtlEvent::Tip;

                    // Propagate up host tree: mapping to ancestor costs losses
                    propagate_losses(&mut cost, &mut event, host, p, h_target, nh, costs);
                }
            }
        } else {
            // Internal parasite node
            let p1 = parasite.left_child[p] as usize;
            let p2 = parasite.right_child[p] as usize;

            for h in 0..nh {
                let mut best = INF_COST;
                let mut best_ev = DtlEvent::Impossible;

                if !host.is_leaf(h) {
                    let h1 = host.left_child[h] as usize;
                    let h2 = host.right_child[h] as usize;

                    // Co-speciation: p1→h1,p2→h2 or p1→h2,p2→h1
                    let spec1 = cost[p1 * nh + h1].saturating_add(cost[p2 * nh + h2]);
                    let spec2 = cost[p1 * nh + h2].saturating_add(cost[p2 * nh + h1]);
                    if spec1 < best {
                        best = spec1;
                        best_ev = DtlEvent::Speciation;
                    }
                    if spec2 < best {
                        best = spec2;
                        best_ev = DtlEvent::Speciation;
                    }

                    // Loss: pass-through to child host
                    for &hc in &[h1, h2] {
                        if cost[p * nh + hc] < INF_COST {
                            let lc = cost[p * nh + hc].saturating_add(costs.loss);
                            if lc < best {
                                best = lc;
                                best_ev = DtlEvent::Loss;
                            }
                        }
                    }
                }

                // Duplication: both parasite children on same host
                let dup = costs
                    .duplication
                    .saturating_add(cost[p1 * nh + h])
                    .saturating_add(cost[p2 * nh + h]);
                if dup < best {
                    best = dup;
                    best_ev = DtlEvent::Duplication;
                }

                // Transfer: one child stays, other goes anywhere
                for ht in 0..nh {
                    if ht == h {
                        continue;
                    }
                    let t1 = costs
                        .transfer
                        .saturating_add(cost[p1 * nh + h])
                        .saturating_add(cost[p2 * nh + ht]);
                    let t2 = costs
                        .transfer
                        .saturating_add(cost[p1 * nh + ht])
                        .saturating_add(cost[p2 * nh + h]);
                    if t1 < best {
                        best = t1;
                        best_ev = DtlEvent::Transfer;
                    }
                    if t2 < best {
                        best = t2;
                        best_ev = DtlEvent::Transfer;
                    }
                }

                cost[p * nh + h] = best;
                event[p * nh + h] = best_ev;
            }
        }
    }

    // Find optimal mapping for parasite root
    let p_root = np - 1;
    let mut opt_cost = INF_COST;
    let mut opt_host_idx = 0;
    for h in 0..nh {
        if cost[p_root * nh + h] < opt_cost {
            opt_cost = cost[p_root * nh + h];
            opt_host_idx = h;
        }
    }

    DtlResult {
        optimal_cost: opt_cost,
        optimal_host: host.names[opt_host_idx].clone(),
        cost_table: cost,
        event_table: event,
    }
}

fn propagate_losses(
    cost: &mut [u32],
    event: &mut [DtlEvent],
    host: &FlatRecTree,
    p: usize,
    h_target: usize,
    nh: usize,
    costs: &DtlCosts,
) {
    // For each host node that is an ancestor of h_target, the cost is
    // the number of losses along the path. We propagate upward.
    for h in (0..nh).rev() {
        if h == h_target || host.is_leaf(h) {
            continue;
        }
        if !host.is_leaf(h) {
            let h1 = host.left_child[h] as usize;
            let h2 = host.right_child[h] as usize;
            // If either child has a valid mapping, we can reach through loss
            for &hc in &[h1, h2] {
                if cost[p * nh + hc] < INF_COST {
                    let lc = cost[p * nh + hc].saturating_add(costs.loss);
                    if lc < cost[p * nh + h] {
                        cost[p * nh + h] = lc;
                        event[p * nh + h] = DtlEvent::Loss;
                    }
                }
            }
        }
    }
}

/// Batch reconciliation: reconcile multiple parasite trees against
/// the same host tree. Each reconciliation is independent — maps to
/// one GPU workgroup per gene family.
#[must_use]
pub fn reconcile_batch(
    host: &FlatRecTree,
    parasites: &[(&FlatRecTree, &[(String, String)])],
    costs: &DtlCosts,
) -> Vec<DtlResult> {
    parasites
        .iter()
        .map(|(para, tip_map)| reconcile_dtl(host, para, tip_map, costs))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2leaf_host() -> FlatRecTree {
        FlatRecTree {
            names: vec!["H_A".into(), "H_B".into(), "H_AB".into()],
            left_child: vec![NO_CHILD, NO_CHILD, 0],
            right_child: vec![NO_CHILD, NO_CHILD, 1],
        }
    }

    fn make_2leaf_parasite() -> FlatRecTree {
        FlatRecTree {
            names: vec!["P_A".into(), "P_B".into(), "P_AB".into()],
            left_child: vec![NO_CHILD, NO_CHILD, 0],
            right_child: vec![NO_CHILD, NO_CHILD, 1],
        }
    }

    fn make_4leaf_host() -> FlatRecTree {
        FlatRecTree {
            names: vec![
                "H_A".into(),
                "H_B".into(),
                "H_AB".into(),
                "H_C".into(),
                "H_D".into(),
                "H_CD".into(),
                "H_root".into(),
            ],
            left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, NO_CHILD, 3, 2],
            right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, NO_CHILD, 4, 5],
        }
    }

    #[test]
    fn congruent_trees_zero_cost() {
        let host = make_2leaf_host();
        let para = make_2leaf_parasite();
        let tip_map = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
        let result = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
        assert_eq!(result.optimal_cost, 0, "congruent trees = zero cost");
    }

    #[test]
    fn duplication_detected() {
        let host = make_4leaf_host();
        // Parasite: P_1 and P_2 both map to H_A (duplication)
        let para = FlatRecTree {
            names: vec![
                "P_1".into(),
                "P_2".into(),
                "P_12".into(),
                "P_3".into(),
                "P_root".into(),
            ],
            left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, 2],
            right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, 3],
        };
        let tip_map = vec![
            ("P_1".into(), "H_A".into()),
            ("P_2".into(), "H_A".into()),
            ("P_3".into(), "H_C".into()),
        ];
        let result = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
        assert!(
            result.optimal_cost > 0,
            "duplication scenario should have non-zero cost: {}",
            result.optimal_cost
        );
    }

    #[test]
    fn cost_increases_with_events() {
        let host = make_2leaf_host();
        // Congruent
        let para = make_2leaf_parasite();
        let tip_map = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
        let r0 = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());

        // Both parasite leaves map to same host = duplication needed
        let tip_map_dup = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_A".into())];
        let r1 = reconcile_dtl(&host, &para, &tip_map_dup, &DtlCosts::default());
        assert!(
            r1.optimal_cost > r0.optimal_cost,
            "duplication should cost more: {} vs {}",
            r1.optimal_cost,
            r0.optimal_cost
        );
    }

    #[test]
    fn dp_table_dimensions() {
        let host = make_2leaf_host();
        let para = make_2leaf_parasite();
        let tip_map = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
        let result = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
        assert_eq!(result.cost_table.len(), 3 * 3); // 3 parasite × 3 host
        assert_eq!(result.event_table.len(), 3 * 3);
    }

    #[test]
    fn batch_reconciliation() {
        let host = make_2leaf_host();
        let para = make_2leaf_parasite();
        let tm1 = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
        let tm2 = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_A".into())];
        let results = reconcile_batch(&host, &[(&para, &tm1), (&para, &tm2)], &DtlCosts::default());
        assert_eq!(results.len(), 2);
        assert!(results[0].optimal_cost <= results[1].optimal_cost);
    }

    #[test]
    fn deterministic() {
        let host = make_4leaf_host();
        let para = FlatRecTree {
            names: vec![
                "P_1".into(),
                "P_2".into(),
                "P_12".into(),
                "P_3".into(),
                "P_root".into(),
            ],
            left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, 2],
            right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, 3],
        };
        let tip_map = vec![
            ("P_1".into(), "H_A".into()),
            ("P_2".into(), "H_A".into()),
            ("P_3".into(), "H_C".into()),
        ];
        let r1 = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
        let r2 = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
        assert_eq!(r1.optimal_cost, r2.optimal_cost);
        assert_eq!(r1.optimal_host, r2.optimal_host);
    }
}
