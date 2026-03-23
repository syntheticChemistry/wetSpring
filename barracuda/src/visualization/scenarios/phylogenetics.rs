// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phylogenetics scenario builders: tree estimation, placement, `UniFrac`,
//! dN/dS, molecular clock, and gene-species reconciliation.

use crate::bio::bootstrap;
use crate::bio::felsenstein::{self, TreeNode};
use crate::bio::molecular_clock;
use crate::bio::placement;
use crate::bio::reconciliation::{self, DtlCosts, FlatRecTree};
use crate::cast::{f64_usize, usize_f64};
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, distribution, gauge, heatmap, node, scaffold, scatter, timeseries};

fn demo_tree() -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: felsenstein::encode_dna("ATCGATCG"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: felsenstein::encode_dna("ATCGATCG"),
            }),
            left_branch: 0.1,
            right_branch: 0.15,
        }),
        right: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "C".into(),
                states: felsenstein::encode_dna("ATCAATCG"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "D".into(),
                states: felsenstein::encode_dna("GCCGATCG"),
            }),
            left_branch: 0.2,
            right_branch: 0.25,
        }),
        left_branch: 0.3,
        right_branch: 0.35,
    }
}

/// Felsenstein pruning + bootstrap support scenario.
///
/// Computes site log-likelihoods and bootstrap replicate distribution using
/// live `BarraCUDA` math (JC69 + RK4 pruning).
#[must_use]
#[expect(clippy::cast_precision_loss)] // site/bootstrap counts < 1000
pub fn felsenstein_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Felsenstein Pruning",
        "JC69 site log-likelihoods and bootstrap replicate support",
    );

    let tree = demo_tree();
    let mu = 1.0;

    let site_lls = felsenstein::site_log_likelihoods(&tree, mu);
    let total_ll = felsenstein::log_likelihood(&tree, mu);

    let alignment = bootstrap::Alignment::from_rows(&[
        felsenstein::encode_dna("ATCGATCG"),
        felsenstein::encode_dna("ATCGATCG"),
        felsenstein::encode_dna("ATCAATCG"),
        felsenstein::encode_dna("GCCGATCG"),
    ]);
    let boot_lls = bootstrap::bootstrap_likelihoods(&tree, &alignment, 100, mu, 42);

    let mut fels_node = node(
        "felsenstein",
        "Felsenstein Pruning",
        "compute",
        &["science.phylogenetics"],
    );
    let site_indices: Vec<f64> = (0..site_lls.len()).map(usize_f64).collect();
    fels_node.data_channels.push(bar(
        "site_ll",
        "Site Log-Likelihoods",
        &site_indices
            .iter()
            .map(|i| format!("site_{}", f64_usize(*i)))
            .collect::<Vec<_>>(),
        &site_lls,
        "ln(L)",
    ));
    fels_node.data_channels.push(gauge(
        "total_ll",
        "Total Log-Likelihood",
        total_ll,
        total_ll * 2.0,
        0.0,
        "ln(L)",
        [total_ll * 1.5, total_ll * 0.5],
        [total_ll * 2.0, total_ll * 1.5],
    ));
    let boot_mean = boot_lls.iter().sum::<f64>() / usize_f64(boot_lls.len());
    let boot_std = (boot_lls
        .iter()
        .map(|v| (v - boot_mean).powi(2))
        .sum::<f64>()
        / boot_lls.len() as f64)
        .sqrt();
    fels_node.data_channels.push(distribution(
        "bootstrap_ll",
        "Bootstrap Replicate Log-Likelihoods",
        "ln(L)",
        &boot_lls,
        boot_mean,
        boot_std,
    ));
    s.nodes.push(fels_node);
    (s, vec![])
}

/// Phylogenetic placement scenario.
///
/// Places a query sequence onto a reference tree and visualises placement
/// likelihoods per edge and top candidate edges.
#[must_use]
pub fn placement_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Phylogenetic Placement",
        "EPA-style query placement onto a reference tree",
    );

    let tree = demo_tree();
    let query = "ATCGATCG";
    let scan = placement::placement_scan(&tree, query, 0.05, 1.0);

    let mut place_node = node(
        "placement",
        "Phylogenetic Placement",
        "compute",
        &["science.placement"],
    );

    let edge_labels: Vec<String> = scan
        .placements
        .iter()
        .map(|p| format!("edge_{}", p.edge_idx))
        .collect();
    let lls: Vec<f64> = scan.placements.iter().map(|p| p.log_likelihood).collect();

    place_node.data_channels.push(heatmap(
        "placement_ll",
        "Placement Log-Likelihoods",
        &edge_labels,
        &["query".into()],
        &lls,
        "ln(L)",
    ));
    place_node.data_channels.push(bar(
        "top_placements",
        "Top Placement Edges",
        &edge_labels,
        &lls,
        "ln(L)",
    ));
    place_node.data_channels.push(gauge(
        "placement_confidence",
        "Placement Confidence",
        scan.confidence,
        0.0,
        1.0,
        "ratio",
        [0.8, 1.0],
        [0.5, 0.8],
    ));
    s.nodes.push(place_node);
    (s, vec![])
}

/// `UniFrac` beta-diversity scenario.
///
/// Computes unweighted and weighted `UniFrac` distance matrices and projects
/// samples via `PCoA` scatter (using first two BC axes as proxy).
#[must_use]
pub fn unifrac_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    use crate::bio::unifrac;
    use std::collections::HashMap;

    let mut s = scaffold(
        "UniFrac Beta Diversity",
        "Weighted and unweighted UniFrac distance matrices",
    );

    let newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35);";
    let tree = unifrac::PhyloTree::from_newick(newick);

    let sample_a: HashMap<String, f64> = [("A", 10.0), ("B", 20.0), ("C", 5.0), ("D", 1.0)]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
    let sample_b: HashMap<String, f64> = [("A", 5.0), ("B", 5.0), ("C", 25.0), ("D", 15.0)]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
    let sample_c: HashMap<String, f64> = [("A", 1.0), ("B", 2.0), ("C", 30.0), ("D", 20.0)]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();

    let mut samples = unifrac::AbundanceTable::new();
    samples.insert("Sample1".into(), sample_a);
    samples.insert("Sample2".into(), sample_b);
    samples.insert("Sample3".into(), sample_c);

    let wu_matrix = unifrac::unifrac_distance_matrix(&tree, &samples, true);
    let uu_matrix = unifrac::unifrac_distance_matrix(&tree, &samples, false);

    let mut uf_node = node("unifrac", "UniFrac", "compute", &["science.unifrac"]);

    uf_node.data_channels.push(heatmap(
        "weighted_unifrac",
        "Weighted UniFrac",
        &wu_matrix.sample_ids,
        &wu_matrix.sample_ids,
        &wu_matrix.condensed,
        "UniFrac",
    ));
    uf_node.data_channels.push(heatmap(
        "unweighted_unifrac",
        "Unweighted UniFrac",
        &uu_matrix.sample_ids,
        &uu_matrix.sample_ids,
        &uu_matrix.condensed,
        "UniFrac",
    ));

    let n = wu_matrix.sample_ids.len();
    let x_coords: Vec<f64> = (0..n)
        .map(|i| {
            if i < wu_matrix.condensed.len() {
                wu_matrix.condensed[i]
            } else {
                0.0
            }
        })
        .collect();
    let y_coords: Vec<f64> = (0..n)
        .map(|i| {
            let idx = n - 1 + i;
            if idx < wu_matrix.condensed.len() {
                wu_matrix.condensed[idx]
            } else {
                0.0
            }
        })
        .collect();
    uf_node.data_channels.push(scatter(
        "unifrac_pcoa",
        "UniFrac PCoA (approx)",
        &x_coords,
        &y_coords,
        &wu_matrix.sample_ids,
        "PC1",
        "PC2",
        "UniFrac",
    ));
    s.nodes.push(uf_node);
    (s, vec![])
}

/// dN/dS selection pressure scenario.
///
/// Computes pairwise dN/dS (omega) ratios for codon-aligned sequences and
/// visualises per-gene omega distributions.
#[must_use]
#[expect(clippy::cast_precision_loss)] // pair counts < 100
pub fn dnds_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    use crate::bio::dnds;

    let mut s = scaffold(
        "dN/dS Selection Pressure",
        "Pairwise omega ratios for detecting positive, neutral, and purifying selection",
    );

    let seq1 = b"ATGATGATGATGATGATG";
    let seq2 = b"ATGATGATGATCATGATG";
    let seq3 = b"GCGATGATGATGATGATG";

    let pairs: Vec<(&[u8], &[u8])> = vec![(seq1, seq2), (seq1, seq3), (seq2, seq3)];
    let results = dnds::pairwise_dnds_batch(&pairs);

    let mut dnds_node = node("dnds", "dN/dS Analysis", "compute", &["science.dnds"]);

    let mut pair_labels = Vec::new();
    let mut omegas = Vec::new();
    let mut nonsyn_rates = Vec::new();
    let mut syn_rates = Vec::new();

    for (i, r) in results.iter().enumerate() {
        if let Ok(res) = r {
            pair_labels.push(format!("pair_{i}"));
            omegas.push(res.omega.unwrap_or(f64::INFINITY).min(10.0));
            nonsyn_rates.push(res.dn);
            syn_rates.push(res.ds);
        }
    }

    dnds_node.data_channels.push(bar(
        "omega_ratios",
        "Omega (dN/dS) Ratios",
        &pair_labels,
        &omegas,
        "ω",
    ));

    if !omegas.is_empty() {
        let mean = omegas.iter().sum::<f64>() / omegas.len() as f64;
        let std =
            (omegas.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / omegas.len() as f64).sqrt();
        dnds_node.data_channels.push(distribution(
            "omega_dist",
            "dN/dS Distribution",
            "ω",
            &omegas,
            mean,
            std,
        ));
    }

    dnds_node.data_channels.push(scatter(
        "dn_vs_ds",
        "dN vs dS",
        &syn_rates,
        &nonsyn_rates,
        &pair_labels,
        "dS (synonymous)",
        "dN (non-synonymous)",
        "substitutions/site",
    ));
    s.nodes.push(dnds_node);
    (s, vec![])
}

/// Molecular clock scenario.
///
/// Applies strict clock to branch lengths with calibration constraints
/// and visualises divergence times and rate variation.
#[must_use]
#[expect(clippy::cast_precision_loss)] // node/branch counts < 100
pub fn molecular_clock_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Molecular Clock",
        "Strict clock divergence time estimation with calibration points",
    );

    let branch_lengths = vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.05];
    let parent_indices: Vec<Option<usize>> =
        vec![Some(4), Some(4), Some(5), Some(5), Some(6), Some(6), None];
    let root_age = 100.0;
    let calibrations = vec![
        molecular_clock::CalibrationPoint {
            node_id: 4,
            min_age_ma: 30.0,
            max_age_ma: 50.0,
        },
        molecular_clock::CalibrationPoint {
            node_id: 5,
            min_age_ma: 40.0,
            max_age_ma: 60.0,
        },
    ];

    let mut clock_node = node(
        "molecular_clock",
        "Molecular Clock",
        "compute",
        &["science.molecular_clock"],
    );

    if let Some(result) =
        molecular_clock::strict_clock(&branch_lengths, &parent_indices, root_age, &calibrations)
    {
        let node_indices: Vec<f64> = (0..result.node_ages.len()).map(|i| i as f64).collect();
        clock_node.data_channels.push(timeseries(
            "divergence_times",
            "Node Divergence Times",
            "Node Index",
            "Age (Ma)",
            "Ma",
            &node_indices,
            &result.node_ages,
        ));

        let rates = molecular_clock::relaxed_clock_rates(
            &branch_lengths,
            &result.node_ages,
            &parent_indices,
        );
        let rate_labels: Vec<String> = (0..rates.len()).map(|i| format!("branch_{i}")).collect();
        clock_node.data_channels.push(bar(
            "rate_categories",
            "Branch Rate Categories",
            &rate_labels,
            &rates,
            "subst/site/Ma",
        ));

        let cv = molecular_clock::rate_variation_cv(&rates);
        clock_node.data_channels.push(gauge(
            "rate_cv",
            "Rate Variation CV",
            cv,
            0.0,
            2.0,
            "CV",
            [0.0, 0.5],
            [0.5, 1.0],
        ));
    }
    s.nodes.push(clock_node);
    (s, vec![])
}

/// Gene-species tree reconciliation scenario.
///
/// Reconciles a gene tree against a species tree using DTL (duplication,
/// transfer, loss) costs and visualises optimal cost breakdown.
#[must_use]
pub fn reconciliation_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Gene-Species Reconciliation",
        "DTL reconciliation optimal cost and event mapping",
    );

    let host = FlatRecTree {
        names: vec!["H0".into(), "H1".into(), "H2".into()],
        left_child: vec![1, u32::MAX, u32::MAX],
        right_child: vec![2, u32::MAX, u32::MAX],
    };
    let parasite = FlatRecTree {
        names: vec!["P0".into(), "P1".into(), "P2".into()],
        left_child: vec![1, u32::MAX, u32::MAX],
        right_child: vec![2, u32::MAX, u32::MAX],
    };
    let tip_mapping = vec![("P1".into(), "H1".into()), ("P2".into(), "H2".into())];
    let costs = DtlCosts::default();

    let result = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);

    let mut rec_node = node(
        "reconciliation",
        "DTL Reconciliation",
        "compute",
        &["science.reconciliation"],
    );

    rec_node.data_channels.push(bar(
        "dtl_costs",
        "DTL Event Costs",
        &["Duplication", "Transfer", "Loss"],
        &[
            f64::from(costs.duplication),
            f64::from(costs.transfer),
            f64::from(costs.loss),
        ],
        "cost units",
    ));
    let cost = f64::from(result.optimal_cost);
    rec_node.data_channels.push(gauge(
        "optimal_cost",
        "Optimal Reconciliation Cost",
        cost,
        0.0,
        cost.mul_add(3.0, 1.0),
        "cost units",
        [0.0, cost * 1.5],
        [cost * 1.5, cost * 3.0],
    ));
    s.nodes.push(rec_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn felsenstein_scenario_builds() {
        let (s, _) = felsenstein_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn placement_scenario_builds() {
        let (s, _) = placement_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn unifrac_scenario_builds() {
        let (s, _) = unifrac_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(s.nodes[0].data_channels.len() >= 2);
    }

    #[test]
    fn dnds_scenario_builds() {
        let (s, _) = dnds_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn molecular_clock_scenario_builds() {
        let (s, _) = molecular_clock_scenario();
        assert_eq!(s.nodes.len(), 1);
    }

    #[test]
    fn reconciliation_scenario_builds() {
        let (s, _) = reconciliation_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }
}
