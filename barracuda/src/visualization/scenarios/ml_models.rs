// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML model scenario builders: decision tree, random forest, and echo
//! state network (ESN) reservoir computing.

use crate::bio::decision_tree::DecisionTree;
use crate::bio::esn::{EsnConfig, LegacyEsn};
use crate::bio::random_forest::RandomForest;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, distribution, gauge, node, scaffold, timeseries};

/// Decision tree scenario.
///
/// Builds a small decision tree, runs prediction on test samples,
/// and visualises feature importance and accuracy.
#[must_use]
pub fn decision_tree_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Decision Tree",
        "Single decision tree classification with feature importance",
    );

    let features = vec![0, 1, -1, -1, -1];
    let thresholds = vec![0.5, 0.3, 0.0, 0.0, 0.0];
    let left_children = vec![1, 3, -1, -1, -1];
    let right_children = vec![2, 4, -1, -1, -1];
    let predictions: Vec<Option<usize>> = vec![None, None, Some(0), Some(1), Some(0)];

    let Ok(tree) = DecisionTree::from_arrays(
        &features,
        &thresholds,
        &left_children,
        &right_children,
        &predictions,
        2,
    ) else {
        return (s, vec![]);
    };

    let test_samples = vec![
        vec![0.2, 0.1],
        vec![0.8, 0.5],
        vec![0.3, 0.9],
        vec![0.7, 0.2],
        vec![0.6, 0.4],
    ];
    let expected = [1, 0, 1, 0, 0];
    let predictions_vec = tree.predict_batch(&test_samples);

    let correct = predictions_vec
        .iter()
        .zip(expected.iter())
        .filter(|(p, e)| p == e)
        .count();
    #[expect(clippy::cast_precision_loss)] // sample counts < 100
    let accuracy = correct as f64 / predictions_vec.len() as f64;

    let mut dt_node = node(
        "decision_tree",
        "Decision Tree",
        "compute",
        &["science.decision_tree"],
    );

    let feature_names = vec!["Feature 0", "Feature 1"];
    let mut importance = vec![0.0f64; 2];
    for i in 0..tree.n_nodes() {
        let n = tree.node_at(i);
        if !n.is_leaf() {
            let feat_idx = n.feature as usize;
            if feat_idx < importance.len() {
                importance[feat_idx] += 1.0;
            }
        }
    }
    let total: f64 = importance.iter().sum();
    if total > 0.0 {
        for v in &mut importance {
            *v /= total;
        }
    }

    dt_node.data_channels.push(bar(
        "feature_importance",
        "Feature Importance",
        &feature_names,
        &importance,
        "importance",
    ));

    dt_node.data_channels.push(gauge(
        "accuracy",
        "Classification Accuracy",
        accuracy,
        0.0,
        1.0,
        "ratio",
        [0.8, 1.0],
        [0.5, 0.8],
    ));
    s.nodes.push(dt_node);
    (s, vec![])
}

/// Random forest scenario.
///
/// Constructs a small ensemble of trees, runs voting predictions,
/// and visualises feature importance and out-of-bag error distribution.
#[must_use]
pub fn random_forest_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Random Forest",
        "Ensemble classification with voting, feature importance, and OOB error",
    );

    let make_tree = |feat: i32, thresh: f64, left_pred: usize, right_pred: usize| {
        DecisionTree::from_arrays(
            &[feat, -1, -1],
            &[thresh, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(left_pred), Some(right_pred)],
            2,
        )
    };

    let tree_results = [
        make_tree(0, 0.5, 0, 1),
        make_tree(1, 0.4, 1, 0),
        make_tree(0, 0.6, 0, 1),
        make_tree(1, 0.3, 1, 0),
        make_tree(0, 0.55, 0, 1),
    ];
    let mut trees = Vec::with_capacity(tree_results.len());
    for r in tree_results {
        let Ok(t) = r else {
            return (s, vec![]);
        };
        trees.push(t);
    }
    let Ok(forest) = RandomForest::from_trees(trees, 2) else {
        return (s, vec![]);
    };

    let test_samples = vec![
        vec![0.2, 0.1],
        vec![0.8, 0.5],
        vec![0.3, 0.9],
        vec![0.7, 0.2],
    ];
    let votes = forest.predict_batch_with_votes(&test_samples);
    let oob_errors: Vec<f64> = votes.iter().map(|v| 1.0 - v.confidence).collect();

    let mut rf_node = node(
        "random_forest",
        "Random Forest",
        "compute",
        &["science.random_forest"],
    );

    let feature_names = vec!["Feature 0", "Feature 1"];
    let f0_count = 3.0;
    let f1_count = 2.0;
    let total = f0_count + f1_count;
    rf_node.data_channels.push(bar(
        "feature_importance",
        "Feature Importance (split frequency)",
        &feature_names,
        &[f0_count / total, f1_count / total],
        "importance",
    ));

    #[expect(clippy::cast_precision_loss)] // vote counts < 100
    let n_oob = oob_errors.len() as f64;
    let mean_oob = oob_errors.iter().sum::<f64>() / n_oob;
    let std_oob = (oob_errors
        .iter()
        .map(|v| (v - mean_oob).powi(2))
        .sum::<f64>()
        / n_oob)
        .sqrt();
    rf_node.data_channels.push(distribution(
        "oob_error",
        "OOB Error Distribution",
        "error rate",
        &oob_errors,
        mean_oob,
        std_oob,
    ));

    rf_node.data_channels.push(gauge(
        "ensemble_accuracy",
        "Ensemble Accuracy",
        1.0 - mean_oob,
        0.0,
        1.0,
        "ratio",
        [0.8, 1.0],
        [0.5, 0.8],
    ));
    s.nodes.push(rf_node);
    (s, vec![])
}

/// Echo State Network (ESN) scenario.
///
/// Trains a small reservoir on a sine wave and visualises predictions
/// vs actual values alongside reservoir state gauge.
#[must_use]
#[expect(clippy::cast_precision_loss)] // loop indices < 300
pub fn esn_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Echo State Network",
        "Reservoir computing: prediction vs actual on time series data",
    );

    let config = EsnConfig {
        input_size: 1,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.1,
        leak_rate: 0.3,
        regularization: 1e-6,
        seed: 42,
    };
    let mut esn = LegacyEsn::new(config);

    let n_train = 200;
    let n_test = 50;
    let train_inputs: Vec<Vec<f64>> = (0..n_train).map(|i| vec![(i as f64 * 0.1).sin()]).collect();
    let train_targets: Vec<Vec<f64>> = (0..n_train)
        .map(|i| vec![((i + 1) as f64 * 0.1).sin()])
        .collect();

    esn.train(&train_inputs, &train_targets);

    esn.reset_state();
    let test_inputs: Vec<Vec<f64>> = (n_train..n_train + n_test)
        .map(|i| vec![(i as f64 * 0.1).sin()])
        .collect();
    let test_actual: Vec<f64> = (n_train..n_train + n_test)
        .map(|i| ((i + 1) as f64 * 0.1).sin())
        .collect();

    let mut predictions = Vec::with_capacity(n_test);
    for input in &test_inputs {
        esn.update(input);
        let out = esn.readout();
        predictions.push(out[0]);
    }

    let time_axis: Vec<f64> = (0..n_test).map(|i| i as f64).collect();

    let mut esn_node = node("esn", "Echo State Network", "compute", &["science.esn"]);

    esn_node.data_channels.push(timeseries(
        "predictions",
        "ESN Predictions",
        "Time Step",
        "Value",
        "AU",
        &time_axis,
        &predictions,
    ));
    esn_node.data_channels.push(timeseries(
        "actual",
        "Actual Values",
        "Time Step",
        "Value",
        "AU",
        &time_axis,
        &test_actual,
    ));

    let reservoir_energy: f64 =
        esn.state().iter().map(|v| v * v).sum::<f64>() / esn.state().len() as f64;
    esn_node.data_channels.push(gauge(
        "reservoir_state",
        "Reservoir Energy",
        reservoir_energy.sqrt(),
        0.0,
        1.0,
        "RMS",
        [0.0, 0.5],
        [0.5, 1.0],
    ));
    s.nodes.push(esn_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_tree_builds() {
        let (s, _) = decision_tree_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn random_forest_builds() {
        let (s, _) = random_forest_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 3);
    }

    #[test]
    fn esn_builds() {
        let (s, _) = esn_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 3);
    }
}
