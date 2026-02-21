// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: PFAS decision tree inference — Exp008 Phase 3.
//!
//! Loads a decision tree trained in Python (sklearn) and exported to JSON,
//! then runs inference on 744 test samples. Validates that every single
//! prediction matches the Python baseline — proving pure math portability.
//!
//! Follows the `hotSpring` pattern: hardcoded expected values from
//! `decision_tree_test_data.json`, explicit pass/fail, exit code 0/1.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | sklearn export (pfas_tree_export.py) |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 scripts/pfas_tree_export.py + decision_tree_test_data.json |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/pfas_tree_export.py` |
//! | Data | 744 test samples, decision_tree_exported.json |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::fs;
use wetspring_barracuda::bio::decision_tree::DecisionTree;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[allow(clippy::cast_possible_truncation)]
fn load_tree() -> DecisionTree {
    let tree_path =
        validation::data_dir("WETSPRING_PFAS_ML_DIR", "experiments/results/008_pfas_ml")
            .join("decision_tree_exported.json");
    let tree_json =
        fs::read_to_string(&tree_path).expect("cannot read decision_tree_exported.json");

    let tree_data: serde_json::Value =
        serde_json::from_str(&tree_json).expect("cannot parse tree JSON");

    let nodes = tree_data["nodes"].as_array().expect("nodes array");
    let n_features = tree_data["n_features"].as_u64().unwrap_or(28) as usize;

    let mut features_arr = Vec::new();
    let mut thresholds_arr = Vec::new();
    let mut left_arr = Vec::new();
    let mut right_arr = Vec::new();
    let mut predictions_arr = Vec::new();

    for node in nodes {
        features_arr.push(node["feature"].as_i64().unwrap_or(-2) as i32);
        thresholds_arr.push(node["threshold"].as_f64().unwrap_or(-2.0));
        left_arr.push(node["left_child"].as_i64().unwrap_or(-1) as i32);
        right_arr.push(node["right_child"].as_i64().unwrap_or(-1) as i32);
        predictions_arr.push(
            node.get("prediction")
                .and_then(serde_json::Value::as_u64)
                .map(|p| p as usize),
        );
    }

    DecisionTree::from_arrays(
        &features_arr,
        &thresholds_arr,
        &left_arr,
        &right_arr,
        &predictions_arr,
        n_features,
    )
    .expect("invalid tree structure")
}

struct TestData {
    samples: Vec<(Vec<f64>, usize, usize)>,
    expected_accuracy: f64,
    expected_f1: f64,
}

#[allow(clippy::cast_possible_truncation)]
fn load_test_data() -> TestData {
    let test_path =
        validation::data_dir("WETSPRING_PFAS_ML_DIR", "experiments/results/008_pfas_ml")
            .join("decision_tree_test_data.json");
    let test_json = fs::read_to_string(&test_path).expect("cannot read test data");

    let test_data: serde_json::Value =
        serde_json::from_str(&test_json).expect("cannot parse test JSON");

    let raw_samples = test_data["samples"].as_array().expect("samples array");
    let expected_accuracy = test_data["test_accuracy"].as_f64().unwrap_or(0.0);
    let expected_f1 = test_data["test_f1"].as_f64().unwrap_or(0.0);

    let samples = raw_samples
        .iter()
        .map(|s| {
            let feats: Vec<f64> = s["features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            let python_pred = s["predicted_label"].as_u64().unwrap_or(0) as usize;
            let true_label = s["true_label"].as_u64().unwrap_or(0) as usize;
            (feats, python_pred, true_label)
        })
        .collect();

    TestData {
        samples,
        expected_accuracy,
        expected_f1,
    }
}

fn main() {
    let mut v = Validator::new("Exp008 Phase 3: PFAS Decision Tree Inference");

    v.section("── Loading Exported Tree ──");
    let tree = load_tree();
    v.check_count("Tree loaded: node count", tree.n_nodes(), 65);
    println!(
        "  Tree: {} nodes, {} leaves, depth {}, {} features",
        tree.n_nodes(),
        tree.n_leaves(),
        tree.depth(),
        tree.n_features()
    );

    v.section("── Loading Test Data ──");
    let data = load_test_data();
    println!("  Test samples: {}", data.samples.len());
    println!("  Python accuracy: {:.4}", data.expected_accuracy);
    println!("  Python F1: {:.4}", data.expected_f1);

    v.section("── Inference Parity ──");

    let mut match_count = 0usize;
    let mut true_pos = 0usize;
    let mut false_pos = 0usize;
    let mut false_neg = 0usize;
    let mut true_neg = 0usize;

    for (feats, python_pred, true_label) in &data.samples {
        let rust_pred = tree.predict(feats);
        if rust_pred == *python_pred {
            match_count += 1;
        }
        match (rust_pred, *true_label) {
            (1, 1) => true_pos += 1,
            (1, 0) => false_pos += 1,
            (0, 1) => false_neg += 1,
            _ => true_neg += 1,
        }
    }

    let n_samples = data.samples.len();
    #[allow(clippy::cast_precision_loss)]
    let parity_rate = match_count as f64 / n_samples as f64;
    #[allow(clippy::cast_precision_loss)]
    let rust_accuracy = (true_pos + true_neg) as f64 / n_samples as f64;
    #[allow(clippy::cast_precision_loss)]
    let precision = if true_pos + false_pos > 0 {
        true_pos as f64 / (true_pos + false_pos) as f64
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let recall = if true_pos + false_neg > 0 {
        true_pos as f64 / (true_pos + false_neg) as f64
    } else {
        0.0
    };
    let rust_f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    v.check(
        "Python↔Rust prediction parity (100%)",
        parity_rate,
        1.0,
        0.0,
    );
    v.check_count("All predictions match", match_count, n_samples);

    v.section("── Accuracy Metrics ──");

    v.check(
        "Rust accuracy matches Python",
        rust_accuracy,
        data.expected_accuracy,
        1e-6,
    );
    v.check(
        "Rust F1 matches Python",
        rust_f1,
        data.expected_f1,
        tolerances::ML_F1_SCORE,
    );
    let acc_ok = rust_accuracy >= 0.80;
    v.check(
        "Accuracy above 0.80 threshold",
        f64::from(u8::from(acc_ok)),
        1.0,
        0.0,
    );
    let f1_ok = rust_f1 >= 0.80;
    v.check(
        "F1 above 0.80 threshold",
        f64::from(u8::from(f1_ok)),
        1.0,
        0.0,
    );

    v.finish();
}
