// SPDX-License-Identifier: AGPL-3.0-or-later
//! petalTongue scenario export for Anderson QS model comparison.

use std::path::PathBuf;

use crate::validation::OrExit;
use crate::validation::Validator;
use crate::visualization::{
    DataChannel, EcologyScenario, ScenarioEdge, ScenarioNode, ScientificRange, scenario_to_json,
};

pub(super) struct ModelData<'a> {
    pub env_names: Vec<String>,
    pub known: Vec<f64>,
    pub p1_vals: Vec<f64>,
    pub p2_vals: Vec<f64>,
    pub p3_vals: Vec<f64>,
    pub o2_levels: Vec<f64>,
    pub corr_h1: f64,
    pub corr_h2: f64,
    pub corr_h3: f64,
    pub mae_h3: f64,
    pub h3_anaerobic: f64,
    pub h3_aerobic: f64,
    pub microaerobic_envs: &'a [usize],
}

fn mean_p(indices: &[usize], vals: &[f64]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f64 = indices.iter().map(|&i| vals[i]).sum();
    sum / indices.len() as f64
}

pub(super) fn export_scenario(v: &mut Validator, data: &ModelData<'_>) {
    println!("\n── S7: petalTongue scenario export ──");

    let mut scenario = EcologyScenario {
        name: "Anderson QS: Model Comparison Across Environments".into(),
        description: "Three W parameterizations tested against known QS biology in 10 environments"
            .into(),
        version: "1.0.0".into(),
        mode: "live-ecosystem".into(),
        domain: "ecology".into(),
        nodes: vec![],
        edges: vec![],
    };

    let mut compare_node = ScenarioNode {
        id: "model_comparison".into(),
        name: "Model vs Biology".into(),
        node_type: "compute".into(),
        family: crate::PRIMAL_NAME.into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.anderson".into(), "science.validation".into()],
        data_channels: vec![],
        scientific_ranges: vec![],
    };

    compare_node.data_channels.push(DataChannel::Bar {
        id: "known_qs".into(),
        label: "Known QS Score (Literature)".into(),
        categories: data.env_names.clone(),
        values: data.known.clone(),
        unit: "QS score".into(),
    });
    compare_node.data_channels.push(DataChannel::Bar {
        id: "p_h1".into(),
        label: "H1: P(QS) Inverse Diversity".into(),
        categories: data.env_names.clone(),
        values: data.p1_vals.clone(),
        unit: "probability".into(),
    });
    compare_node.data_channels.push(DataChannel::Bar {
        id: "p_h2".into(),
        label: "H2: P(QS) Signal Dilution".into(),
        categories: data.env_names.clone(),
        values: data.p2_vals.clone(),
        unit: "probability".into(),
    });
    compare_node.data_channels.push(DataChannel::Bar {
        id: "p_h3".into(),
        label: "H3: P(QS) O₂-Modulated".into(),
        categories: data.env_names.clone(),
        values: data.p3_vals.clone(),
        unit: "probability".into(),
    });

    compare_node.data_channels.push(DataChannel::Scatter {
        id: "h3_vs_known".into(),
        label: "H3 P(QS) vs Known QS Score".into(),
        x: data.known.clone(),
        y: data.p3_vals.clone(),
        point_labels: data.env_names.clone(),
        x_label: "Known QS Score".into(),
        y_label: "H3 P(QS)".into(),
        unit: "score".into(),
    });
    scenario.nodes.push(compare_node);

    let mut o2_node = ScenarioNode {
        id: "o2_regimes".into(),
        name: "Oxygen Regime Analysis".into(),
        node_type: "compute".into(),
        family: crate::PRIMAL_NAME.into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.anderson.oxygen".into()],
        data_channels: vec![],
        scientific_ranges: vec![
            ScientificRange {
                label: "Anaerobic QS zone".into(),
                min: 0.6,
                max: 1.0,
                status: "normal".into(),
            },
            ScientificRange {
                label: "Aerobic dilution zone".into(),
                min: 0.0,
                max: 0.4,
                status: "warning".into(),
            },
        ],
    };
    o2_node.data_channels.push(DataChannel::Bar {
        id: "regime_means".into(),
        label: "Mean P(QS) by O₂ Regime (H3)".into(),
        categories: vec!["Anaerobic".into(), "Microaerobic".into(), "Aerobic".into()],
        values: vec![
            data.h3_anaerobic,
            mean_p(data.microaerobic_envs, &data.p3_vals),
            data.h3_aerobic,
        ],
        unit: "P(QS)".into(),
    });
    o2_node.data_channels.push(DataChannel::Scatter {
        id: "o2_vs_pqs".into(),
        label: "O₂ Level vs P(QS) [H3]".into(),
        x: data.o2_levels.clone(),
        y: data.p3_vals.clone(),
        point_labels: data.env_names.clone(),
        x_label: "O₂ (fraction)".into(),
        y_label: "P(QS) H3".into(),
        unit: "probability".into(),
    });
    scenario.nodes.push(o2_node);

    let mut corr_node = ScenarioNode {
        id: "correlations".into(),
        name: "Model Correlation (Pearson r)".into(),
        node_type: "compute".into(),
        family: crate::PRIMAL_NAME.into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.validation".into()],
        data_channels: vec![],
        scientific_ranges: vec![],
    };
    corr_node.data_channels.push(DataChannel::Gauge {
        id: "corr_h1".into(),
        label: "H1 Correlation (inverse)".into(),
        value: data.corr_h1,
        min: -1.0,
        max: 1.0,
        unit: "r".into(),
        normal_range: [0.7, 1.0],
        warning_range: [-1.0, 0.7],
    });
    corr_node.data_channels.push(DataChannel::Gauge {
        id: "corr_h2".into(),
        label: "H2 Correlation (dilution)".into(),
        value: data.corr_h2,
        min: -1.0,
        max: 1.0,
        unit: "r".into(),
        normal_range: [0.7, 1.0],
        warning_range: [-1.0, 0.7],
    });
    corr_node.data_channels.push(DataChannel::Gauge {
        id: "corr_h3".into(),
        label: "H3 Correlation (O₂-modulated)".into(),
        value: data.corr_h3,
        min: -1.0,
        max: 1.0,
        unit: "r".into(),
        normal_range: [0.7, 1.0],
        warning_range: [-1.0, 0.7],
    });
    corr_node.data_channels.push(DataChannel::Gauge {
        id: "mae_h3".into(),
        label: "H3 Mean Absolute Error".into(),
        value: data.mae_h3,
        min: 0.0,
        max: 0.5,
        unit: "MAE".into(),
        normal_range: [0.0, 0.15],
        warning_range: [0.15, 0.3],
    });
    scenario.nodes.push(corr_node);

    scenario.edges = vec![
        ScenarioEdge {
            from: "model_comparison".into(),
            to: "o2_regimes".into(),
            edge_type: "data_flow".into(),
            label: "model → O₂ analysis".into(),
        },
        ScenarioEdge {
            from: "model_comparison".into(),
            to: "correlations".into(),
            edge_type: "validation".into(),
            label: "predictions → correlation".into(),
        },
    ];

    let json = scenario_to_json(&scenario).or_exit("serialize");
    let output_dir = PathBuf::from("output");
    let _ = std::fs::create_dir_all(&output_dir);
    let path = output_dir.join("anderson_qs_model_comparison.json");
    std::fs::write(&path, &json).or_exit("write JSON");
    v.check_pass("scenario JSON written", path.exists());
    let size = std::fs::metadata(&path).or_exit("meta").len();
    println!("  → File: {} ({} bytes)", path.display(), size);
    println!("  → Load: petaltongue ui --scenario {}", path.display());
    v.check_pass("scenario has 3 nodes", scenario.nodes.len() == 3);
}
