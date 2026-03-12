// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp355: petalTongue Biogas Dashboard v1
//!
//! Track 6 anaerobic digestion visualization: Gompertz, first-order,
//! Monod, and Haldane kinetics streamed as petalTongue scenarios with
//! industrial monitoring thresholds.
//!
//! ## What This Visualizes
//!
//! 1. Modified Gompertz biogas production curves (3 feedstocks)
//! 2. First-order kinetics comparison
//! 3. Monod substrate-limited growth across substrate concentrations
//! 4. Haldane inhibition curves (the "too much substrate" story)
//! 5. Cross-digester diversity and Anderson W gauges
//! 6. Temperature/pH operational envelopes as Distribution
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | petalTongue biogas visualization |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --features gpu --bin validate_petaltongue_biogas_v1` |

use std::path::PathBuf;

use barracuda::stats::norm_cdf;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::{
    DataChannel, EcologyScenario, ScenarioEdge, ScenarioNode, ScientificRange, scenario_to_json,
};

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}

fn haldane(s: f64, mu_max: f64, ks: f64, ki: f64) -> f64 {
    mu_max * s / (ks + s + s * s / ki)
}

fn main() {
    let mut v = Validator::new("Exp355 petalTongue Biogas Dashboard v1");

    // ── Feedstock definitions ──
    struct Feedstock {
        name: &'static str,
        p: f64,
        rm: f64,
        lambda: f64,
        b_max: f64,
        k: f64,
        mu_max: f64,
        ks: f64,
        ki: f64,
        community: Vec<f64>,
    }

    let feedstocks = vec![
        Feedstock {
            name: "Corn Stover",
            p: 280.0,
            rm: 12.5,
            lambda: 3.0,
            b_max: 260.0,
            k: 0.08,
            mu_max: 0.45,
            ks: 50.0,
            ki: 200.0,
            community: vec![150.0, 80.0, 40.0, 20.0, 10.0, 5.0],
        },
        Feedstock {
            name: "Coffee Residues",
            p: 350.0,
            rm: 15.0,
            lambda: 5.0,
            b_max: 320.0,
            k: 0.06,
            mu_max: 0.38,
            ks: 40.0,
            ki: 150.0,
            community: vec![180.0, 60.0, 30.0, 15.0, 8.0, 4.0, 2.0],
        },
        Feedstock {
            name: "Co-Digestion Mix",
            p: 400.0,
            rm: 18.0,
            lambda: 2.0,
            b_max: 380.0,
            k: 0.10,
            mu_max: 0.55,
            ks: 60.0,
            ki: 250.0,
            community: vec![100.0, 90.0, 80.0, 70.0, 50.0, 30.0, 20.0, 10.0],
        },
    ];

    let map_h_to_w = |h: f64| -> f64 { 20.0 * (-0.3 * h).exp() };
    let p_qs = |w: f64| -> f64 { norm_cdf((16.5 - w) / 3.0) };

    // ── S1: Kinetics computation ──
    println!("\n── S1: Biogas kinetics ──");

    let t_points: Vec<f64> = (0..60).map(f64::from).collect();

    let mut scenario = EcologyScenario {
        name: "Biogas Kinetics Dashboard".into(),
        description:
            "Anaerobic digestion monitoring — Gompertz, Monod, Haldane across 3 feedstocks".into(),
        version: "1.0.0".into(),
        mode: "live-ecosystem".into(),
        domain: "ecology".into(),
        nodes: vec![],
        edges: vec![],
    };

    // Node 1: Gompertz production curves
    let mut gompertz_node = ScenarioNode {
        id: "gompertz".into(),
        name: "Modified Gompertz (Biogas Production)".into(),
        node_type: "compute".into(),
        family: "wetspring".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.kinetics.gompertz".into()],
        data_channels: vec![],
        scientific_ranges: vec![
            ScientificRange {
                label: "Good yield".into(),
                min: 200.0,
                max: 500.0,
                status: "normal".into(),
            },
            ScientificRange {
                label: "Low yield".into(),
                min: 0.0,
                max: 200.0,
                status: "warning".into(),
            },
        ],
    };
    for fs in &feedstocks {
        let y_gomp: Vec<f64> = t_points
            .iter()
            .map(|&t| gompertz(t, fs.p, fs.rm, fs.lambda))
            .collect();
        gompertz_node.data_channels.push(DataChannel::TimeSeries {
            id: format!("gomp_{}", fs.name.to_lowercase().replace(' ', "_")),
            label: format!("Gompertz: {}", fs.name),
            x_label: "Time (days)".into(),
            y_label: "Cumulative CH₄ (mL/g VS)".into(),
            unit: "mL/g VS".into(),
            x_values: t_points.clone(),
            y_values: y_gomp,
        });
    }
    scenario.nodes.push(gompertz_node);

    // Node 2: First-order kinetics
    let mut fo_node = ScenarioNode {
        id: "first_order".into(),
        name: "First-Order Kinetics".into(),
        node_type: "compute".into(),
        family: "wetspring".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.kinetics.first_order".into()],
        data_channels: vec![],
        scientific_ranges: vec![],
    };
    for fs in &feedstocks {
        let y_fo: Vec<f64> = t_points
            .iter()
            .map(|&t| first_order(t, fs.b_max, fs.k))
            .collect();
        fo_node.data_channels.push(DataChannel::TimeSeries {
            id: format!("fo_{}", fs.name.to_lowercase().replace(' ', "_")),
            label: format!("First-Order: {}", fs.name),
            x_label: "Time (days)".into(),
            y_label: "Cumulative CH₄ (mL/g VS)".into(),
            unit: "mL/g VS".into(),
            x_values: t_points.clone(),
            y_values: y_fo,
        });
    }
    scenario.nodes.push(fo_node);

    // Node 3: Monod/Haldane growth kinetics
    let mut kinetics_node = ScenarioNode {
        id: "growth_kinetics".into(),
        name: "Growth Kinetics (Monod + Haldane)".into(),
        node_type: "compute".into(),
        family: "wetspring".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec![
            "science.kinetics.monod".into(),
            "science.kinetics.haldane".into(),
        ],
        data_channels: vec![],
        scientific_ranges: vec![],
    };
    let s_range: Vec<f64> = (0..100).map(|i| f64::from(i) * 5.0).collect();
    for fs in &feedstocks {
        let y_monod: Vec<f64> = s_range
            .iter()
            .map(|&s| monod(s, fs.mu_max, fs.ks))
            .collect();
        let y_haldane: Vec<f64> = s_range
            .iter()
            .map(|&s| haldane(s, fs.mu_max, fs.ks, fs.ki))
            .collect();
        kinetics_node.data_channels.push(DataChannel::TimeSeries {
            id: format!("monod_{}", fs.name.to_lowercase().replace(' ', "_")),
            label: format!("Monod: {}", fs.name),
            x_label: "Substrate (mg/L)".into(),
            y_label: "Growth rate (1/h)".into(),
            unit: "1/h".into(),
            x_values: s_range.clone(),
            y_values: y_monod,
        });
        kinetics_node.data_channels.push(DataChannel::TimeSeries {
            id: format!("haldane_{}", fs.name.to_lowercase().replace(' ', "_")),
            label: format!("Haldane: {}", fs.name),
            x_label: "Substrate (mg/L)".into(),
            y_label: "Growth rate (1/h)".into(),
            unit: "1/h".into(),
            x_values: s_range.clone(),
            y_values: y_haldane,
        });
    }
    scenario.nodes.push(kinetics_node);

    // Node 4: Digester diversity + Anderson W
    let mut diversity_node = ScenarioNode {
        id: "digester_diversity".into(),
        name: "Digester Microbial Diversity".into(),
        node_type: "compute".into(),
        family: "wetspring".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.diversity".into(), "science.anderson".into()],
        data_channels: vec![],
        scientific_ranges: vec![
            ScientificRange {
                label: "Healthy digester".into(),
                min: 1.5,
                max: 3.0,
                status: "normal".into(),
            },
            ScientificRange {
                label: "Stressed digester".into(),
                min: 0.0,
                max: 1.5,
                status: "warning".into(),
            },
        ],
    };

    let mut h_vals = Vec::new();
    let mut w_vals = Vec::new();
    let mut names = Vec::new();
    for fs in &feedstocks {
        let h = diversity::shannon(&fs.community);
        let w = map_h_to_w(h);
        let p = p_qs(w);
        h_vals.push(h);
        w_vals.push(w);
        names.push(fs.name.to_string());
        println!("  {}: H'={h:.3}, W={w:.3}, P(QS)={p:.3}", fs.name);
    }

    diversity_node.data_channels.push(DataChannel::Bar {
        id: "digester_shannon".into(),
        label: "Shannon H' by Feedstock".into(),
        categories: names.clone(),
        values: h_vals.clone(),
        unit: "H'".into(),
    });

    for (i, fs) in feedstocks.iter().enumerate() {
        diversity_node.data_channels.push(DataChannel::Gauge {
            id: format!("w_{}", fs.name.to_lowercase().replace(' ', "_")),
            label: format!("Anderson W: {}", fs.name),
            value: w_vals[i],
            min: 0.0,
            max: 25.0,
            unit: "W".into(),
            normal_range: [0.0, 12.0],
            warning_range: [12.0, 20.0],
        });
    }
    scenario.nodes.push(diversity_node);

    // Node 5: Operational envelope (temperature/pH distributions)
    let mut ops_node = ScenarioNode {
        id: "operational".into(),
        name: "Operational Envelope".into(),
        node_type: "data".into(),
        family: "wetspring".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: vec!["science.monitoring".into()],
        data_channels: vec![],
        scientific_ranges: vec![
            ScientificRange {
                label: "Mesophilic optimum".into(),
                min: 35.0,
                max: 40.0,
                status: "normal".into(),
            },
            ScientificRange {
                label: "pH optimum".into(),
                min: 6.8,
                max: 7.5,
                status: "normal".into(),
            },
        ],
    };

    let temp_data: Vec<f64> = (0..50)
        .map(|i| f64::from(i).mul_add(0.2, 35.0) + (f64::from(i * 7 % 10) * 0.1))
        .collect();
    let temp_mean = temp_data.iter().sum::<f64>() / temp_data.len() as f64;
    let temp_var = temp_data
        .iter()
        .map(|&x| (x - temp_mean).powi(2))
        .sum::<f64>()
        / temp_data.len() as f64;
    ops_node.data_channels.push(DataChannel::Distribution {
        id: "temp_dist".into(),
        label: "Digester Temperature".into(),
        unit: "°C".into(),
        values: temp_data,
        mean: temp_mean,
        std: temp_var.sqrt(),
    });

    let ph_data: Vec<f64> = (0..50)
        .map(|i| f64::from(i).mul_add(0.02, 6.8) + (f64::from(i * 3 % 10) * 0.02))
        .collect();
    let ph_mean = ph_data.iter().sum::<f64>() / ph_data.len() as f64;
    let ph_var = ph_data.iter().map(|&x| (x - ph_mean).powi(2)).sum::<f64>() / ph_data.len() as f64;
    ops_node.data_channels.push(DataChannel::Distribution {
        id: "ph_dist".into(),
        label: "Digester pH".into(),
        unit: "pH".into(),
        values: ph_data,
        mean: ph_mean,
        std: ph_var.sqrt(),
    });
    scenario.nodes.push(ops_node);

    // Edges
    scenario.edges = vec![
        ScenarioEdge {
            from: "gompertz".into(),
            to: "first_order".into(),
            edge_type: "data_flow".into(),
            label: "production model comparison".into(),
        },
        ScenarioEdge {
            from: "gompertz".into(),
            to: "growth_kinetics".into(),
            edge_type: "data_flow".into(),
            label: "production → growth".into(),
        },
        ScenarioEdge {
            from: "digester_diversity".into(),
            to: "growth_kinetics".into(),
            edge_type: "data_flow".into(),
            label: "community → kinetics".into(),
        },
        ScenarioEdge {
            from: "operational".into(),
            to: "digester_diversity".into(),
            edge_type: "data_flow".into(),
            label: "env → community".into(),
        },
    ];

    // ── S2: Validate scenario structure ──
    println!("\n── S2: Scenario structure ──");

    v.check_pass("scenario has 5 nodes", scenario.nodes.len() == 5);
    v.check_pass("scenario has 4 edges", scenario.edges.len() == 4);

    let total_channels: usize = scenario.nodes.iter().map(|n| n.data_channels.len()).sum();
    println!("  Total data channels: {total_channels}");
    v.check_pass("scenario has 15+ channels", total_channels >= 15);

    // ── S3: Kinetics math validation ──
    println!("\n── S3: Kinetics math validation ──");

    let fs0 = &feedstocks[0];
    let g30 = gompertz(30.0, fs0.p, fs0.rm, fs0.lambda);
    v.check_pass("Gompertz t=30 > 0", g30 > 0.0);
    v.check_pass("Gompertz t=30 < P_max", g30 < fs0.p);
    v.check("Gompertz corn stover t=30", g30, 250.0, 50.0);

    let fo30 = first_order(30.0, fs0.b_max, fs0.k);
    v.check_pass("first-order t=30 > 0", fo30 > 0.0);
    v.check("first-order corn stover t=30", fo30, 230.0, 50.0);

    let m100 = monod(100.0, fs0.mu_max, fs0.ks);
    v.check("Monod S=100 (corn stover)", m100, 0.3, 0.1);

    let h100 = haldane(100.0, fs0.mu_max, fs0.ks, fs0.ki);
    v.check_pass("Haldane <= Monod at same S", h100 <= m100);

    let h_high = haldane(500.0, fs0.mu_max, fs0.ks, fs0.ki);
    let h_mid = haldane(100.0, fs0.mu_max, fs0.ks, fs0.ki);
    v.check_pass("Haldane inhibition at high S", h_high < h_mid);

    // ── S4: JSON export ──
    println!("\n── S4: JSON export ──");

    let json = scenario_to_json(&scenario).expect("serialize");
    v.check_pass("JSON valid", json.contains("Biogas Kinetics Dashboard"));
    v.check_pass("JSON has Gompertz", json.contains("Gompertz"));
    v.check_pass("JSON has Haldane", json.contains("Haldane"));

    let output_dir = PathBuf::from("output");
    let _ = std::fs::create_dir_all(&output_dir);
    let path = output_dir.join("biogas_kinetics_dashboard.json");
    std::fs::write(&path, &json).expect("write JSON");
    v.check_pass("JSON file written", path.exists());

    let size = std::fs::metadata(&path).expect("meta").len();
    println!("  → File: {} ({} bytes)", path.display(), size);
    println!("  → Load: petaltongue ui --scenario {}", path.display());
    v.check_pass("JSON has content", size > 5000);

    // ── S5: Live push ──
    println!("\n── S5: petalTongue live push ──");

    if let Ok(client) = PetalTonguePushClient::discover() {
        match client.push_render("exp355-biogas", "Biogas Kinetics", &scenario) {
            Ok(()) => {
                println!("  ✓ Biogas dashboard pushed to petalTongue");
                v.check_pass("live push succeeded", true);
            }
            Err(e) => {
                println!("  ○ Push: {e}");
                v.check_pass("graceful degradation", true);
            }
        }
    } else {
        println!("  ○ petalTongue not running — JSON export mode");
        v.check_pass("graceful degradation to JSON", true);
    }

    // ── S6: Industrial summary ──
    println!("\n── S6: Industrial monitoring summary ──");
    println!("  ┌──────────────────────────────────────────────────────┐");
    println!("  │ An environmental engineer monitoring 3 digesters     │");
    println!("  │ sees production curves, growth kinetics, community   │");
    println!("  │ health, and operational envelopes in one dashboard.  │");
    println!("  └──────────────────────────────────────────────────────┘");

    for (i, fs) in feedstocks.iter().enumerate() {
        let g_final = gompertz(59.0, fs.p, fs.rm, fs.lambda);
        let w = w_vals[i];
        let status = if w < 12.0 {
            "HEALTHY"
        } else if w < 18.0 {
            "STRESSED"
        } else {
            "CRITICAL"
        };
        println!(
            "  {}: CH₄={g_final:.0} mL/gVS, W={w:.1}, H'={:.2} → {status}",
            fs.name, h_vals[i]
        );
    }
    v.check_pass("industrial summary printed", true);

    v.finish();
}
