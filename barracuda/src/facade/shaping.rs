// SPDX-License-Identifier: AGPL-3.0-or-later
//! Response shaping: transforms wetSpring IPC results into petalTongue
//! `DataChannel`-compatible scenario format for explorer.js consumption.
//!
//! The output JSON matches the format produced by `--export-scenarios` in the
//! guideStone binary, so `explorer.js` renders it identically whether the
//! data comes from static files or the live facade.

use serde_json::{Value, json};

/// Wrap a dose-response IPC result as a scenario node with DataChannels.
pub fn shape_dose_response(result: &Value) -> Value {
    let doses = result.get("doses").cloned().unwrap_or(json!([]));
    let hill_n = result.get("hill_n").and_then(Value::as_f64).unwrap_or(1.0);
    let curves = result.get("curves").and_then(Value::as_array);

    let mut channels: Vec<Value> = Vec::new();

    if let Some(curves) = curves {
        for curve in curves {
            let pathway = curve.get("pathway").and_then(Value::as_str).unwrap_or("?");
            let ic50 = curve.get("ic50_nm").and_then(Value::as_f64).unwrap_or(0.0);
            let responses = curve.get("responses").cloned().unwrap_or(json!([]));

            channels.push(json!({
                "channel_type": "timeseries",
                "id": format!("dr_{}", pathway.to_lowercase().replace('-', "")),
                "label": format!("{pathway} (IC50 = {ic50} nM)"),
                "x_label": "Dose (nM)",
                "y_label": "Fractional Inhibition",
                "unit": "fraction",
                "x_values": doses,
                "y_values": responses,
            }));
        }

        let categories: Vec<String> = curves
            .iter()
            .filter_map(|c| c.get("pathway").and_then(Value::as_str).map(String::from))
            .collect();
        let ic50_values: Vec<f64> = curves
            .iter()
            .filter_map(|c| c.get("ic50_nm").and_then(Value::as_f64))
            .collect();

        channels.push(json!({
            "channel_type": "bar",
            "id": "ic50_bar",
            "label": "IC50 Values",
            "categories": categories,
            "values": ic50_values,
            "unit": "nM",
        }));
    }

    json!({
        "id": "gonzales_ic50",
        "label": format!("IC50 Dose-Response (Hill n={hill_n})"),
        "data_channels": channels,
    })
}

/// Wrap a PK decay IPC result as a scenario node.
pub fn shape_pk_decay(result: &Value) -> Value {
    let times = result.get("times_days").cloned().unwrap_or(json!([]));
    let k_decay = result.get("k_decay").and_then(Value::as_f64).unwrap_or(0.0);
    let dose_profiles = result.get("dose_profiles").and_then(Value::as_array);

    let mut channels: Vec<Value> = Vec::new();

    if let Some(profiles) = dose_profiles {
        for p in profiles {
            let dose = p.get("dose_mg_kg").and_then(Value::as_f64).unwrap_or(0.0);
            let efficacy = p.get("efficacy").cloned().unwrap_or(json!([]));
            channels.push(json!({
                "channel_type": "timeseries",
                "id": format!("pk_{}", dose.to_bits()),
                "label": format!("Lokivetmab {dose} mg/kg"),
                "x_label": "Time (days)",
                "y_label": "Efficacy",
                "unit": "fraction",
                "x_values": times,
                "y_values": efficacy,
            }));
        }
    }

    channels.push(json!({
        "channel_type": "gauge",
        "id": "k_decay",
        "label": "Decay Constant",
        "value": k_decay,
        "min": 0.0,
        "max": 0.5,
        "unit": "1/day",
        "normal_range": [0.05, 0.2],
        "warning_range": [0.2, 0.4],
    }));

    json!({
        "id": "gonzales_pk",
        "label": "Lokivetmab PK Decay",
        "data_channels": channels,
    })
}

/// Wrap a tissue lattice IPC result as a scenario node.
pub fn shape_tissue_lattice(result: &Value) -> Value {
    let profiles = result.get("profiles").and_then(Value::as_array);

    let mut channels: Vec<Value> = Vec::new();

    if let Some(profiles) = profiles {
        let names: Vec<String> = profiles
            .iter()
            .filter_map(|p| p.get("profile").and_then(Value::as_str).map(String::from))
            .collect();
        let shannon: Vec<f64> = profiles
            .iter()
            .filter_map(|p| p.get("shannon").and_then(Value::as_f64))
            .collect();
        let pielou: Vec<f64> = profiles
            .iter()
            .filter_map(|p| p.get("pielou").and_then(Value::as_f64))
            .collect();
        let anderson_w: Vec<f64> = profiles
            .iter()
            .filter_map(|p| p.get("anderson_w").and_then(Value::as_f64))
            .collect();

        channels.push(json!({
            "channel_type": "bar",
            "id": "shannon_bar",
            "label": "Shannon Diversity by AD Severity",
            "categories": names,
            "values": shannon,
            "unit": "bits",
        }));
        channels.push(json!({
            "channel_type": "bar",
            "id": "pielou_bar",
            "label": "Pielou Evenness by AD Severity",
            "categories": names,
            "values": pielou,
            "unit": "dimensionless",
        }));
        channels.push(json!({
            "channel_type": "bar",
            "id": "anderson_w_bar",
            "label": "Anderson Disorder W by AD Severity",
            "categories": names,
            "values": anderson_w,
            "unit": "dimensionless",
        }));
    }

    json!({
        "id": "tissue_geometry",
        "label": "Tissue Geometry — AD Severity Profiles",
        "data_channels": channels,
    })
}

/// Wrap a hormesis IPC result as a scenario node.
pub fn shape_hormesis(result: &Value) -> Value {
    let doses = result.get("doses").cloned().unwrap_or(json!([]));
    let responses = result.get("responses").cloned().unwrap_or(json!([]));
    let stim_component = result.get("stimulatory_component").cloned().unwrap_or(json!([]));
    let surv_component = result.get("survival_component").cloned().unwrap_or(json!([]));
    let disorder_w = result.get("disorder_mapping").and_then(|m| m.get("w_values")).cloned().unwrap_or(json!([]));

    let mut channels = vec![
        json!({
            "channel_type": "timeseries",
            "id": "hormesis_total",
            "label": "Hormetic Response (Total)",
            "x_label": "Dose",
            "y_label": "Response",
            "unit": "fraction",
            "x_values": doses,
            "y_values": responses,
        }),
        json!({
            "channel_type": "timeseries",
            "id": "hormesis_stim",
            "label": "Stimulatory Component",
            "x_label": "Dose",
            "y_label": "Stimulation",
            "unit": "fraction",
            "x_values": doses,
            "y_values": stim_component,
        }),
        json!({
            "channel_type": "timeseries",
            "id": "hormesis_surv",
            "label": "Survival Component",
            "x_label": "Dose",
            "y_label": "Survival",
            "unit": "fraction",
            "x_values": doses,
            "y_values": surv_component,
        }),
    ];

    if disorder_w != json!([]) {
        channels.push(json!({
            "channel_type": "timeseries",
            "id": "hormesis_disorder",
            "label": "Dose → Anderson Disorder W",
            "x_label": "Dose",
            "y_label": "Disorder W",
            "unit": "dimensionless",
            "x_values": doses,
            "y_values": disorder_w,
        }));
    }

    json!({
        "id": "hormesis",
        "label": "Hormesis — Biphasic Dose-Response",
        "data_channels": channels,
    })
}

/// Wrap a cross-species IPC result as a scenario node.
pub fn shape_cross_species(result: &Value) -> Value {
    let species_data = result.get("species").and_then(Value::as_array);
    let mut channels: Vec<Value> = Vec::new();

    if let Some(species) = species_data {
        let names: Vec<String> = species
            .iter()
            .filter_map(|s| s.get("name").and_then(Value::as_str).map(String::from))
            .collect();
        let epidermis: Vec<f64> = species
            .iter()
            .filter_map(|s| s.get("epidermis_um").and_then(Value::as_f64))
            .collect();
        let d_eff: Vec<f64> = species
            .iter()
            .filter_map(|s| s.get("d_eff").and_then(Value::as_f64))
            .collect();
        let barrier_w: Vec<f64> = species
            .iter()
            .filter_map(|s| s.get("barrier_w").and_then(Value::as_f64))
            .collect();

        channels.push(json!({
            "channel_type": "bar",
            "id": "morphometry_bar",
            "label": "Epidermis Thickness by Species",
            "categories": names,
            "values": epidermis,
            "unit": "μm",
        }));

        channels.push(json!({
            "channel_type": "scatter",
            "id": "d_eff_vs_w",
            "label": "Effective Dimension vs. Barrier Disorder",
            "x": d_eff,
            "y": barrier_w,
            "point_labels": names,
            "x_label": "d_eff",
            "y_label": "Barrier W",
            "unit": "dimensionless",
        }));
    }

    json!({
        "id": "cross_species",
        "label": "Cross-Species Tissue Comparison",
        "data_channels": channels,
    })
}

/// Build a full scenario envelope from shaped nodes + provenance.
pub fn scenario_envelope(
    name: &str,
    description: &str,
    nodes: Vec<Value>,
    provenance: &Value,
) -> Value {
    json!({
        "name": name,
        "description": description,
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "live",
        "domain": "gonzales_dermatitis",
        "nodes": nodes,
        "provenance": provenance,
    })
}
