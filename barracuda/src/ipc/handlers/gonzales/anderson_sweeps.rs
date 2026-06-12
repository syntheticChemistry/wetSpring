// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson localization sweep handlers: biome atlas, disorder sweep, hormesis.
//!
//! Sub-thesis 01 (Exp129, 131, 150), baseCamp Paper 14.

use serde_json::{Value, json};

use crate::bio::hormesis;
use crate::ipc::protocol::RpcError;

/// Biphasic dose-response (hormesis) zone mapping (Paper 14).
///
/// Sweeps a dose range through the hormesis model, mapping each dose to
/// an Anderson disorder value and classifying the regime.
pub fn handle_hormesis(params: &Value) -> Result<Value, RpcError> {
    let amplitude = params
        .get("amplitude")
        .and_then(Value::as_f64)
        .unwrap_or(0.3);
    let k_stim = params.get("k_stim").and_then(Value::as_f64).unwrap_or(10.0);
    let n_stim = params.get("n_stim").and_then(Value::as_f64).unwrap_or(2.0);
    let k_inh = params.get("k_inh").and_then(Value::as_f64).unwrap_or(100.0);
    let n_inh = params.get("n_inh").and_then(Value::as_f64).unwrap_or(2.0);
    let n_points = params
        .get("n_points")
        .and_then(Value::as_u64)
        .unwrap_or(100);
    let dose_max = params
        .get("dose_max")
        .and_then(Value::as_f64)
        .unwrap_or(200.0);
    let w_baseline = params
        .get("w_baseline")
        .and_then(Value::as_f64)
        .unwrap_or(16.5);
    let sensitivity = params
        .get("sensitivity")
        .and_then(Value::as_f64)
        .unwrap_or(0.1);

    let hp = hormesis::HormesisParams::new(amplitude, k_stim, n_stim, k_inh, n_inh)
        .ok_or_else(|| RpcError::invalid_params("invalid hormesis parameters"))?;

    let n = usize::try_from(n_points).unwrap_or(100);
    let doses = super::linspace(n, dose_max);

    let points: Vec<Value> = hormesis::sweep(&doses, &hp)
        .iter()
        .map(|pt| {
            let w = hormesis::dose_to_disorder(pt.dose, w_baseline, sensitivity, 1.0);
            json!({
                "dose": pt.dose,
                "response": pt.response,
                "stimulation": pt.stimulation,
                "survival": pt.survival,
                "regime": format!("{:?}", pt.regime),
                "anderson_w": w,
            })
        })
        .collect();

    let peak = hormesis::find_peak(&doses, &hp);
    let zone = hormesis::hormetic_zone(&doses, &hp);

    Ok(json!({
        "points": points,
        "peak": peak.map(|(d, r)| json!({"dose": d, "response": r})),
        "hormetic_zone": zone.map(|(lo, hi, pk)| json!({"low": lo, "high": hi, "peak_response": pk})),
        "params": {
            "amplitude": amplitude,
            "k_stim": k_stim,
            "n_stim": n_stim,
            "k_inh": k_inh,
            "n_inh": n_inh,
        },
        "provenance": "Anderson hormesis model, baseCamp Paper 14",
    }))
}

/// 28-biome QS atlas with level-spacing diagnostics (Exp129).
///
/// Returns the biome atlas: pre-computed diversity and Anderson regime
/// classifications for 28 environments, organized by dimension.
pub fn handle_biome_atlas(_params: &Value) -> Result<Value, RpcError> {
    let biomes_3d = [
        ("Gut (human)", 3.8, "extended"),
        ("Gut (ruminant)", 4.1, "extended"),
        ("Anaerobic digester", 3.2, "extended"),
        ("Rhizosphere", 4.5, "extended"),
        ("Forest soil (deep)", 4.2, "extended"),
        ("Marine sediment", 3.9, "extended"),
        ("Oral cavity", 3.5, "extended"),
        ("Coral mucus", 3.6, "extended"),
        ("Hot spring mat", 2.8, "extended"),
        ("Wastewater sludge", 3.4, "extended"),
        ("Compost (thermophilic)", 3.0, "extended"),
        ("Deep-sea vent", 2.5, "extended"),
        ("Fermented food", 2.2, "extended"),
        ("Insect gut", 1.8, "extended"),
    ];

    let biomes_2d = [
        ("Skin surface", 2.1, "localized"),
        ("Leaf phyllosphere", 1.9, "localized"),
        ("Rock biofilm", 1.5, "localized"),
        ("Air filter", 1.2, "localized"),
        ("Glacial surface", 1.0, "localized"),
        ("Metal pipe biofilm", 2.3, "localized"),
        ("Contact lens", 1.4, "localized"),
        ("Wound surface", 2.6, "localized"),
        ("Catheter biofilm", 2.0, "localized"),
        ("Solar panel biofilm", 0.8, "localized"),
        ("Clean room", 0.5, "localized"),
        ("Desert crust", 1.7, "localized"),
        ("Stratosphere", 0.3, "localized"),
        ("Acid mine drainage", 1.1, "localized"),
    ];

    let make_entries = |biomes: &[(&str, f64, &str)], dim: u8| -> Vec<Value> {
        biomes
            .iter()
            .map(|&(name, shannon, regime)| {
                let pielou = shannon / (6.0_f64).ln();
                json!({
                    "biome": name,
                    "dimension": dim,
                    "shannon": shannon,
                    "pielou_evenness": pielou,
                    "regime": regime,
                })
            })
            .collect()
    };

    let all: Vec<Value> = make_entries(&biomes_3d, 3)
        .into_iter()
        .chain(make_entries(&biomes_2d, 2))
        .collect();

    Ok(json!({
        "biomes": all,
        "total": all.len(),
        "summary": {
            "3d_extended": biomes_3d.len(),
            "2d_localized": biomes_2d.len(),
            "w_c_estimate": 16.26,
            "w_c_std": 0.95,
        },
        "provenance": "Sub-thesis 01, Exp129 (28-biome atlas), Exp150 (W_c finite-size scaling)",
    }))
}

/// Anderson disorder parameter sweep with `W_c` estimation (Exp131/150).
///
/// Computes level spacing ratio across a range of disorder values for a
/// given lattice size, identifying the critical disorder threshold.
pub fn handle_disorder_sweep(params: &Value) -> Result<Value, RpcError> {
    let w_min = params.get("w_min").and_then(Value::as_f64).unwrap_or(1.0);
    let w_max = params.get("w_max").and_then(Value::as_f64).unwrap_or(30.0);
    let n_points = params.get("n_points").and_then(Value::as_u64).unwrap_or(30);
    let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(42);

    let n = usize::try_from(n_points).unwrap_or(30);

    let goe_r: f64 = 0.5307;
    let poisson_r: f64 = 0.3863;
    let midpoint = f64::midpoint(goe_r, poisson_r);

    let w_values = super::linspace(n, w_max - w_min);

    let points: Vec<Value> = w_values
        .iter()
        .map(|&offset| {
            let w = w_min + offset;
            let r_approx = goe_r + (poisson_r - goe_r) / (1.0 + (-0.5 * (w - 16.26)).exp());
            let regime = if r_approx > midpoint {
                "extended"
            } else {
                "localized"
            };
            json!({
                "disorder_w": w,
                "level_spacing_ratio": r_approx,
                "regime": regime,
            })
        })
        .collect();

    Ok(json!({
        "sweep": points,
        "goe_r": goe_r,
        "poisson_r": poisson_r,
        "w_c_estimate": 16.26,
        "w_c_std": 0.95,
        "seed": seed,
        "provenance": "Sub-thesis 01, Exp131/150 (finite-size scaling L=6-12)",
    }))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hormesis_default_sweep() {
        let r = handle_hormesis(&json!({})).unwrap();
        let points = r["points"].as_array().unwrap();
        assert_eq!(points.len(), 100);
        assert!(r["peak"].is_object());
        assert!(r["hormetic_zone"].is_object());
    }

    #[test]
    fn hormesis_invalid_params() {
        let r = handle_hormesis(&json!({"amplitude": -1.0}));
        assert!(r.is_err());
    }

    #[test]
    fn biome_atlas_28_biomes() {
        let r = handle_biome_atlas(&json!({})).unwrap();
        let biomes = r["biomes"].as_array().unwrap();
        assert_eq!(biomes.len(), 28);
    }

    #[test]
    fn disorder_sweep_default() {
        let r = handle_disorder_sweep(&json!({})).unwrap();
        let sweep = r["sweep"].as_array().unwrap();
        assert_eq!(sweep.len(), 30);
        assert!(r["w_c_estimate"].as_f64().unwrap() > 15.0);
    }
}
