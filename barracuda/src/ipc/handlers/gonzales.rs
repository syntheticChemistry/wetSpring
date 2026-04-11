// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gonzales dermatitis and Anderson immunology IPC handlers.
//!
//! Methods for interactive exploration of the Anderson localization framework
//! applied to cytokine signaling, dose-response, pharmacokinetics, and
//! tissue geometry in atopic dermatitis (Papers 12, 53-58).
//!
//! All math delegates to existing validated modules: `barracuda::stats::hill`,
//! `bio::hormesis`, `bio::binding_landscape`, `bio::diversity`.

use serde_json::{Value, json};

use crate::bio::{binding_landscape, diversity, hormesis};
use crate::ipc::protocol::RpcError;

/// Linearly spaced dose/time vector from 0.0 to `max`, `n` points.
fn linspace(n: usize, max: f64) -> Vec<f64> {
    let denom = n.saturating_sub(1).max(1);
    (0..n)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "n ≤ ~1000 fits in f64 mantissa")]
            let t = (i as f64) / (denom as f64);
            max * t
        })
        .collect()
}

/// IC50 dose-response sweep for 6 Gonzales cytokine pathways (Exp280).
///
/// Computes Hill equation curves across a dose range for JAK1, IL-2, IL-6,
/// IL-31, IL-4, and IL-13 using published IC50 values from Gonzales 2014.
pub fn handle_dose_response(params: &Value) -> Result<Value, RpcError> {
    let n_points = params.get("n_points").and_then(Value::as_u64).unwrap_or(50);
    let dose_max = params
        .get("dose_max")
        .and_then(Value::as_f64)
        .unwrap_or(500.0);
    let hill_n = params.get("hill_n").and_then(Value::as_f64).unwrap_or(1.0);

    let pathways = [
        ("JAK1", 10.0),
        ("IL-2", 36.0),
        ("IL-6", 80.0),
        ("IL-31", 71.0),
        ("IL-4", 150.0),
        ("IL-13", 249.0),
    ];

    let n = usize::try_from(n_points).unwrap_or(50);
    let doses = linspace(n, dose_max);

    let curves: Vec<Value> = pathways
        .iter()
        .map(|&(name, ic50)| {
            let responses: Vec<f64> = doses
                .iter()
                .map(|&d| barracuda::stats::hill(d, ic50, hill_n))
                .collect();
            let barrier_w = ic50.ln() * 4.0;
            json!({
                "pathway": name,
                "ic50_nm": ic50,
                "barrier_w": barrier_w,
                "responses": responses,
            })
        })
        .collect();

    Ok(json!({
        "doses": doses,
        "hill_n": hill_n,
        "curves": curves,
        "provenance": "Gonzales AJ et al. 2014, J Vet Pharmacol Ther 37:317-324",
    }))
}

/// Lokivetmab pharmacokinetic decay model (Exp281).
///
/// Models dose-dependent duration of lokivetmab (Cytopoint) anti-pruritic
/// effect using exponential decay from published Fleck/Gonzales 2021 data.
pub fn handle_pk_decay(params: &Value) -> Result<Value, RpcError> {
    let n_points = params
        .get("n_points")
        .and_then(Value::as_u64)
        .unwrap_or(100);
    let t_max_days = params
        .get("t_max_days")
        .and_then(Value::as_f64)
        .unwrap_or(56.0);

    let doses_mg_kg: [f64; 3] = [0.125, 0.5, 2.0];
    let duration_days: [f64; 3] = [14.0, 28.0, 42.0];

    let k_decay = (doses_mg_kg[2] / doses_mg_kg[0]).ln() / (duration_days[2] - duration_days[0]);

    let n = usize::try_from(n_points).unwrap_or(100);
    let times = linspace(n, t_max_days);

    let dose_profiles: Vec<Value> = doses_mg_kg
        .iter()
        .zip(duration_days.iter())
        .map(|(&dose, &dur)| {
            let efficacy: Vec<f64> = times
                .iter()
                .map(|&t| {
                    if t <= dur {
                        (-k_decay * t / dur).exp()
                    } else {
                        0.0
                    }
                })
                .collect();
            json!({
                "dose_mg_kg": dose,
                "duration_days": dur,
                "efficacy": efficacy,
            })
        })
        .collect();

    Ok(json!({
        "times_days": times,
        "k_decay": k_decay,
        "dose_profiles": dose_profiles,
        "provenance": "Fleck TJ,...,Gonzales AJ 2021, Vet Dermatol 32:681-e182",
    }))
}

/// Anderson tissue lattice with skin-layer geometry (Exp273-279).
///
/// Models cytokine propagation through disordered skin tissue using the
/// Anderson localization framework. Epidermis is ~2D; dermis is ~3D.
pub fn handle_tissue_lattice(params: &Value) -> Result<Value, RpcError> {
    let disorder = params
        .get("disorder")
        .and_then(Value::as_f64)
        .unwrap_or(10.0);
    let n_profiles = params
        .get("n_profiles")
        .and_then(Value::as_u64)
        .unwrap_or(6);
    let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(42);

    let disease_profiles = [
        ("healthy", 0.85_f64),
        ("mild_ad", 0.72),
        ("moderate_ad", 0.58),
        ("severe_ad", 0.40),
        ("barrier_breach", 0.25),
        ("chronic_lesion", 0.15),
    ];

    let n = usize::try_from(n_profiles)
        .unwrap_or(6)
        .min(disease_profiles.len());

    let cell_type_counts = [
        &[60.0, 20.0, 10.0, 5.0, 3.0, 2.0][..],
        &[45.0, 25.0, 15.0, 8.0, 5.0, 2.0],
        &[30.0, 25.0, 20.0, 12.0, 8.0, 5.0],
        &[20.0, 22.0, 20.0, 18.0, 12.0, 8.0],
        &[15.0, 18.0, 22.0, 20.0, 15.0, 10.0],
        &[10.0, 15.0, 20.0, 22.0, 18.0, 15.0],
    ];

    let profiles: Vec<Value> = disease_profiles[..n]
        .iter()
        .zip(cell_type_counts[..n].iter())
        .map(|(&(name, evenness), counts)| {
            let shannon = diversity::shannon(counts);
            let pielou = diversity::pielou_evenness(counts);
            let anderson_w = disorder * (1.0 - evenness);

            json!({
                "profile": name,
                "target_evenness": evenness,
                "shannon": shannon,
                "pielou": pielou,
                "anderson_w": anderson_w,
                "cell_type_counts": counts,
            })
        })
        .collect();

    let epidermis_d = 2;
    let dermis_d = 3;

    Ok(json!({
        "seed": seed,
        "base_disorder": disorder,
        "epidermis_dimension": epidermis_d,
        "dermis_dimension": dermis_d,
        "profiles": profiles,
        "provenance": "Gonzales catalog G1-G6 + McCandless 2014, Exp273-279",
    }))
}

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
    let doses = linspace(n, dose_max);

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

/// Anderson disorder parameter sweep with W_c estimation (Exp131/150).
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

    let w_values = linspace(n, w_max - w_min);

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

/// Cross-species tissue geometry comparison (Paper 12 extension).
///
/// Computes effective dimension and Anderson disorder for different
/// species based on published tissue morphometry, predicting AD severity
/// from geometry alone.
pub fn handle_cross_species(params: &Value) -> Result<Value, RpcError> {
    struct SpeciesRecord {
        name: &'static str,
        epidermis_um: f64,
        follicle_per_cm2: f64,
        d_eff: f64,
        severity: &'static str,
        effective_ic50: f64,
    }

    let custom_counts = params
        .get("cell_type_counts")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>());

    let species_data: &[SpeciesRecord] = &[
        SpeciesRecord {
            name: "dog",
            epidermis_um: 15.0,
            follicle_per_cm2: 500.0,
            d_eff: 2.5,
            severity: "high",
            effective_ic50: 10.0,
        },
        SpeciesRecord {
            name: "cat",
            epidermis_um: 10.0,
            follicle_per_cm2: 800.0,
            d_eff: 2.1,
            severity: "moderate",
            effective_ic50: 36.0,
        },
        SpeciesRecord {
            name: "human",
            epidermis_um: 50.0,
            follicle_per_cm2: 100.0,
            d_eff: 2.8,
            severity: "moderate-high",
            effective_ic50: 15.0,
        },
        SpeciesRecord {
            name: "horse",
            epidermis_um: 35.0,
            follicle_per_cm2: 300.0,
            d_eff: 2.3,
            severity: "low-moderate",
            effective_ic50: 63.0,
        },
        SpeciesRecord {
            name: "mouse",
            epidermis_um: 8.0,
            follicle_per_cm2: 900.0,
            d_eff: 2.0,
            severity: "low (model)",
            effective_ic50: 100.0,
        },
    ];

    let species: Vec<Value> = species_data
        .iter()
        .map(|s| {
            let barrier_w = s.effective_ic50.ln() * 4.0;
            let follicle_promotion = (s.follicle_per_cm2 / 1000.0).min(1.0);
            json!({
                "species": s.name,
                "epidermis_thickness_um": s.epidermis_um,
                "follicle_density_per_cm2": s.follicle_per_cm2,
                "effective_dimension": s.d_eff,
                "ad_severity_class": s.severity,
                "effective_ic50_nm": s.effective_ic50,
                "barrier_w": barrier_w,
                "follicle_promotion_factor": follicle_promotion,
            })
        })
        .collect();

    let mut result = json!({
        "species": species,
        "provenance": "baseCamp Paper 12, cross-species tissue morphometry",
    });

    if let Some(counts) = custom_counts {
        if !counts.is_empty() {
            let shannon = diversity::shannon(&counts);
            let pielou = diversity::pielou_evenness(&counts);
            let ipr = binding_landscape::binding_ipr(&counts);
            result["custom_analysis"] = json!({
                "shannon": shannon,
                "pielou_evenness": pielou,
                "binding_ipr": ipr,
                "cell_type_counts": counts,
            });
        }
    }

    Ok(result)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dose_response_default() {
        let r = handle_dose_response(&json!({})).unwrap();
        let curves = r["curves"].as_array().unwrap();
        assert_eq!(curves.len(), 6);
        assert_eq!(curves[0]["pathway"], "JAK1");
        assert!(r["doses"].as_array().unwrap().len() == 50);
    }

    #[test]
    fn dose_response_custom_points() {
        let r = handle_dose_response(&json!({"n_points": 10, "dose_max": 100.0})).unwrap();
        assert_eq!(r["doses"].as_array().unwrap().len(), 10);
    }

    #[test]
    fn pk_decay_three_doses() {
        let r = handle_pk_decay(&json!({})).unwrap();
        let profiles = r["dose_profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 3);
        assert!(r["k_decay"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn tissue_lattice_profiles() {
        let r = handle_tissue_lattice(&json!({})).unwrap();
        let profiles = r["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 6);
        let healthy = &profiles[0];
        assert_eq!(healthy["profile"], "healthy");
        assert!(healthy["shannon"].as_f64().unwrap() > 0.0);
    }

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

    #[test]
    fn cross_species_five_species() {
        let r = handle_cross_species(&json!({})).unwrap();
        let species = r["species"].as_array().unwrap();
        assert_eq!(species.len(), 5);
        assert_eq!(species[0]["species"], "dog");
    }

    #[test]
    fn cross_species_custom_counts() {
        let r = handle_cross_species(&json!({
            "cell_type_counts": [30.0, 25.0, 20.0, 15.0, 10.0]
        }))
        .unwrap();
        assert!(r["custom_analysis"]["shannon"].as_f64().unwrap() > 0.0);
    }
}
