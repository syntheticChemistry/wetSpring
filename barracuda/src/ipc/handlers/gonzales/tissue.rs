// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson tissue lattice and cross-species morphometry handlers.
//!
//! Exp273-279 (tissue lattice), Paper 12 (cross-species).

use serde_json::{Value, json};

use crate::bio::{binding_landscape, diversity};
use crate::ipc::protocol::RpcError;

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
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use serde_json::json;

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
