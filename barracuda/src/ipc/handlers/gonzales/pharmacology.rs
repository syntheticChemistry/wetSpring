// SPDX-License-Identifier: AGPL-3.0-or-later
//! IC50 dose-response and lokivetmab PK decay handlers.
//!
//! Gonzales AJ et al. 2014 (dose-response), Fleck/Gonzales 2021 (PK decay).

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

/// IC50 dose-response sweep for 6 Gonzales cytokine pathways (Exp280).
///
/// Computes Hill equation curves across a dose range for JAK1, IL-2, IL-6,
/// IL-31, IL-4, and IL-13 using published IC50 values from Gonzales 2014.
/// When `primal-proof` is active, attempts to forward to a live barraCuda
/// primal via IPC before falling back to in-process compute.
pub fn handle_dose_response(params: &Value) -> Result<Value, RpcError> {
    #[cfg(feature = "primal-proof")]
    if let Some(result) = super::super::super::barracuda_route::try_forward("stats.hill_sweep", params) {
        return Ok(result);
    }

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
    let doses = super::linspace(n, dose_max);

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
    let times = super::linspace(n, t_max_days);

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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
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
}
