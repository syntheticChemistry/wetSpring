// SPDX-License-Identifier: AGPL-3.0-or-later
//! Brain IPC handlers — bio observation, attention, urgency.

use std::sync::Mutex;

use serde_json::{Value, json};

use crate::bio::brain::{BioBrain, BioObservation};
use crate::bio::esn::heads::NUM_HEADS;
use crate::ipc::protocol::RpcError;

use super::extract_f64_array;

static BIO_BRAIN: Mutex<Option<BioBrain>> = Mutex::new(None);

/// Process a bio observation through the brain pipeline.
///
/// Expects: `sample_id`, `shannon`, `simpson`, `evenness`, `chao1`,
/// `bray_curtis_mean`, `anderson_w`, `anderson_phase`, `amr_load`,
/// and `head_outputs` (array of 36 floats).
///
/// Returns attention state, urgency, and disagreement signals.
pub fn handle_brain_observe(params: &Value) -> Result<Value, RpcError> {
    let obs = parse_bio_observation(params);
    let head_outputs = extract_f64_array(params, "head_outputs")?;
    if head_outputs.len() < NUM_HEADS {
        return Err(RpcError::invalid_params(format!(
            "head_outputs requires {NUM_HEADS} elements, got {}",
            head_outputs.len()
        )));
    }

    let mut guard = BIO_BRAIN
        .lock()
        .map_err(|e| RpcError::server_error(-32001, format!("brain lock poisoned: {e}")))?;
    let brain = guard.get_or_insert_with(|| BioBrain::new(10));
    let update = brain.observe(&obs, &head_outputs);
    let status = brain.status_snapshot();
    drop(guard);

    Ok(json!({
        "status": "observed",
        "sample_id": obs.sample_id,
        "attention": format!("{:?}", update.attention),
        "urgency": status.urgency,
        "n_species": update.n_species,
        "shannon_h": update.shannon_h,
        "evenness": update.evenness,
        "observation_count": status.observation_count,
        "delta_regime": status.delta_regime,
        "delta_phase": status.delta_phase,
        "delta_anomaly": status.delta_anomaly,
        "delta_priority": status.delta_priority,
    }))
}

/// Query the current brain attention state and disagreement signals.
pub fn handle_brain_attention(_params: &Value) -> Result<Value, RpcError> {
    let guard = BIO_BRAIN
        .lock()
        .map_err(|e| RpcError::server_error(-32001, format!("brain lock poisoned: {e}")))?;

    let result = guard.as_ref().map_or_else(
        || {
            json!({
                "attention": "Healthy",
                "urgency": 0.0,
                "observation_count": 0,
                "message": "brain not initialized (no observations yet)",
            })
        },
        |brain| {
            let status = brain.status_snapshot();
            json!({
                "attention": format!("{:?}", status.attention),
                "urgency": status.urgency,
                "observation_count": status.observation_count,
                "delta_regime": status.delta_regime,
                "delta_phase": status.delta_phase,
                "delta_anomaly": status.delta_anomaly,
                "delta_priority": status.delta_priority,
                "is_healthy": brain.is_healthy(),
                "is_critical": brain.is_critical(),
            })
        },
    );
    drop(guard);
    Ok(result)
}

/// Get the current urgency score (scalar 0–1).
///
/// Lightweight endpoint for dashboard polling.
pub fn handle_brain_urgency(_params: &Value) -> Result<Value, RpcError> {
    let guard = BIO_BRAIN
        .lock()
        .map_err(|e| RpcError::server_error(-32001, format!("brain lock poisoned: {e}")))?;

    let urgency = guard.as_ref().map_or(0.0, BioBrain::smoothed_urgency);
    let attention = guard.as_ref().map_or(
        crate::bio::esn::heads::AttentionState::Healthy,
        BioBrain::attention,
    );
    drop(guard);

    Ok(json!({
        "urgency": urgency,
        "attention": format!("{attention:?}"),
    }))
}

fn parse_bio_observation(params: &Value) -> BioObservation {
    let get_f64 = |key: &str| -> f64 { params.get(key).and_then(Value::as_f64).unwrap_or(0.0) };
    let sample_id = params
        .get("sample_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    BioObservation {
        sample_id,
        shannon: get_f64("shannon"),
        simpson: get_f64("simpson"),
        evenness: get_f64("evenness"),
        chao1: get_f64("chao1"),
        bray_curtis_mean: get_f64("bray_curtis_mean"),
        anderson_w: get_f64("anderson_w"),
        anderson_phase: get_f64("anderson_phase"),
        amr_load: get_f64("amr_load"),
    }
}
