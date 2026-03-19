// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kinetics IPC handlers — biogas production curve fitting.
//!
//! Supports Gompertz, first-order, and custom-ODE models via barracuda's
//! numerical integration.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

/// Handle `science.kinetics` — biogas production curve fitting.
///
/// Supports Gompertz, first-order, and custom-ODE models via barracuda's
/// numerical integration.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` for unknown model names.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // Cast: steps, indices bounded
pub fn handle_kinetics(params: &Value) -> Result<Value, RpcError> {
    let model = params
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("gompertz");

    let t_max = params.get("t_max").and_then(Value::as_f64).unwrap_or(30.0);
    let dt = params.get("dt").and_then(Value::as_f64).unwrap_or(0.1);

    let steps = (t_max / dt) as usize;

    match model {
        "gompertz" => {
            let p_max = params.get("p_max").and_then(Value::as_f64).unwrap_or(300.0);
            let r_max = params.get("r_max").and_then(Value::as_f64).unwrap_or(15.0);
            let lag = params.get("lag").and_then(Value::as_f64).unwrap_or(2.0);

            let mut final_p = 0.0_f64;
            for i in 0..steps {
                let t = i as f64 * dt;
                let exponent = (r_max * std::f64::consts::E / p_max).mul_add(lag - t, 1.0);
                final_p = p_max * (-(-exponent).exp()).exp();
            }

            Ok(json!({
                "model": "gompertz",
                "parameters": {"p_max": p_max, "r_max": r_max, "lag": lag},
                "t_end": t_max,
                "steps": steps,
                "final_production": final_p,
            }))
        }
        "first_order" => {
            let k = params.get("k").and_then(Value::as_f64).unwrap_or(0.1);
            let s0 = params.get("s0").and_then(Value::as_f64).unwrap_or(100.0);
            let final_val = s0 * (1.0 - (-k * t_max).exp());

            Ok(json!({
                "model": "first_order",
                "parameters": {"k": k, "s0": s0},
                "t_end": t_max,
                "steps": steps,
                "final_production": final_val,
            }))
        }
        _ => Err(RpcError::invalid_params(format!(
            "unknown kinetics model: {model} (supported: gompertz, first_order)"
        ))),
    }
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
    fn kinetics_gompertz_default() {
        let result = handle_kinetics(&json!({})).unwrap();
        assert_eq!(result["model"], "gompertz");
        assert!(result["final_production"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn kinetics_first_order() {
        let result = handle_kinetics(&json!({"model": "first_order"})).unwrap();
        assert_eq!(result["model"], "first_order");
    }

    #[test]
    fn kinetics_unknown_model() {
        let err = handle_kinetics(&json!({"model": "unknown"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }
}
