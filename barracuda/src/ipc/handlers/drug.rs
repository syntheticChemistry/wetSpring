// SPDX-License-Identifier: AGPL-3.0-or-later
//! Drug repurposing IPC handlers — Non-negative Matrix Factorization (NMF).
//!
//! Delegates to `barracuda::linalg::nmf` (Lee & Seung multiplicative updates,
//! absorbed from wetSpring Feb 2026). No local math — single implementation
//! in barraCuda.

use barracuda::linalg::nmf::{NmfConfig, NmfObjective};
use serde_json::{Value, json};

use crate::cast::u64_usize;
use crate::ipc::protocol::RpcError;

use super::extract_f64_array;

/// Handle `science.nmf` — Non-negative Matrix Factorization (multiplicative update).
///
/// Delegates to [`barracuda::linalg::nmf::nmf`] for the actual factorisation.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` for dimension mismatches or invalid rank.
pub fn handle_nmf(params: &Value) -> Result<Value, RpcError> {
    let data = extract_f64_array(params, "data")?;
    let n_rows = u64_usize(
        params
            .get("n_rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| RpcError::invalid_params("missing required param: n_rows"))?,
    );
    let n_cols = u64_usize(
        params
            .get("n_cols")
            .and_then(Value::as_u64)
            .ok_or_else(|| RpcError::invalid_params("missing required param: n_cols"))?,
    );
    let rank = u64_usize(params.get("rank").and_then(Value::as_u64).unwrap_or(2));
    let max_iter = u64_usize(
        params
            .get("max_iter")
            .and_then(Value::as_u64)
            .unwrap_or(200),
    );

    let config = NmfConfig {
        rank,
        max_iter,
        tol: 1e-6,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };

    let result = barracuda::linalg::nmf::nmf(&data, n_rows, n_cols, &config)
        .map_err(|e| RpcError::invalid_params(format!("{e}")))?;

    let final_error = result.errors.last().copied().unwrap_or(0.0);
    let iterations = result.errors.len();

    Ok(json!({
        "rank": rank,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "iterations": iterations,
        "converged": iterations < max_iter,
        "reconstruction_error": final_error,
    }))
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
    fn nmf_basic() {
        let params = json!({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "n_rows": 2,
            "n_cols": 3,
            "rank": 1,
            "max_iter": 50,
        });
        let result = handle_nmf(&params).unwrap();
        assert_eq!(result["rank"], 1);
        assert!(result.get("reconstruction_error").is_some());
    }

    #[test]
    fn nmf_invalid_dimensions() {
        let params = json!({
            "data": [1.0, 2.0],
            "n_rows": 2,
            "n_cols": 3,
            "rank": 1,
        });
        let err = handle_nmf(&params).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn nmf_invalid_rank() {
        let params = json!({
            "data": [1.0, 2.0, 3.0, 4.0],
            "n_rows": 2,
            "n_cols": 2,
            "rank": 5,
        });
        let err = handle_nmf(&params).unwrap_err();
        assert_eq!(err.code, -32602);
    }
}
