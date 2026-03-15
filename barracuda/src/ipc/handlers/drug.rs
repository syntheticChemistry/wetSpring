// SPDX-License-Identifier: AGPL-3.0-or-later
//! Drug repurposing IPC handlers — Non-negative Matrix Factorization (NMF).
//!
//! Self-contained NMF implementation; will wire to `barracuda::gpu` primitives
//! (e.g. `SparseGemmF64`) when available.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::tolerances;

use super::extract_f64_array;

/// Handle `science.nmf` — Non-negative Matrix Factorization (multiplicative update).
///
/// Self-contained NMF implementation; will wire to `barracuda::gpu` primitives
/// (e.g. `SparseGemmF64`) when available.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` for dimension mismatches or invalid rank.
pub fn handle_nmf(params: &Value) -> Result<Value, RpcError> {
    let data = extract_f64_array(params, "data")?;
    let n_rows = params
        .get("n_rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| RpcError::invalid_params("missing required param: n_rows"))?
        as usize;
    let n_cols = params
        .get("n_cols")
        .and_then(Value::as_u64)
        .ok_or_else(|| RpcError::invalid_params("missing required param: n_cols"))?
        as usize;
    let rank = params.get("rank").and_then(Value::as_u64).unwrap_or(2) as usize;
    let max_iter = params
        .get("max_iter")
        .and_then(Value::as_u64)
        .unwrap_or(200) as usize;

    if data.len() != n_rows * n_cols {
        return Err(RpcError::invalid_params(format!(
            "data length {} != n_rows({n_rows}) * n_cols({n_cols})",
            data.len()
        )));
    }

    if rank == 0 || rank > n_rows.min(n_cols) {
        return Err(RpcError::invalid_params(format!(
            "rank must be in [1, min(n_rows, n_cols)], got {rank}"
        )));
    }

    let (w, h, error, iterations) = nmf_mu(&data, n_rows, n_cols, rank, max_iter);
    let _ = (w, h);

    Ok(json!({
        "rank": rank,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "iterations": iterations,
        "converged": iterations < max_iter,
        "reconstruction_error": error,
    }))
}

/// Multiplicative-update NMF (Lee & Seung 2001).
/// Returns (W, H, `final_error`, `iterations_used`).
#[expect(clippy::cast_precision_loss)] // Precision: rank and loop indices small (< 2^53)
fn nmf_mu(
    v: &[f64],
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, f64, usize) {
    let epsilon = tolerances::MATRIX_EPS;
    let tol = tolerances::NMF_CONVERGENCE;

    let mut w = vec![1.0 / rank as f64; n_rows * rank];
    let mut h = vec![1.0 / rank as f64; rank * n_cols];

    // Initialize W and H with simple deterministic values
    for (i, val) in w.iter_mut().enumerate() {
        *val = 0.5f64.mul_add((i % 7) as f64 / 7.0, 0.5);
    }
    for (i, val) in h.iter_mut().enumerate() {
        *val = 0.5f64.mul_add((i % 11) as f64 / 11.0, 0.5);
    }

    let mut prev_error = f64::MAX;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Update H: H *= (W^T V) / (W^T W H + eps)
        let wt_v = matmul_t_a(n_rows, rank, n_cols, &w, v);
        let wt_w = matmul_t_a(n_rows, rank, rank, &w, &w);
        let wt_w_h = matmul(rank, rank, n_cols, &wt_w, &h);
        for i in 0..h.len() {
            h[i] *= wt_v[i] / (wt_w_h[i] + epsilon);
        }

        // Update W: W *= (V H^T) / (W H H^T + eps)
        let v_ht = matmul_t_b(n_rows, n_cols, rank, v, &h);
        let h_ht = matmul_t_b(rank, n_cols, rank, &h, &h);
        let w_h_ht = matmul(n_rows, rank, rank, &w, &h_ht);
        for i in 0..w.len() {
            w[i] *= v_ht[i] / (w_h_ht[i] + epsilon);
        }

        // Reconstruction error (Frobenius norm)
        let wh = matmul(n_rows, rank, n_cols, &w, &h);
        let error: f64 = v
            .iter()
            .zip(wh.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        if (prev_error - error).abs() < tol {
            return (w, h, error, iterations);
        }
        prev_error = error;
    }

    (w, h, prev_error, iterations)
}

/// C = A * B where A is (m x k), B is (k x n).
#[expect(clippy::many_single_char_names)]
fn matmul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum = a[i * k + p].mul_add(b[p * n + j], sum);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// C = A^T * B where A is (m x k), B is (m x n) → C is (k x n).
#[expect(clippy::many_single_char_names)]
fn matmul_t_a(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..m {
                sum = a[p * k + i].mul_add(b[p * n + j], sum);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// C = A * B^T where A is (m x k), B is (n x k) → C is (m x n).
#[expect(clippy::many_single_char_names)]
fn matmul_t_b(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum = a[i * k + p].mul_add(b[j * k + p], sum);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
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
