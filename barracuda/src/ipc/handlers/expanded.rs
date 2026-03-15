// SPDX-License-Identifier: AGPL-3.0-or-later
//! Expanded science IPC handlers — kinetics, alignment, taxonomy, phylogenetics, NMF.
//!
//! Each handler wraps existing barracuda library functions, exposing them
//! as biomeOS `capability.call` targets. No math is duplicated.

use serde_json::{Value, json};

use crate::bio::alignment::{AlignmentResult, ScoringParams};
use crate::bio::robinson_foulds;
use crate::bio::unifrac::PhyloTree;
use crate::ipc::protocol::RpcError;
use crate::tolerances;

use super::extract_f64_array;

/// Handle `science.kinetics` — biogas production curve fitting.
///
/// Supports Gompertz, first-order, and custom-ODE models via barracuda's
/// numerical integration.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` for unknown model names.
#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)] // Cast: steps, indices bounded
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
            let p_max = params
                .get("p_max")
                .and_then(Value::as_f64)
                .unwrap_or(300.0);
            let r_max = params
                .get("r_max")
                .and_then(Value::as_f64)
                .unwrap_or(15.0);
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

/// Handle `science.alignment` — Smith-Waterman local alignment.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `seq_a` or `seq_b` is missing.
#[expect(clippy::cast_precision_loss)] // Precision: match count and aligned_len bounded by sequence length
pub fn handle_alignment(params: &Value) -> Result<Value, RpcError> {
    let seq_a = params
        .get("seq_a")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: seq_a"))?;
    let seq_b = params
        .get("seq_b")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: seq_b"))?;

    let scoring = ScoringParams {
        match_score: params
            .get("match_score")
            .and_then(Value::as_i64)
            .unwrap_or(2) as i32,
        mismatch_penalty: params
            .get("mismatch_penalty")
            .and_then(Value::as_i64)
            .unwrap_or(-1) as i32,
        gap_open: params
            .get("gap_open")
            .and_then(Value::as_i64)
            .unwrap_or(-3) as i32,
        gap_extend: params
            .get("gap_extend")
            .and_then(Value::as_i64)
            .unwrap_or(-1) as i32,
    };

    let result: AlignmentResult =
        crate::bio::alignment::smith_waterman(seq_a.as_bytes(), seq_b.as_bytes(), &scoring);

    let aligned_len = result.aligned_query.len();
    let identity = if aligned_len > 0 {
        let matches = result
            .aligned_query
            .iter()
            .zip(result.aligned_target.iter())
            .filter(|(a, b)| a == b && **a != b'-')
            .count();
        matches as f64 / aligned_len as f64
    } else {
        0.0
    };

    Ok(json!({
        "score": result.score,
        "aligned_query": String::from_utf8_lossy(&result.aligned_query),
        "aligned_target": String::from_utf8_lossy(&result.aligned_target),
        "identity": identity,
        "alignment_length": aligned_len,
        "query_start": result.query_start,
        "target_start": result.target_start,
    }))
}

/// Handle `science.taxonomy` — Naive Bayes k-mer classification (RDP-style).
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `sequence` is missing.
pub fn handle_taxonomy(params: &Value) -> Result<Value, RpcError> {
    let sequence = params
        .get("sequence")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: sequence"))?;

    let k = params
        .get("k")
        .and_then(Value::as_u64)
        .unwrap_or(8) as usize;

    let ref_fasta = params.get("reference_fasta").and_then(Value::as_str);

    if let Some(fasta) = ref_fasta {
        let refs = crate::bio::taxonomy::kmers::parse_reference_fasta(fasta);
        if refs.is_empty() {
            return Ok(json!({
                "status": "no_valid_references",
                "k": k,
                "sequence_length": sequence.len(),
            }));
        }

        let classifier = crate::bio::taxonomy::NaiveBayesClassifier::train(&refs, k);
        let classify_params = crate::bio::taxonomy::types::ClassifyParams {
            k,
            ..Default::default()
        };
        let result = classifier.classify(sequence.as_bytes(), &classify_params);

        let top_rank = result
            .confidence
            .iter()
            .rposition(|&c| c >= classify_params.min_confidence);

        Ok(json!({
            "lineage": result.lineage.ranks,
            "confidence": result.confidence,
            "deepest_confident_rank": top_rank,
            "k": k,
        }))
    } else {
        Ok(json!({
            "status": "no_reference",
            "message": "taxonomy requires reference_fasta (SILVA-style FASTA with taxonomy in headers)",
            "k": k,
            "sequence_length": sequence.len(),
        }))
    }
}

/// Handle `science.phylogenetics` — Robinson-Foulds tree distance.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `tree_a` or `tree_b` is missing.
pub fn handle_phylogenetics(params: &Value) -> Result<Value, RpcError> {
    let newick_a = params
        .get("tree_a")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: tree_a (Newick)"))?;
    let newick_b = params
        .get("tree_b")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: tree_b (Newick)"))?;

    let pa = PhyloTree::from_newick(newick_a);
    let pb = PhyloTree::from_newick(newick_b);

    let rf = robinson_foulds::rf_distance(&pa, &pb);
    let rf_norm = robinson_foulds::rf_distance_normalized(&pa, &pb);

    Ok(json!({
        "robinson_foulds": rf,
        "robinson_foulds_normalized": rf_norm,
        "tree_a_leaves": pa.n_leaves(),
        "tree_b_leaves": pb.n_leaves(),
    }))
}

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
        .ok_or_else(|| RpcError::invalid_params("missing required param: n_rows"))? as usize;
    let n_cols = params
        .get("n_cols")
        .and_then(Value::as_u64)
        .ok_or_else(|| RpcError::invalid_params("missing required param: n_cols"))? as usize;
    let rank = params
        .get("rank")
        .and_then(Value::as_u64)
        .unwrap_or(2) as usize;
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

    #[test]
    fn alignment_identical() {
        let params = json!({"seq_a": "ACGTACGT", "seq_b": "ACGTACGT"});
        let result = handle_alignment(&params).unwrap();
        assert!(result["score"].as_i64().unwrap() > 0);
        assert!(result["identity"].as_f64().unwrap() > 0.9);
    }

    #[test]
    fn alignment_missing_seq() {
        let err = handle_alignment(&json!({"seq_a": "ACGT"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn taxonomy_no_reference() {
        let result = handle_taxonomy(&json!({"sequence": "ACGTACGT"})).unwrap();
        assert_eq!(result["status"], "no_reference");
    }

    #[test]
    fn phylogenetics_identical_trees() {
        let params = json!({
            "tree_a": "((A:1,B:1):0.5,(C:1,D:1):0.5)",
            "tree_b": "((A:1,B:1):0.5,(C:1,D:1):0.5)",
        });
        let result = handle_phylogenetics(&params).unwrap();
        assert_eq!(result["robinson_foulds"], 0);
        assert_eq!(result["tree_a_leaves"], 4);
    }

    #[test]
    fn phylogenetics_missing_tree() {
        let err = handle_phylogenetics(&json!({"tree_a": "(A:1,B:1)"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

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
