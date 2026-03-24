// SPDX-License-Identifier: AGPL-3.0-or-later
//! Alignment IPC handlers — Smith-Waterman local sequence alignment.

use serde_json::{Value, json};

use crate::bio::alignment::{AlignmentResult, ScoringParams};
use crate::cast::usize_f64;
use crate::ipc::protocol::RpcError;

/// Handle `science.alignment` — Smith-Waterman local alignment.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `seq_a` or `seq_b` is missing.
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
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(2),
        mismatch_penalty: params
            .get("mismatch_penalty")
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(-1),
        gap_open: params
            .get("gap_open")
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(-3),
        gap_extend: params
            .get("gap_extend")
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(-1),
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
        usize_f64(matches) / usize_f64(aligned_len)
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
