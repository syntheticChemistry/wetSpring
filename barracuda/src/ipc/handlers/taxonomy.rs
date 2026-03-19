// SPDX-License-Identifier: AGPL-3.0-or-later
//! Taxonomy IPC handlers — Naive Bayes k-mer classification (RDP-style).

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

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

    let k = params.get("k").and_then(Value::as_u64).unwrap_or(8) as usize;

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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn taxonomy_no_reference() {
        let result = handle_taxonomy(&json!({"sequence": "ACGTACGT"})).unwrap();
        assert_eq!(result["status"], "no_reference");
    }
}
