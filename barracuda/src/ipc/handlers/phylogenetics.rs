// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phylogenetics IPC handlers — Robinson-Foulds tree distance.

use serde_json::{Value, json};

use crate::bio::robinson_foulds;
use crate::bio::unifrac::PhyloTree;
use crate::ipc::protocol::RpcError;

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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
