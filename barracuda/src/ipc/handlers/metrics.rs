// SPDX-License-Identifier: AGPL-3.0-or-later
//! Server metrics snapshot handler.
//!
//! Returns basic operational metrics: uptime, capabilities count, and
//! dispatch surface stats. Designed for biomeOS monitoring and Squirrel
//! performance observations.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::niche;

/// Handle `metrics.snapshot` — return server operational metrics.
///
/// # Errors
///
/// Currently infallible (always returns a valid snapshot).
pub fn handle_metrics_snapshot(_params: &Value) -> Result<Value, RpcError> {
    Ok(json!({
        "primal": niche::NICHE_NAME,
        "capabilities_count": niche::CAPABILITIES.len(),
        "dependencies_total": niche::DEPENDENCIES.len(),
        "dependencies_required": niche::required_dependency_count(),
        "consumed_capabilities": niche::CONSUMED_CAPABILITIES.len(),
        "deploy_graph": niche::deploy_graph_path(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_primal_name() {
        let r = handle_metrics_snapshot(&json!({})).unwrap();
        assert_eq!(r["primal"], "wetspring");
    }

    #[test]
    fn snapshot_includes_capability_counts() {
        let r = handle_metrics_snapshot(&json!({})).unwrap();
        assert!(r["capabilities_count"].as_u64().unwrap() >= 30);
        assert!(r["dependencies_total"].as_u64().unwrap() >= 5);
    }
}
