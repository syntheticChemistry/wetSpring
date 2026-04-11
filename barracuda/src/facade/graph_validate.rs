// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural validation for deploy graph TOML files.
//!
//! Checks that a deploy graph satisfies the ecosystem's structural invariants
//! (every node has `by_capability`, no duplicate names, dependency closure,
//! order consistency). Pattern from primalSpring's `graph.validate` capability.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
struct DeployGraph {
    graph: GraphBody,
}

#[derive(Deserialize)]
struct GraphBody {
    name: String,
    #[expect(
        dead_code,
        reason = "deserialized for structural completeness but not inspected"
    )]
    version: Option<String>,
    node: Vec<GraphNode>,
}

#[derive(Deserialize)]
struct GraphNode {
    name: String,
    #[expect(
        dead_code,
        reason = "deserialized for structural completeness but not inspected"
    )]
    binary: Option<String>,
    order: Option<u32>,
    by_capability: Option<String>,
    capabilities: Option<Vec<String>>,
    depends_on: Option<Vec<String>>,
}

/// Result of structural graph validation.
pub struct ValidationResult {
    /// `true` if no structural errors were found.
    pub valid: bool,
    /// `[graph].name` from the TOML.
    pub graph_name: String,
    /// Total number of `[[graph.node]]` entries.
    pub node_count: usize,
    /// Hard errors (duplicate names, missing by_capability, broken deps).
    pub errors: Vec<String>,
    /// Soft warnings (order inconsistencies, missing capabilities, shared domains).
    pub warnings: Vec<String>,
}

/// Validate a deploy graph TOML string structurally.
pub fn validate_graph(toml_source: &str) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let graph: DeployGraph = match toml::from_str(toml_source) {
        Ok(g) => g,
        Err(e) => {
            return ValidationResult {
                valid: false,
                graph_name: "parse_error".into(),
                node_count: 0,
                errors: vec![format!("TOML parse error: {e}")],
                warnings: vec![],
            };
        }
    };

    let graph_name = graph.graph.name.clone();
    let nodes = &graph.graph.node;
    let node_count = nodes.len();

    let mut names: HashSet<&str> = HashSet::new();
    let mut name_to_order: HashMap<&str, u32> = HashMap::new();

    for node in nodes {
        if !names.insert(&node.name) {
            errors.push(format!("duplicate node name: `{}`", node.name));
        }

        if node.by_capability.is_none() {
            errors.push(format!(
                "node `{}` missing `by_capability` (required for capability-first discovery)",
                node.name
            ));
        }

        if node.capabilities.as_ref().map_or(true, |c| c.is_empty()) {
            warnings.push(format!("node `{}` has no capabilities declared", node.name));
        }

        if let Some(order) = node.order {
            name_to_order.insert(&node.name, order);
        }
    }

    for node in nodes {
        if let Some(deps) = &node.depends_on {
            for dep in deps {
                if !names.contains(dep.as_str()) {
                    errors.push(format!(
                        "node `{}` depends on `{dep}` which is not in the graph",
                        node.name
                    ));
                }

                if let (Some(&dep_order), Some(my_order)) =
                    (name_to_order.get(dep.as_str()), node.order)
                {
                    if dep_order >= my_order {
                        warnings.push(format!(
                            "node `{}` (order {my_order}) depends on `{dep}` (order {dep_order}) — dependency should have lower order",
                            node.name
                        ));
                    }
                }
            }
        }
    }

    let mut cap_domains: HashSet<&str> = HashSet::new();
    for node in nodes {
        if let Some(by_cap) = &node.by_capability {
            if !cap_domains.insert(by_cap) {
                warnings.push(format!(
                    "by_capability `{by_cap}` claimed by multiple nodes"
                ));
            }
        }
    }

    ValidationResult {
        valid: errors.is_empty(),
        graph_name,
        node_count,
        errors,
        warnings,
    }
}

impl ValidationResult {
    /// Serialize to JSON for the composition endpoint.
    pub fn to_json(&self) -> Value {
        json!({
            "valid": self.valid,
            "graph_name": self.graph_name,
            "node_count": self.node_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": self.errors.len(),
            "warning_count": self.warnings.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAPH: &str = include_str!("../../../graphs/wetspring_science_nucleus.toml");

    #[test]
    fn wetspring_nucleus_graph_is_structurally_valid() {
        let result = validate_graph(GRAPH);
        assert!(
            result.valid,
            "graph has structural errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn all_nodes_have_by_capability() {
        let result = validate_graph(GRAPH);
        let missing: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.contains("missing `by_capability`"))
            .collect();
        assert!(
            missing.is_empty(),
            "nodes missing by_capability: {missing:?}"
        );
    }

    #[test]
    fn no_duplicate_node_names() {
        let result = validate_graph(GRAPH);
        let dupes: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.contains("duplicate node name"))
            .collect();
        assert!(dupes.is_empty(), "duplicate nodes: {dupes:?}");
    }

    #[test]
    fn dependency_closure_holds() {
        let result = validate_graph(GRAPH);
        let broken: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.contains("depends on") && e.contains("not in the graph"))
            .collect();
        assert!(broken.is_empty(), "broken dependencies: {broken:?}");
    }
}
