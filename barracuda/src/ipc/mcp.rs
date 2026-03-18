// SPDX-License-Identifier: AGPL-3.0-or-later
//! MCP (Model Context Protocol) tool definitions for Squirrel AI integration.
//!
//! Typed tool schemas that allow Squirrel (or any MCP-compliant AI client)
//! to discover wetSpring's science capabilities. Each tool maps to an
//! existing JSON-RPC method — no new functionality, just structured
//! discoverability.
//!
//! # Discovery
//!
//! When Squirrel queries `tools/list`, wetSpring returns these definitions.
//! Squirrel then calls `tools/call` with the tool name and arguments,
//! which the server dispatches to the existing JSON-RPC handler.

use serde_json::{Value, json};

/// A typed MCP tool definition per the MCP specification.
pub struct McpTool {
    /// Tool name (e.g. `"wetspring_diversity"`).
    pub name: &'static str,
    /// Human-readable description for the AI model.
    pub description: &'static str,
    /// JSON Schema for the input parameters.
    pub input_schema: fn() -> Value,
}

/// All MCP tools exposed by wetSpring.
pub const TOOLS: &[McpTool] = &[
    McpTool {
        name: "wetspring_diversity",
        description: "Compute alpha diversity metrics (Shannon H', Simpson D, Chao1, Pielou J) \
                      for a microbial community abundance vector. Returns all four indices \
                      plus observed feature count.",
        input_schema: diversity_schema,
    },
    McpTool {
        name: "wetspring_anderson",
        description: "Run Anderson spectral disorder analysis on a lattice to determine \
                      the localization regime (extended/GOE vs localized/Poisson). \
                      Maps community diversity to a 3D lattice disorder parameter W.",
        input_schema: anderson_schema,
    },
    McpTool {
        name: "wetspring_qs_model",
        description: "Integrate a quorum-sensing biofilm ODE model (Waters 2008). \
                      Returns time-series concentrations for N, AI, R, C, B species \
                      given initial conditions and kinetic parameters.",
        input_schema: qs_model_schema,
    },
    McpTool {
        name: "wetspring_alignment",
        description: "Perform Smith-Waterman local sequence alignment between a query \
                      and reference nucleotide sequence. Returns alignment score, \
                      identity percentage, and aligned subsequences.",
        input_schema: alignment_schema,
    },
    McpTool {
        name: "wetspring_kinetics",
        description: "Compute biogas production curves using Gompertz, first-order, Monod, \
                      or Haldane kinetic models. Returns time-series production data \
                      for anaerobic digestion analysis.",
        input_schema: kinetics_schema,
    },
    McpTool {
        name: "wetspring_ecology_interpret",
        description: "Forward an ecology question to Squirrel AI with domain context. \
                      Provides ecology-specific grounding for diversity metrics, \
                      Anderson disorder, and biofilm dynamics interpretation.",
        input_schema: ai_interpret_schema,
    },
    McpTool {
        name: "wetspring_nmf",
        description: "Run Non-negative Matrix Factorization on an abundance/expression \
                      matrix. Returns W (basis) and H (coefficient) matrices for \
                      community typing or drug repurposing analysis.",
        input_schema: nmf_schema,
    },
    McpTool {
        name: "wetspring_phylogenetics",
        description: "Compute Robinson-Foulds tree distance between two Newick trees. \
                      Returns normalized and unnormalized RF distance metrics.",
        input_schema: phylogenetics_schema,
    },
];

fn diversity_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "abundances": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Relative or absolute abundance vector (one entry per OTU/ASV)"
            }
        },
        "required": ["abundances"]
    })
}

fn anderson_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "disorder": {
                "type": "number",
                "description": "Anderson disorder strength W (higher = more localized)"
            },
            "lattice_size": {
                "type": "integer",
                "description": "Linear lattice dimension L (total sites = L³ for 3D)",
                "default": 8
            },
            "realizations": {
                "type": "integer",
                "description": "Number of disorder realizations for averaging",
                "default": 4
            }
        },
        "required": ["disorder"]
    })
}

fn qs_model_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "scenario": {
                "type": "string",
                "enum": ["standard", "high_density", "hapr_mutant", "dgc_overexpression"],
                "description": "Pre-defined ODE scenario from Waters 2008"
            },
            "dt": {
                "type": "number",
                "description": "Integration time step",
                "default": 0.001
            }
        },
        "required": ["scenario"]
    })
}

fn alignment_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query nucleotide sequence (ACGT)"
            },
            "reference": {
                "type": "string",
                "description": "Reference nucleotide sequence (ACGT)"
            }
        },
        "required": ["query", "reference"]
    })
}

fn kinetics_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["gompertz", "first_order", "monod", "haldane"],
                "description": "Kinetic model type"
            },
            "parameters": {
                "type": "object",
                "description": "Model-specific parameters (P, rm, lambda for Gompertz; etc.)"
            },
            "t_end": {
                "type": "number",
                "description": "End time for integration (days)",
                "default": 60.0
            }
        },
        "required": ["model", "parameters"]
    })
}

fn ai_interpret_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Ecology question for AI interpretation"
            },
            "context": {
                "type": "object",
                "description": "Domain context (diversity metrics, Anderson results, etc.)"
            }
        },
        "required": ["query"]
    })
}

fn nmf_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Flattened input matrix V (row-major, n_samples × n_features)"
            },
            "n_samples": { "type": "integer", "description": "Number of rows" },
            "n_features": { "type": "integer", "description": "Number of columns" },
            "rank": { "type": "integer", "description": "Factorization rank k" }
        },
        "required": ["matrix", "n_samples", "n_features", "rank"]
    })
}

fn phylogenetics_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tree1": {
                "type": "string",
                "description": "First Newick tree string"
            },
            "tree2": {
                "type": "string",
                "description": "Second Newick tree string"
            }
        },
        "required": ["tree1", "tree2"]
    })
}

/// Build the `tools/list` response payload per MCP specification.
#[must_use]
pub fn list_tools() -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": (t.input_schema)(),
            })
        })
        .collect();

    json!({ "tools": tools })
}

/// Map an MCP tool name to the corresponding JSON-RPC method.
#[must_use]
pub fn tool_to_method(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "wetspring_diversity" => Some("science.diversity"),
        "wetspring_anderson" => Some("science.anderson"),
        "wetspring_qs_model" => Some("science.qs_model"),
        "wetspring_alignment" => Some("science.alignment"),
        "wetspring_kinetics" => Some("science.kinetics"),
        "wetspring_ecology_interpret" => Some("ai.ecology_interpret"),
        "wetspring_nmf" => Some("science.nmf"),
        "wetspring_phylogenetics" => Some("science.phylogenetics"),
        _ => None,
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn list_tools_returns_all() {
        let response = list_tools();
        let tools = response["tools"].as_array().unwrap();
        assert_eq!(tools.len(), TOOLS.len());
    }

    #[test]
    fn all_tools_have_input_schema() {
        for tool in TOOLS {
            let schema = (tool.input_schema)();
            assert_eq!(
                schema["type"], "object",
                "{} schema must be object",
                tool.name
            );
            assert!(
                schema.get("properties").is_some(),
                "{} schema must have properties",
                tool.name
            );
        }
    }

    #[test]
    fn tool_names_are_prefixed() {
        for tool in TOOLS {
            assert!(
                tool.name.starts_with("wetspring_"),
                "MCP tool '{}' must be prefixed with wetspring_",
                tool.name
            );
        }
    }

    #[test]
    fn all_tools_have_method_mapping() {
        for tool in TOOLS {
            assert!(
                tool_to_method(tool.name).is_some(),
                "MCP tool '{}' has no JSON-RPC method mapping",
                tool.name
            );
        }
    }

    #[test]
    fn method_mappings_are_valid_capabilities() {
        let all_methods = crate::ipc::capability_domains::all_methods();
        for tool in TOOLS {
            let method = tool_to_method(tool.name).unwrap();
            assert!(
                all_methods.contains(&method),
                "tool '{}' maps to '{}' which is not a registered capability",
                tool.name,
                method
            );
        }
    }
}
