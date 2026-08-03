// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ionic bond integration for wetSpring's IPC layer.
//!
//! Bridges `primalspring::bonding::ionic_runtime::IonicContractRegistry` into
//! wetSpring's JSON-RPC handler surface. Provides the `bonding.*` method family:
//!
//! - `bonding.propose` — create a new ionic bond proposal
//! - `bonding.accept` — accept a pending proposal
//! - `bonding.reject` — reject a pending proposal
//! - `bonding.status` — query contract state
//! - `bonding.terminate` — initiate graceful termination
//! - `bonding.list` — enumerate active contracts
//!
//! The registry lives in-process as a singleton behind `OnceLock`. Cross-gate
//! operations (Ed25519 signing via bearDog, remote dispatch via songbird) are
//! deferred until those primals are live on the deployed mesh.

use std::sync::{Mutex, OnceLock};

use primalspring::bonding::ionic::{AttributionTerms, CreditMethod, DataReturnPolicy};
use primalspring::bonding::ionic::{
    ContractState, IonicProposal, TerminationReason, TerminationRequest,
};
use primalspring::bonding::ionic_runtime::{IonicContractRegistry, IonicProtocolError};
use primalspring::bonding::{BondingConstraint, TrustModel};

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::primal_names;

static REGISTRY: OnceLock<Mutex<IonicContractRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<IonicContractRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(IonicContractRegistry::new()))
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "IonicProtocolError consumed by to_string()"
)]
fn map_ionic_err(e: IonicProtocolError) -> RpcError {
    RpcError {
        code: -32001,
        message: e.to_string(),
    }
}

fn lock_err() -> RpcError {
    RpcError {
        code: -32603,
        message: "registry lock poisoned".into(),
    }
}

/// `bonding.propose` — create a new ionic bond proposal.
///
/// Params: `{ proposer_identity, requested_capabilities, duration_secs, trust_model, rate_limit_rps }`
pub fn handle_propose(params: &Value) -> Result<Value, RpcError> {
    let proposer_identity = params["proposer_identity"]
        .as_str()
        .unwrap_or(primal_names::SELF_NAME)
        .to_owned();

    let requested_capabilities: Vec<String> = params["requested_capabilities"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if requested_capabilities.is_empty() {
        return Err(RpcError {
            code: -32602,
            message: "requested_capabilities must be non-empty".into(),
        });
    }

    let duration_secs = params["duration_secs"].as_u64().unwrap_or(3600);
    let rate_limit_rps =
        u32::try_from(params["rate_limit_rps"].as_u64().unwrap_or(100)).unwrap_or(u32::MAX);

    let trust_model = match params["trust_model"].as_str() {
        Some("nuclear") => TrustModel::NuclearLineage,
        Some("mito_beacon") => TrustModel::MitoBeaconFamily,
        Some("organizational") => TrustModel::Organizational,
        Some("zero_trust") => TrustModel::ZeroTrust,
        _ => TrustModel::Contractual,
    };

    let proposal = IonicProposal {
        proposer_identity,
        requested_capabilities,
        duration_secs,
        trust_model,
        attribution: AttributionTerms {
            credit_method: CreditMethod::CapabilityProvider,
            propagate_to_derivatives: true,
        },
        data_return_policy: DataReturnPolicy::ProvenanceRetentionOnly,
        rate_limit_rps,
    };

    let mut reg = registry().lock().map_err(|_| lock_err())?;
    let id = reg.propose(proposal).map_err(map_ionic_err)?;

    Ok(json!({
        "contract_id": id,
        "state": "Proposed",
    }))
}

/// `bonding.accept` — accept a pending proposal with constraints.
pub fn handle_accept(params: &Value) -> Result<Value, RpcError> {
    let id = extract_id(params)?;

    let constraints = BondingConstraint::default();

    let mut reg = registry().lock().map_err(|_| lock_err())?;
    let resp = reg.accept(&id, constraints).map_err(map_ionic_err)?;

    Ok(json!({
        "contract_id": resp.contract_id,
        "accepted": resp.accepted,
        "state": "Active",
    }))
}

/// `bonding.reject` — reject a pending proposal.
pub fn handle_reject(params: &Value) -> Result<Value, RpcError> {
    let id = extract_id(params)?;
    let reason = params["reason"].as_str().unwrap_or("rejected by wetspring");

    let mut reg = registry().lock().map_err(|_| lock_err())?;
    let resp = reg.reject(&id, reason).map_err(map_ionic_err)?;

    Ok(json!({
        "contract_id": resp.contract_id,
        "accepted": false,
        "state": "Rejected",
        "reason": resp.rejection_reason,
    }))
}

/// `bonding.status` — query a contract's current state.
pub fn handle_status(params: &Value) -> Result<Value, RpcError> {
    let id = extract_id(params)?;

    let reg = registry().lock().map_err(|_| lock_err())?;
    let contract = reg.get(&id).ok_or_else(|| RpcError {
        code: -32001,
        message: format!("contract not found: {id}"),
    })?;

    Ok(json!({
        "contract_id": contract.contract_id,
        "state": format!("{:?}", contract.state),
        "proposer": contract.proposal.proposer_identity,
        "capabilities": contract.proposal.requested_capabilities,
        "usage": {
            "total_calls": contract.usage.total_calls,
            "total_bytes": contract.usage.total_bytes,
            "distinct_methods": contract.usage.distinct_methods,
        },
        "accepted_at": contract.accepted_at,
        "expires_at": contract.expires_at,
    }))
}

/// `bonding.terminate` — initiate graceful termination.
pub fn handle_terminate(params: &Value) -> Result<Value, RpcError> {
    let id = extract_id(params)?;

    let reason = match params["reason"].as_str() {
        Some("violation") => TerminationReason::PolicyViolation,
        Some("agreement") => TerminationReason::MutualAgreement,
        Some("expired") => TerminationReason::Expired,
        _ => TerminationReason::Complete,
    };

    let request = TerminationRequest {
        contract_id: id,
        reason,
    };

    let mut reg = registry().lock().map_err(|_| lock_err())?;
    let seal = reg.terminate(&request).map_err(map_ionic_err)?;

    Ok(json!({
        "contract_id": seal.contract_id,
        "state": "Sealed",
        "seal": {
            "merkle_root": seal.merkle_root,
            "commit_id": seal.commit_id,
            "braid_id": seal.braid_id,
            "sealed_at": seal.sealed_at,
            "final_usage": {
                "total_calls": seal.final_usage.total_calls,
                "total_bytes": seal.final_usage.total_bytes,
            },
        },
    }))
}

/// `bonding.list` — enumerate contracts, optionally filtered by state.
pub fn handle_list(params: &Value) -> Result<Value, RpcError> {
    let filter: Option<ContractState> =
        params["state"]
            .as_str()
            .and_then(|s| match s.to_lowercase().as_str() {
                "proposed" => Some(ContractState::Proposed),
                "active" => Some(ContractState::Active),
                "sealed" => Some(ContractState::Sealed),
                "expired" => Some(ContractState::Expired),
                "rejected" => Some(ContractState::Rejected),
                _ => None,
            });

    let reg = registry().lock().map_err(|_| lock_err())?;

    let contracts: Vec<Value> = reg
        .by_state(filter.unwrap_or(ContractState::Active))
        .iter()
        .map(|c| {
            json!({
                "contract_id": c.contract_id,
                "state": format!("{:?}", c.state),
                "proposer": c.proposal.proposer_identity,
                "capabilities": c.proposal.requested_capabilities,
            })
        })
        .collect();

    Ok(json!({
        "count": contracts.len(),
        "contracts": contracts,
    }))
}

fn extract_id(params: &Value) -> Result<String, RpcError> {
    params["contract_id"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| RpcError {
            code: -32602,
            message: "missing 'contract_id' field".into(),
        })
}
