// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live composition health probing for `composition.science_health`.
//!
//! Replaces static `"deferred_check"` declarations with runtime socket
//! discovery and RPC liveness probes. All probes use a short timeout
//! and degrade gracefully — the health endpoint stays responsive even
//! when primals are absent or unresponsive.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write as _};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use super::discover;
use super::primal_names::{LOAMSPINE, NESTGATE, RHIZOCRYPT, SWEETGRASS};

const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

// ── Trio Status ─────────────────────────────────────────────────────

/// Per-component status for the Provenance Trio (`rhizoCrypt`, `loamSpine`, `sweetGrass`).
#[derive(Debug)]
pub struct TrioStatus {
    /// `rhizoCrypt` DAG/dehydration primal.
    pub rhizocrypt: ComponentStatus,
    /// `loamSpine` ledger/commit primal.
    pub loamspine: ComponentStatus,
    /// `sweetGrass` braid/attribution primal.
    pub sweetgrass: ComponentStatus,
}

/// Status of a single primal component discovered via socket probing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentStatus {
    /// Socket found and `health.liveness` responded positively.
    Live,
    /// Socket file exists but RPC did not succeed (timeout, parse error, etc.).
    Discovered,
    /// No socket file found at any standard location.
    Absent,
}

impl ComponentStatus {
    /// String representation for JSON serialization.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Discovered => "discovered",
            Self::Absent => "absent",
        }
    }
}

impl TrioStatus {
    /// Serialize to a JSON object with per-component status strings.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "rhizocrypt": self.rhizocrypt.as_str(),
            "loamspine": self.loamspine.as_str(),
            "sweetgrass": self.sweetgrass.as_str(),
            "summary": self.summary(),
        })
    }

    /// Overall trio summary: `"live"` if all live, `"partial"` if any live,
    /// `"discovered"` if any discovered but none live, `"absent"` if all absent.
    #[must_use]
    pub fn summary(&self) -> &'static str {
        let components = [self.rhizocrypt, self.loamspine, self.sweetgrass];
        let live_count = components.iter().filter(|c| **c == ComponentStatus::Live).count();
        let discovered_count = components
            .iter()
            .filter(|c| **c == ComponentStatus::Discovered)
            .count();

        if live_count == 3 {
            "live"
        } else if live_count > 0 {
            "partial"
        } else if discovered_count > 0 {
            "discovered"
        } else {
            "absent"
        }
    }
}

/// Probe all three Provenance Trio components.
#[must_use]
pub fn probe_trio_status() -> TrioStatus {
    TrioStatus {
        rhizocrypt: probe_primal(RHIZOCRYPT),
        loamspine: probe_primal(LOAMSPINE),
        sweetgrass: probe_primal(SWEETGRASS),
    }
}

// ── NestGate Status ─────────────────────────────────────────────────

/// Probe `NestGate` socket and liveness.
#[must_use]
pub fn probe_nestgate_status() -> ComponentStatus {
    probe_primal(NESTGATE)
}

// ── BiomeOS Status ──────────────────────────────────────────────────

/// Live status of the biomeOS Neural API.
#[derive(Debug)]
pub struct BiomeOsLiveStatus {
    /// Whether the Neural API socket was found and responded.
    pub socket_status: ComponentStatus,
    /// Number of primals reported by `primal.list` (Wave 20), if available.
    pub primal_count: Option<u64>,
}

impl BiomeOsLiveStatus {
    /// Serialize to a JSON object for inclusion in health responses.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "neural_api": self.socket_status.as_str(),
            "primal_count": self.primal_count,
        })
    }
}

/// Probe the biomeOS Neural API socket. If reachable, attempt `primal.list`
/// to get a live primal count.
#[must_use]
pub fn probe_biomeos_status() -> BiomeOsLiveStatus {
    let Some(socket_path) = super::provenance::neural_api_socket() else {
        return BiomeOsLiveStatus {
            socket_status: ComponentStatus::Absent,
            primal_count: None,
        };
    };

    let primal_count = try_primal_list(&socket_path);
    let rpc_reachable = primal_count.is_some() || try_health_liveness(&socket_path);
    let socket_status = if rpc_reachable {
        ComponentStatus::Live
    } else {
        ComponentStatus::Discovered
    };

    BiomeOsLiveStatus {
        socket_status,
        primal_count,
    }
}

// ── Schema Parity ───────────────────────────────────────────────────

/// Self-check: validate that our own `capability.list` response conforms
/// to the Wave 20 canonical schema (`capabilities` array + `count` field).
#[derive(Debug)]
#[expect(clippy::struct_excessive_bools, reason = "each bool maps to a distinct schema check")]
pub struct SchemaParity {
    /// Overall conformance: all three sub-checks pass.
    pub conformant: bool,
    /// `capabilities` key is present and is a JSON array.
    pub has_capabilities_array: bool,
    /// `count` key is present and is a JSON number.
    pub has_count: bool,
    /// `count` value equals `capabilities` array length.
    pub count_matches: bool,
}

impl SchemaParity {
    /// Serialize to a JSON object for inclusion in health responses.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "conformant": self.conformant,
            "has_capabilities_array": self.has_capabilities_array,
            "has_count": self.has_count,
            "count_matches": self.count_matches,
        })
    }
}

/// Validate our own `capability.list` response against Wave 20 canonical shape.
pub fn probe_schema_parity() -> SchemaParity {
    let Ok(response) = super::handlers::handle_capability_list() else {
        return SchemaParity {
            conformant: false,
            has_capabilities_array: false,
            has_count: false,
            count_matches: false,
        };
    };

    let has_capabilities_array = response
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some();

    let has_count = response.get("count").and_then(Value::as_u64).is_some();

    let count_matches = match (
        response.get("capabilities").and_then(Value::as_array),
        response.get("count").and_then(Value::as_u64),
    ) {
        (Some(arr), Some(count)) => arr.len() as u64 == count,
        _ => false,
    };

    let conformant = has_capabilities_array && has_count && count_matches;

    SchemaParity {
        conformant,
        has_capabilities_array,
        has_count,
        count_matches,
    }
}

// ── Internal Helpers ────────────────────────────────────────────────

/// Probe a single primal by name: discover socket, then try `health.liveness`.
fn probe_primal(primal: &str) -> ComponentStatus {
    let env_var = discover::socket_env_var(primal);
    let Some(socket_path) = discover::discover_socket(&env_var, primal) else {
        return ComponentStatus::Absent;
    };

    if try_health_liveness(&socket_path) {
        ComponentStatus::Live
    } else {
        ComponentStatus::Discovered
    }
}

/// Attempt a `health.liveness` RPC with a short timeout. Returns `true` if
/// the primal responds with any valid JSON-RPC result.
fn try_health_liveness(socket_path: &Path) -> bool {
    let request = r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1}"#;
    probe_rpc(socket_path, request).is_some()
}

/// Attempt `primal.list` via the Neural API. Returns the primal count if
/// biomeOS responds with a valid `{ "primals": [...], "count": N }` shape.
fn try_primal_list(socket_path: &Path) -> Option<u64> {
    let request = r#"{"jsonrpc":"2.0","method":"primal.list","params":{},"id":1}"#;
    let response = probe_rpc(socket_path, request)?;
    response
        .get("result")
        .and_then(|r| r.get("count"))
        .and_then(Value::as_u64)
}

/// Low-level probe: connect to a Unix socket, send one JSON-RPC request,
/// read one response line, parse as JSON. Returns `None` on any failure.
fn probe_rpc(socket_path: &Path, request: &str) -> Option<Value> {
    let stream = UnixStream::connect(socket_path).ok()?;
    stream.set_read_timeout(Some(HEALTH_PROBE_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(HEALTH_PROBE_TIMEOUT)).ok()?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer.write_all(request.as_bytes()).ok()?;
    writer.write_all(b"\n").ok()?;
    writer.flush().ok()?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;

    if line.is_empty() {
        return None;
    }

    serde_json::from_str(&line).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trio_absent_when_no_sockets() {
        let status = probe_trio_status();
        assert_eq!(status.rhizocrypt, ComponentStatus::Absent);
        assert_eq!(status.loamspine, ComponentStatus::Absent);
        assert_eq!(status.sweetgrass, ComponentStatus::Absent);
        assert_eq!(status.summary(), "absent");
    }

    #[test]
    fn nestgate_absent_when_no_socket() {
        assert_eq!(probe_nestgate_status(), ComponentStatus::Absent);
    }

    #[test]
    fn biomeos_absent_when_no_socket() {
        let status = probe_biomeos_status();
        assert_eq!(status.socket_status, ComponentStatus::Absent);
        assert!(status.primal_count.is_none());
    }

    #[test]
    fn schema_parity_self_check() {
        let parity = probe_schema_parity();
        assert!(parity.conformant, "own capability.list must pass Wave 20 schema check");
        assert!(parity.has_capabilities_array);
        assert!(parity.has_count);
        assert!(parity.count_matches);
    }

    #[test]
    fn trio_status_summary_logic() {
        let all_live = TrioStatus {
            rhizocrypt: ComponentStatus::Live,
            loamspine: ComponentStatus::Live,
            sweetgrass: ComponentStatus::Live,
        };
        assert_eq!(all_live.summary(), "live");

        let partial = TrioStatus {
            rhizocrypt: ComponentStatus::Live,
            loamspine: ComponentStatus::Absent,
            sweetgrass: ComponentStatus::Absent,
        };
        assert_eq!(partial.summary(), "partial");

        let discovered_only = TrioStatus {
            rhizocrypt: ComponentStatus::Discovered,
            loamspine: ComponentStatus::Absent,
            sweetgrass: ComponentStatus::Absent,
        };
        assert_eq!(discovered_only.summary(), "discovered");

        let absent = TrioStatus {
            rhizocrypt: ComponentStatus::Absent,
            loamspine: ComponentStatus::Absent,
            sweetgrass: ComponentStatus::Absent,
        };
        assert_eq!(absent.summary(), "absent");
    }

    #[test]
    fn component_status_as_str() {
        assert_eq!(ComponentStatus::Live.as_str(), "live");
        assert_eq!(ComponentStatus::Discovered.as_str(), "discovered");
        assert_eq!(ComponentStatus::Absent.as_str(), "absent");
    }

    #[test]
    fn trio_to_json_shape() {
        let status = probe_trio_status();
        let j = status.to_json();
        assert!(j.get("rhizocrypt").is_some());
        assert!(j.get("loamspine").is_some());
        assert!(j.get("sweetgrass").is_some());
        assert!(j.get("summary").is_some());
    }
}
