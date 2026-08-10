// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gossip injection for wetSpring science pipeline events.
//!
//! Emits lifecycle events to the swarmVine gossip mesh via `gossip.spread`
//! JSON-RPC. Cross-gate consumers observe these events for pipeline
//! coordination, provenance tracking, and composition orchestration.
//!
//! All calls degrade gracefully — gossip failures never block science.
//! Fire-and-forget: errors are logged via `tracing`, never propagated.
//!
//! # Events
//!
//! | Event | Trigger | Domain |
//! |-------|---------|--------|
//! | `PipelineComplete` | Full 16S rRNA pipeline finishes | `science` |
//! | `ValidationPass` | Science validation confirms results | `science` |
//! | `ProvenanceWitness` | Provenance record committed | `provenance` |
//! | `DataIngested` | New dataset processed into CAS | `data` |
//!
//! # Socket discovery
//!
//! Uses swarmVine's gossip relay socket:
//! 1. `GOSSIP_RELAY_SOCKET` env var (explicit override)
//! 2. `$BIOMEOS_SOCKET_DIR/swarmvine-{family_id}.sock`
//! 3. Standard cascade via [`super::discover`]

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};

use super::primal_names;

const GOSSIP_TIMEOUT: Duration = Duration::from_millis(200);
const GOSSIP_METHOD: &str = "gossip.spread";

/// Gossip event domains for wetSpring.
#[derive(Debug, Clone, Copy)]
pub enum Domain {
    /// Science pipeline lifecycle.
    Science,
    /// Provenance and audit trail.
    Provenance,
    /// Data ingest and CAS.
    Data,
}

impl Domain {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Science => "science",
            Self::Provenance => "provenance",
            Self::Data => "data",
        }
    }
}

/// Gossip event kinds emitted by wetSpring.
#[derive(Debug, Clone)]
pub enum GossipEvent {
    /// A full pipeline run completed (16S rRNA or similar).
    PipelineComplete {
        /// Unique identifier for this pipeline run.
        pipeline_id: String,
        /// Number of processing stages completed.
        stage_count: u32,
        /// Optional sample identifier for traceability.
        sample_id: Option<String>,
    },
    /// A science validation passed.
    ValidationPass {
        /// Unique identifier for this validation run.
        validation_id: String,
        /// Validation method name (e.g. "16s_chimera_check").
        method: String,
    },
    /// A provenance witness was committed.
    ProvenanceWitness {
        /// Provenance session identifier.
        session_id: String,
        /// Number of steps recorded in the session.
        step_count: u32,
    },
    /// A dataset was ingested into CAS.
    DataIngested {
        /// CAS-addressable dataset identifier.
        dataset_id: String,
        /// Number of records ingested.
        record_count: u64,
        /// Data format (e.g. "fastq", "fasta", "biom").
        format: String,
    },
}

impl GossipEvent {
    const fn domain(&self) -> Domain {
        match self {
            Self::PipelineComplete { .. } | Self::ValidationPass { .. } => Domain::Science,
            Self::ProvenanceWitness { .. } => Domain::Provenance,
            Self::DataIngested { .. } => Domain::Data,
        }
    }

    const fn kind(&self) -> &'static str {
        match self {
            Self::PipelineComplete { .. } => "PipelineComplete",
            Self::ValidationPass { .. } => "ValidationPass",
            Self::ProvenanceWitness { .. } => "ProvenanceWitness",
            Self::DataIngested { .. } => "DataIngested",
        }
    }

    fn payload(&self) -> Value {
        match self {
            Self::PipelineComplete {
                pipeline_id,
                stage_count,
                sample_id,
            } => json!({
                "kind": self.kind(),
                "pipeline_id": pipeline_id,
                "stage_count": stage_count,
                "sample_id": sample_id,
            }),
            Self::ValidationPass {
                validation_id,
                method,
            } => json!({
                "kind": self.kind(),
                "validation_id": validation_id,
                "method": method,
            }),
            Self::ProvenanceWitness {
                session_id,
                step_count,
            } => json!({
                "kind": self.kind(),
                "session_id": session_id,
                "step_count": step_count,
            }),
            Self::DataIngested {
                dataset_id,
                record_count,
                format,
            } => json!({
                "kind": self.kind(),
                "dataset_id": dataset_id,
                "record_count": record_count,
                "format": format,
            }),
        }
    }
}

/// Discover the gossip relay socket (swarmVine).
///
/// Returns `None` if no relay is reachable (standalone mode / mesh unavailable).
#[must_use]
pub fn discover_relay() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("GOSSIP_RELAY_SOCKET") {
        let p = PathBuf::from(&path);
        if super::discover::socket_is_alive(&p) {
            return Some(p);
        }
    }

    super::discover::discover_socket(&super::discover::socket_env_var("swarmvine"), "swarmvine")
}

/// Emit a gossip event to the mesh. Fire-and-forget.
///
/// Returns `true` if the event was successfully sent, `false` if the relay
/// is unavailable or the send failed. Never panics or blocks beyond timeout.
pub fn emit(event: &GossipEvent) -> bool {
    let Some(socket) = discover_relay() else {
        tracing::debug!(
            "gossip relay unavailable — event {} not propagated",
            event.kind()
        );
        return false;
    };

    emit_to(&socket, event)
}

/// Emit a gossip event to a specific relay socket. Fire-and-forget.
///
/// Useful when the relay socket is already known (cached from a previous
/// discovery) to avoid repeated discovery overhead.
pub fn emit_to(socket: &std::path::Path, event: &GossipEvent) -> bool {
    let request = json!({
        "jsonrpc": "2.0",
        "method": GOSSIP_METHOD,
        "params": {
            "source_primal": primal_names::SELF,
            "domain": event.domain().as_str(),
            "event": event.payload(),
        },
        "id": 1
    });

    let request_line = match serde_json::to_string(&request) {
        Ok(line) => line,
        Err(e) => {
            tracing::warn!("gossip serialize failed: {e}");
            return false;
        }
    };

    match send_fire_and_forget(socket, &request_line) {
        Ok(()) => {
            tracing::debug!(
                event = event.kind(),
                domain = event.domain().as_str(),
                "gossip event emitted"
            );
            true
        }
        Err(e) => {
            tracing::debug!("gossip emit failed (non-fatal): {e}");
            false
        }
    }
}

/// Send a JSON-RPC line to a socket without waiting for response.
/// Uses a short timeout to prevent blocking the caller.
fn send_fire_and_forget(socket: &std::path::Path, request_line: &str) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    let stream = UnixStream::connect(socket)
        .map_err(|e| format!("gossip connect {}: {e}", socket.display()))?;

    stream
        .set_write_timeout(Some(GOSSIP_TIMEOUT))
        .map_err(|e| format!("gossip set timeout: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request_line.as_bytes())
        .map_err(|e| format!("gossip write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("gossip write newline: {e}"))?;
    writer.flush().map_err(|e| format!("gossip flush: {e}"))?;

    Ok(())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn event_domains_correct() {
        let pipeline = GossipEvent::PipelineComplete {
            pipeline_id: "test".into(),
            stage_count: 3,
            sample_id: None,
        };
        assert_eq!(pipeline.domain().as_str(), "science");

        let witness = GossipEvent::ProvenanceWitness {
            session_id: "s1".into(),
            step_count: 5,
        };
        assert_eq!(witness.domain().as_str(), "provenance");

        let ingest = GossipEvent::DataIngested {
            dataset_id: "ds1".into(),
            record_count: 1000,
            format: "fastq".into(),
        };
        assert_eq!(ingest.domain().as_str(), "data");
    }

    #[test]
    fn event_payload_structure() {
        let event = GossipEvent::PipelineComplete {
            pipeline_id: "p123".into(),
            stage_count: 4,
            sample_id: Some("sample_abc".into()),
        };
        let payload = event.payload();
        assert_eq!(payload["kind"], "PipelineComplete");
        assert_eq!(payload["pipeline_id"], "p123");
        assert_eq!(payload["stage_count"], 4);
        assert_eq!(payload["sample_id"], "sample_abc");
    }

    #[test]
    fn validation_event_payload() {
        let event = GossipEvent::ValidationPass {
            validation_id: "v456".into(),
            method: "16s_chimera_check".into(),
        };
        let payload = event.payload();
        assert_eq!(payload["kind"], "ValidationPass");
        assert_eq!(payload["validation_id"], "v456");
        assert_eq!(payload["method"], "16s_chimera_check");
    }

    #[test]
    fn discover_relay_returns_none_without_socket() {
        temp_env::with_vars(
            [
                ("GOSSIP_RELAY_SOCKET", None::<&str>),
                ("BIOMEOS_SOCKET_DIR", None::<&str>),
                ("XDG_RUNTIME_DIR", Some("/nonexistent_gossip_test")),
            ],
            || {
                assert!(discover_relay().is_none());
            },
        );
    }

    #[test]
    fn emit_returns_false_without_relay() {
        temp_env::with_vars(
            [
                ("GOSSIP_RELAY_SOCKET", None::<&str>),
                ("BIOMEOS_SOCKET_DIR", None::<&str>),
                ("XDG_RUNTIME_DIR", Some("/nonexistent_gossip_test2")),
            ],
            || {
                let event = GossipEvent::PipelineComplete {
                    pipeline_id: "test".into(),
                    stage_count: 1,
                    sample_id: None,
                };
                assert!(!emit(&event));
            },
        );
    }

    #[test]
    fn emit_to_nonexistent_socket_returns_false() {
        let fake_socket = std::path::Path::new("/tmp/wetspring-gossip-nonexistent.sock");
        let event = GossipEvent::DataIngested {
            dataset_id: "ds".into(),
            record_count: 42,
            format: "fasta".into(),
        };
        assert!(!emit_to(fake_socket, &event));
    }

    #[test]
    fn gossip_wire_format_correct() {
        let event = GossipEvent::ProvenanceWitness {
            session_id: "sess_789".into(),
            step_count: 3,
        };

        let request = json!({
            "jsonrpc": "2.0",
            "method": GOSSIP_METHOD,
            "params": {
                "source_primal": primal_names::SELF,
                "domain": event.domain().as_str(),
                "event": event.payload(),
            },
            "id": 1
        });

        assert_eq!(request["method"], "gossip.spread");
        assert_eq!(request["params"]["source_primal"], "wetspring");
        assert_eq!(request["params"]["domain"], "provenance");
        assert_eq!(request["params"]["event"]["kind"], "ProvenanceWitness");
        assert_eq!(request["params"]["event"]["session_id"], "sess_789");
    }
}
