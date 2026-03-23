// SPDX-License-Identifier: AGPL-3.0-or-later
//! Push visualization data to `petalTongue` via JSON-RPC IPC.
//!
//! Springs discover `petalTongue` at runtime and push [`DataChannel`] payloads
//! without compile-time coupling. Uses the `visualization.render` and
//! `visualization.render.stream` JSON-RPC methods.

#[cfg(feature = "ipc")]
use crate::ipc::discover;
use crate::primal_names::PETALTONGUE;

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use super::types::{DataChannel, EcologyScenario, UiConfig};

/// Client for pushing visualization data to `petalTongue`.
pub struct PetalTonguePushClient {
    socket_path: PathBuf,
}

/// Error type for push operations.
#[derive(Debug)]
pub enum PushError {
    /// `petalTongue` socket not found at any candidate path.
    NotFound(String),
    /// Connection to `petalTongue` socket failed.
    ConnectionFailed(std::io::Error),
    /// JSON serialization error.
    SerializationError(String),
    /// JSON-RPC error response from `petalTongue`.
    RpcError {
        /// JSON-RPC error code.
        code: i64,
        /// Error message from `petalTongue`.
        message: String,
    },
}

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "petalTongue not found: {msg}"),
            Self::ConnectionFailed(e) => write!(f, "connection failed: {e}"),
            Self::SerializationError(e) => write!(f, "serialization error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
        }
    }
}

impl std::error::Error for PushError {}

/// Result type for push operations.
pub type PushResult<T> = Result<T, PushError>;

/// Standard discovery logic when `ipc` feature is disabled (no access to `discover` module).
/// Mirrors `discover::discover_socket` resolution order.
#[cfg(not(feature = "ipc"))]
fn discover_petaltongue_fallback(env_var: &str, primal: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg)
            .join(crate::primal_names::BIOMEOS)
            .join(format!("{primal}-default.sock"));
        if p.exists() {
            return Some(p);
        }
    }
    let fallback = std::env::temp_dir().join(format!("{primal}-default.sock"));
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

fn build_render_params(
    session_id: &str,
    title: &str,
    scenario: &EcologyScenario,
) -> serde_json::Value {
    let bindings: Vec<&DataChannel> = scenario
        .nodes
        .iter()
        .flat_map(|n| n.data_channels.iter())
        .collect();

    serde_json::json!({
        "session_id": session_id,
        "title": title,
        "bindings": bindings,
        "domain": scenario.domain,
    })
}

fn build_append_params(
    session_id: &str,
    binding_id: &str,
    x_values: &[f64],
    y_values: &[f64],
) -> serde_json::Value {
    serde_json::json!({
        "session_id": session_id,
        "binding_id": binding_id,
        "operation": {
            "type": "append",
            "x_values": x_values,
            "y_values": y_values,
        },
    })
}

fn build_gauge_params(session_id: &str, binding_id: &str, value: f64) -> serde_json::Value {
    serde_json::json!({
        "session_id": session_id,
        "binding_id": binding_id,
        "operation": {
            "type": "set_value",
            "value": value,
        },
    })
}

impl PetalTonguePushClient {
    /// Discover `petalTongue` socket at runtime using the standard discovery pattern.
    ///
    /// Resolution order (via [`crate::ipc::discover::discover_socket`]):
    /// 1. `{PRIMAL}_SOCKET` env var (e.g. `PETALTONGUE_SOCKET`)
    /// 2. `$XDG_RUNTIME_DIR/biomeos/{primal}-default.sock`
    /// 3. `<temp_dir>/{primal}-default.sock` (platform-agnostic fallback)
    ///
    /// # Errors
    ///
    /// Returns [`PushError::NotFound`] if no `petalTongue` socket is found.
    pub fn discover() -> PushResult<Self> {
        #[cfg(feature = "ipc")]
        {
            discover::discover_socket(&discover::socket_env_var(PETALTONGUE), PETALTONGUE)
                .map_or_else(
                    || Err(PushError::NotFound("no petalTongue socket found".into())),
                    |p| Ok(Self { socket_path: p }),
                )
        }
        #[cfg(not(feature = "ipc"))]
        {
            let env_var = format!("{}_SOCKET", PETALTONGUE.to_ascii_uppercase());
            let path = discover_petaltongue_fallback(&env_var, PETALTONGUE);
            match path {
                Some(p) => Ok(Self { socket_path: p }),
                None => Err(PushError::NotFound("no petalTongue socket found".into())),
            }
        }
    }

    /// Create client with an explicit socket path.
    #[must_use]
    pub const fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// Push a full visualization render request.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_render(
        &self,
        session_id: &str,
        title: &str,
        scenario: &EcologyScenario,
    ) -> PushResult<()> {
        let params = build_render_params(session_id, title, scenario);
        self.send_rpc("visualization.render", &params)?;
        Ok(())
    }

    /// Push a streaming append (add data points to a `TimeSeries` channel).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_append(
        &self,
        session_id: &str,
        binding_id: &str,
        x_values: &[f64],
        y_values: &[f64],
    ) -> PushResult<()> {
        let params = build_append_params(session_id, binding_id, x_values, y_values);
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Push a gauge value update.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_gauge_update(
        &self,
        session_id: &str,
        binding_id: &str,
        value: f64,
    ) -> PushResult<()> {
        let params = build_gauge_params(session_id, binding_id, value);
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Replace an entire channel with new data.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_replace(&self, session_id: &str, channel: &DataChannel) -> PushResult<()> {
        let params = serde_json::json!({
            "session_id": session_id,
            "operation": {
                "type": "replace",
                "channel": channel,
            },
        });
        self.send_rpc("visualization.render.stream", &params)?;
        Ok(())
    }

    /// Push a full render with domain theme and UI configuration.
    ///
    /// Follows healthSpring's `push_render_with_config` pattern for
    /// domain-specific theming (panel visibility, zoom, theme name).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_render_with_config(
        &self,
        session_id: &str,
        title: &str,
        scenario: &EcologyScenario,
        config: &UiConfig,
    ) -> PushResult<()> {
        let bindings: Vec<&DataChannel> = scenario
            .nodes
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let params = serde_json::json!({
            "session_id": session_id,
            "title": title,
            "bindings": bindings,
            "domain": scenario.domain,
            "ui_config": config,
        });
        self.send_rpc("visualization.render", &params)?;
        Ok(())
    }

    /// Push a render with explicit domain string for themed rendering.
    ///
    /// Shorthand for `push_render_with_config` when only the domain matters
    /// and default panel layout is acceptable.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection, serialization, or RPC failure.
    pub fn push_render_with_domain(
        &self,
        session_id: &str,
        title: &str,
        scenario: &EcologyScenario,
        domain: &str,
    ) -> PushResult<()> {
        let bindings: Vec<&DataChannel> = scenario
            .nodes
            .iter()
            .flat_map(|n| n.data_channels.iter())
            .collect();
        let params = serde_json::json!({
            "session_id": session_id,
            "title": title,
            "bindings": bindings,
            "domain": domain,
        });
        self.send_rpc("visualization.render", &params)?;
        Ok(())
    }

    /// Query petalTongue's supported capabilities.
    ///
    /// Returns the raw JSON response containing supported channel types,
    /// streaming modes, and feature flags.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn query_capabilities(&self) -> PushResult<serde_json::Value> {
        let params = serde_json::json!({});
        self.send_rpc("visualization.capabilities", &params)
    }

    /// Subscribe to interaction events from petalTongue.
    ///
    /// Registers interest in user interactions (clicks, selections, zooms)
    /// on the specified session. Returns the subscription acknowledgment.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn subscribe_interactions(&self, session_id: &str) -> PushResult<serde_json::Value> {
        let params = serde_json::json!({
            "session_id": session_id,
        });
        self.send_rpc("visualization.interact.subscribe", &params)
    }

    /// Dismiss (close) a visualization session.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn dismiss_session(&self, session_id: &str) -> PushResult<()> {
        let params = serde_json::json!({
            "session_id": session_id,
        });
        self.send_rpc("visualization.dismiss", &params)?;
        Ok(())
    }

    fn send_rpc(&self, method: &str, params: &serde_json::Value) -> PushResult<serde_json::Value> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        });

        let payload = serde_json::to_vec(&request)
            .map_err(|e| PushError::SerializationError(e.to_string()))?;

        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(PushError::ConnectionFailed)?;
        stream
            .write_all(&payload)
            .map_err(PushError::ConnectionFailed)?;
        stream
            .write_all(b"\n")
            .map_err(PushError::ConnectionFailed)?;
        stream.flush().map_err(PushError::ConnectionFailed)?;

        let mut buf = vec![0u8; 65_536];
        let n = stream.read(&mut buf).map_err(PushError::ConnectionFailed)?;

        let response: serde_json::Value = serde_json::from_slice(&buf[..n])
            .map_err(|e| PushError::SerializationError(e.to_string()))?;

        if let Some(error) = response.get("error") {
            return Err(PushError::RpcError {
                code: error
                    .get("code")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(-1),
                message: error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }

        Ok(response)
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
    fn discover_returns_not_found_when_no_socket() {
        temp_env::with_vars(
            [
                ("PETALTONGUE_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", Some("/nonexistent_xdg_wetspring_test")),
            ],
            || {
                let result = PetalTonguePushClient::discover();
                assert!(result.is_err());
            },
        );
    }

    #[test]
    fn new_creates_client_with_path() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test.sock");
        let client = PetalTonguePushClient::new(sock_path.clone());
        assert_eq!(client.socket_path, sock_path);
    }

    #[test]
    fn build_render_params_produces_valid_json() {
        let scenario = EcologyScenario {
            name: "test".into(),
            description: "test scenario".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        };
        let params = build_render_params("s1", "Test", &scenario);
        assert_eq!(params["session_id"], "s1");
        assert_eq!(params["title"], "Test");
        assert_eq!(params["domain"], "ecology");
    }

    #[test]
    fn build_append_params_structure() {
        let params = build_append_params("s1", "ch1", &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(params["binding_id"], "ch1");
        assert_eq!(params["operation"]["type"], "append");
    }

    #[test]
    fn build_gauge_params_structure() {
        let params = build_gauge_params("s1", "g1", 42.0);
        assert_eq!(params["binding_id"], "g1");
        assert_eq!(params["operation"]["type"], "set_value");
    }

    #[test]
    fn build_render_with_domain_params() {
        let scenario = EcologyScenario {
            name: "test".into(),
            description: "desc".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test.sock");
        let client = PetalTonguePushClient::new(sock_path);
        let _ = client.push_render_with_domain("s1", "Test", &scenario, "measurement");
    }

    #[test]
    fn ipc_buffer_is_64kb() {
        assert_eq!(65_536, 64 * 1024);
    }

    #[test]
    fn push_error_display() {
        let e = PushError::NotFound("test".into());
        assert!(e.to_string().contains("petalTongue not found"));

        let e = PushError::RpcError {
            code: -32600,
            message: "invalid".into(),
        };
        assert!(e.to_string().contains("-32600"));
    }
}
