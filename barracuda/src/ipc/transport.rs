// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport abstraction for IPC connections.
//!
//! Supports Unix domain sockets (default), TCP for cross-gate communication,
//! and mesh relay via Songbird federation. [`TransportEndpoint`] is the
//! ecosystem-standard wire type — launcher/Tower Atomic injects it via the
//! `TRANSPORT_ENDPOINT` env var so primals never self-select transport.
//!
//! [`unix_jsonrpc_line`] and [`tcp_jsonrpc_line`] implement newline-delimited
//! JSON-RPC 2.0 client calls to peer primals.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default timeout for client JSON-RPC over Unix sockets to peer primals
/// (toadStool, sweetGrass, …) when not using workload-specific limits.
pub const UNIX_JSONRPC_TIMEOUT: Duration = super::timeouts::STANDARD_RPC;

/// Send one newline-terminated JSON-RPC request and read one response line
/// over a Unix domain socket.
///
/// # Errors
///
/// Returns `Err(String)` if the socket connection fails or the RPC transport errors.
pub fn unix_jsonrpc_line(socket: &Path, request_line: &str) -> Result<String, String> {
    let stream =
        UnixStream::connect(socket).map_err(|e| format!("connect {}: {e}", socket.display()))?;

    stream
        .set_read_timeout(Some(UNIX_JSONRPC_TIMEOUT))
        .map_err(|e| format!("set read timeout: {e}"))?;
    stream
        .set_write_timeout(Some(UNIX_JSONRPC_TIMEOUT))
        .map_err(|e| format!("set write timeout: {e}"))?;

    super::ribocipher::send_clear_signal(&stream)
        .map_err(|e| format!("riboCipher signal: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request_line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response from peer".to_string());
    }

    Ok(line)
}

/// Send one newline-terminated JSON-RPC request and read one response line
/// over TCP. Used for cross-gate communication where Unix sockets are not
/// available.
///
/// # Errors
///
/// Returns `Err(String)` if the TCP connection fails or the RPC transport errors.
pub fn tcp_jsonrpc_line(addr: &str, request_line: &str) -> Result<String, String> {
    let socket_addr = addr
        .to_socket_addrs()
        .map_err(|e| format!("resolve {addr}: {e}"))?
        .next()
        .ok_or_else(|| format!("no addresses resolved for {addr}"))?;

    let stream = TcpStream::connect_timeout(&socket_addr, UNIX_JSONRPC_TIMEOUT)
        .map_err(|e| format!("tcp connect {addr}: {e}"))?;

    stream
        .set_read_timeout(Some(UNIX_JSONRPC_TIMEOUT))
        .map_err(|e| format!("set read timeout: {e}"))?;
    stream
        .set_write_timeout(Some(UNIX_JSONRPC_TIMEOUT))
        .map_err(|e| format!("set write timeout: {e}"))?;

    super::ribocipher::send_clear_signal_tcp(&stream)
        .map_err(|e| format!("riboCipher signal: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request_line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response from tcp peer".to_string());
    }

    Ok(line)
}

/// Send a JSON-RPC request over the appropriate transport.
///
/// `MeshRelay` endpoints are not directly connectable — the caller
/// should route through Songbird `capability.call` instead. This
/// function returns an error for relay endpoints to keep the type
/// system honest.
///
/// # Errors
///
/// Returns `Err(String)` on connection, protocol, or unsupported transport errors.
pub fn jsonrpc_line(transport: &Transport, request_line: &str) -> Result<String, String> {
    match transport {
        Transport::Unix(path) => unix_jsonrpc_line(path, request_line),
        Transport::Tcp(addr) => tcp_jsonrpc_line(addr, request_line),
        Transport::MeshRelay { peer_id, capability } => Err(format!(
            "mesh_relay({peer_id}/{capability}) is not directly connectable — \
             route via Songbird capability.call"
        )),
    }
}

/// Supported transport types for the Primal IPC Protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Unix domain socket at a filesystem path.
    Unix(PathBuf),
    /// TCP socket at `host:port` for cross-gate communication.
    Tcp(String),
    /// Mesh relay — cross-gate routing via local Songbird `capability.call`.
    /// Not directly connectable; callers must forward through Songbird.
    MeshRelay {
        /// Remote gate node ID (e.g. `"eastGate"`).
        peer_id: String,
        /// Capability being routed.
        capability: String,
    },
}

impl Transport {
    /// Resolve the transport from environment configuration.
    ///
    /// Checks `{ENV_VAR}_TCP` for a TCP address first (cross-gate),
    /// then falls back to Unix domain socket discovery (local gate).
    #[must_use]
    pub fn resolve(env_var: &str, primal: &str) -> Self {
        let tcp_var = format!("{env_var}_TCP");
        if let Ok(addr) = std::env::var(&tcp_var) {
            if !addr.is_empty() {
                return Self::Tcp(addr);
            }
        }
        Self::Unix(super::discover::resolve_bind_path(env_var, primal))
    }

    /// The filesystem path for Unix transports, `None` for TCP/relay.
    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Unix(p) => Some(p),
            Self::Tcp(_) | Self::MeshRelay { .. } => None,
        }
    }

    /// The TCP address for TCP transports, `None` for Unix/relay.
    #[must_use]
    pub fn tcp_addr(&self) -> Option<&str> {
        match self {
            Self::Tcp(addr) => Some(addr),
            Self::Unix(_) | Self::MeshRelay { .. } => None,
        }
    }

    /// Whether this transport requires Songbird relay (cross-gate mesh).
    #[must_use]
    pub const fn is_mesh_relay(&self) -> bool {
        matches!(self, Self::MeshRelay { .. })
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
            Self::Tcp(addr) => write!(f, "tcp:{addr}"),
            Self::MeshRelay { peer_id, capability } => {
                write!(f, "mesh_relay:{peer_id}/{capability}")
            }
        }
    }
}

// ── Ecosystem-standard transport injection ──────────────────────────

/// Env var name for launcher-injected transport endpoint.
pub const TRANSPORT_ENDPOINT_ENV: &str = "TRANSPORT_ENDPOINT";

/// Ecosystem-standard transport endpoint for launcher injection.
///
/// Injected by the launcher or Tower Atomic via `TRANSPORT_ENDPOINT`.
/// Primals accept this at startup instead of self-binding — the primal
/// never chooses its own transport.
///
/// Wire format (JSON, serde internally tagged):
/// ```json
/// {"transport":"uds","path":"/run/membrane/wetspring.sock"}
/// {"transport":"tcp","host":"127.0.0.1","port":9100}
/// {"transport":"mesh_relay","peer_id":"strand-gate","capability":"science"}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum TransportEndpoint {
    /// Unix domain socket (preferred on Linux — local primal on same host).
    Uds {
        /// Absolute path to the socket file.
        path: PathBuf,
    },
    /// TCP socket (loopback or network — cross-gate).
    Tcp {
        /// Hostname or IP address.
        host: String,
        /// Port number.
        port: u16,
    },
    /// Mesh relay via Songbird federation (cross-gate, cross-WAN).
    MeshRelay {
        /// Mesh peer identifier (e.g. `"strand-gate"`).
        peer_id: String,
        /// Capability domain being relayed.
        capability: String,
    },
}

impl TransportEndpoint {
    /// Parse from the `TRANSPORT_ENDPOINT` env var (JSON string).
    ///
    /// # Errors
    ///
    /// Returns `NotSet` if the env var is absent, `InvalidJson` if the
    /// value doesn't match the expected tagged format.
    pub fn from_env() -> Result<Self, TransportEndpointError> {
        let raw = std::env::var(TRANSPORT_ENDPOINT_ENV)
            .map_err(|_| TransportEndpointError::NotSet)?;
        serde_json::from_str(&raw).map_err(TransportEndpointError::InvalidJson)
    }

    /// Convert to the internal [`Transport`] used by `jsonrpc_line`.
    ///
    /// `MeshRelay` has no direct transport — it must be resolved through
    /// Songbird, so this returns `None` for relay endpoints.
    #[must_use]
    pub fn to_transport(&self) -> Option<Transport> {
        match self {
            Self::Uds { path } => Some(Transport::Unix(path.clone())),
            Self::Tcp { host, port } => Some(Transport::Tcp(format!("{host}:{port}"))),
            Self::MeshRelay { .. } => None,
        }
    }

    /// Convert to [`Transport`], including `MeshRelay` as a first-class
    /// transport variant for topology-aware routing (Wave 107 M1).
    ///
    /// Unlike [`to_transport`], this never returns `None` — mesh relay
    /// endpoints become `Transport::MeshRelay` which callers must route
    /// through local Songbird `capability.call`.
    #[must_use]
    pub fn to_transport_or_relay(&self) -> Option<Transport> {
        match self {
            Self::Uds { path } => Some(Transport::Unix(path.clone())),
            Self::Tcp { host, port } => Some(Transport::Tcp(format!("{host}:{port}"))),
            Self::MeshRelay {
                peer_id,
                capability,
            } => Some(Transport::MeshRelay {
                peer_id: peer_id.clone(),
                capability: capability.clone(),
            }),
        }
    }
}

impl std::fmt::Display for TransportEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uds { path } => write!(f, "uds:{}", path.display()),
            Self::Tcp { host, port } => write!(f, "tcp:{host}:{port}"),
            Self::MeshRelay { peer_id, capability } => {
                write!(f, "mesh_relay:{peer_id}/{capability}")
            }
        }
    }
}

/// Errors from [`TransportEndpoint`] env-var parsing.
#[derive(Debug)]
pub enum TransportEndpointError {
    /// `TRANSPORT_ENDPOINT` env var is not set.
    NotSet,
    /// `TRANSPORT_ENDPOINT` contains invalid JSON.
    InvalidJson(serde_json::Error),
}

impl std::fmt::Display for TransportEndpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSet => f.write_str("`TRANSPORT_ENDPOINT` environment variable is not set"),
            Self::InvalidJson(e) => write!(f, "invalid `TRANSPORT_ENDPOINT` JSON: {e}"),
        }
    }
}

impl std::error::Error for TransportEndpointError {}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_unix() {
        temp_env::with_vars(
            [
                ("WETSPRING_TRANSPORT_TEST", None::<&str>),
                ("WETSPRING_TRANSPORT_TEST_TCP", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let t = Transport::resolve("WETSPRING_TRANSPORT_TEST", "test_primal");
                assert!(matches!(t, Transport::Unix(_)));
                assert!(t.path().is_some());
                assert!(t.tcp_addr().is_none());
            },
        );
    }

    #[test]
    fn resolve_prefers_tcp_when_set() {
        temp_env::with_vars(
            [("WETSPRING_TCP_TEST_TCP", Some("127.0.0.1:9200"))],
            || {
                let t = Transport::resolve("WETSPRING_TCP_TEST", "test_primal");
                assert!(matches!(t, Transport::Tcp(_)));
                assert_eq!(t.tcp_addr(), Some("127.0.0.1:9200"));
                assert!(t.path().is_none());
            },
        );
    }

    #[test]
    fn display_format_unix() {
        let sock_path = crate::ipc::test_socket_path("transport_display_format");
        crate::ipc::cleanup_test_socket(&sock_path);
        let t = Transport::Unix(sock_path.clone());
        assert_eq!(t.to_string(), format!("unix:{}", sock_path.display()));
        crate::ipc::cleanup_test_socket(&sock_path);
    }

    #[test]
    fn display_format_tcp() {
        let t = Transport::Tcp("10.0.0.5:9100".to_string());
        assert_eq!(t.to_string(), "tcp:10.0.0.5:9100");
    }

    #[test]
    fn jsonrpc_line_unix_no_server() {
        let sock_path = crate::ipc::test_socket_path("transport_jsonrpc_noserver");
        crate::ipc::cleanup_test_socket(&sock_path);
        let t = Transport::Unix(sock_path.clone());
        let result = jsonrpc_line(&t, r#"{"jsonrpc":"2.0","method":"health.ping","id":1}"#);
        assert!(result.is_err());
        crate::ipc::cleanup_test_socket(&sock_path);
    }

    #[test]
    fn jsonrpc_line_tcp_no_server() {
        let t = Transport::Tcp("127.0.0.1:1".to_string());
        let result = jsonrpc_line(&t, r#"{"jsonrpc":"2.0","method":"health.ping","id":1}"#);
        assert!(result.is_err());
    }

    // ── TransportEndpoint tests ──

    #[test]
    fn transport_endpoint_serde_uds() {
        let ep = TransportEndpoint::Uds {
            path: PathBuf::from("/run/membrane/wetspring.sock"),
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains(r#""transport":"uds""#));
        assert!(json.contains(r#""path":"/run/membrane/wetspring.sock""#));

        let parsed: TransportEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ep);
    }

    #[test]
    fn transport_endpoint_serde_tcp() {
        let ep = TransportEndpoint::Tcp {
            host: "192.168.1.173".to_string(),
            port: 9100,
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains(r#""transport":"tcp""#));

        let parsed: TransportEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ep);
    }

    #[test]
    fn transport_endpoint_serde_mesh_relay() {
        let json = r#"{"transport":"mesh_relay","peer_id":"strand-gate","capability":"science"}"#;
        let ep: TransportEndpoint = serde_json::from_str(json).unwrap();
        assert!(matches!(ep, TransportEndpoint::MeshRelay { .. }));
    }

    #[test]
    fn transport_endpoint_to_transport_uds() {
        let ep = TransportEndpoint::Uds {
            path: PathBuf::from("/tmp/test.sock"),
        };
        let t = ep.to_transport().unwrap();
        assert!(matches!(t, Transport::Unix(_)));
    }

    #[test]
    fn transport_endpoint_to_transport_tcp() {
        let ep = TransportEndpoint::Tcp {
            host: "10.0.0.1".to_string(),
            port: 7700,
        };
        let t = ep.to_transport().unwrap();
        assert_eq!(t.tcp_addr(), Some("10.0.0.1:7700"));
    }

    #[test]
    fn transport_endpoint_to_transport_mesh_relay_is_none() {
        let ep = TransportEndpoint::MeshRelay {
            peer_id: "east-gate".to_string(),
            capability: "compute".to_string(),
        };
        assert!(ep.to_transport().is_none());
    }

    #[test]
    fn transport_endpoint_to_transport_or_relay_uds() {
        let ep = TransportEndpoint::Uds {
            path: PathBuf::from("/tmp/test.sock"),
        };
        let t = ep.to_transport_or_relay().unwrap();
        assert!(matches!(t, Transport::Unix(_)));
    }

    #[test]
    fn transport_endpoint_to_transport_or_relay_tcp() {
        let ep = TransportEndpoint::Tcp {
            host: "10.0.0.1".to_string(),
            port: 7700,
        };
        let t = ep.to_transport_or_relay().unwrap();
        assert_eq!(t.tcp_addr(), Some("10.0.0.1:7700"));
    }

    #[test]
    fn transport_endpoint_to_transport_or_relay_mesh() {
        let ep = TransportEndpoint::MeshRelay {
            peer_id: "eastGate".to_string(),
            capability: "science".to_string(),
        };
        let t = ep.to_transport_or_relay().unwrap();
        assert!(t.is_mesh_relay());
        assert!(t.path().is_none());
        assert!(t.tcp_addr().is_none());
        assert_eq!(t.to_string(), "mesh_relay:eastGate/science");
    }

    #[test]
    fn transport_mesh_relay_display() {
        let t = Transport::MeshRelay {
            peer_id: "golgiBody".to_string(),
            capability: "security".to_string(),
        };
        assert_eq!(t.to_string(), "mesh_relay:golgiBody/security");
    }

    #[test]
    fn transport_mesh_relay_jsonrpc_returns_error() {
        let t = Transport::MeshRelay {
            peer_id: "peer".to_string(),
            capability: "cap".to_string(),
        };
        let result = jsonrpc_line(&t, r#"{"jsonrpc":"2.0","method":"test","id":1}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mesh_relay"));
    }

    #[test]
    fn transport_is_mesh_relay() {
        let unix = Transport::Unix(PathBuf::from("/tmp/t.sock"));
        assert!(!unix.is_mesh_relay());
        let tcp = Transport::Tcp("127.0.0.1:9000".to_string());
        assert!(!tcp.is_mesh_relay());
        let relay = Transport::MeshRelay {
            peer_id: "p".to_string(),
            capability: "c".to_string(),
        };
        assert!(relay.is_mesh_relay());
    }

    #[test]
    fn transport_endpoint_from_env_not_set() {
        temp_env::with_vars([("TRANSPORT_ENDPOINT", None::<&str>)], || {
            let result = TransportEndpoint::from_env();
            assert!(matches!(result, Err(TransportEndpointError::NotSet)));
        });
    }

    #[test]
    fn transport_endpoint_from_env_valid() {
        temp_env::with_vars(
            [(
                "TRANSPORT_ENDPOINT",
                Some(r#"{"transport":"tcp","host":"127.0.0.1","port":9200}"#),
            )],
            || {
                let ep = TransportEndpoint::from_env().unwrap();
                assert!(matches!(ep, TransportEndpoint::Tcp { port: 9200, .. }));
            },
        );
    }

    #[test]
    fn transport_endpoint_from_env_invalid_json() {
        temp_env::with_vars(
            [("TRANSPORT_ENDPOINT", Some("not-json"))],
            || {
                let result = TransportEndpoint::from_env();
                assert!(matches!(result, Err(TransportEndpointError::InvalidJson(_))));
            },
        );
    }

    #[test]
    fn transport_endpoint_display() {
        let uds = TransportEndpoint::Uds {
            path: PathBuf::from("/run/membrane/ws.sock"),
        };
        assert_eq!(uds.to_string(), "uds:/run/membrane/ws.sock");

        let tcp = TransportEndpoint::Tcp {
            host: "10.0.0.5".to_string(),
            port: 9100,
        };
        assert_eq!(tcp.to_string(), "tcp:10.0.0.5:9100");

        let relay = TransportEndpoint::MeshRelay {
            peer_id: "sg".to_string(),
            capability: "security".to_string(),
        };
        assert_eq!(relay.to_string(), "mesh_relay:sg/security");
    }
}
