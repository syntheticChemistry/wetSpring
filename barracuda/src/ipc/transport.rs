// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport abstraction for IPC connections.
//!
//! Supports Unix domain sockets (default) and TCP for cross-gate communication.
//! [`unix_jsonrpc_line`] and [`tcp_jsonrpc_line`] implement newline-delimited
//! JSON-RPC 2.0 client calls to peer primals.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
/// # Errors
///
/// Returns `Err(String)` on connection or protocol errors.
pub fn jsonrpc_line(transport: &Transport, request_line: &str) -> Result<String, String> {
    match transport {
        Transport::Unix(path) => unix_jsonrpc_line(path, request_line),
        Transport::Tcp(addr) => tcp_jsonrpc_line(addr, request_line),
    }
}

/// Supported transport types for the Primal IPC Protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Unix domain socket at a filesystem path.
    Unix(PathBuf),
    /// TCP socket at `host:port` for cross-gate communication.
    Tcp(String),
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

    /// The filesystem path for Unix transports, `None` for TCP.
    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Unix(p) => Some(p),
            Self::Tcp(_) => None,
        }
    }

    /// The TCP address for TCP transports, `None` for Unix.
    #[must_use]
    pub fn tcp_addr(&self) -> Option<&str> {
        match self {
            Self::Unix(_) => None,
            Self::Tcp(addr) => Some(addr),
        }
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
            Self::Tcp(addr) => write!(f, "tcp:{addr}"),
        }
    }
}

#[cfg(test)]
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
}
