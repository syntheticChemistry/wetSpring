// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport abstraction for IPC connections.
//!
//! Currently supports Unix domain sockets only. TCP transport is documented
//! in the `wetspring_server` binary (`WETSPRING_TCP_ADDR`) but not yet
//! implemented. This module provides the abstraction layer for future
//! multi-transport support without changing the server or dispatch logic.
//!
//! [`unix_jsonrpc_line`] implements newline-delimited JSON-RPC 2.0 client
//! calls to peer primals (shared by lightweight IPC helpers).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Default timeout for client JSON-RPC over Unix sockets to peer primals
/// (toadStool, sweetGrass, …) when not using workload-specific limits.
pub const UNIX_JSONRPC_TIMEOUT: Duration = Duration::from_secs(10);

/// Send one newline-terminated JSON-RPC request and read one response line.
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

/// Supported transport types for the Primal IPC Protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Unix domain socket at a filesystem path.
    Unix(PathBuf),
}

impl Transport {
    /// Resolve the transport from environment configuration.
    ///
    /// Currently only resolves Unix domain sockets via [`super::discover`].
    #[must_use]
    pub fn resolve(env_var: &str, primal: &str) -> Self {
        Self::Unix(super::discover::resolve_bind_path(env_var, primal))
    }

    /// The filesystem path for Unix transports.
    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Unix(p) => Some(p),
        }
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
        }
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
    fn resolve_returns_unix() {
        temp_env::with_vars(
            [
                ("WETSPRING_TRANSPORT_TEST", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let t = Transport::resolve("WETSPRING_TRANSPORT_TEST", "test_primal");
                assert!(matches!(t, Transport::Unix(_)));
                assert!(t.path().is_some());
            },
        );
    }

    #[test]
    fn display_format() {
        let sock_path = crate::ipc::test_socket_path("transport_display_format");
        crate::ipc::cleanup_test_socket(&sock_path);
        let t = Transport::Unix(sock_path.clone());
        assert_eq!(t.to_string(), format!("unix:{}", sock_path.display()));
        crate::ipc::cleanup_test_socket(&sock_path);
    }
}
