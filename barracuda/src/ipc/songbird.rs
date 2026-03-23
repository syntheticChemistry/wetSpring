// SPDX-License-Identifier: AGPL-3.0-or-later
//! Songbird discovery registration for the wetSpring science primal.
//!
//! Registers wetSpring's capabilities with the Songbird discovery service
//! so biomeOS can route `capability.call` requests to this primal. Falls
//! back gracefully when Songbird is not running (standalone mode).
//!
//! # Protocol
//!
//! JSON-RPC 2.0, newline-delimited, over Unix domain socket.
//! Follows the same wire format as [`crate::ncbi::nestgate`].
//!
//! # Discovery
//!
//! Socket path resolution:
//! 1. `SONGBIRD_SOCKET` env var
//! 2. `$XDG_RUNTIME_DIR/biomeos/songbird-default.sock`
//! 3. `<temp_dir>/songbird-default.sock`

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Discover the Songbird Unix socket path.
///
/// Returns `None` if no Songbird socket is found (standalone mode).
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    super::discover::discover_socket(
        &super::discover::socket_env_var(super::primal_names::SONGBIRD),
        super::primal_names::SONGBIRD,
    )
}

/// Register wetSpring capabilities with Songbird.
///
/// Sends a `discovery.register` JSON-RPC call declaring the primal's
/// socket path and capability tags.
///
/// # Errors
///
/// Returns `Err` if Songbird is unreachable or the registration is rejected.
pub fn register(songbird_socket: &Path, wetspring_socket: &Path) -> crate::error::Result<()> {
    use crate::error::IpcError;

    let ws_path = wetspring_socket.display();
    let version = env!("CARGO_PKG_VERSION");
    let primal = crate::PRIMAL_NAME;
    let caps = super::handlers::CAPABILITIES;
    let caps_json: Vec<String> = caps.iter().map(|c| format!("\"{c}\"")).collect();
    let caps_str = caps_json.join(",");

    let niche = crate::niche::NICHE_NAME;
    let niche_desc = crate::niche::NICHE_DESCRIPTION;
    let dep_count = crate::niche::required_dependency_count();

    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"discovery.register","params":{{"primal":"{primal}","socket":"{ws_path}","capabilities":[{caps_str}],"version":"{version}","niche":"{niche}","niche_description":"{niche_desc}","required_dependencies":{dep_count}}},"id":1}}"#,
    );

    let response = rpc_call(songbird_socket, &request)?;
    if let Some((code, msg)) = super::protocol::extract_rpc_error(&response) {
        Err(IpcError::RpcReject {
            code,
            message: format!("Songbird registration: {msg}"),
        })?
    } else {
        Ok(())
    }
}

/// Send a heartbeat to Songbird to maintain registration.
///
/// # Errors
///
/// Returns `Err` if Songbird is unreachable or the heartbeat is rejected.
pub fn heartbeat(songbird_socket: &Path) -> crate::error::Result<()> {
    use crate::error::IpcError;

    let primal = crate::PRIMAL_NAME;
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"discovery.heartbeat","params":{{"primal":"{primal}"}},"id":2}}"#
    );
    let response = rpc_call(songbird_socket, &request)?;
    if let Some((code, msg)) = super::protocol::extract_rpc_error(&response) {
        Err(IpcError::RpcReject {
            code,
            message: format!("Songbird heartbeat: {msg}"),
        })?
    } else {
        Ok(())
    }
}

/// Start a background heartbeat loop that maintains Songbird registration.
///
/// The returned [`AtomicBool`] can be set to `true` to signal the loop to stop.
/// The loop performs an initial registration, then sends heartbeats at 30-second
/// intervals. On heartbeat failure, it attempts re-registration.
#[must_use]
pub fn start_heartbeat_loop(
    songbird_socket: PathBuf,
    wetspring_socket: PathBuf,
) -> Arc<AtomicBool> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);

    std::thread::spawn(move || {
        match register(&songbird_socket, &wetspring_socket) {
            Ok(()) => tracing::info!("registered with Songbird"),
            Err(e) => tracing::warn!(error = %e, "Songbird registration failed"),
        }

        while !shutdown_clone.load(Ordering::Relaxed) {
            std::thread::sleep(HEARTBEAT_INTERVAL);
            if shutdown_clone.load(Ordering::Relaxed) {
                break;
            }
            if let Err(e) = heartbeat(&songbird_socket) {
                tracing::warn!(error = %e, "heartbeat failed");
                match register(&songbird_socket, &wetspring_socket) {
                    Ok(()) => tracing::info!("re-registered with Songbird"),
                    Err(e2) => tracing::warn!(error = %e2, "re-registration failed"),
                }
            }
        }

        tracing::info!("heartbeat loop stopped");
    });

    shutdown
}

/// Send a JSON-RPC request to Songbird and read the response.
fn rpc_call(socket: &Path, request: &str) -> crate::error::Result<String> {
    use crate::error::IpcError;

    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| IpcError::SocketPath(format!("{}: {e}", socket.display())))?;

    let stream = UnixStream::connect_addr(&addr)
        .map_err(|e| IpcError::Connect(format!("Songbird {}: {e}", socket.display())))?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| IpcError::Transport(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| IpcError::Transport(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| IpcError::Transport(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| IpcError::Transport(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| IpcError::Transport(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| IpcError::Transport(format!("read: {e}")))?;

    if line.is_empty() {
        return Err(IpcError::EmptyResponse.into());
    }

    Ok(line)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn discover_socket_explicit() {
        let sock = crate::ipc::test_socket_path("songbird_discover_socket_explicit");
        crate::ipc::cleanup_test_socket(&sock);
        std::fs::write(&sock, "").unwrap();

        temp_env::with_var("SONGBIRD_SOCKET", Some(sock.to_str().unwrap()), || {
            let found = discover_socket();
            assert_eq!(found, Some(sock.clone()));
        });
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn register_nonexistent_socket_errors() {
        let bad_path = crate::ipc::test_socket_path("songbird_register_nonexistent_bad");
        let ws_path = crate::ipc::test_socket_path("songbird_register_nonexistent_ws");
        crate::ipc::cleanup_test_socket(&bad_path);
        crate::ipc::cleanup_test_socket(&ws_path);
        let err = register(&bad_path, &ws_path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") || msg.contains("socket"),
            "unexpected error: {msg}"
        );
        crate::ipc::cleanup_test_socket(&bad_path);
        crate::ipc::cleanup_test_socket(&ws_path);
    }

    #[test]
    fn heartbeat_nonexistent_socket_errors() {
        let bad_path = crate::ipc::test_socket_path("songbird_heartbeat_nonexistent");
        crate::ipc::cleanup_test_socket(&bad_path);
        let err = heartbeat(&bad_path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") || msg.contains("socket"),
            "unexpected error: {msg}"
        );
        crate::ipc::cleanup_test_socket(&bad_path);
    }
}
