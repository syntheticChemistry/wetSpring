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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Discover the Songbird Unix socket path.
///
/// Returns `None` if no Songbird socket is found (standalone mode).
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SONGBIRD_SOCKET") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join("biomeos/songbird-default.sock");
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join("songbird-default.sock");
    if fallback.exists() {
        return Some(fallback);
    }

    None
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
    let ws_path = wetspring_socket.display();
    let version = env!("CARGO_PKG_VERSION");
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"discovery.register","params":{{"primal":"wetspring","socket":"{ws_path}","capabilities":["science","biodiversity","spectral","metagenomics","qs_model"],"version":"{version}"}},"id":1}}"#,
    );

    let response = rpc_call(songbird_socket, &request)?;
    if response.contains("\"error\"") {
        Err(crate::error::Error::Ipc(format!(
            "Songbird registration rejected: {}",
            &response[..response.len().min(200)]
        )))
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
    let request =
        r#"{"jsonrpc":"2.0","method":"discovery.heartbeat","params":{"primal":"wetspring"},"id":2}"#;
    let response = rpc_call(songbird_socket, request)?;
    if response.contains("\"error\"") {
        Err(crate::error::Error::Ipc(format!(
            "Songbird heartbeat failed: {}",
            &response[..response.len().min(200)]
        )))
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
            Ok(()) => eprintln!("[wetspring-server] registered with Songbird"),
            Err(e) => eprintln!("[wetspring-server] Songbird registration failed: {e}"),
        }

        while !shutdown_clone.load(Ordering::Relaxed) {
            std::thread::sleep(HEARTBEAT_INTERVAL);
            if shutdown_clone.load(Ordering::Relaxed) {
                break;
            }
            if let Err(e) = heartbeat(&songbird_socket) {
                eprintln!("[wetspring-server] heartbeat failed: {e}");
                match register(&songbird_socket, &wetspring_socket) {
                    Ok(()) => eprintln!("[wetspring-server] re-registered with Songbird"),
                    Err(e2) => eprintln!("[wetspring-server] re-registration failed: {e2}"),
                }
            }
        }

        eprintln!("[wetspring-server] heartbeat loop stopped");
    });

    shutdown
}

/// Send a JSON-RPC request to Songbird and read the response.
fn rpc_call(socket: &Path, request: &str) -> crate::error::Result<String> {
    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| crate::error::Error::Ipc(format!("invalid socket path: {e}")))?;

    let stream = UnixStream::connect_addr(&addr).map_err(|e| {
        crate::error::Error::Ipc(format!("Songbird connect {}: {e}", socket.display()))
    })?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| crate::error::Error::Ipc(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| crate::error::Error::Ipc(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| crate::error::Error::Ipc(format!("write to Songbird: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| crate::error::Error::Ipc(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| crate::error::Error::Ipc(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| crate::error::Error::Ipc(format!("read from Songbird: {e}")))?;

    if line.is_empty() {
        return Err(crate::error::Error::Ipc(
            "Songbird returned empty response".to_string(),
        ));
    }

    Ok(line)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn discover_socket_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("songbird.sock");
        std::fs::write(&sock, "").unwrap();

        // SAFETY: single-threaded test; no other thread reads SONGBIRD_SOCKET.
        unsafe { std::env::set_var("SONGBIRD_SOCKET", sock.to_str().unwrap()) };
        let found = discover_socket();
        unsafe { std::env::remove_var("SONGBIRD_SOCKET") };

        assert_eq!(found, Some(sock));
    }

    #[test]
    fn register_nonexistent_socket_errors() {
        let bad_path = PathBuf::from("/tmp/wetspring_test_no_songbird.sock");
        let ws_path = PathBuf::from("/tmp/wetspring.sock");
        let err = register(&bad_path, &ws_path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Songbird connect") || msg.contains("invalid socket"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn heartbeat_nonexistent_socket_errors() {
        let bad_path = PathBuf::from("/tmp/wetspring_test_no_songbird_hb.sock");
        let err = heartbeat(&bad_path).unwrap_err();
        assert!(err.to_string().contains("Songbird connect") || err.to_string().contains("invalid socket"));
    }
}
