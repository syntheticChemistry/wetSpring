// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unix socket server for the wetSpring science primal.
//!
//! Accepts connections on a Unix domain socket and handles JSON-RPC 2.0
//! requests (newline-delimited), dispatching to barracuda library functions.
//! Each connection is handled in its own thread.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::dispatch;
use super::metrics::Metrics;
use super::protocol;
#[cfg(test)]
use crate::tolerances;

const CONNECTION_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// wetSpring IPC server.
///
/// Listens on a Unix domain socket and handles JSON-RPC 2.0 requests,
/// dispatching to barracuda library functions for each registered capability.
pub struct Server {
    listener: UnixListener,
    socket_path: PathBuf,
    metrics: Arc<Metrics>,
}

impl Server {
    /// Bind to the default socket path following the Primal IPC Protocol.
    ///
    /// Discovery order:
    /// 1. `WETSPRING_SOCKET` env var (explicit override)
    /// 2. `$XDG_RUNTIME_DIR/biomeos/wetspring-default.sock`
    /// 3. `<temp_dir>/wetspring-default.sock` (platform-agnostic fallback)
    ///
    /// # Errors
    ///
    /// Returns `Err` if the socket directory cannot be created or the bind fails.
    pub fn bind_default() -> crate::error::Result<Self> {
        let path = resolve_bind_path();
        Self::bind(&path)
    }

    /// Bind to a specific socket path.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the directory cannot be created, a stale socket cannot
    /// be removed, or the bind fails.
    pub fn bind(path: &Path) -> crate::error::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::error::Error::Ipc(format!(
                    "cannot create socket directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        // Remove stale socket from a previous run
        if path.exists() {
            std::fs::remove_file(path).map_err(|e| {
                crate::error::Error::Ipc(format!(
                    "cannot remove stale socket {}: {e}",
                    path.display()
                ))
            })?;
        }

        let listener = UnixListener::bind(path)
            .map_err(|e| crate::error::Error::Ipc(format!("bind {}: {e}", path.display())))?;

        Ok(Self {
            listener,
            socket_path: path.to_path_buf(),
            metrics: Arc::new(Metrics::new()),
        })
    }

    /// The filesystem path this server is listening on.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// A reference to the server's metrics collector.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    /// Accept connections and dispatch requests until the process exits.
    ///
    /// Each connection is handled in its own OS thread. The call blocks
    /// on `accept()` and never returns under normal operation.
    pub fn run(&self) {
        eprintln!(
            "[wetspring] listening on {}",
            self.socket_path.display()
        );

        for stream in self.listener.incoming() {
            match stream {
                Ok(stream) => {
                    let metrics = Arc::clone(&self.metrics);
                    std::thread::spawn(move || {
                        handle_connection(&stream, &metrics);
                    });
                }
                Err(e) => {
                    eprintln!("[wetspring] accept error: {e}");
                }
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Handle a single client connection: read newline-delimited JSON-RPC, dispatch, respond.
fn handle_connection(stream: &std::os::unix::net::UnixStream, metrics: &Metrics) {
    if let Err(e) = stream.set_read_timeout(Some(CONNECTION_READ_TIMEOUT)) {
        eprintln!("[wetspring] set timeout: {e}");
        return;
    }

    let reader = BufReader::new(stream);
    let mut writer = std::io::BufWriter::new(stream);

    for line in reader.lines() {
        let line = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(_) => break,
        };

        let start = std::time::Instant::now();

        let response = match protocol::parse_request(&line) {
            Ok(req) => {
                // Server-level intercept for metrics (needs Metrics reference)
                let result = if req.method == "metrics.snapshot" {
                    Ok(metrics.snapshot())
                } else {
                    dispatch::dispatch(&req.method, &req.params)
                };
                match result {
                    Ok(result) => {
                        metrics.record_success(&req.method, start.elapsed());
                        protocol::success_response(&req.id, &result)
                    }
                    Err(rpc_err) => {
                        metrics.record_error(&req.method, start.elapsed());
                        protocol::error_response(&req.id, rpc_err.code, &rpc_err.message)
                    }
                }
            }
            Err(parse_err) => {
                metrics.record_error("_parse", start.elapsed());
                protocol::error_response(
                    &parse_err.id,
                    parse_err.error.code,
                    &parse_err.error.message,
                )
            }
        };

        let write_ok = writer.write_all(response.as_bytes()).is_ok()
            && writer.write_all(b"\n").is_ok()
            && writer.flush().is_ok();
        if !write_ok {
            break;
        }
    }
}

/// Resolve the socket path for binding.
fn resolve_bind_path() -> PathBuf {
    super::discover::resolve_bind_path("WETSPRING_SOCKET", crate::PRIMAL_NAME)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    #[test]
    fn server_bind_and_health_check() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_server.sock");
        let server = Server::bind(&sock).unwrap();

        assert!(sock.exists());

        let server_path = server.socket_path().to_path_buf();
        let handle = std::thread::spawn(move || {
            server.run();
        });

        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(val["result"]["status"], "healthy");
        assert_eq!(val["id"], 1);

        drop(handle);
    }

    #[test]
    fn server_diversity_over_socket() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_diversity.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"science.diversity","params":{"counts":[25.0,25.0,25.0,25.0]},"id":2}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        let shannon = val["result"]["shannon"].as_f64().unwrap();
        assert!((shannon - 4.0_f64.ln()).abs() < tolerances::PYTHON_PARITY);
    }

    #[test]
    fn server_qs_model_over_socket() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_qs.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"science.qs_model","params":{"scenario":"standard_growth","dt":0.1},"id":3}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(val["result"]["t_end"].as_f64().unwrap() > 0.0);
        assert!(val["result"]["peak_biofilm"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn server_multiple_requests_single_connection() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_multi.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let mut reader = BufReader::new(&stream);

        for i in 1..=3 {
            let request =
                format!(r#"{{"jsonrpc":"2.0","method":"health.check","params":{{}},"id":{i}}}"#);
            writer.write_all(request.as_bytes()).unwrap();
            writer.write_all(b"\n").unwrap();
            writer.flush().unwrap();

            let mut response = String::new();
            reader.read_line(&mut response).unwrap();
            let val: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert_eq!(val["id"], i);
        }
    }

    #[test]
    fn server_unknown_method_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_unknown.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"bogus","params":{},"id":99}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(val["error"]["code"], -32601);
    }

    #[test]
    fn server_cleanup_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_cleanup.sock");
        {
            let _server = Server::bind(&sock).unwrap();
            assert!(sock.exists());
        }
        assert!(!sock.exists());
    }

    #[test]
    fn server_metrics_snapshot_via_rpc() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_metrics_snap.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        // First make a health check to generate some metrics
        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let mut reader = BufReader::new(&stream);

        let health_req = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        writer.write_all(health_req.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();

        // Now request metrics snapshot
        let metrics_req = r#"{"jsonrpc":"2.0","method":"metrics.snapshot","params":{},"id":2}"#;
        writer.write_all(metrics_req.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();
        let mut metrics_resp = String::new();
        reader.read_line(&mut metrics_resp).unwrap();

        let val: serde_json::Value = serde_json::from_str(&metrics_resp).unwrap();
        assert_eq!(val["result"]["primal"], "wetspring");
        assert!(val["result"]["total_calls"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn resolve_bind_path_default() {
        let path = resolve_bind_path();
        assert!(path.to_string_lossy().contains("wetspring"));
    }

    #[test]
    fn metrics_tracked_after_requests() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test_metrics.sock");
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();
        let metrics = Arc::clone(server.metrics());

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        std::thread::sleep(Duration::from_millis(50));

        let total = metrics
            .total_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(total >= 1, "expected at least 1 call, got {total}");
    }
}
