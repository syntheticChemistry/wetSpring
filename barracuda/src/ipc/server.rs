// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unix socket server for the wetSpring science primal.
//!
//! Accepts connections on a Unix domain socket and handles JSON-RPC 2.0
//! requests (newline-delimited), dispatching to barracuda library functions.
//! Each connection is handled in its own thread.
//!
//! Connection processing (read/dispatch/write pipeline) lives in the
//! sibling `connection` module — this module owns only the server lifecycle.

use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::connection;
use super::metrics::Metrics;
#[cfg(test)]
use crate::tolerances;

/// wetSpring IPC server.
///
/// Listens on a Unix domain socket and handles JSON-RPC 2.0 requests,
/// dispatching to barracuda library functions for each registered capability.
pub struct Server {
    listener: UnixListener,
    socket_path: PathBuf,
    metrics: Arc<Metrics>,
    /// riboCipher enforcement policy for incoming connections.
    policy: super::ribocipher::Policy,
}

impl Server {
    /// Bind to the default socket path following the Primal IPC Protocol.
    ///
    /// Reads riboCipher policy from `RIBOCIPHER_POLICY` env var (production default: Reject).
    ///
    /// Discovery order:
    /// 1. `WETSPRING_SOCKET` env var (explicit override)
    /// 2. `$XDG_RUNTIME_DIR/biomeos/wetspring-{family_id}.sock`
    /// 3. `<temp_dir>/wetspring-{family_id}.sock` (platform-agnostic fallback)
    ///
    /// # Errors
    ///
    /// Returns `Err` if the socket directory cannot be created or the bind fails.
    pub fn bind_default() -> crate::error::Result<Self> {
        let path = resolve_bind_path();
        Self::bind_with_policy(&path, super::ribocipher::Policy::from_env())
    }

    /// Bind to a specific socket path with `Error` policy (accept unsignalled with ERROR log).
    ///
    /// For production deployments that should reject unsignalled connections, use
    /// [`bind_default`](Self::bind_default) or [`bind_with_policy`](Self::bind_with_policy)
    /// with [`Policy::Reject`](super::ribocipher::Policy::Reject).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the directory cannot be created, a stale socket cannot
    /// be removed, or the bind fails.
    pub fn bind(path: &Path) -> crate::error::Result<Self> {
        Self::bind_with_policy(path, super::ribocipher::Policy::Error)
    }

    /// Bind to a specific socket path with an explicit riboCipher policy.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the directory cannot be created, a stale socket cannot
    /// be removed, or the bind fails.
    pub fn bind_with_policy(
        path: &Path,
        policy: super::ribocipher::Policy,
    ) -> crate::error::Result<Self> {
        use crate::error::IpcError;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                IpcError::SocketPath(format!("create dir {}: {e}", parent.display()))
            })?;
        }

        if path.exists() {
            std::fs::remove_file(path).map_err(|e| {
                IpcError::SocketPath(format!("remove stale {}: {e}", path.display()))
            })?;
        }

        let listener = UnixListener::bind(path)
            .map_err(|e| IpcError::SocketPath(format!("bind {}: {e}", path.display())))?;

        Ok(Self {
            listener,
            socket_path: path.to_path_buf(),
            metrics: Arc::new(Metrics::new()),
            policy,
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
        tracing::info!(socket = %self.socket_path.display(), "listening");

        for stream in self.listener.incoming() {
            match stream {
                Ok(stream) => {
                    let metrics = Arc::clone(&self.metrics);
                    let policy = self.policy;
                    std::thread::spawn(move || {
                        connection::handle_connection(&stream, &metrics, policy);
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "accept error");
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

/// Resolve the socket path for binding.
fn resolve_bind_path() -> PathBuf {
    super::discover::resolve_bind_path("WETSPRING_SOCKET", super::primal_names::SELF)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use crate::ipc::{cleanup_test_socket, test_socket_path};
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    #[test]
    fn server_bind_and_health_check() {
        let sock = test_socket_path("server_bind_and_health_check");
        cleanup_test_socket(&sock);
        let server = Server::bind_with_policy(&sock, crate::ipc::ribocipher::Policy::Error).unwrap();

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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_diversity_wetspring_prefixed_method_normalized() {
        let sock = test_socket_path("server_diversity_wetspring_prefixed_method_normalized");
        cleanup_test_socket(&sock);
        let server = Server::bind_with_policy(&sock, crate::ipc::ribocipher::Policy::Error).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"wetspring.science.diversity","params":{"counts":[25.0,25.0,25.0,25.0]},"id":2}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        let shannon = val["result"]["shannon"].as_f64().unwrap();
        assert!((shannon - 4.0_f64.ln()).abs() < tolerances::PYTHON_PARITY);
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_diversity_over_socket() {
        let sock = test_socket_path("server_diversity_over_socket");
        cleanup_test_socket(&sock);
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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_qs_model_over_socket() {
        let sock = test_socket_path("server_qs_model_over_socket");
        cleanup_test_socket(&sock);
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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_multiple_requests_single_connection() {
        let sock = test_socket_path("server_multiple_requests_single_connection");
        cleanup_test_socket(&sock);
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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_unknown_method_returns_error() {
        let sock = test_socket_path("server_unknown_method_returns_error");
        cleanup_test_socket(&sock);
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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_cleanup_on_drop() {
        let sock = test_socket_path("server_cleanup_on_drop");
        cleanup_test_socket(&sock);
        {
            let _server = Server::bind(&sock).unwrap();
            assert!(sock.exists());
        }
        assert!(!sock.exists());
    }

    #[test]
    fn server_metrics_snapshot_via_rpc() {
        let sock = test_socket_path("server_metrics_snapshot_via_rpc");
        cleanup_test_socket(&sock);
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
        assert_eq!(val["result"]["primal"], super::super::primal_names::SELF);
        assert!(val["result"]["total_calls"].as_u64().unwrap() >= 1);
        cleanup_test_socket(&sock);
    }

    #[test]
    fn resolve_bind_path_default() {
        let path = resolve_bind_path();
        assert!(
            path.to_string_lossy()
                .contains(super::super::primal_names::SELF)
        );
    }

    #[test]
    fn metrics_tracked_after_requests() {
        let sock = test_socket_path("metrics_tracked_after_requests");
        cleanup_test_socket(&sock);
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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_empty_batch_returns_invalid_request() {
        let sock = test_socket_path("server_empty_batch_returns_invalid_request");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        writer.write_all(b"[]\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(val["error"]["code"], -32600);
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_all_notification_batch_no_response() {
        let sock = test_socket_path("server_all_notification_batch_no_response");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let mut reader = BufReader::new(&stream);

        let batch = r#"[{"jsonrpc":"2.0","method":"health.check","params":{}},{"jsonrpc":"2.0","method":"health.check","params":{}}]"#;
        writer.write_all(batch.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let probe = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":999}"#;
        writer.write_all(probe.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut response = String::new();
        reader.read_line(&mut response).unwrap();
        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            val["id"], 999,
            "probe after all-notification batch gets response"
        );
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_mixed_batch_returns_array() {
        let sock = test_socket_path("server_mixed_batch_returns_array");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let batch = r#"[{"jsonrpc":"2.0","method":"health.check","params":{}},{"jsonrpc":"2.0","method":"health.check","params":{},"id":42}]"#;
        writer.write_all(batch.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], 42);
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_single_notification_no_response() {
        let sock = test_socket_path("server_single_notification_no_response");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let mut reader = BufReader::new(&stream);

        let req = r#"{"jsonrpc":"2.0","method":"health.check","params":{}}"#;
        writer.write_all(req.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let probe = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":888}"#;
        writer.write_all(probe.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut response = String::new();
        reader.read_line(&mut response).unwrap();
        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(val["id"], 888, "probe after notification gets response");
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_id_null_gets_response() {
        let sock = test_socket_path("server_id_null_gets_response");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        let mut writer = std::io::BufWriter::new(&stream);
        let req = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":null}"#;
        writer.write_all(req.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(val.get("result").is_some());
        assert!(val["id"].is_null());
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_ribocipher_signalled_connection() {
        use crate::ipc::ribocipher;

        let sock = test_socket_path("server_ribocipher_signalled");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        ribocipher::send_clear_signal(&stream).unwrap();

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
        cleanup_test_socket(&sock);
    }

    #[test]
    fn server_ribocipher_reject_drops_unsignalled() {
        use std::io::Read;

        let sock = test_socket_path("server_ribocipher_reject");
        cleanup_test_socket(&sock);
        let server = Server::bind_with_policy(
            &sock,
            crate::ipc::ribocipher::Policy::Reject,
        )
        .unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();

        // Send JSON-RPC without riboCipher signal — should be rejected
        let mut writer = std::io::BufWriter::new(&stream);
        writer
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"method\":\"health.check\",\"params\":{},\"id\":1}\n",
            )
            .unwrap();
        writer.flush().unwrap();

        // Read should fail (connection dropped by server)
        let mut buf = [0u8; 1];
        let result = (&stream).read(&mut buf);
        assert!(
            result.is_err() || result.unwrap() == 0,
            "server should have dropped the connection"
        );

        cleanup_test_socket(&sock);
    }

    /// DEPLOY-THEN-STALE validation: lifecycle.status must return version + git_sha
    /// so mesh health audits can detect stale deploys (Wave 113 exit criterion 3).
    #[test]
    fn server_lifecycle_status_exposes_skew_fields() {
        use crate::ipc::ribocipher;

        let sock = test_socket_path("server_lifecycle_skew");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        ribocipher::send_clear_signal(&stream).unwrap();

        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"lifecycle.status","params":{},"id":42}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(val["id"], 42);

        let result = &val["result"];
        assert_eq!(result["primal"], "wetspring");
        assert_eq!(result["status"], "running");

        let version = result["version"].as_str().unwrap();
        assert!(!version.is_empty(), "version must be non-empty");

        let git_sha = result["git_sha"].as_str().unwrap();
        assert!(!git_sha.is_empty(), "git_sha must be non-empty");
        assert_ne!(git_sha, "unknown", "git_sha must resolve at compile time");
        assert!(
            git_sha.len() >= 7,
            "git_sha should be at least 7 hex chars, got: {git_sha}"
        );

        let uptime = result["uptime_s"].as_f64().unwrap();
        assert!(uptime >= 0.0, "uptime must be non-negative");

        cleanup_test_socket(&sock);
    }

    /// DEPLOY-THEN-STALE: health.liveness must include version per guideStone amendment.
    #[test]
    fn server_health_liveness_includes_version() {
        use crate::ipc::ribocipher;

        let sock = test_socket_path("server_health_version");
        cleanup_test_socket(&sock);
        let server = Server::bind(&sock).unwrap();
        let server_path = server.socket_path().to_path_buf();

        std::thread::spawn(move || server.run());
        std::thread::sleep(Duration::from_millis(50));

        let stream = UnixStream::connect(&server_path).unwrap();
        ribocipher::send_clear_signal(&stream).unwrap();

        let mut writer = std::io::BufWriter::new(&stream);
        let request = r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":7}"#;
        writer.write_all(request.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).unwrap();

        let val: serde_json::Value = serde_json::from_str(&response).unwrap();
        let result = &val["result"];
        assert_eq!(result["status"], "alive");
        assert_eq!(result["primal"], "wetspring");
        assert!(
            result["version"].as_str().is_some_and(|v| !v.is_empty()),
            "health.liveness must include version per guideStone amendment"
        );
        cleanup_test_socket(&sock);
    }
}
