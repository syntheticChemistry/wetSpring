// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC client for wetSpring IPC socket.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};

/// Resolve the wetSpring IPC socket path.
///
/// Priority: `WETSPRING_SOCKET` env var > XDG runtime dir > /tmp fallback.
pub fn socket_path() -> PathBuf {
    if let Ok(p) = std::env::var("WETSPRING_SOCKET") {
        return PathBuf::from(p);
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    PathBuf::from(runtime)
        .join("biomeos")
        .join("wetspring-default.sock")
}

/// Send a JSON-RPC 2.0 request and return the result or error.
pub fn call(method: &str, params: &Value) -> Result<Value, String> {
    let path = socket_path();
    let stream =
        UnixStream::connect(&path).map_err(|e| format!("connect to {}: {e}", path.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    call_on_stream(stream, method, params)
}

fn call_on_stream(mut stream: UnixStream, method: &str, params: &Value) -> Result<Value, String> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    });

    let mut line = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|e| format!("read: {e}"))?;

    let resp: Value =
        serde_json::from_str(response_line.trim()).map_err(|e| format!("parse: {e}"))?;

    if let Some(err) = resp.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown RPC error");
        return Err(msg.to_string());
    }

    resp.get("result")
        .cloned()
        .ok_or_else(|| "missing result field".to_string())
}
