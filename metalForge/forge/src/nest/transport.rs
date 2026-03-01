// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC transport over Unix socket.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

const NESTGATE_TIMEOUT: Duration = Duration::from_secs(5);

/// Send a JSON-RPC request and read the response.
pub(super) fn rpc(socket: &Path, request: &str) -> Result<String, String> {
    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| format!("invalid NestGate socket: {e}"))?;
    let stream = UnixStream::connect_addr(&addr).map_err(|e| format!("NestGate connect: {e}"))?;
    stream
        .set_read_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;
    stream
        .set_write_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer.write_all(b"\n").map_err(|e| format!("write: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("NestGate returned empty response".to_string());
    }
    Ok(line)
}
