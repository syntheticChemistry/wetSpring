// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC transport over Unix socket.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

const NESTGATE_TIMEOUT: Duration = Duration::from_secs(5);

/// Send a JSON-RPC request and read the response.
pub(super) fn rpc(socket: &Path, request: &str) -> Result<String, crate::error::NestError> {
    use crate::error::NestError;

    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| NestError::InvalidSocket(e.to_string()))?;
    let stream =
        UnixStream::connect_addr(&addr).map_err(|e| NestError::Connect(e.to_string()))?;
    stream
        .set_read_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| NestError::Timeout(e.to_string()))?;
    stream
        .set_write_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| NestError::Timeout(e.to_string()))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| NestError::Write(e.to_string()))?;
    writer
        .write_all(b"\n")
        .map_err(|e| NestError::Write(e.to_string()))?;
    writer
        .flush()
        .map_err(|e| NestError::Flush(e.to_string()))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| NestError::Read(e.to_string()))?;

    if line.is_empty() {
        return Err(NestError::EmptyResponse);
    }
    Ok(line)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    #[test]
    fn rpc_nonexistent_socket() {
        let bad = Path::new("/tmp/wetspring_nest_nonexistent_test.sock");
        let result = rpc(bad, r#"{"jsonrpc":"2.0","method":"ping","id":1}"#);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("connect") || msg.contains("NestGate"),
            "expected connect error, got: {msg}"
        );
    }

    #[test]
    fn rpc_roundtrip_echo() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test_rpc.sock");
        let listener = UnixListener::bind(&sock_path).unwrap();

        let handle = std::thread::spawn(move || {
            rpc(&sock_path, r#"{"jsonrpc":"2.0","method":"ping","id":1}"#)
        });

        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(&stream);
        let mut req = String::new();
        reader.read_line(&mut req).unwrap();
        assert!(req.contains("ping"));

        let mut writer = std::io::BufWriter::new(&stream);
        writer
            .write_all(br#"{"jsonrpc":"2.0","result":"pong","id":1}"#)
            .unwrap();
        writer.write_all(b"\n").unwrap();
        writer.flush().unwrap();

        let response = handle.join().unwrap().unwrap();
        assert!(response.contains("pong"));
    }

    #[test]
    fn rpc_server_closes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("close_rpc.sock");
        let listener = UnixListener::bind(&sock_path).unwrap();

        let handle = std::thread::spawn(move || {
            rpc(&sock_path, r#"{"jsonrpc":"2.0","method":"ping","id":1}"#)
        });

        let (stream, _) = listener.accept().unwrap();
        drop(stream);

        let result = handle.join().unwrap();
        assert!(result.is_err() || result.as_ref().is_ok_and(String::is_empty));
    }

    #[test]
    fn rpc_invalid_socket_path() {
        let result = rpc(Path::new(""), r#"{"jsonrpc":"2.0","method":"ping","id":1}"#);
        assert!(result.is_err());
    }
}
