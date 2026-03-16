// SPDX-License-Identifier: AGPL-3.0-or-later
//! AI-assisted ecology interpretation via Squirrel.
//!
//! Forwards ecology-contextualized queries to the Squirrel AI primal.
//! Graceful degradation when Squirrel is offline — returns a "Squirrel
//! unavailable" message instead of failing.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::ipc::discover;
use crate::ipc::protocol::RpcError;

const RPC_TIMEOUT: Duration = Duration::from_secs(30);

/// Send a JSON-RPC `ai.query` request to Squirrel.
fn squirrel_query(
    socket_path: &Path,
    query: &str,
    context: &Value,
) -> Result<Value, crate::error::Error> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": "ai.query",
        "params": {
            "query": query,
            "context": context,
        },
        "id": 1,
    });

    let payload = serde_json::to_string(&request)
        .map_err(|e| crate::error::Error::Ipc(format!("serialize: {e}")))?;

    let stream = std::os::unix::net::UnixStream::connect(socket_path)
        .map_err(|e| crate::error::Error::Ipc(format!("connect {}: {e}", socket_path.display())))?;
    stream.set_read_timeout(Some(RPC_TIMEOUT)).ok();
    stream.set_write_timeout(Some(RPC_TIMEOUT)).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(payload.as_bytes())
        .map_err(|e| crate::error::Error::Ipc(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| crate::error::Error::Ipc(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| crate::error::Error::Ipc(format!("flush: {e}")))?;
    drop(writer);

    stream
        .shutdown(std::net::Shutdown::Write)
        .map_err(|e| crate::error::Error::Ipc(format!("shutdown: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| crate::error::Error::Ipc(format!("read: {e}")))?;

    let parsed: Value = serde_json::from_str(line.trim())
        .map_err(|e| crate::error::Error::Ipc(format!("parse: {e}")))?;

    if let Some(err) = parsed.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        return Err(crate::error::Error::Ipc(format!("rpc error: {msg}")));
    }

    parsed
        .get("result")
        .cloned()
        .ok_or_else(|| crate::error::Error::Ipc("no result in response".to_string()))
}

/// Handle `ai.ecology_interpret` — forward ecology-contextualized queries to Squirrel.
///
/// Never fails. When Squirrel is unavailable, returns `Ok` with
/// `squirrel: "unavailable"` and a message. When Squirrel times out,
/// returns `Ok` with `squirrel: "timeout"`.
pub fn handle_ai_ecology_interpret(params: &Value) -> Result<Value, RpcError> {
    let query = params
        .get("query")
        .or_else(|| params.get("prompt"))
        .and_then(Value::as_str)
        .unwrap_or("Interpret this ecology context.");

    let context = params.get("context").cloned().unwrap_or_else(|| json!({}));

    let Some(socket) = discover::discover_squirrel() else {
        return Ok(json!({
            "squirrel": "unavailable",
            "message": "Squirrel AI unavailable — socket not discovered",
            "query": query,
        }));
    };

    match squirrel_query(&socket, query, &context) {
        Ok(result) => Ok(json!({
            "squirrel": "available",
            "response": result,
            "query": query,
        })),
        Err(e) => {
            let msg = e.to_string();
            let (status, message) = if msg.contains("timeout") || msg.contains("timed out") {
                ("timeout", "Squirrel AI request timed out".to_string())
            } else {
                ("error", msg)
            };
            Ok(json!({
                "squirrel": status,
                "message": message,
                "query": query,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_degrades_gracefully_when_squirrel_unavailable() {
        let result =
            handle_ai_ecology_interpret(&json!({"query": "Explain Shannon diversity"})).unwrap();
        assert_eq!(result["squirrel"], "unavailable");
        assert!(result["message"].as_str().unwrap().contains("unavailable"));
        assert_eq!(result["query"], "Explain Shannon diversity");
    }

    #[test]
    fn handle_accepts_prompt_param() {
        let result = handle_ai_ecology_interpret(&json!({"prompt": "What is Chao1?"})).unwrap();
        assert_eq!(result["query"], "What is Chao1?");
    }

    #[test]
    fn handle_default_query_when_empty_params() {
        let result = handle_ai_ecology_interpret(&json!({})).unwrap();
        assert_eq!(result["query"], "Interpret this ecology context.");
    }
}
