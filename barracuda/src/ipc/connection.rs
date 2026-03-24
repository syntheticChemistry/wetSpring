// SPDX-License-Identifier: AGPL-3.0-or-later
//! Connection processing pipeline for JSON-RPC 2.0 over Unix domain sockets.
//!
//! Handles reading newline-delimited JSON-RPC from a stream, dispatching
//! requests through the capability handler, and writing responses.
//! Separated from the server lifecycle (`Server` bind/run/drop) so that
//! the protocol pipeline can be tested and reasoned about independently.

use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

use super::dispatch;
use super::message::Request;
use super::metrics::Metrics;
use super::protocol::{self, RpcError};

const CONNECTION_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Dispatch a method call, routing `metrics.snapshot` internally and
/// all other methods through the capability handler.
fn execute_method(
    method: &str,
    params: &serde_json::Value,
    metrics: &Metrics,
) -> Result<serde_json::Value, RpcError> {
    let normalized = protocol::normalize_method(method);
    if normalized.as_ref() == "metrics.snapshot" {
        Ok(metrics.snapshot())
    } else {
        dispatch::dispatch(method, params)
    }
}

/// Dispatch a request and produce a JSON-RPC response string.
fn dispatch_request(req: &Request, metrics: &Metrics, start: Instant) -> String {
    let method_key = protocol::normalize_method(&req.method);
    match execute_method(&req.method, &req.params, metrics) {
        Ok(result) => {
            metrics.record_success(method_key.as_ref(), start.elapsed());
            protocol::success_response(&req.id, &result)
        }
        Err(rpc_err) => {
            metrics.record_error(method_key.as_ref(), start.elapsed());
            protocol::error_response(&req.id, rpc_err.code, &rpc_err.message)
        }
    }
}

/// Dispatch a notification (fire-and-forget, no response).
fn dispatch_notification(line: &str, metrics: &Metrics, start: Instant) {
    if let Ok(req) = protocol::parse_request(line) {
        let method_key = protocol::normalize_method(&req.method);
        match execute_method(&req.method, &req.params, metrics) {
            Ok(_) => metrics.record_success(method_key.as_ref(), start.elapsed()),
            Err(_) => metrics.record_error(method_key.as_ref(), start.elapsed()),
        }
    }
}

/// Process a single JSON-RPC line. Returns `None` for notifications.
fn process_single(line: &str, metrics: &Metrics, start: Instant) -> Option<String> {
    if protocol::is_notification(line) {
        dispatch_notification(line, metrics, start);
        return None;
    }

    Some(match protocol::parse_request(line) {
        Ok(req) => dispatch_request(&req, metrics, start),
        Err(parse_err) => {
            metrics.record_error("_parse", start.elapsed());
            protocol::error_response(
                &parse_err.id,
                parse_err.error.code,
                &parse_err.error.message,
            )
        }
    })
}

/// Process a JSON-RPC batch. Returns `None` if every element is a notification.
fn process_batch(line: &str, metrics: &Metrics, start: Instant) -> Option<String> {
    let elements = protocol::parse_batch(line);
    if elements.is_empty() {
        return Some(protocol::error_response(
            &serde_json::Value::Null,
            -32600,
            "Invalid Request",
        ));
    }

    let mut responses: Vec<String> = Vec::new();
    for elem in elements {
        if protocol::is_notification(&elem) {
            dispatch_notification(&elem, metrics, start);
            continue;
        }

        let resp = match protocol::parse_request(&elem) {
            Ok(req) => dispatch_request(&req, metrics, start),
            Err(parse_err) => {
                metrics.record_error("_parse", start.elapsed());
                protocol::error_response(
                    &parse_err.id,
                    parse_err.error.code,
                    &parse_err.error.message,
                )
            }
        };
        responses.push(resp);
    }

    if responses.is_empty() {
        return None;
    }

    Some(format!("[{}]", responses.join(",")))
}

/// Handle a client connection: read newline-delimited JSON-RPC, dispatch, respond.
pub(super) fn handle_connection(stream: &std::os::unix::net::UnixStream, metrics: &Metrics) {
    if let Err(e) = stream.set_read_timeout(Some(CONNECTION_READ_TIMEOUT)) {
        tracing::warn!(error = %e, "set read timeout");
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

        let start = Instant::now();

        let response_opt = if protocol::is_batch(&line) {
            process_batch(&line, metrics, start)
        } else {
            process_single(&line, metrics, start)
        };

        if let Some(response) = response_opt {
            let write_ok = writer.write_all(response.as_bytes()).is_ok()
                && writer.write_all(b"\n").is_ok()
                && writer.flush().is_ok();
            if !write_ok {
                break;
            }
        }
    }
}
