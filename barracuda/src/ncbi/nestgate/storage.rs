// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NestGate` storage API (store, retrieve, exists).

use crate::error::Error;
use crate::ncbi::nestgate::rpc::{escape_json, extract_error, extract_result_value, rpc_call};
use std::path::Path;

const FAMILY_ID: &str = "default";

/// Store data in `NestGate` via `storage.store`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the RPC fails.
pub fn store(socket: &Path, key: &str, value: &str) -> crate::error::Result<()> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{}","value":"{}","family_id":"{}"}},"id":1}}"#,
        escape_json(key),
        escape_json(value),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        Ok(())
    }
}

/// Retrieve data from `NestGate` via `storage.retrieve`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable, the key does not exist,
/// or the RPC fails.
pub fn retrieve(socket: &Path, key: &str) -> crate::error::Result<String> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{}","family_id":"{}"}},"id":2}}"#,
        escape_json(key),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        extract_result_value(&response)
    }
}

/// Check if a key exists in `NestGate` via `storage.exists`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the RPC fails.
pub fn exists(socket: &Path, key: &str) -> crate::error::Result<bool> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{}","family_id":"{}"}},"id":3}}"#,
        escape_json(key),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    Ok(response.contains("true"))
}
