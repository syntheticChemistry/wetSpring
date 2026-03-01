// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI fetch via `NestGate` and biomeOS.

use crate::error::Error;
use crate::ncbi::nestgate::rpc::{
    escape_json, extract_error, extract_fasta_from_response, rpc_call,
};
use crate::ncbi::nestgate::storage::{exists, retrieve, store};
use std::path::Path;

/// Route an NCBI fetch through biomeOS Neural API `capability.call`.
///
/// Sends `capability.call("science.ncbi_fetch", ...)` to the biomeOS
/// orchestrator, which routes it to `NestGate` or the registered provider.
///
/// # Errors
///
/// Returns `Err` if biomeOS is unreachable or the capability call fails.
pub fn fetch_via_biomeos(
    biomeos_socket: &Path,
    db: &str,
    id: &str,
    api_key: &str,
) -> crate::error::Result<String> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"capability.call","params":{{"capability":"science.ncbi_fetch","params":{{"db":"{}","id":"{}","api_key":"{}"}}}},"id":1}}"#,
        escape_json(db),
        escape_json(id),
        escape_json(api_key),
    );
    let response = rpc_call(biomeos_socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(format!(
            "biomeOS capability.call failed: {}",
            extract_error(&response)
        )))
    } else {
        extract_fasta_from_response(&response)
    }
}

/// Fetch FASTA using the three-tier routing strategy.
///
/// 1. **biomeOS**: If biomeOS Neural API is available, use `capability.call`
/// 2. **`NestGate`**: If direct `NestGate` socket is available, use cache + fetch
/// 3. **Sovereign**: Fall back to direct NCBI HTTP
///
/// # Errors
///
/// Returns `Err` if all three tiers fail.
pub fn fetch_tiered(db: &str, id: &str, api_key: &str) -> crate::error::Result<String> {
    use crate::ncbi::nestgate::discovery::{discover_biomeos_socket, discover_socket, is_enabled};

    if let Some(biomeos_socket) = discover_biomeos_socket() {
        match fetch_via_biomeos(&biomeos_socket, db, id, api_key) {
            Ok(fasta) => return Ok(fasta),
            Err(e) => {
                eprintln!("[nestgate] biomeOS routing failed, falling back: {e}");
            }
        }
    }

    if is_enabled() {
        if let Some(nestgate_socket) = discover_socket() {
            match fetch_or_fallback(&nestgate_socket, db, id, api_key) {
                Ok(fasta) => return Ok(fasta),
                Err(e) => {
                    eprintln!("[nestgate] NestGate failed, falling back to sovereign: {e}");
                }
            }
        }
    }

    super::super::efetch::efetch_fasta(db, id, api_key)
}

/// Fetch FASTA from NCBI via `NestGate`, caching in `NestGate`'s storage.
///
/// 1. Check if `NestGate` has the sequence cached (`ncbi:{db}:{id}`)
/// 2. If not, fall back to sovereign `efetch_fasta` and store in `NestGate`
///
/// # Errors
///
/// Returns `Err` if both `NestGate` retrieval and sovereign fallback fail.
pub fn fetch_or_fallback(
    socket: &Path,
    db: &str,
    id: &str,
    api_key: &str,
) -> crate::error::Result<String> {
    let cache_key = format!("ncbi:{db}:{id}");

    if exists(socket, &cache_key).unwrap_or(false) {
        if let Ok(cached) = retrieve(socket, &cache_key) {
            return Ok(cached);
        }
    }

    let fasta = super::super::efetch::efetch_fasta(db, id, api_key)?;

    let _ = store(socket, &cache_key, &fasta);

    Ok(fasta)
}
