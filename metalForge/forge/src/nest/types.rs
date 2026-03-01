// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nest storage types — client and result structs.

use std::path::PathBuf;

/// `NestGate` storage client.
///
/// Wraps a Unix socket path and provides typed methods for all
/// `NestGate` storage operations.
#[derive(Debug, Clone)]
pub struct NestClient {
    pub(crate) socket: PathBuf,
    pub(crate) family_id: String,
}

/// Result of a storage operation.
#[derive(Debug)]
pub struct StoreResult {
    /// Whether the operation succeeded.
    pub ok: bool,
    /// Raw JSON-RPC response (for inspection).
    pub raw: String,
}

/// Result of a list operation.
#[derive(Debug)]
pub struct ListResult {
    /// Keys matching the prefix.
    pub keys: Vec<String>,
    /// Raw JSON-RPC response.
    pub raw: String,
}

/// Result of a retrieve operation.
#[derive(Debug)]
pub struct RetrieveResult {
    /// The stored value as a JSON string.
    pub value: Option<String>,
    /// Raw JSON-RPC response.
    pub raw: String,
}
