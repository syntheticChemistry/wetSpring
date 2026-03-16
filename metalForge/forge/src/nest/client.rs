// SPDX-License-Identifier: AGPL-3.0-or-later

//! `NestClient` implementation — storage operations.

use std::path::Path;

use crate::nest::base64;
use crate::nest::discovery;
use crate::nest::json;
use crate::nest::time;
use crate::nest::transport;
use crate::nest::types::{ListResult, NestClient, RetrieveResult, StoreResult};

const DEFAULT_FAMILY: &str = "default";

impl NestClient {
    /// Create a client for a discovered `NestGate` socket.
    ///
    /// Returns `None` if no `NestGate` socket is found.
    #[must_use]
    pub fn discover() -> Option<Self> {
        discovery::discover_nestgate_socket().map(|socket| Self {
            socket,
            family_id: DEFAULT_FAMILY.to_string(),
        })
    }

    /// Create a client for a specific socket path.
    #[must_use]
    pub fn new(socket: std::path::PathBuf) -> Self {
        Self {
            socket,
            family_id: DEFAULT_FAMILY.to_string(),
        }
    }

    /// Set the family ID for all subsequent operations.
    #[must_use]
    pub fn with_family(mut self, family_id: impl Into<String>) -> Self {
        self.family_id = family_id.into();
        self
    }

    /// The socket path this client is connected to.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket
    }

    /// Family ID for storage operations.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn family_id(&self) -> &str {
        &self.family_id
    }

    /// Check if a key exists in `NestGate` storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn exists(&self, key: &str) -> Result<bool, crate::error::NestError> {
        let escaped = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{escaped}","family_id":"{family}"}},"id":1}}"#,
        );
        let resp = transport::rpc(&self.socket, &req)?;
        Ok(resp.contains("true"))
    }

    /// Store a JSON value by key.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn store(&self, key: &str, value: &str) -> Result<StoreResult, crate::error::NestError> {
        let escaped_key = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{escaped_key}","value":{value},"family_id":"{family}"}},"id":1}}"#,
        );
        let raw = transport::rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// Store a binary blob (base64-encoded) by key.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn store_blob(
        &self,
        key: &str,
        data: &[u8],
    ) -> Result<StoreResult, crate::error::NestError> {
        let escaped_key = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let encoded = base64::base64_encode(data);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store_blob","params":{{"key":"{escaped_key}","blob":"{encoded}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = transport::rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// Retrieve a JSON value by key.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn retrieve(&self, key: &str) -> Result<RetrieveResult, crate::error::NestError> {
        let escaped_key = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = transport::rpc(&self.socket, &req)?;
        let value = json::extract_result_value(&raw);
        Ok(RetrieveResult { value, raw })
    }

    /// Retrieve a binary blob by key.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn retrieve_blob(&self, key: &str) -> Result<Option<Vec<u8>>, crate::error::NestError> {
        let escaped_key = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve_blob","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = transport::rpc(&self.socket, &req)?;
        if raw.contains("\"error\"") {
            return Ok(None);
        }
        let b64 = json::extract_result_string(&raw, "blob")
            .or_else(|| json::extract_result_string(&raw, "data"));
        Ok(b64.map(|encoded| base64::base64_decode(&encoded)))
    }

    /// Delete a key from storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn delete(&self, key: &str) -> Result<StoreResult, crate::error::NestError> {
        let escaped_key = json::escape_json_str(key);
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.delete","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = transport::rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// List keys with an optional prefix filter.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn list(&self, prefix: Option<&str>) -> Result<ListResult, crate::error::NestError> {
        let family = json::escape_json_str(&self.family_id);
        let params = prefix.map_or_else(
            || format!(r#"{{"family_id":"{family}"}}"#),
            |p| {
                let escaped = json::escape_json_str(p);
                format!(r#"{{"family_id":"{family}","prefix":"{escaped}"}}"#)
            },
        );
        let req =
            format!(r#"{{"jsonrpc":"2.0","method":"storage.list","params":{params},"id":1}}"#);
        let raw = transport::rpc(&self.socket, &req)?;
        let keys = json::extract_string_array(&raw, "keys")
            .or_else(|| json::extract_result_array(&raw))
            .unwrap_or_default();
        Ok(ListResult { keys, raw })
    }

    /// Get storage statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn stats(&self) -> Result<String, crate::error::NestError> {
        let family = json::escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.stats","params":{{"family_id":"{family}"}},"id":1}}"#,
        );
        transport::rpc(&self.socket, &req)
    }

    /// Ingest a local file into `NestGate` blob storage.
    ///
    /// Reads the file and stores it as a base64-encoded blob under the
    /// given key. For large files, this loads the entire file into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or storage operations fail.
    pub fn ingest_file(
        &self,
        key: &str,
        path: &Path,
    ) -> Result<StoreResult, crate::error::NestError> {
        let data = std::fs::read(path).map_err(|e| crate::error::NestError::FileRead {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        self.store_blob(key, &data)
    }

    /// Store dataset metadata as structured JSON.
    ///
    /// Stores provenance (source, accession, file count, total bytes)
    /// under the key `meta:<dataset>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is invalid, connection fails, or the RPC
    /// response cannot be read.
    pub fn store_dataset_metadata(
        &self,
        dataset: &str,
        source: &str,
        file_count: usize,
        total_bytes: u64,
    ) -> Result<StoreResult, crate::error::NestError> {
        let meta_key = format!("meta:{dataset}");
        let value = format!(
            r#"{{"dataset":"{dataset}","source":"{source}","file_count":{file_count},"total_bytes":{total_bytes},"stored_at":"{}"}}"#,
            time::chrono_lite_now(),
        );
        self.store(&meta_key, &value)
    }
}
