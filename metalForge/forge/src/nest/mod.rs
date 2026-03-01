// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nest atomic storage client — JSON-RPC over Unix socket to `NestGate`.
//!
//! Provides a typed client for `NestGate`'s storage API using the biomeOS
//! JSON-RPC 2.0 protocol. Supports both structured JSON storage and
//! binary blob storage (base64-encoded).
//!
//! # Socket Discovery
//!
//! `NestGate` socket is discovered in order:
//! 1. `NESTGATE_SOCKET` environment variable (if path exists)
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock` (if exists)
//! 3. Default path from `default_socket_path()` (`/run/nestgate/default.sock` if env unset)
//!
//! # NUCLEUS Role
//!
//! Nest = Tower + `NestGate` + Squirrel. This module implements the `NestGate`
//! storage interface that validation binaries and pipelines use to persist
//! and retrieve datasets without hardcoded paths.

mod base64;
mod client;
mod discovery;
mod json;
mod time;
mod transport;
mod types;

#[cfg(test)]
mod tests;

// Re-export public API
pub use discovery::{default_socket_path, discover_nestgate_socket};
pub use types::{ListResult, NestClient, RetrieveResult, StoreResult};
