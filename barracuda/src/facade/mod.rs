// SPDX-License-Identifier: AGPL-3.0-or-later
//! Science Facade — HTTP gateway to wetSpring IPC for primals.eco.
//!
//! Translates browser REST calls into wetSpring JSON-RPC, shapes responses
//! into petalTongue `DataChannel`-compatible JSON, and attaches progressive
//! provenance metadata. Designed to sit behind a cloudflared tunnel with
//! biomeOS Dark Forest gate.

pub mod dark_forest;
pub mod grammar;
pub mod graph_validate;
pub mod ipc_client;
pub mod provenance;
pub mod routes;
pub mod shaping;
