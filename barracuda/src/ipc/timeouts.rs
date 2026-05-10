// SPDX-License-Identifier: AGPL-3.0-or-later
//! Centralized IPC timeout constants.
//!
//! Timeout tiers reflect the expected latency profile of each primal category:
//! - **Discovery** (Songbird): lightweight metadata, 5 s
//! - **Standard** (provenance trio, general RPC): 10 s
//! - **Compute** (toadStool, barraCuda dispatch): 30 s
//! - **AI** (Squirrel inference): 30 s
//! - **Connection** (server-side long-lived): 120 s
//!
//! Facade timeouts mirror the same tiers but are declared separately for the
//! HTTP gateway context.

use std::time::Duration;

/// Songbird discovery and registration calls.
pub const DISCOVERY: Duration = Duration::from_secs(5);

/// Standard JSON-RPC calls (provenance trio, general transport).
pub const STANDARD_RPC: Duration = Duration::from_secs(10);

/// Heavy compute dispatch (toadStool, barraCuda).
pub const COMPUTE: Duration = Duration::from_secs(30);

/// AI inference calls (Squirrel).
pub const AI_INFERENCE: Duration = Duration::from_secs(30);

/// Server-side long-lived connection reads.
pub const CONNECTION: Duration = Duration::from_secs(120);

/// Facade → provenance trio and Dark Forest auth.
pub const FACADE_SHORT: Duration = Duration::from_secs(5);

/// Facade → standard IPC client calls.
pub const FACADE_STANDARD: Duration = Duration::from_secs(10);

/// Facade → petalTongue grammar rendering.
pub const FACADE_RENDER: Duration = Duration::from_secs(15);
