// SPDX-License-Identifier: AGPL-3.0-or-later
//! loamSpine immutable ledger operations — permanent session commit.
//!
//! Wraps the `ledger.commit` capability call routed through biomeOS Neural
//! API. Phase 2 of the three-phase provenance completion pattern:
//! rhizoCrypt dehydrate → **loamSpine commit** → sweetGrass braid.
//!
//! Capability domain: `ledger` (per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` v2.0).

use serde_json::{Value, json};

use super::capability_call;

/// Commit a dehydrated session summary to the loamSpine immutable ledger.
///
/// Routes via `capability.call` with `capability = "ledger"` and
/// `operation = "commit"` per the canonical trio integration guide.
///
/// Returns the commit result containing `commit_id` / `entry_id`, or an
/// error if the ledger is unreachable.
pub(super) fn commit_session(
    socket: &std::path::Path,
    dehydration: &Value,
    merkle_root: &str,
) -> Result<Value, crate::error::Error> {
    capability_call(
        socket,
        "ledger",
        "commit",
        &json!({"summary": dehydration, "content_hash": merkle_root}),
    )
}
