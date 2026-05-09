// SPDX-License-Identifier: AGPL-3.0-or-later
//! loamSpine immutable ledger operations — permanent session commit.
//!
//! Wraps the `session.commit` capability call routed through biomeOS Neural
//! API. Phase 2 of the three-phase provenance completion pattern:
//! rhizoCrypt dehydrate → **loamSpine commit** → sweetGrass braid.

use serde_json::{Value, json};

use super::capability_call;

/// Commit a dehydrated session summary to the loamSpine immutable ledger.
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
        "session",
        "commit",
        &json!({"summary": dehydration, "content_hash": merkle_root}),
    )
}
