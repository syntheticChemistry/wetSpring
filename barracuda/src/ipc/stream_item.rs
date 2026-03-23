// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared `StreamItem` envelope for NDJSON pipeline output across ecoPrimals primals.
//!
//! Used by loamSpine, petalTongue, and rhizoCrypt for streaming progress, payloads,
//! completion summaries, and errors on a single line-delimited JSON channel.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One frame in a pipeline NDJSON stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamItem {
    /// Application payload chunk (opaque JSON).
    Data(Value),
    /// Human-readable progress update with a normalized percentage in `[0, 100]`.
    Progress {
        /// Completion ratio as a percentage (e.g. `50.0` for half done).
        percent: f64,
        /// Short status line for operators.
        message: String,
    },
    /// Terminal success frame with an opaque summary object.
    End {
        /// Aggregated result metadata or final payload reference.
        summary: Value,
    },
    /// Terminal error frame (application-level).
    Error {
        /// Stable error code (primal- or pipeline-specific).
        code: i32,
        /// Human-readable error description.
        message: String,
    },
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: assertions use expect for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn stream_item_roundtrips_json() {
        let item = StreamItem::Progress {
            percent: 42.5,
            message: "halfway".to_string(),
        };
        let s = serde_json::to_string(&item).expect("serialize");
        let back: StreamItem = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, item);
    }
}
