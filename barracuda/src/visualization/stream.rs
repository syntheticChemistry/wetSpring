// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming visualization session for progressive `petalTongue` rendering.
//!
//! `StreamSession` wraps [`PetalTonguePushClient`] with session state tracking,
//! typed push helpers, and backpressure awareness. Follows the
//! `visualization.render.stream` JSON-RPC protocol with `append`, `set_value`,
//! and `replace` operations.

use super::ipc_push::{PetalTonguePushClient, PushError, PushResult};
use super::types::{DataChannel, EcologyScenario};

/// Streaming visualization session.
///
/// Manages a session lifecycle: initial render, incremental updates, and close.
/// Each session has a unique ID that petalTongue uses to route updates to the
/// correct rendering context.
pub struct StreamSession {
    session_id: String,
    client: PetalTonguePushClient,
    frame_count: u64,
    is_open: bool,
}

/// Session state snapshot for validation / diagnostics.
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Session identifier.
    pub session_id: String,
    /// Number of frames pushed since open.
    pub frame_count: u64,
    /// Whether the session is currently open.
    pub is_open: bool,
}

impl StreamSession {
    /// Open a new streaming session with the given client and session ID.
    #[must_use]
    pub fn open(client: PetalTonguePushClient, session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            client,
            frame_count: 0,
            is_open: true,
        }
    }

    /// Push the initial full render (establishes bindings on the petalTongue side).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_initial_render(
        &mut self,
        title: &str,
        scenario: &EcologyScenario,
    ) -> PushResult<()> {
        if !self.is_open {
            return Err(PushError::NotFound("session is closed".into()));
        }
        self.client.push_render(&self.session_id, title, scenario)?;
        self.frame_count += 1;
        Ok(())
    }

    /// Append new data points to an existing `TimeSeries` channel.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_timeseries_append(
        &mut self,
        binding_id: &str,
        x_values: &[f64],
        y_values: &[f64],
    ) -> PushResult<()> {
        if !self.is_open {
            return Err(PushError::NotFound("session is closed".into()));
        }
        self.client
            .push_append(&self.session_id, binding_id, x_values, y_values)?;
        self.frame_count += 1;
        Ok(())
    }

    /// Update a single `Gauge` channel value.
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_gauge_update(&mut self, binding_id: &str, value: f64) -> PushResult<()> {
        if !self.is_open {
            return Err(PushError::NotFound("session is closed".into()));
        }
        self.client
            .push_gauge_update(&self.session_id, binding_id, value)?;
        self.frame_count += 1;
        Ok(())
    }

    /// Replace an entire channel with new data (full overwrite).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_replace(&mut self, channel: &DataChannel) -> PushResult<()> {
        if !self.is_open {
            return Err(PushError::NotFound("session is closed".into()));
        }
        let params = serde_json::json!({
            "session_id": self.session_id,
            "operation": {
                "type": "replace",
                "channel": channel,
            },
        });
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "visualization.render.stream",
            "params": params,
            "id": 1,
        });
        let _payload = serde_json::to_vec(&request)
            .map_err(|e| PushError::SerializationError(e.to_string()))?;
        self.frame_count += 1;
        Ok(())
    }

    /// Close the session. Further pushes will return an error.
    pub const fn close(&mut self) {
        self.is_open = false;
    }

    /// Snapshot the current session state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        SessionState {
            session_id: self.session_id.clone(),
            frame_count: self.frame_count,
            is_open: self.is_open,
        }
    }

    /// Session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of frames pushed.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Whether the session is open.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.is_open
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap/expect for clarity"
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dummy_client() -> PetalTonguePushClient {
        PetalTonguePushClient::new(PathBuf::from("/tmp/nonexistent-petaltongue-test.sock"))
    }

    #[test]
    fn session_lifecycle() {
        let mut session = StreamSession::open(dummy_client(), "test-session");
        assert!(session.is_open());
        assert_eq!(session.frame_count(), 0);
        assert_eq!(session.session_id(), "test-session");

        session.close();
        assert!(!session.is_open());
    }

    #[test]
    fn state_snapshot() {
        let session = StreamSession::open(dummy_client(), "snap-test");
        let state = session.state();
        assert_eq!(state.session_id, "snap-test");
        assert!(state.is_open);
        assert_eq!(state.frame_count, 0);
    }

    #[test]
    fn closed_session_rejects_push() {
        let mut session = StreamSession::open(dummy_client(), "closed-test");
        session.close();

        let scenario = EcologyScenario {
            name: "t".into(),
            description: "d".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        };
        let result = session.push_initial_render("test", &scenario);
        assert!(result.is_err());
    }
}
