// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming visualization session for progressive `petalTongue` rendering.
//!
//! `StreamSession` wraps [`PetalTonguePushClient`] with session state tracking,
//! typed push helpers, and backpressure awareness. Follows the
//! `visualization.render.stream` JSON-RPC protocol with `append`, `set_value`,
//! and `replace` operations.

use std::time::{Duration, Instant};

use super::ipc_push::{PetalTonguePushClient, PushError, PushResult};
use super::types::{DataChannel, EcologyScenario};

/// Backpressure configuration for streaming sessions.
///
/// Follows healthSpring's pattern: after repeated slow pushes the session
/// enters cooldown to avoid overwhelming petalTongue.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum time to wait for a push to complete before counting it as slow.
    pub timeout: Duration,
    /// Cooldown period after too many slow pushes.
    pub cooldown: Duration,
    /// Number of consecutive slow pushes before entering cooldown.
    pub max_slow_pushes: u32,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(500),
            cooldown: Duration::from_millis(200),
            max_slow_pushes: 3,
        }
    }
}

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
    backpressure: BackpressureConfig,
    consecutive_slow: u32,
    cooldown_until: Option<Instant>,
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
            backpressure: BackpressureConfig::default(),
            consecutive_slow: 0,
            cooldown_until: None,
        }
    }

    /// Open a streaming session with explicit backpressure configuration.
    #[must_use]
    pub fn open_with_backpressure(
        client: PetalTonguePushClient,
        session_id: impl Into<String>,
        config: BackpressureConfig,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            client,
            frame_count: 0,
            is_open: true,
            backpressure: config,
            consecutive_slow: 0,
            cooldown_until: None,
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
        self.check_open()?;
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
        self.check_open()?;
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
        self.check_open()?;
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
        self.check_open()?;
        self.client.push_replace(&self.session_id, channel)?;
        self.frame_count += 1;
        Ok(())
    }

    // ── Domain-specific push helpers ───────────────────────────────────

    /// Push a diversity metrics update (Shannon, Simpson, evenness).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_diversity_update(
        &mut self,
        binding_id: &str,
        shannon: f64,
        simpson: f64,
        evenness: f64,
    ) -> PushResult<()> {
        self.check_open()?;
        self.check_backpressure()?;
        let start = Instant::now();
        let channel = DataChannel::Bar {
            id: binding_id.into(),
            label: "Diversity Update".into(),
            categories: vec!["Shannon".into(), "Simpson".into(), "Evenness".into()],
            values: vec![shannon, simpson, evenness],
            unit: "index".into(),
        };
        self.client.push_replace(&self.session_id, &channel)?;
        self.record_push_timing(start);
        self.frame_count += 1;
        Ok(())
    }

    /// Push an ODE time-step update (append new point to all state variables).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_ode_step(
        &mut self,
        binding_id: &str,
        t: f64,
        state_vars: &[f64],
    ) -> PushResult<()> {
        self.check_open()?;
        self.check_backpressure()?;
        let start = Instant::now();
        self.client
            .push_append(&self.session_id, binding_id, &[t], state_vars)?;
        self.record_push_timing(start);
        self.frame_count += 1;
        Ok(())
    }

    /// Push a pipeline progress update (reads processed, pass rate).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] on connection or RPC failure.
    pub fn push_pipeline_progress(
        &mut self,
        stage_id: &str,
        reads_processed: f64,
        pass_rate: f64,
    ) -> PushResult<()> {
        self.check_open()?;
        self.check_backpressure()?;
        let start = Instant::now();
        self.client
            .push_gauge_update(&self.session_id, stage_id, reads_processed)?;
        self.client
            .push_gauge_update(&self.session_id, &format!("{stage_id}_rate"), pass_rate)?;
        self.record_push_timing(start);
        self.frame_count += 1;
        Ok(())
    }

    // ── Session lifecycle ───────────────────────────────────────────────

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

    /// Current backpressure configuration.
    #[must_use]
    pub const fn backpressure(&self) -> &BackpressureConfig {
        &self.backpressure
    }

    /// Whether the session is currently in cooldown.
    #[must_use]
    pub fn in_cooldown(&self) -> bool {
        self.cooldown_until
            .is_some_and(|until| Instant::now() < until)
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn check_open(&self) -> PushResult<()> {
        if self.is_open {
            Ok(())
        } else {
            Err(PushError::NotFound("session is closed".into()))
        }
    }

    fn check_backpressure(&self) -> PushResult<()> {
        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                return Err(PushError::NotFound(
                    "session in backpressure cooldown".into(),
                ));
            }
        }
        Ok(())
    }

    fn record_push_timing(&mut self, start: Instant) {
        if start.elapsed() > self.backpressure.timeout {
            self.consecutive_slow += 1;
            if self.consecutive_slow >= self.backpressure.max_slow_pushes {
                self.cooldown_until = Some(Instant::now() + self.backpressure.cooldown);
                self.consecutive_slow = 0;
            }
        } else {
            self.consecutive_slow = 0;
        }
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

    fn empty_scenario() -> EcologyScenario {
        EcologyScenario {
            name: "t".into(),
            description: "d".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        }
    }

    #[test]
    fn closed_session_rejects_initial_render() {
        let mut session = StreamSession::open(dummy_client(), "closed-test");
        session.close();
        let result = session.push_initial_render("test", &empty_scenario());
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_timeseries() {
        let mut session = StreamSession::open(dummy_client(), "closed-ts");
        session.close();
        let result = session.push_timeseries_append("ch1", &[1.0], &[2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_gauge() {
        let mut session = StreamSession::open(dummy_client(), "closed-gauge");
        session.close();
        let result = session.push_gauge_update("g1", 42.0);
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_replace() {
        let mut session = StreamSession::open(dummy_client(), "closed-replace");
        session.close();
        let channel = DataChannel::Gauge {
            id: "ch".into(),
            label: "test".into(),
            value: 1.0,
            min: 0.0,
            max: 10.0,
            unit: "x".into(),
            normal_range: [0.0, 5.0],
            warning_range: [5.0, 8.0],
        };
        let result = session.push_replace(&channel);
        assert!(result.is_err());
    }

    #[test]
    fn open_session_push_render_fails_on_missing_socket() {
        let mut session = StreamSession::open(dummy_client(), "open-fail");
        let result = session.push_initial_render("test", &empty_scenario());
        assert!(result.is_err());
    }

    #[test]
    fn open_session_push_timeseries_fails_on_missing_socket() {
        let mut session = StreamSession::open(dummy_client(), "open-ts-fail");
        let result = session.push_timeseries_append("ch1", &[1.0], &[2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn open_session_push_gauge_fails_on_missing_socket() {
        let mut session = StreamSession::open(dummy_client(), "open-gauge-fail");
        let result = session.push_gauge_update("g1", 42.0);
        assert!(result.is_err());
    }

    #[test]
    fn close_is_idempotent() {
        let mut session = StreamSession::open(dummy_client(), "idempotent");
        session.close();
        session.close();
        assert!(!session.is_open());
    }

    #[test]
    fn backpressure_config_default() {
        let config = BackpressureConfig::default();
        assert_eq!(config.timeout, Duration::from_millis(500));
        assert_eq!(config.cooldown, Duration::from_millis(200));
        assert_eq!(config.max_slow_pushes, 3);
    }

    #[test]
    fn open_with_backpressure_sets_config() {
        let config = BackpressureConfig {
            timeout: Duration::from_millis(100),
            cooldown: Duration::from_millis(50),
            max_slow_pushes: 2,
        };
        let session = StreamSession::open_with_backpressure(dummy_client(), "bp-test", config);
        assert!(session.is_open());
        assert_eq!(session.backpressure().timeout, Duration::from_millis(100));
        assert!(!session.in_cooldown());
    }

    #[test]
    fn closed_session_rejects_diversity_update() {
        let mut session = StreamSession::open(dummy_client(), "closed-div");
        session.close();
        let result = session.push_diversity_update("div", 1.0, 0.5, 0.8);
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_ode_step() {
        let mut session = StreamSession::open(dummy_client(), "closed-ode");
        session.close();
        let result = session.push_ode_step("ode", 1.0, &[0.1, 0.2]);
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_pipeline_progress() {
        let mut session = StreamSession::open(dummy_client(), "closed-pp");
        session.close();
        let result = session.push_pipeline_progress("stage", 100.0, 0.95);
        assert!(result.is_err());
    }
}
