// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ecology-specific streaming helpers for `petalTongue` `StreamSession`.
//!
//! Extends the generic stream session with domain-aware push methods
//! that compute and push diversity metrics, Bray-Curtis updates,
//! rarefaction points, and Anderson W/P(QS) gauges in a single call.
//!
//! These methods compose real `barraCuda` math with IPC push — the
//! petalTongue session sees typed, labeled channels rather than raw
//! numbers.

use super::DataChannel;
use super::ipc_push::PushResult;
use super::stream::StreamSession;
use crate::bio::diversity;

/// Push a complete diversity metrics frame for a single community sample.
///
/// Computes Shannon H', Simpson D, and Pielou J from raw counts and
/// pushes a `Bar` channel replacement. The channel ID is `{prefix}_diversity`.
///
///
/// # Errors
///
/// Returns [`super::ipc_push::PushError`] on IPC failure.
pub fn push_diversity_frame(
    session: &mut StreamSession,
    prefix: &str,
    counts: &[f64],
) -> PushResult<()> {
    let shannon = diversity::shannon(counts);
    let simpson = diversity::simpson(counts);
    let pielou = diversity::pielou_evenness(counts);

    let channel = DataChannel::Bar {
        id: format!("{prefix}_diversity"),
        label: format!("{prefix} Diversity"),
        categories: vec!["Shannon H'".into(), "Simpson D".into(), "Pielou J".into()],
        values: vec![shannon, simpson, pielou],
        unit: "index".into(),
    };
    session.push_replace(&channel)
}

/// Push a Bray-Curtis distance matrix update.
///
/// Recomputes the full distance matrix from the given samples and pushes
/// it as a `Heatmap` channel replacement.
///
/// # Errors
///
/// Returns [`super::ipc_push::PushError`] on IPC failure.
pub fn push_bray_curtis_update(
    session: &mut StreamSession,
    channel_id: &str,
    samples: &[Vec<f64>],
    labels: &[String],
) -> PushResult<()> {
    let bc_matrix = diversity::bray_curtis_matrix(samples);
    let channel = DataChannel::Heatmap {
        id: channel_id.into(),
        label: "Bray-Curtis Dissimilarity".into(),
        x_labels: labels.to_vec(),
        y_labels: labels.to_vec(),
        values: bc_matrix,
        unit: "BC index".into(),
    };
    session.push_replace(&channel)
}

/// Push a rarefaction curve point (append to existing `TimeSeries`).
///
/// # Errors
///
/// Returns [`super::ipc_push::PushError`] on IPC failure.
pub fn push_rarefaction_point(
    session: &mut StreamSession,
    channel_id: &str,
    depth: f64,
    richness: f64,
) -> PushResult<()> {
    session.push_timeseries_append(channel_id, &[depth], &[richness])
}

/// Push Anderson W disorder parameter and P(QS) probability as gauges.
///
/// Updates two gauge channels: `{prefix}_w` and `{prefix}_pqs`.
///
/// # Errors
///
/// Returns [`super::ipc_push::PushError`] on IPC failure.
pub fn push_anderson_w(
    session: &mut StreamSession,
    prefix: &str,
    w_value: f64,
    p_qs: f64,
) -> PushResult<()> {
    session.push_gauge_update(&format!("{prefix}_w"), w_value)?;
    session.push_gauge_update(&format!("{prefix}_pqs"), p_qs)?;
    Ok(())
}

/// Push a kinetics time-step (Gompertz, Monod, etc.) as `TimeSeries` append.
///
/// Appends a single (t, y) point to the named channel.
///
/// # Errors
///
/// Returns [`super::ipc_push::PushError`] on IPC failure.
pub fn push_kinetics_step(
    session: &mut StreamSession,
    channel_id: &str,
    t: f64,
    y: f64,
) -> PushResult<()> {
    session.push_timeseries_append(channel_id, &[t], &[y])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::ipc_push::PetalTonguePushClient;

    fn dummy_session() -> StreamSession {
        let client = PetalTonguePushClient::new(
            std::env::temp_dir().join("wetspring-stream-ecology-test.sock"),
        );
        StreamSession::open(client, "ecology-test")
    }

    #[test]
    fn push_diversity_frame_fails_on_missing_socket() {
        let mut s = dummy_session();
        let result = push_diversity_frame(&mut s, "soil", &[10.0, 20.0, 30.0]);
        assert!(result.is_err());
    }

    #[test]
    fn push_bray_curtis_fails_on_missing_socket() {
        let mut s = dummy_session();
        let samples = vec![vec![10.0, 20.0], vec![15.0, 25.0]];
        let labels = vec!["S1".into(), "S2".into()];
        let result = push_bray_curtis_update(&mut s, "bc", &samples, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn push_anderson_w_fails_on_missing_socket() {
        let mut s = dummy_session();
        let result = push_anderson_w(&mut s, "digester", 12.0, 0.85);
        assert!(result.is_err());
    }

    #[test]
    fn push_rarefaction_point_fails_on_missing_socket() {
        let mut s = dummy_session();
        let result = push_rarefaction_point(&mut s, "rare_soil", 100.0, 15.0);
        assert!(result.is_err());
    }

    #[test]
    fn push_kinetics_step_fails_on_missing_socket() {
        let mut s = dummy_session();
        let result = push_kinetics_step(&mut s, "gomp_corn", 5.0, 120.0);
        assert!(result.is_err());
    }

    #[test]
    fn closed_session_rejects_ecology_push() {
        let mut s = dummy_session();
        s.close();
        let result = push_diversity_frame(&mut s, "closed", &[1.0, 2.0]);
        assert!(result.is_err());
    }
}
