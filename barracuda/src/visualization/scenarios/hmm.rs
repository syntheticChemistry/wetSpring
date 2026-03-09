// SPDX-License-Identifier: AGPL-3.0-or-later
//! HMM scenario: forward log-alpha, Viterbi path, posterior probability heatmap.

use crate::bio::hmm::{self, HmmModel};
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, heatmap, node, scaffold, timeseries};

/// Build an HMM visualization scenario from a model and observation sequence.
///
/// Produces:
/// - **`TimeSeries`**: forward log-alpha per state over time steps
/// - **Bar**: Viterbi decoded state path (most likely state per timestep)
/// - **Heatmap**: posterior probability matrix (states × time)
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "timestep counts ≤ thousands")]
pub fn hmm_scenario(
    model: &HmmModel,
    observations: &[usize],
    state_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring HMM Analysis",
        "Forward algorithm, Viterbi decoding, posterior probabilities",
    );

    let forward_result = hmm::forward(model, observations);
    let viterbi_result = hmm::viterbi(model, observations);

    let mut hmm_node = node("hmm", "HMM Analysis", "compute", &["science.hmm"]);

    let time_steps: Vec<f64> = (0..observations.len()).map(|t| t as f64).collect();

    for (state_idx, label) in state_labels.iter().enumerate() {
        let log_alphas: Vec<f64> = (0..forward_result.n_steps)
            .map(|t| forward_result.log_alpha[t * forward_result.n_states + state_idx])
            .collect();
        hmm_node.data_channels.push(timeseries(
            &format!("forward_state_{state_idx}"),
            &format!("Forward log-α: {label}"),
            "Time step",
            "log P(state)",
            "log-prob",
            &time_steps,
            &log_alphas,
        ));
    }

    let viterbi_f64: Vec<f64> = viterbi_result.path.iter().map(|&s| s as f64).collect();
    hmm_node.data_channels.push(bar(
        "viterbi_path",
        "Viterbi Decoded Path",
        &(0..observations.len())
            .map(|t| format!("t{t}"))
            .collect::<Vec<_>>(),
        &viterbi_f64,
        "state",
    ));

    build_posterior_heatmap(&mut hmm_node, &forward_result, state_labels);

    s.nodes.push(hmm_node);
    (s, vec![])
}

fn build_posterior_heatmap(
    hmm_node: &mut crate::visualization::ScenarioNode,
    fwd: &hmm::ForwardResult,
    state_labels: &[String],
) {
    let time_labels: Vec<String> = (0..fwd.n_steps).map(|t| format!("t{t}")).collect();
    let mut posterior: Vec<f64> = Vec::with_capacity(fwd.n_states * fwd.n_steps);

    for t in 0..fwd.n_steps {
        let row_start = t * fwd.n_states;
        let row = &fwd.log_alpha[row_start..row_start + fwd.n_states];
        let max_log = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = max_log + row.iter().map(|&v| (v - max_log).exp()).sum::<f64>().ln();
        for &log_a in row {
            posterior.push((log_a - log_sum).exp());
        }
    }

    hmm_node.data_channels.push(heatmap(
        "posterior",
        "Posterior Probabilities",
        &time_labels,
        state_labels,
        &posterior,
        "probability",
    ));
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap/expect for clarity"
)]
mod tests {
    use super::*;

    fn two_state_model() -> HmmModel {
        HmmModel {
            n_states: 2,
            log_pi: vec![(-0.5_f64).ln(), (-0.5_f64).ln()],
            log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
            n_symbols: 2,
            log_emit: vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
        }
    }

    #[test]
    fn hmm_produces_channels() {
        let model = two_state_model();
        let obs = vec![0, 1, 0, 1, 0];
        let labels = vec!["State0".into(), "State1".into()];
        let (scenario, edges) = hmm_scenario(&model, &obs, &labels);
        assert_eq!(scenario.nodes.len(), 1);
        assert!(scenario.nodes[0].data_channels.len() >= 3);
        assert!(edges.is_empty());
    }

    #[test]
    fn hmm_serializes() {
        let model = two_state_model();
        let obs = vec![0, 1, 0];
        let labels = vec!["S0".into(), "S1".into()];
        let (scenario, _) = hmm_scenario(&model, &obs, &labels);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("viterbi_path"));
        assert!(json.contains("posterior"));
    }
}
