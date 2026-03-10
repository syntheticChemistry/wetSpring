// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stochastic scenario: Gillespie SSA trajectories and ensemble statistics.

use crate::bio::gillespie::{self, EnsembleStats, Trajectory};
use crate::visualization::ScientificRange;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{distribution, gauge, node, scaffold, timeseries};

/// Build a stochastic simulation scenario from ensemble results.
///
/// Produces:
/// - **`TimeSeries`**: individual trajectory traces from the ensemble
/// - **Distribution**: final-state distribution across replicates
/// - **Gauge**: mean population ± SD
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "population counts ≤ millions")]
pub fn stochastic_scenario(
    trajectories: &[Trajectory],
    stats: &EnsembleStats,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Stochastic Simulation",
        "Gillespie SSA birth-death ensemble trajectories",
    );

    let mut stoch_node = node(
        "stochastic",
        "Stochastic Simulation",
        "compute",
        &["science.stochastic"],
    );

    let max_traces = trajectories.len().min(10);
    for (i, traj) in trajectories.iter().take(max_traces).enumerate() {
        let y: Vec<f64> = traj
            .times
            .iter()
            .enumerate()
            .map(|(step, _)| {
                let st = traj.state_at(step);
                st.first().copied().unwrap_or(0) as f64
            })
            .collect();
        stoch_node.data_channels.push(timeseries(
            &format!("trajectory_{i}"),
            &format!("Trajectory {i}"),
            "Time",
            "Population",
            "count",
            &traj.times,
            &y,
        ));
    }

    let finals: Vec<f64> = stats.final_counts.iter().map(|&c| c as f64).collect();
    stoch_node.data_channels.push(distribution(
        "final_state_dist",
        "Final State Distribution",
        "count",
        &finals,
        stats.mean,
        stats.std_dev,
    ));

    let upper = 3.0_f64.mul_add(stats.std_dev, stats.mean);
    stoch_node.data_channels.push(gauge(
        "mean_population",
        "Mean Final Population",
        stats.mean,
        0.0,
        if upper > 0.0 { upper } else { 100.0 },
        "count",
        [
            (stats.mean - stats.std_dev).max(0.0),
            stats.mean + stats.std_dev,
        ],
        [0.0, (stats.mean - stats.std_dev).max(0.0)],
    ));

    let lo = 2.0_f64.mul_add(-stats.std_dev, stats.mean).max(0.0);
    let hi = 2.0_f64.mul_add(stats.std_dev, stats.mean);
    let gauge_max = if upper > 0.0 { upper } else { 100.0 };
    stoch_node.scientific_ranges.push(ScientificRange {
        label: "Within 2 SD of mean".into(),
        min: lo,
        max: hi,
        status: "normal".into(),
    });
    stoch_node.scientific_ranges.push(ScientificRange {
        label: "Below 2 SD of mean".into(),
        min: 0.0,
        max: lo,
        status: "warning".into(),
    });
    stoch_node.scientific_ranges.push(ScientificRange {
        label: "Above 2 SD of mean".into(),
        min: hi,
        max: gauge_max,
        status: "warning".into(),
    });

    s.nodes.push(stoch_node);
    (s, vec![])
}

/// Convenience: run a birth-death ensemble and build the scenario.
#[must_use]
pub fn birth_death_scenario(
    k_birth: f64,
    k_death: f64,
    t_max: f64,
    n_runs: usize,
    seed: u64,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let stats = gillespie::birth_death_ensemble(k_birth, k_death, t_max, n_runs, seed);

    let trajectories: Vec<Trajectory> = (0..n_runs.min(10))
        .map(|i| gillespie::birth_death_ssa(k_birth, k_death, t_max, seed + i as u64))
        .collect();

    stochastic_scenario(&trajectories, &stats)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use unwrap/expect for clarity")]
mod tests {
    use super::*;

    #[test]
    fn birth_death_produces_channels() {
        let (scenario, edges) = birth_death_scenario(1.0, 0.5, 10.0, 5, 42);
        assert_eq!(scenario.nodes.len(), 1);
        assert!(scenario.nodes[0].data_channels.len() >= 3);
        assert!(edges.is_empty());
    }

    #[test]
    fn stochastic_serializes() {
        let (scenario, _) = birth_death_scenario(1.0, 0.5, 10.0, 3, 99);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("trajectory_"));
        assert!(json.contains("final_state_dist"));
        assert!(json.contains("mean_population"));
    }
}
