// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark scenario: three-tier Galaxy vs CPU vs GPU comparisons.

use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, node, scaffold};

/// A single benchmark tier result for visualization.
pub struct TierResult {
    /// Stage name (e.g. "DADA2", "Taxonomy", "Total").
    pub stage: String,
    /// Galaxy/Python baseline time (seconds).
    pub galaxy_s: f64,
    /// Rust CPU time (seconds).
    pub cpu_s: f64,
    /// Rust GPU time (seconds).
    pub gpu_s: f64,
}

/// Build a benchmark comparison scenario from three-tier results.
///
/// Produces grouped bar charts comparing Galaxy, Rust CPU, and Rust GPU
/// performance for each pipeline stage.
#[must_use]
pub fn benchmark_scenario(tiers: &[TierResult]) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Three-Tier Benchmark",
        "Galaxy/Python → Rust CPU → Rust GPU performance comparison",
    );
    s.domain = "default".into();

    let mut bench_node = node(
        "benchmark",
        "Three-Tier Benchmark",
        "data",
        &["science.benchmark"],
    );

    let stages: Vec<String> = tiers.iter().map(|t| t.stage.clone()).collect();
    let galaxy_times: Vec<f64> = tiers.iter().map(|t| t.galaxy_s).collect();
    let cpu_times: Vec<f64> = tiers.iter().map(|t| t.cpu_s).collect();
    let gpu_times: Vec<f64> = tiers.iter().map(|t| t.gpu_s).collect();

    bench_node.data_channels.push(bar(
        "galaxy_times",
        "Galaxy/Python (baseline)",
        &stages,
        &galaxy_times,
        "seconds",
    ));
    bench_node.data_channels.push(bar(
        "cpu_times",
        "Rust CPU (barraCuda)",
        &stages,
        &cpu_times,
        "seconds",
    ));
    bench_node.data_channels.push(bar(
        "gpu_times",
        "Rust GPU (barraCuda WGSL)",
        &stages,
        &gpu_times,
        "seconds",
    ));

    let speedups: Vec<f64> = tiers
        .iter()
        .map(|t| {
            if t.gpu_s > 0.0 {
                t.galaxy_s / t.gpu_s
            } else {
                0.0
            }
        })
        .collect();
    bench_node.data_channels.push(bar(
        "gpu_speedup",
        "GPU Speedup vs Galaxy",
        &stages,
        &speedups,
        "×",
    ));

    s.nodes.push(bench_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_scenario_builds() {
        let tiers = vec![
            TierResult {
                stage: "DADA2".into(),
                galaxy_s: 6.8,
                cpu_s: 0.33,
                gpu_s: 0.013,
            },
            TierResult {
                stage: "Taxonomy".into(),
                galaxy_s: 3.0,
                cpu_s: 0.115,
                gpu_s: 0.011,
            },
        ];
        let (scenario, _) = benchmark_scenario(&tiers);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 4);
    }
}
