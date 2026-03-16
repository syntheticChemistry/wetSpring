// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live pipeline streaming session for progressive visualization.
//!
//! Wraps [`StreamSession`] with domain-aware stage progression for 16S
//! amplicon, LC-MS analytical, and phylogenetic pipelines. Each stage
//! pushes results to `petalTongue` as they complete, enabling scientists
//! to watch their data transform in real time.
//!
//! The session auto-discovers `petalTongue` at runtime; when unavailable,
//! it falls back to writing JSON snapshots to disk for offline viewing.

mod stages;

pub use stages::{amplicon_16s_stages, lcms_stages, phylo_stages};

use std::path::Path;

use super::ipc_push::{PetalTonguePushClient, PushResult};
use super::stream::StreamSession;
use super::types::{DataChannel, EcologyScenario, ScenarioNode, ScientificRange};

/// Pipeline domain for themed rendering and stage configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineDomain {
    /// 16S amplicon sequencing (FASTQ, QC, DADA2, diversity, taxonomy, `PCoA`).
    Amplicon16S,
    /// LC-MS analytical chemistry (mzML → peaks → EIC → spectral match → quantitation).
    LcMs,
    /// Phylogenetic analysis (alignment → tree → dN/dS → molecular clock).
    Phylogenetics,
    /// General-purpose pipeline (custom stages).
    General,
}

impl PipelineDomain {
    const fn domain_str(self) -> &'static str {
        match self {
            Self::LcMs => "measurement",
            Self::Amplicon16S | Self::Phylogenetics | Self::General => "ecology",
        }
    }

    const fn pipeline_name(self) -> &'static str {
        match self {
            Self::Amplicon16S => "16S Amplicon Pipeline",
            Self::LcMs => "LC-MS Analytical Pipeline",
            Self::Phylogenetics => "Phylogenetic Analysis",
            Self::General => "wetSpring Pipeline",
        }
    }
}

/// A stage in the live pipeline with visualization configuration.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage identifier (e.g. "qc", "dada2", "diversity").
    pub id: String,
    /// Human-readable stage name.
    pub name: String,
    /// Stage status: "pending", "running", "complete", "error".
    pub status: String,
    /// Progress gauge (0.0–1.0).
    pub progress: f64,
    /// Data channels produced by this stage.
    pub channels: Vec<DataChannel>,
    /// Scientific ranges for actionable thresholds.
    pub ranges: Vec<ScientificRange>,
}

/// Live pipeline session that streams stage results to petalTongue.
pub struct LivePipelineSession {
    stream: StreamSession,
    domain: PipelineDomain,
    stages: Vec<PipelineStage>,
    current_stage: usize,
}

impl LivePipelineSession {
    /// Create a new live pipeline session.
    ///
    /// Discovers petalTongue at runtime. If unavailable, streaming calls
    /// will return errors that callers can handle gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if petalTongue discovery fails.
    pub fn new(session_id: impl Into<String>, domain: PipelineDomain) -> PushResult<Self> {
        let client = PetalTonguePushClient::discover()?;
        let stream = StreamSession::open(client, session_id);
        Ok(Self {
            stream,
            domain,
            stages: Vec::new(),
            current_stage: 0,
        })
    }

    /// Create a session with an explicit client (for testing or manual socket).
    #[must_use]
    pub fn with_client(
        client: PetalTonguePushClient,
        session_id: impl Into<String>,
        domain: PipelineDomain,
    ) -> Self {
        let stream = StreamSession::open(client, session_id);
        Self {
            stream,
            domain,
            stages: Vec::new(),
            current_stage: 0,
        }
    }

    /// Initialize the pipeline with its stages and push the initial scenario.
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn initialize(&mut self, stages: Vec<PipelineStage>) -> PushResult<()> {
        self.stages = stages;
        let scenario = self.build_scenario();
        self.stream
            .push_initial_render(self.domain.pipeline_name(), &scenario)
    }

    /// Mark the current stage as running and update the gauge.
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn begin_stage(&mut self) -> PushResult<()> {
        if self.current_stage < self.stages.len() {
            self.stages[self.current_stage].status = "running".into();
            self.stages[self.current_stage].progress = 0.0;
            let gauge_id = format!("{}_progress", self.stages[self.current_stage].id);
            self.stream.push_gauge_update(&gauge_id, 0.0)?;
        }
        Ok(())
    }

    /// Update progress for the current stage (0.0–1.0).
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn update_progress(&mut self, progress: f64) -> PushResult<()> {
        if self.current_stage < self.stages.len() {
            self.stages[self.current_stage].progress = progress;
            let gauge_id = format!("{}_progress", self.stages[self.current_stage].id);
            self.stream.push_gauge_update(&gauge_id, progress)?;
        }
        Ok(())
    }

    /// Complete the current stage with result channels and advance to the next.
    ///
    /// Replaces the stage's channels with the actual computed results and
    /// pushes them to petalTongue.
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn complete_stage(&mut self, channels: Vec<DataChannel>) -> PushResult<()> {
        if self.current_stage < self.stages.len() {
            self.stages[self.current_stage].status = "complete".into();
            self.stages[self.current_stage].progress = 1.0;

            for channel in &channels {
                self.stream.push_replace(channel)?;
            }
            self.stages[self.current_stage].channels = channels;

            let gauge_id = format!("{}_progress", self.stages[self.current_stage].id);
            self.stream.push_gauge_update(&gauge_id, 1.0)?;

            self.current_stage += 1;
        }
        Ok(())
    }

    /// Push a live data update to a specific channel (e.g. streaming peaks).
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn push_live_update(&mut self, channel: &DataChannel) -> PushResult<()> {
        self.stream.push_replace(channel)
    }

    /// Push a gauge value (e.g. diversity score, pass rate).
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn push_gauge(&mut self, binding_id: &str, value: f64) -> PushResult<()> {
        self.stream.push_gauge_update(binding_id, value)
    }

    /// Append time-series data (e.g. chromatogram, rarefaction curve).
    ///
    /// # Errors
    ///
    /// Returns [`super::ipc_push::PushError`] on IPC failure.
    pub fn push_timeseries(&mut self, binding_id: &str, x: &[f64], y: &[f64]) -> PushResult<()> {
        self.stream.push_timeseries_append(binding_id, x, y)
    }

    /// Close the session.
    pub const fn close(&mut self) {
        self.stream.close();
    }

    /// Build the current scenario state for snapshot/JSON export.
    #[must_use]
    pub fn build_scenario(&self) -> EcologyScenario {
        let mut nodes: Vec<ScenarioNode> = self
            .stages
            .iter()
            .map(|stage| {
                let health = match stage.status.as_str() {
                    "complete" => 100,
                    "running" => 50,
                    "error" => 0,
                    _ => 25,
                };
                ScenarioNode {
                    id: stage.id.clone(),
                    name: stage.name.clone(),
                    node_type: "pipeline".into(),
                    family: "wetspring".into(),
                    status: stage.status.clone(),
                    health,
                    confidence: 100,
                    capabilities: vec![],
                    data_channels: stage.channels.clone(),
                    scientific_ranges: stage.ranges.clone(),
                }
            })
            .collect();

        for stage in &self.stages {
            let gauge = DataChannel::Gauge {
                id: format!("{}_progress", stage.id),
                label: format!("{} Progress", stage.name),
                value: stage.progress,
                min: 0.0,
                max: 1.0,
                unit: "fraction".into(),
                normal_range: [0.8, 1.0],
                warning_range: [0.0, 0.8],
            };
            if let Some(node) = nodes.iter_mut().find(|n| n.id == stage.id) {
                node.data_channels.push(gauge);
            }
        }

        EcologyScenario {
            name: self.domain.pipeline_name().into(),
            description: format!(
                "Live {} pipeline — {} stages",
                self.domain.domain_str(),
                self.stages.len()
            ),
            version: "1.0.0".into(),
            mode: "live-ecosystem".into(),
            domain: self.domain.domain_str().into(),
            nodes,
            edges: vec![],
        }
    }

    /// Export the current scenario state to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be written.
    pub fn export_json(&self, path: &Path) -> std::io::Result<()> {
        let scenario = self.build_scenario();
        let json = serde_json::to_string_pretty(&scenario).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Access the underlying stream session.
    #[must_use]
    pub const fn stream(&self) -> &StreamSession {
        &self.stream
    }

    /// Current pipeline domain.
    #[must_use]
    pub const fn domain(&self) -> PipelineDomain {
        self.domain
    }

    /// Number of stages in the pipeline.
    #[must_use]
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Index of the current (active or next) stage.
    #[must_use]
    pub const fn current_stage_index(&self) -> usize {
        self.current_stage
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_domain_strings() {
        assert_eq!(PipelineDomain::Amplicon16S.domain_str(), "ecology");
        assert_eq!(PipelineDomain::LcMs.domain_str(), "measurement");
        assert_eq!(PipelineDomain::Phylogenetics.domain_str(), "ecology");
    }

    #[test]
    fn amplicon_stages_count() {
        let stages = amplicon_16s_stages();
        assert_eq!(stages.len(), 5);
        assert_eq!(stages[0].id, "qc");
        assert_eq!(stages[4].id, "ordination");
    }

    #[test]
    fn lcms_stages_count() {
        let stages = lcms_stages();
        assert_eq!(stages.len(), 5);
        assert_eq!(stages[0].id, "parse");
        assert_eq!(stages[4].id, "quantitation");
    }

    #[test]
    fn phylo_stages_count() {
        let stages = phylo_stages();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].id, "alignment");
        assert_eq!(stages[3].id, "clock");
    }

    #[test]
    fn build_scenario_from_stages() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent-live-test.sock"));
        let session =
            LivePipelineSession::with_client(client, "test-live", PipelineDomain::Amplicon16S);
        let scenario = session.build_scenario();
        assert_eq!(scenario.name, "16S Amplicon Pipeline");
        assert_eq!(scenario.domain, "ecology");
        assert!(scenario.nodes.is_empty());
    }

    #[test]
    fn build_scenario_with_stages() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent-live-test2.sock"));
        let mut session =
            LivePipelineSession::with_client(client, "test-stages", PipelineDomain::LcMs);
        session.stages = lcms_stages();
        let scenario = session.build_scenario();
        assert_eq!(scenario.nodes.len(), 5);
        assert_eq!(scenario.domain, "measurement");
        for node in &scenario.nodes {
            assert!(
                !node.data_channels.is_empty(),
                "each stage should have a progress gauge"
            );
        }
    }

    #[test]
    fn export_json_writes_file() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent-export-test.sock"));
        let mut session =
            LivePipelineSession::with_client(client, "test-export", PipelineDomain::Phylogenetics);
        session.stages = phylo_stages();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pipeline.json");
        session.export_json(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Phylogenetic Analysis"));
        assert!(content.contains("alignment"));
    }

    #[test]
    fn session_lifecycle() {
        let client = PetalTonguePushClient::new(
            std::env::temp_dir().join("nonexistent-lifecycle-test.sock"),
        );
        let mut session =
            LivePipelineSession::with_client(client, "test-lifecycle", PipelineDomain::General);
        assert_eq!(session.domain(), PipelineDomain::General);
        assert_eq!(session.stage_count(), 0);
        assert_eq!(session.current_stage_index(), 0);

        session.stages = amplicon_16s_stages();
        assert_eq!(session.stage_count(), 5);

        session.close();
        assert!(!session.stream().is_open());
    }

    #[test]
    fn scientific_ranges_on_stages() {
        let stages = amplicon_16s_stages();
        assert!(!stages[0].ranges.is_empty(), "QC should have ranges");
        assert!(!stages[2].ranges.is_empty(), "diversity should have ranges");
        assert!(!stages[3].ranges.is_empty(), "taxonomy should have ranges");
    }
}
