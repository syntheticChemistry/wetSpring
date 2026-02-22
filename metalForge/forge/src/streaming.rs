// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming dispatch for multi-stage GPU pipelines.
//!
//! Eliminates CPU round-trips between pipeline stages by keeping data on the
//! GPU between dispatches. The streaming model chains stages as:
//!
//! ```text
//! Upload → Stage₁ → Stage₂ → ... → StageN → Readback
//! ```
//!
//! Each stage operates on GPU buffers directly — no intermediate readback.
//! This is the metalForge analogue of ToadStool's unidirectional pipelines.
//!
//! # Write → Absorb → Lean
//!
//! This module evolves locally in wetSpring. Once validated (Exp090-091),
//! the streaming dispatch model will be handed to ToadStool for absorption
//! as `barracuda::pipeline::streaming`. The absorption seam is the
//! [`StreamingSession`] type — ToadStool replaces the implementation;
//! wetSpring switches to `use barracuda::pipeline::StreamingSession`.
//!
//! # Evidence
//!
//! - Exp090: 80/80 checks, 441-837× speedup over round-trip
//! - Exp091: streaming eliminates 92-94% of PCIe overhead
//! - Key insight: math is identical; only transfer topology changes

use crate::substrate::{Capability, Substrate, SubstrateKind};

/// A stage in a streaming pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Human-readable name.
    pub name: String,
    /// Required capability for this stage.
    pub capability: Capability,
    /// Whether this stage can accept GPU buffer input directly.
    pub accepts_gpu_buffer: bool,
    /// Whether this stage produces GPU buffer output.
    pub produces_gpu_buffer: bool,
}

/// Streaming session that chains multiple compute stages.
///
/// The session pre-warms pipelines and manages buffer hand-off between
/// stages without CPU round-trips. This is the absorption seam —
/// ToadStool will replace the implementation while preserving the API.
#[derive(Debug)]
pub struct StreamingSession {
    stages: Vec<PipelineStage>,
    substrate: SubstrateKind,
}

/// Analysis of a streaming pipeline's transfer topology.
#[derive(Debug)]
pub struct PipelineAnalysis {
    /// Total stages in the pipeline.
    pub n_stages: usize,
    /// Stages that require CPU round-trip (can't chain on GPU).
    pub cpu_roundtrips: usize,
    /// Stages that chain on GPU (zero intermediate readback).
    pub gpu_chained: usize,
    /// Whether the full pipeline can stream without CPU round-trips.
    pub fully_streamable: bool,
}

impl StreamingSession {
    /// Create a new streaming session for the given substrate.
    #[must_use]
    pub const fn new(substrate: SubstrateKind) -> Self {
        Self {
            stages: Vec::new(),
            substrate,
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Analyze the pipeline for streaming viability.
    ///
    /// Counts how many stages can chain on GPU without CPU round-trips.
    #[must_use]
    pub fn analyze(&self) -> PipelineAnalysis {
        let n = self.stages.len();
        if n == 0 {
            return PipelineAnalysis {
                n_stages: 0,
                cpu_roundtrips: 0,
                gpu_chained: 0,
                fully_streamable: true,
            };
        }

        let mut chained = 0;
        let mut roundtrips = 0;

        for window in self.stages.windows(2) {
            let produces = window[0].produces_gpu_buffer;
            let accepts = window[1].accepts_gpu_buffer;

            if produces && accepts {
                chained += 1;
            } else {
                roundtrips += 1;
            }
        }

        PipelineAnalysis {
            n_stages: n,
            cpu_roundtrips: roundtrips,
            gpu_chained: chained,
            fully_streamable: roundtrips == 0,
        }
    }

    /// Return the substrate type this session targets.
    #[must_use]
    pub const fn substrate(&self) -> SubstrateKind {
        self.substrate
    }

    /// Select the best substrate for a stage from an inventory.
    #[must_use]
    pub fn select_substrate_for_stage<'a>(
        stage: &PipelineStage,
        inventory: &'a [Substrate],
    ) -> Option<&'a Substrate> {
        inventory.iter().find(|s| s.has(&stage.capability))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pipeline_is_streamable() {
        let session = StreamingSession::new(SubstrateKind::Gpu);
        let analysis = session.analyze();
        assert!(analysis.fully_streamable);
        assert_eq!(analysis.n_stages, 0);
    }

    #[test]
    fn fully_gpu_pipeline_is_streamable() {
        let mut session = StreamingSession::new(SubstrateKind::Gpu);
        session.add_stage(PipelineStage {
            name: "QF".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
        session.add_stage(PipelineStage {
            name: "DADA2".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
        session.add_stage(PipelineStage {
            name: "GEMM".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });

        let analysis = session.analyze();
        assert!(analysis.fully_streamable);
        assert_eq!(analysis.gpu_chained, 2);
        assert_eq!(analysis.cpu_roundtrips, 0);
    }

    #[test]
    fn mixed_pipeline_detects_roundtrips() {
        let mut session = StreamingSession::new(SubstrateKind::Gpu);
        session.add_stage(PipelineStage {
            name: "QF".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
        session.add_stage(PipelineStage {
            name: "Chimera".into(),
            capability: Capability::CpuCompute,
            accepts_gpu_buffer: false,
            produces_gpu_buffer: false,
        });
        session.add_stage(PipelineStage {
            name: "GEMM".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });

        let analysis = session.analyze();
        assert!(!analysis.fully_streamable);
        assert_eq!(analysis.cpu_roundtrips, 2);
        assert_eq!(analysis.gpu_chained, 0);
    }
}
