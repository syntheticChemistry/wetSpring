// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-built stage definitions for domain-specific live pipelines.
//!
//! Each function returns a `Vec<PipelineStage>` with the canonical stages
//! for a pipeline domain, including scientific threshold ranges that
//! `petalTongue` renders as actionable indicators.

use super::PipelineStage;
use crate::visualization::types::ScientificRange;

/// Pre-built stage definitions for a 16S amplicon pipeline.
#[must_use]
pub fn amplicon_16s_stages() -> Vec<PipelineStage> {
    vec![
        PipelineStage {
            id: "qc".into(),
            name: "Quality Control".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![
                ScientificRange {
                    label: "Good pass rate".into(),
                    min: 0.8,
                    max: 1.0,
                    status: "normal".into(),
                },
                ScientificRange {
                    label: "Low pass rate".into(),
                    min: 0.0,
                    max: 0.8,
                    status: "warning".into(),
                },
            ],
        },
        PipelineStage {
            id: "dada2".into(),
            name: "DADA2 Denoising".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "diversity".into(),
            name: "Alpha/Beta Diversity".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![
                ScientificRange {
                    label: "Healthy diversity".into(),
                    min: 2.0,
                    max: 6.0,
                    status: "normal".into(),
                },
                ScientificRange {
                    label: "Low diversity".into(),
                    min: 0.0,
                    max: 2.0,
                    status: "warning".into(),
                },
            ],
        },
        PipelineStage {
            id: "taxonomy".into(),
            name: "Taxonomy Classification".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![
                ScientificRange {
                    label: "Good confidence".into(),
                    min: 0.8,
                    max: 1.0,
                    status: "normal".into(),
                },
                ScientificRange {
                    label: "Low confidence".into(),
                    min: 0.0,
                    max: 0.8,
                    status: "warning".into(),
                },
            ],
        },
        PipelineStage {
            id: "ordination".into(),
            name: "PCoA Ordination".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
    ]
}

/// Pre-built stage definitions for an LC-MS analytical pipeline.
#[must_use]
pub fn lcms_stages() -> Vec<PipelineStage> {
    vec![
        PipelineStage {
            id: "parse".into(),
            name: "Parse mzML/mzXML".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "peaks".into(),
            name: "Peak Detection".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "eic".into(),
            name: "EIC Extraction".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "match".into(),
            name: "Spectral Matching".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![
                ScientificRange {
                    label: "High confidence match".into(),
                    min: 0.8,
                    max: 1.0,
                    status: "normal".into(),
                },
                ScientificRange {
                    label: "Low confidence match".into(),
                    min: 0.0,
                    max: 0.8,
                    status: "warning".into(),
                },
            ],
        },
        PipelineStage {
            id: "quantitation".into(),
            name: "Quantitation".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
    ]
}

/// Pre-built stage definitions for a phylogenetic pipeline.
#[must_use]
pub fn phylo_stages() -> Vec<PipelineStage> {
    vec![
        PipelineStage {
            id: "alignment".into(),
            name: "Multiple Sequence Alignment".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "tree".into(),
            name: "Tree Construction".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
        PipelineStage {
            id: "selection".into(),
            name: "Selection Analysis (dN/dS)".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![
                ScientificRange {
                    label: "Purifying selection".into(),
                    min: 0.0,
                    max: 1.0,
                    status: "normal".into(),
                },
                ScientificRange {
                    label: "Positive selection".into(),
                    min: 1.0,
                    max: 10.0,
                    status: "warning".into(),
                },
            ],
        },
        PipelineStage {
            id: "clock".into(),
            name: "Molecular Clock".into(),
            status: "pending".into(),
            progress: 0.0,
            channels: vec![],
            ranges: vec![],
        },
    ]
}
