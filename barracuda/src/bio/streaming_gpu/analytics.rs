// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming analytics — multi-stage GPU pipelines and result types.
//!
//! Combines taxonomy, alpha diversity, and beta diversity into single
//! streaming sessions. Also provides a CPU-equivalent benchmark function.

use super::GpuPipelineSession;
use crate::bio::diversity;
use crate::bio::taxonomy::{Classification, ClassifyParams, NaiveBayesClassifier};
use crate::error::Result;
use std::time::Instant;

/// Per-sample alpha diversity metrics.
#[derive(Debug)]
pub struct AlphaDiversity {
    /// Shannon entropy.
    pub shannon: f64,
    /// Simpson diversity (1 - dominance).
    pub simpson: f64,
    /// Observed species count.
    pub observed: f64,
}

/// Results from the full streaming analytics pipeline.
#[derive(Debug)]
pub struct FullStreamingResult {
    /// Per-read taxonomy classifications.
    pub classifications: Vec<Classification>,
    /// Per-sample alpha diversity.
    pub alpha: Vec<AlphaDiversity>,
    /// Condensed Bray-Curtis distance matrix.
    pub bray_curtis: Vec<f64>,
    /// Taxonomy stage time in milliseconds.
    pub taxonomy_ms: f64,
    /// Diversity stage time in milliseconds.
    pub diversity_ms: f64,
    /// Bray-Curtis stage time in milliseconds.
    pub bray_curtis_ms: f64,
    /// Total pipeline time in milliseconds.
    pub total_ms: f64,
}

/// Results from the streaming GPU pipeline.
#[derive(Debug)]
pub struct StreamingGpuResult {
    /// Per-read taxonomy classifications.
    pub classifications: Vec<Classification>,
    /// Shannon diversity index.
    pub shannon: f64,
    /// Simpson diversity index.
    pub simpson: f64,
    /// Observed species count.
    pub observed: f64,
    /// Taxonomy classification time in milliseconds.
    pub taxonomy_ms: f64,
    /// Diversity computation time in milliseconds.
    pub diversity_ms: f64,
    /// Total GPU pipeline time in milliseconds.
    pub total_gpu_ms: f64,
}

impl GpuPipelineSession {
    /// Run taxonomy + diversity on GPU in a single streaming session.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails for taxonomy or diversity.
    #[expect(clippy::cast_precision_loss)]
    pub fn stream_sample(
        &self,
        classifier: &NaiveBayesClassifier,
        sequences: &[&[u8]],
        counts: &[f64],
        params: &ClassifyParams,
    ) -> Result<StreamingGpuResult> {
        let session_start = Instant::now();

        let tax_start = Instant::now();
        let classifications = self.classify_batch(classifier, sequences, params)?;
        let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

        let div_start = Instant::now();
        let (shannon, simpson, observed) = if counts.len() < 2 {
            (0.0, 0.0, counts.len() as f64)
        } else {
            (
                self.shannon(counts)?,
                self.simpson(counts)?,
                self.observed_features(counts)?,
            )
        };
        let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

        let total_gpu_ms = session_start.elapsed().as_secs_f64() * 1000.0;

        Ok(StreamingGpuResult {
            classifications,
            shannon,
            simpson,
            observed,
            taxonomy_ms,
            diversity_ms,
            total_gpu_ms,
        })
    }

    /// Run taxonomy + diversity + Bray-Curtis in a single streaming session.
    ///
    /// Chains: GEMM taxonomy → FMR diversity → `BrayCurtisF64` beta diversity.
    /// All use pre-warmed pipelines — zero shader recompilation.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails for any stage.
    pub fn stream_full_analytics(
        &self,
        classifier: &NaiveBayesClassifier,
        sequences: &[&[u8]],
        sample_counts: &[&[f64]],
        params: &ClassifyParams,
    ) -> Result<FullStreamingResult> {
        let session_start = Instant::now();

        let tax_start = Instant::now();
        let classifications = self.classify_batch(classifier, sequences, params)?;
        let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

        let div_start = Instant::now();
        let mut alpha = Vec::with_capacity(sample_counts.len());
        for counts in sample_counts {
            let shannon = if counts.len() < 2 {
                0.0
            } else {
                self.shannon(counts)?
            };
            let simpson = if counts.len() < 2 {
                0.0
            } else {
                self.simpson(counts)?
            };
            let observed = if counts.is_empty() {
                0.0
            } else {
                self.observed_features(counts)?
            };
            alpha.push(AlphaDiversity {
                shannon,
                simpson,
                observed,
            });
        }
        let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

        let bc_start = Instant::now();
        let bray_curtis = if sample_counts.len() >= 2 {
            self.bray_curtis_matrix(sample_counts)?
        } else {
            vec![]
        };
        let bray_curtis_ms = bc_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = session_start.elapsed().as_secs_f64() * 1000.0;

        Ok(FullStreamingResult {
            classifications,
            alpha,
            bray_curtis,
            taxonomy_ms,
            diversity_ms,
            bray_curtis_ms,
            total_ms,
        })
    }
}

/// CPU equivalent for benchmarking comparison.
#[must_use]
pub fn stream_classify_and_diversity_cpu(
    classifier: &NaiveBayesClassifier,
    sequences: &[&[u8]],
    counts: &[f64],
    params: &ClassifyParams,
) -> StreamingGpuResult {
    let session_start = Instant::now();

    let tax_start = Instant::now();
    let classifications: Vec<Classification> = sequences
        .iter()
        .map(|seq| classifier.classify(seq, params))
        .collect();
    let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

    let div_start = Instant::now();
    let shannon = diversity::shannon(counts);
    let simpson = diversity::simpson(counts);
    let observed = diversity::observed_features(counts);
    let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

    let total_gpu_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    StreamingGpuResult {
        classifications,
        shannon,
        simpson,
        observed,
        taxonomy_ms,
        diversity_ms,
        total_gpu_ms,
    }
}
