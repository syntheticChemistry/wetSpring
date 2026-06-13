// SPDX-License-Identifier: AGPL-3.0-or-later
//! MAPQ calibration via simulated reads (WS-11).
//!
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    reason = "Calibration code uses small counts (< 10^6) and MAPQ values (0-60); \
              precision loss and truncation are acceptable and expected. \
              const fn is inapplicable to PRNG and Vec methods."
)]
//! Generates reads at known reference positions with controlled sequencing
//! error rates, maps them back, and builds an empirical model mapping
//! (score_gap, n_candidates) → P(wrong mapping) → Phred MAPQ.
//!
//! # Approach
//!
//! Standard MAPQ calibration (BWA-MEM, minimap2) uses simulated data:
//! 1. Sample positions uniformly from the reference
//! 2. Extract read-length fragments, inject errors at a controlled rate
//! 3. Map reads back to the reference
//! 4. Record (score_gap, mapped_correctly) for each read
//! 5. Bin by score_gap, compute P(wrong) per bin
//! 6. Convert to Phred: MAPQ = -10 × log10(P(wrong))
//!
//! The resulting lookup table replaces the linear `gap * 6` formula.

#[cfg(test)]
mod tests;

use crate::bio::ref_index::FmIndex;
use super::{MapperConfig, map_read};
use crate::io::sam::SamRecord;

/// Configuration for MAPQ calibration.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of simulated reads to generate.
    pub n_reads: usize,
    /// Read length in base pairs.
    pub read_length: usize,
    /// Per-base sequencing error rate (substitution probability).
    pub error_rate: f64,
    /// Maximum positional distance (bp) to consider a mapping correct.
    pub position_tolerance: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            n_reads: 10_000,
            read_length: 150,
            error_rate: 0.01,
            position_tolerance: 5,
            seed: 42,
        }
    }
}

/// A simulated read with its known true position.
#[derive(Debug, Clone)]
pub struct SimulatedRead {
    /// 0-based true position on the reference.
    pub true_position: usize,
    /// Simulated read sequence (with errors injected).
    pub sequence: Vec<u8>,
    /// Simulated quality scores (Phred33 ASCII).
    pub quality: Vec<u8>,
}

/// Result of mapping a simulated read back to the reference.
#[derive(Debug, Clone)]
struct CalibrationSample {
    /// Score gap between best and second-best alignment.
    score_gap: u32,
    /// Number of mapping candidates found (reserved for 2D model extension).
    #[expect(dead_code, reason = "reserved for 2D calibration model (gap × candidates)")]
    n_candidates: usize,
    /// Whether the mapping position matched the true position.
    correct: bool,
}

/// Calibrated MAPQ model — lookup table from score_gap → MAPQ.
#[derive(Debug, Clone)]
pub struct MapqModel {
    /// Lookup table: index = score_gap (0..=max_gap), value = MAPQ (0..=60).
    gap_to_mapq: Vec<u8>,
}

impl Default for MapqModel {
    fn default() -> Self {
        Self::linear_fallback()
    }
}

impl MapqModel {
    /// Linear fallback model (current behavior): MAPQ = min(60, gap * 6).
    #[must_use]
    pub fn linear_fallback() -> Self {
        let gap_to_mapq: Vec<u8> = (0..=60)
            .map(|gap: u32| (gap.saturating_mul(6).min(60)) as u8)
            .collect();
        Self { gap_to_mapq }
    }

    /// Build a calibrated model from training data.
    ///
    /// Groups samples by score_gap, computes P(wrong) per bin,
    /// and converts to Phred-scaled MAPQ.
    #[must_use]
    fn from_training_data(samples: &[CalibrationSample]) -> Self {
        if samples.is_empty() {
            return Self::linear_fallback();
        }

        let max_gap = samples.iter().map(|s| s.score_gap).max().unwrap_or(60);
        let table_size = (max_gap as usize + 1).max(61);

        let mut correct_counts = vec![0u64; table_size];
        let mut total_counts = vec![0u64; table_size];

        for sample in samples {
            let idx = (sample.score_gap as usize).min(table_size - 1);
            total_counts[idx] += 1;
            if sample.correct {
                correct_counts[idx] += 1;
            }
        }

        let gap_to_mapq: Vec<u8> = (0..table_size)
            .map(|idx| {
                if total_counts[idx] == 0 {
                    // No data for this gap — interpolate from linear model
                    let gap = idx as u32;
                    (gap.saturating_mul(6).min(60)) as u8
                } else {
                    let p_correct = correct_counts[idx] as f64 / total_counts[idx] as f64;
                    let p_wrong = (1.0 - p_correct).max(1e-7); // floor to avoid log(0)
                    let mapq = (-10.0 * p_wrong.log10()).round() as u8;
                    mapq.min(60)
                }
            })
            .collect();

        Self { gap_to_mapq }
    }

    /// Look up MAPQ for a given score gap.
    #[must_use]
    pub fn lookup(&self, score_gap: u32) -> u8 {
        let idx = (score_gap as usize).min(self.gap_to_mapq.len() - 1);
        self.gap_to_mapq[idx]
    }

    /// Maximum score gap in the table.
    #[must_use]
    pub fn max_gap(&self) -> usize {
        self.gap_to_mapq.len() - 1
    }
}

/// Generate simulated reads from a reference sequence.
///
/// Samples `n_reads` positions uniformly, extracts fragments of `read_length`,
/// and injects substitution errors at `error_rate`.
#[must_use]
pub fn simulate_reads(
    reference: &[u8],
    config: &CalibrationConfig,
) -> Vec<SimulatedRead> {
    let mut reads = Vec::with_capacity(config.n_reads);
    let max_start = reference.len().saturating_sub(config.read_length);

    if max_start == 0 {
        return reads;
    }

    let mut rng = SimpleRng::new(config.seed);

    for _ in 0..config.n_reads {
        let pos = rng.next_usize() % max_start;
        let fragment = &reference[pos..pos + config.read_length];

        let mut seq = fragment.to_vec();
        let mut qual = vec![b'I'; config.read_length]; // Q40 default

        for (i, base) in seq.iter_mut().enumerate() {
            if rng.next_f64() < config.error_rate {
                *base = mutate_base(*base, &mut rng);
                qual[i] = b'5'; // Q20 for error positions
            }
        }

        reads.push(SimulatedRead {
            true_position: pos,
            sequence: seq,
            quality: qual,
        });
    }

    reads
}

/// Run the calibration pipeline: simulate → map → compare → build model.
///
/// Returns the calibrated [`MapqModel`] and the raw training samples for
/// inspection/validation.
#[must_use]
pub fn calibrate(
    reference: &[u8],
    ref_name: &str,
    index: &FmIndex,
    mapper_config: &MapperConfig,
    cal_config: &CalibrationConfig,
) -> (MapqModel, CalibrationStats) {
    let reads = simulate_reads(reference, cal_config);
    let mut samples = Vec::with_capacity(reads.len());
    let mut correct = 0usize;
    let mut mapped = 0usize;

    for (i, sim_read) in reads.iter().enumerate() {
        let read_id = format!("sim_{i}");
        let record = map_read(
            &read_id,
            &sim_read.sequence,
            &sim_read.quality,
            index,
            reference,
            ref_name,
            mapper_config,
        );

        if let Some(ref rec) = record {
            if rec.is_mapped() {
                mapped += 1;
                let mapped_pos = rec.pos.saturating_sub(1) as usize;
                let is_correct = mapped_pos.abs_diff(sim_read.true_position)
                    <= cal_config.position_tolerance;

                if is_correct {
                    correct += 1;
                }

                let score_gap = extract_score_gap(rec);
                let n_candidates = 1; // record doesn't carry this; default

                samples.push(CalibrationSample {
                    score_gap,
                    n_candidates,
                    correct: is_correct,
                });
            }
        }
    }

    let model = MapqModel::from_training_data(&samples);
    let stats = CalibrationStats {
        total_reads: reads.len(),
        mapped,
        correct,
        samples_collected: samples.len(),
    };

    (model, stats)
}

/// Statistics from a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Total simulated reads generated.
    pub total_reads: usize,
    /// Reads that mapped to the reference.
    pub mapped: usize,
    /// Reads that mapped to the correct position.
    pub correct: usize,
    /// Calibration samples collected (mapped reads).
    pub samples_collected: usize,
}

impl CalibrationStats {
    /// Fraction of mapped reads at the correct position.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.mapped == 0 {
            return 0.0;
        }
        self.correct as f64 / self.mapped as f64
    }

    /// Mapping rate (fraction of reads that aligned).
    #[must_use]
    pub fn mapping_rate(&self) -> f64 {
        if self.total_reads == 0 {
            return 0.0;
        }
        self.mapped as f64 / self.total_reads as f64
    }
}

/// Extract score gap from a mapped SAM record.
///
/// Since the SAM record doesn't carry the candidate list, we encode
/// score gap via the MAPQ field (which was computed from the gap).
/// This reverses the current linear formula for bootstrapping.
fn extract_score_gap(rec: &SamRecord) -> u32 {
    // Current formula: mapq = min(60, gap * 6)
    // Inverse: gap = mapq / 6 (integer division)
    if rec.mapq >= 60 {
        10 // 60/6 = 10, but could be higher — cap at 10 for calibration
    } else {
        u32::from(rec.mapq) / 6
    }
}

fn mutate_base(base: u8, rng: &mut SimpleRng) -> u8 {
    let bases = [b'A', b'C', b'G', b'T'];
    let alt_bases: Vec<u8> = bases.iter().copied().filter(|&b| b != base.to_ascii_uppercase()).collect();
    if alt_bases.is_empty() {
        return base;
    }
    alt_bases[rng.next_usize() % alt_bases.len()]
}

/// Minimal xorshift64 PRNG for reproducible simulation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}
