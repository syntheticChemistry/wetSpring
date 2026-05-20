// SPDX-License-Identifier: AGPL-3.0-or-later
//! Seed-and-extend read mapper for short-read resequencing.
//!
//! Orchestrates:
//! 1. **Seeding**: FM-index exact k-mer matching (CPU, via [`ref_index::FmIndex`])
//! 2. **Extension**: Smith-Waterman local alignment
//!    - CPU fallback via [`alignment::smith_waterman`]
//!    - GPU via `SmithWatermanGpu` (barraCuda) when `--features gpu` is active
//! 3. **Selection**: best alignment by score, MAPQ from score ratio
//!
//! This module is the orchestration glue — the math comes from barraCuda.

#[cfg(test)]
mod tests;

use crate::bio::alignment::{self, AlignmentResult, ScoringParams};
use crate::bio::ref_index::FmIndex;
use crate::io::sam::{CigarOp, CigarType, SamRecord};

#[cfg(feature = "gpu")]
use barracuda::{SmithWatermanGpu, SwConfig};
#[cfg(feature = "gpu")]
use barracuda::device::WgpuDevice;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Mapping configuration.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Seed k-mer length.
    pub seed_k: usize,
    /// Maximum seed hits before a k-mer is considered repetitive.
    pub max_seed_hits: usize,
    /// Extension window: bases around a seed hit to extract for SW alignment.
    pub extension_window: usize,
    /// Minimum alignment score to report a mapping.
    pub min_score: i32,
    /// Smith-Waterman scoring parameters.
    pub scoring: ScoringParams,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            seed_k: 20,
            max_seed_hits: 500,
            extension_window: 50,
            min_score: 30,
            scoring: ScoringParams::default(),
        }
    }
}

/// A single mapping candidate from seed extension.
#[derive(Debug, Clone)]
struct MappingCandidate {
    ref_start: usize,
    score: i32,
    alignment: AlignmentResult,
}

/// Map a single read against the reference using seed-and-extend.
///
/// Returns a `SamRecord` for the best alignment, or `None` if unmapped.
///
/// # Arguments
///
/// * `read_id` - Query name for the SAM record
/// * `read_seq` - Read sequence (forward strand)
/// * `read_qual` - Quality scores (Phred33 ASCII bytes)
/// * `index` - FM-index of the reference genome
/// * `reference` - Full reference sequence
/// * `ref_name` - Reference sequence name for SAM output
/// * `config` - Mapping configuration
#[must_use]
#[expect(clippy::too_many_arguments, reason = "mapper needs all these inputs")]
pub fn map_read(
    read_id: &str,
    read_seq: &[u8],
    read_qual: &[u8],
    index: &FmIndex,
    reference: &[u8],
    ref_name: &str,
    config: &MapperConfig,
) -> Option<SamRecord> {
    let seeds = index.seed_kmers(read_seq, config.seed_k, config.max_seed_hits);

    if seeds.is_empty() {
        return Some(unmapped_record(read_id, read_seq, read_qual));
    }

    let mut candidates: Vec<MappingCandidate> = Vec::new();

    for (read_offset, ref_positions) in &seeds {
        for &ref_pos in ref_positions {
            let ref_start = ref_pos.saturating_sub(*read_offset + config.extension_window);
            let ref_end = (ref_pos + config.seed_k + config.extension_window)
                .min(reference.len());

            if ref_start >= ref_end {
                continue;
            }

            // Check if we already have a candidate near this position (dedup)
            if candidates.iter().any(|c| {
                c.ref_start.abs_diff(ref_start) < config.seed_k
            }) {
                continue;
            }

            let ref_window = &reference[ref_start..ref_end];
            let result = alignment::smith_waterman(read_seq, ref_window, &config.scoring);

            if result.score >= config.min_score {
                candidates.push(MappingCandidate {
                    ref_start: ref_start + result.target_start,
                    score: result.score,
                    alignment: result,
                });
            }
        }
    }

    if candidates.is_empty() {
        return Some(unmapped_record(read_id, read_seq, read_qual));
    }

    candidates.sort_by_key(|c| std::cmp::Reverse(c.score));

    let best = &candidates[0];
    let mapq = compute_mapq(&candidates);
    let cigar = alignment_to_cigar(&best.alignment);

    #[expect(clippy::cast_possible_truncation, reason = "MAPQ fits in u8")]
    Some(SamRecord {
        qname: read_id.to_string(),
        flag: 0,
        rname: ref_name.to_string(),
        pos: (best.ref_start + 1) as u64, // SAM is 1-based
        mapq: mapq as u8,
        cigar,
        rnext: "*".to_string(),
        pnext: 0,
        tlen: 0,
        seq: read_seq.to_vec(),
        qual: read_qual.to_vec(),
    })
}

/// Map a batch of reads (CPU path).
///
/// For GPU-accelerated mapping, use [`map_reads_gpu`] with `--features gpu`.
pub fn map_reads(
    reads: &[(String, Vec<u8>, Vec<u8>)], // (id, seq, qual)
    index: &FmIndex,
    reference: &[u8],
    ref_name: &str,
    config: &MapperConfig,
) -> Vec<SamRecord> {
    reads
        .iter()
        .filter_map(|(id, seq, qual)| map_read(id, seq, qual, index, reference, ref_name, config))
        .collect()
}

// ── GPU-accelerated mapping ──────────────────────────────────────

/// Encode a DNA base to u32 index for barraCuda SW (A=0, C=1, G=2, T=3).
#[cfg(feature = "gpu")]
const fn encode_base_gpu(b: u8) -> u32 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

/// DNA 4x4 substitution matrix (match=2, mismatch=-1) as flat f64.
#[cfg(feature = "gpu")]
const DNA_SUBST_MATRIX: [f64; 16] = [
    2.0, -1.0, -1.0, -1.0, // A vs A,C,G,T
    -1.0, 2.0, -1.0, -1.0, // C vs A,C,G,T
    -1.0, -1.0, 2.0, -1.0, // G vs A,C,G,T
    -1.0, -1.0, -1.0, 2.0, // T vs A,C,G,T
];

/// Map a batch of reads using GPU-accelerated Smith-Waterman extension.
///
/// Seeds are found on CPU via FM-index, then candidate extensions are
/// batch-dispatched to `SmithWatermanGpu` on the GPU. Each (query, target)
/// pair runs as an independent GPU alignment.
///
/// Falls back to CPU SW for individual pairs if GPU dispatch fails.
#[cfg(feature = "gpu")]
#[expect(clippy::too_many_arguments, reason = "GPU mapper needs device handle")]
pub fn map_reads_gpu(
    reads: &[(String, Vec<u8>, Vec<u8>)],
    index: &FmIndex,
    reference: &[u8],
    ref_name: &str,
    config: &MapperConfig,
    device: &Arc<WgpuDevice>,
) -> Vec<SamRecord> {
    let sw_gpu = SmithWatermanGpu::with_device(Arc::clone(device));
    let sw_config = SwConfig {
        gap_open: f64::from(-config.scoring.gap_open),
        gap_extend: f64::from(-config.scoring.gap_extend),
        band_width: 64,
    };

    reads
        .iter()
        .map(|(id, seq, qual)| {
            map_read_gpu(id, seq, qual, index, reference, ref_name, config, &sw_gpu, &sw_config)
        })
        .collect()
}

/// Map a single read using GPU SW for extension.
#[cfg(feature = "gpu")]
#[expect(clippy::too_many_arguments, reason = "GPU path needs SW handle")]
#[expect(clippy::cast_possible_truncation, reason = "f64 score → i32: SW scores fit i32")]
fn map_read_gpu(
    read_id: &str,
    read_seq: &[u8],
    read_qual: &[u8],
    index: &FmIndex,
    reference: &[u8],
    ref_name: &str,
    config: &MapperConfig,
    sw_gpu: &SmithWatermanGpu,
    sw_config: &SwConfig,
) -> SamRecord {
    let seeds = index.seed_kmers(read_seq, config.seed_k, config.max_seed_hits);

    if seeds.is_empty() {
        return unmapped_record(read_id, read_seq, read_qual);
    }

    // Collect candidate windows (dedup by proximity)
    let mut candidate_windows: Vec<(usize, usize)> = Vec::new();

    for (read_offset, ref_positions) in &seeds {
        for &ref_pos in ref_positions {
            let ref_start = ref_pos.saturating_sub(*read_offset + config.extension_window);
            let ref_end = (ref_pos + config.seed_k + config.extension_window).min(reference.len());

            if ref_start >= ref_end {
                continue;
            }

            if candidate_windows.iter().any(|&(s, _)| s.abs_diff(ref_start) < config.seed_k) {
                continue;
            }

            candidate_windows.push((ref_start, ref_end));
        }
    }

    if candidate_windows.is_empty() {
        return unmapped_record(read_id, read_seq, read_qual);
    }

    let query_encoded: Vec<u32> = read_seq.iter().map(|&b| encode_base_gpu(b)).collect();

    let mut candidates: Vec<MappingCandidate> = Vec::new();

    for &(ref_start, ref_end) in &candidate_windows {
        let ref_window = &reference[ref_start..ref_end];
        let target_encoded: Vec<u32> = ref_window.iter().map(|&b| encode_base_gpu(b)).collect();

        if let Ok(sw_result) = sw_gpu.align(&query_encoded, &target_encoded, &DNA_SUBST_MATRIX, sw_config) {
            let score = sw_result.score as i32;
            if score >= config.min_score {
                let aligned_len = read_seq.len().min(ref_window.len());
                candidates.push(MappingCandidate {
                    ref_start: ref_start + sw_result.col.saturating_sub(1),
                    score,
                    alignment: AlignmentResult {
                        score,
                        aligned_query: read_seq[..aligned_len].to_vec(),
                        aligned_target: ref_window[..aligned_len].to_vec(),
                        query_start: 0,
                        target_start: sw_result.col.saturating_sub(1),
                    },
                });
            }
        } else {
            // GPU dispatch failed — fall back to CPU for this pair
            let result = alignment::smith_waterman(read_seq, ref_window, &config.scoring);
            if result.score >= config.min_score {
                candidates.push(MappingCandidate {
                    ref_start: ref_start + result.target_start,
                    score: result.score,
                    alignment: result,
                });
            }
        }
    }

    if candidates.is_empty() {
        return unmapped_record(read_id, read_seq, read_qual);
    }

    candidates.sort_by_key(|c| std::cmp::Reverse(c.score));

    let best = &candidates[0];
    let mapq = compute_mapq(&candidates);
    let cigar = alignment_to_cigar(&best.alignment);

    #[expect(clippy::cast_possible_truncation, reason = "MAPQ fits in u8")]
    SamRecord {
        qname: read_id.to_string(),
        flag: 0,
        rname: ref_name.to_string(),
        pos: (best.ref_start + 1) as u64,
        mapq: mapq as u8,
        cigar,
        rnext: "*".to_string(),
        pnext: 0,
        tlen: 0,
        seq: read_seq.to_vec(),
        qual: read_qual.to_vec(),
    }
}

/// Convert an `AlignmentResult` (with aligned sequences) into CIGAR operations.
fn alignment_to_cigar(alignment: &AlignmentResult) -> Vec<CigarOp> {
    if alignment.aligned_query.is_empty() {
        return vec![];
    }

    let mut ops = Vec::new();
    let mut current_op = None;
    let mut current_len = 0u32;

    for (&q, &t) in alignment
        .aligned_query
        .iter()
        .zip(alignment.aligned_target.iter())
    {
        let op = if q == b'-' {
            CigarType::Deletion
        } else if t == b'-' {
            CigarType::Insertion
        } else {
            CigarType::Match
        };

        if Some(op) == current_op {
            current_len += 1;
        } else {
            if let Some(prev_op) = current_op {
                ops.push(CigarOp {
                    len: current_len,
                    op: prev_op,
                });
            }
            current_op = Some(op);
            current_len = 1;
        }
    }

    if let Some(op) = current_op {
        ops.push(CigarOp {
            len: current_len,
            op,
        });
    }

    ops
}

/// Compute mapping quality from best and second-best alignment scores.
///
/// Uses score gap (best - second_best) rather than ratio, following BWA-MEM's
/// approach. The ratio-based formula `(1 - second/best) * 60` produces near-zero
/// MAPQ for almost all reads when candidates have similar scores (common with
/// FM-index seeding + Smith-Waterman extension on bacterial genomes).
///
/// Gap-based: MAPQ = min(60, gap * 6). A gap of 10 alignment score points
/// yields MAPQ 60 (confident unique placement). Gap of 0 yields MAPQ 0
/// (ambiguous). Gap of 2 yields MAPQ 12 (passes typical min_mapq=10 filter).
fn compute_mapq(candidates: &[MappingCandidate]) -> u32 {
    if candidates.len() < 2 {
        return 60;
    }

    let best = candidates[0].score;
    let second = candidates[1].score;

    if best <= 0 {
        return 0;
    }

    if second <= 0 {
        return 60;
    }

    let gap = (best - second).max(0) as u32;
    gap.saturating_mul(6).min(60)
}

fn unmapped_record(read_id: &str, read_seq: &[u8], read_qual: &[u8]) -> SamRecord {
    SamRecord {
        qname: read_id.to_string(),
        flag: crate::io::sam::FLAG_UNMAPPED,
        rname: "*".to_string(),
        pos: 0,
        mapq: 0,
        cigar: vec![],
        rnext: "*".to_string(),
        pnext: 0,
        tlen: 0,
        seq: read_seq.to_vec(),
        qual: read_qual.to_vec(),
    }
}

/// Reverse complement a DNA sequence.
#[must_use]
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            _ => b'N',
        })
        .collect()
}
