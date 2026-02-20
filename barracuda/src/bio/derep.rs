// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dereplication — unique sequence collapsing with abundance tracking.
//!
//! Replaces VSEARCH `--derep_fulllength` / QIIME2 `dereplicate-sequences`
//! in the 16S amplicon pipeline. Each unique sequence is kept once with
//! a count of its occurrences and the highest-quality representative.
//!
//! # Algorithm
//!
//! 1. Hash each sequence (case-insensitive) into a hash map.
//! 2. For each unique sequence, track:
//!    - Abundance (count of occurrences).
//!    - Best representative (read with highest mean quality).
//! 3. Sort results by abundance (descending) or sequence.
//!
//! This is a critical pre-processing step before denoising (DADA2/UNOISE)
//! and chimera detection, as it collapses redundant sequences and provides
//! the abundance information needed for error modelling.

use crate::io::fastq::FastqRecord;
use std::collections::HashMap;
use std::fmt::Write;

/// A unique sequence with its abundance and representative.
#[derive(Debug, Clone)]
pub struct UniqueSequence {
    /// The unique sequence (uppercase).
    pub sequence: Vec<u8>,
    /// Number of times this sequence was observed.
    pub abundance: usize,
    /// Mean quality score of the best representative.
    pub best_quality: f64,
    /// Identifier of the best representative read.
    pub representative_id: String,
    /// Quality scores of the best representative.
    pub representative_quality: Vec<u8>,
}

/// Sort order for dereplicated output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerepSort {
    /// Sort by abundance (descending), then by sequence.
    Abundance,
    /// Sort by sequence (lexicographic).
    Sequence,
}

/// Statistics from dereplication.
#[derive(Debug, Clone)]
pub struct DerepStats {
    /// Total input sequences.
    pub input_sequences: usize,
    /// Number of unique sequences.
    pub unique_sequences: usize,
    /// Most abundant sequence count.
    pub max_abundance: usize,
    /// Number of singletons (abundance == 1).
    pub singletons: usize,
    /// Mean abundance across unique sequences.
    pub mean_abundance: f64,
}

/// Dereplicate a collection of FASTQ records.
///
/// Collapses identical sequences (case-insensitive), tracking abundance
/// and keeping the highest-quality representative for each unique sequence.
///
/// # Arguments
///
/// * `records` — Input FASTQ records.
/// * `sort` — Sort order for output.
/// * `min_abundance` — Minimum abundance to include in output (0 = keep all).
///
/// # Returns
///
/// Tuple of (unique sequences, statistics).
#[allow(clippy::cast_precision_loss)]
pub fn dereplicate(
    records: &[FastqRecord],
    sort: DerepSort,
    min_abundance: usize,
) -> (Vec<UniqueSequence>, DerepStats) {
    let mut map: HashMap<Vec<u8>, UniqueSequence> = HashMap::new();

    for record in records {
        let key: Vec<u8> = record.sequence.iter().map(u8::to_ascii_uppercase).collect();
        let mean_q = mean_quality(&record.quality);

        map.entry(key.clone())
            .and_modify(|entry| {
                entry.abundance += 1;
                if mean_q > entry.best_quality {
                    entry.best_quality = mean_q;
                    entry.representative_id.clone_from(&record.id);
                    entry.representative_quality.clone_from(&record.quality);
                }
            })
            .or_insert_with(|| UniqueSequence {
                sequence: key,
                abundance: 1,
                best_quality: mean_q,
                representative_id: record.id.clone(),
                representative_quality: record.quality.clone(),
            });
    }

    // Filter by minimum abundance
    let effective_min = if min_abundance == 0 { 1 } else { min_abundance };
    let mut uniques: Vec<UniqueSequence> = map
        .into_values()
        .filter(|u| u.abundance >= effective_min)
        .collect();

    // Sort
    match sort {
        DerepSort::Abundance => {
            uniques.sort_by(|a, b| {
                b.abundance
                    .cmp(&a.abundance)
                    .then_with(|| a.sequence.cmp(&b.sequence))
            });
        }
        DerepSort::Sequence => {
            uniques.sort_by(|a, b| a.sequence.cmp(&b.sequence));
        }
    }

    let n_unique = uniques.len();
    let max_abundance = uniques.iter().map(|u| u.abundance).max().unwrap_or(0);
    let singletons = uniques.iter().filter(|u| u.abundance == 1).count();
    let total_abundance: usize = uniques.iter().map(|u| u.abundance).sum();

    let stats = DerepStats {
        input_sequences: records.len(),
        unique_sequences: n_unique,
        max_abundance,
        singletons,
        mean_abundance: if n_unique > 0 {
            total_abundance as f64 / n_unique as f64
        } else {
            0.0
        },
    };

    (uniques, stats)
}

/// Compute mean Phred quality (offset 33) from quality bytes.
#[allow(clippy::cast_precision_loss)]
fn mean_quality(qual: &[u8]) -> f64 {
    if qual.is_empty() {
        return 0.0;
    }
    qual.iter()
        .map(|&q| f64::from(q.saturating_sub(33)))
        .sum::<f64>()
        / qual.len() as f64
}

/// Write dereplicated sequences to FASTA format with abundance annotations.
///
/// Output format (USEARCH/VSEARCH style):
/// ```text
/// >seq1;size=42
/// ACGTACGTACGT
/// >seq2;size=10
/// GCTAGCTAGCTA
/// ```
#[must_use]
pub fn to_fasta_with_abundance(uniques: &[UniqueSequence]) -> String {
    let mut out = String::new();
    for (i, u) in uniques.iter().enumerate() {
        out.push('>');
        if u.representative_id.is_empty() {
            let _ = write!(out, "seq{}", i + 1);
        } else {
            out.push_str(&u.representative_id);
        }
        let _ = writeln!(out, ";size={}", u.abundance);
        out.push_str(&String::from_utf8_lossy(&u.sequence));
        out.push('\n');
    }
    out
}

/// Create an abundance vector suitable for diversity analysis.
///
/// Returns a vector of f64 counts (one per unique sequence), sorted
/// by abundance descending — ready for Shannon/Simpson/Chao1.
#[must_use]
#[allow(clippy::cast_precision_loss)] // abundance to f64 for diversity calculations
pub fn abundance_vector(uniques: &[UniqueSequence]) -> Vec<f64> {
    let mut counts: Vec<f64> = uniques.iter().map(|u| u.abundance as f64).collect();
    counts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, seq: &[u8], q: u8) -> FastqRecord {
        FastqRecord {
            id: id.to_string(),
            sequence: seq.to_vec(),
            quality: vec![33 + q; seq.len()],
        }
    }

    #[test]
    fn basic_dereplication() {
        let records = vec![
            make_record("r1", b"ACGT", 30),
            make_record("r2", b"ACGT", 35), // duplicate, higher quality
            make_record("r3", b"GCTA", 30),
            make_record("r4", b"ACGT", 25), // duplicate, lower quality
        ];

        let (uniques, stats) = dereplicate(&records, DerepSort::Abundance, 0);

        assert_eq!(stats.input_sequences, 4);
        assert_eq!(stats.unique_sequences, 2);
        assert_eq!(stats.max_abundance, 3); // ACGT appears 3 times

        // ACGT should be first (abundance 3)
        assert_eq!(uniques[0].sequence, b"ACGT");
        assert_eq!(uniques[0].abundance, 3);
        assert_eq!(uniques[0].representative_id, "r2"); // best quality

        // GCTA should be second (abundance 1)
        assert_eq!(uniques[1].sequence, b"GCTA");
        assert_eq!(uniques[1].abundance, 1);
    }

    #[test]
    fn case_insensitive() {
        let records = vec![
            make_record("r1", b"acgt", 30),
            make_record("r2", b"ACGT", 30),
            make_record("r3", b"AcGt", 30),
        ];

        let (uniques, stats) = dereplicate(&records, DerepSort::Abundance, 0);
        assert_eq!(stats.unique_sequences, 1);
        assert_eq!(uniques[0].abundance, 3);
    }

    #[test]
    fn min_abundance_filter() {
        let records = vec![
            make_record("r1", b"AAAA", 30),
            make_record("r2", b"AAAA", 30),
            make_record("r3", b"CCCC", 30), // singleton
            make_record("r4", b"GGGG", 30), // singleton
        ];

        let (uniques, stats) = dereplicate(&records, DerepSort::Abundance, 2);
        assert_eq!(stats.unique_sequences, 1); // only AAAA (abundance 2)
        assert_eq!(uniques[0].sequence, b"AAAA");
    }

    #[test]
    fn sort_by_sequence() {
        let records = vec![
            make_record("r1", b"ZZZZ", 30),
            make_record("r2", b"AAAA", 30),
            make_record("r3", b"MMMM", 30),
        ];

        let (uniques, _) = dereplicate(&records, DerepSort::Sequence, 0);
        assert_eq!(uniques[0].sequence, b"AAAA");
        assert_eq!(uniques[1].sequence, b"MMMM");
        assert_eq!(uniques[2].sequence, b"ZZZZ");
    }

    #[test]
    fn empty_input() {
        let (uniques, stats) = dereplicate(&[], DerepSort::Abundance, 0);
        assert_eq!(stats.input_sequences, 0);
        assert_eq!(stats.unique_sequences, 0);
        assert!(uniques.is_empty());
    }

    #[test]
    fn all_unique() {
        let records = vec![
            make_record("r1", b"AAAA", 30),
            make_record("r2", b"CCCC", 30),
            make_record("r3", b"GGGG", 30),
            make_record("r4", b"TTTT", 30),
        ];

        let (_, stats) = dereplicate(&records, DerepSort::Abundance, 0);
        assert_eq!(stats.unique_sequences, 4);
        assert_eq!(stats.singletons, 4);
        assert!((stats.mean_abundance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fasta_output() {
        let records = vec![
            make_record("r1", b"ACGT", 30),
            make_record("r2", b"ACGT", 30),
            make_record("r3", b"GCTA", 30),
        ];

        let (uniques, _) = dereplicate(&records, DerepSort::Abundance, 0);
        let fasta = to_fasta_with_abundance(&uniques);

        assert!(fasta.contains(";size=2"));
        assert!(fasta.contains(";size=1"));
        assert!(fasta.contains("ACGT"));
        assert!(fasta.contains("GCTA"));
    }

    #[test]
    fn abundance_vector_sorted() {
        let records = vec![
            make_record("r1", b"AAAA", 30),
            make_record("r2", b"AAAA", 30),
            make_record("r3", b"AAAA", 30),
            make_record("r4", b"CCCC", 30),
            make_record("r5", b"GGGG", 30),
            make_record("r6", b"GGGG", 30),
        ];

        let (uniques, _) = dereplicate(&records, DerepSort::Abundance, 0);
        let counts = abundance_vector(&uniques);

        assert_eq!(counts, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn best_representative_quality() {
        let records = vec![
            make_record("low", b"ACGT", 10),
            make_record("high", b"ACGT", 40),
            make_record("mid", b"ACGT", 25),
        ];

        let (uniques, _) = dereplicate(&records, DerepSort::Abundance, 0);
        assert_eq!(uniques[0].representative_id, "high");
    }
}
