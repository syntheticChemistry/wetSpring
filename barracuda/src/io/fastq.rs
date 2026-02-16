//! FASTQ parser — gzip-aware, Phred33 quality scoring.
//!
//! Wraps `needletail` for fast FASTQ/FASTA parsing and adds
//! quality statistics (mean Q, GC content, length distribution).

use needletail::parse_fastx_file;
use std::collections::HashMap;
use std::path::Path;

/// Summary statistics from parsing a FASTQ file.
#[derive(Debug, Clone)]
pub struct FastqStats {
    /// Total number of sequences
    pub num_sequences: usize,
    /// Total number of bases
    pub total_bases: u64,
    /// Minimum sequence length
    pub min_length: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Mean sequence length
    pub mean_length: f64,
    /// Mean Phred quality score (Phred33)
    pub mean_quality: f64,
    /// GC content as fraction [0, 1]
    pub gc_content: f64,
    /// Number of sequences with mean Q >= 30
    pub q30_count: usize,
    /// Length distribution: length -> count
    pub length_distribution: HashMap<usize, usize>,
}

/// A parsed FASTQ record with owned data.
#[derive(Debug, Clone)]
pub struct FastqRecord {
    pub id: String,
    pub sequence: Vec<u8>,
    pub quality: Vec<u8>,
}

/// Parse a FASTQ file and collect all records.
pub fn parse_fastq(path: &Path) -> Result<Vec<FastqRecord>, String> {
    let mut reader = parse_fastx_file(path)
        .map_err(|e| format!("Failed to open FASTQ {}: {}", path.display(), e))?;

    let mut records = Vec::new();
    while let Some(result) = reader.next() {
        let record = result.map_err(|e| format!("Parse error: {}", e))?;
        let id = String::from_utf8_lossy(record.id()).to_string();
        let seq = record.seq().to_vec();
        let qual = record.qual().map(|q| q.to_vec()).unwrap_or_default();
        records.push(FastqRecord {
            id,
            sequence: seq,
            quality: qual,
        });
    }
    Ok(records)
}

/// Compute summary statistics from parsed records.
pub fn compute_stats(records: &[FastqRecord]) -> FastqStats {
    if records.is_empty() {
        return FastqStats {
            num_sequences: 0,
            total_bases: 0,
            min_length: 0,
            max_length: 0,
            mean_length: 0.0,
            mean_quality: 0.0,
            gc_content: 0.0,
            q30_count: 0,
            length_distribution: HashMap::new(),
        };
    }

    let mut total_bases: u64 = 0;
    let mut total_quality_sum: u64 = 0;
    let mut total_quality_count: u64 = 0;
    let mut gc_count: u64 = 0;
    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut q30_count = 0usize;
    let mut length_dist: HashMap<usize, usize> = HashMap::new();

    for rec in records {
        let len = rec.sequence.len();
        total_bases += len as u64;
        min_len = min_len.min(len);
        max_len = max_len.max(len);
        *length_dist.entry(len).or_insert(0) += 1;

        // GC content
        for &base in &rec.sequence {
            if base == b'G' || base == b'C' || base == b'g' || base == b'c' {
                gc_count += 1;
            }
        }

        // Quality (Phred33: ASCII - 33)
        if !rec.quality.is_empty() {
            let mut q_sum: u64 = 0;
            for &q in &rec.quality {
                q_sum += (q.saturating_sub(33)) as u64;
            }
            total_quality_sum += q_sum;
            total_quality_count += rec.quality.len() as u64;

            let mean_q = q_sum as f64 / rec.quality.len() as f64;
            if mean_q >= 30.0 {
                q30_count += 1;
            }
        }
    }

    let n = records.len();
    FastqStats {
        num_sequences: n,
        total_bases,
        min_length: min_len,
        max_length: max_len,
        mean_length: total_bases as f64 / n as f64,
        mean_quality: if total_quality_count > 0 {
            total_quality_sum as f64 / total_quality_count as f64
        } else {
            0.0
        },
        gc_content: if total_bases > 0 {
            gc_count as f64 / total_bases as f64
        } else {
            0.0
        },
        q30_count,
        length_distribution: length_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_stats() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.num_sequences, 0);
    }

    #[test]
    fn test_single_record() {
        let rec = FastqRecord {
            id: "test".to_string(),
            sequence: b"ACGTACGT".to_vec(),
            quality: b"IIIIIIII".to_vec(), // Q=40
        };
        let stats = compute_stats(&[rec]);
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.total_bases, 8);
        assert!((stats.gc_content - 0.5).abs() < 1e-10);
        assert!((stats.mean_quality - 40.0).abs() < 1e-10);
    }
}
