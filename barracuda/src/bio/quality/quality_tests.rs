// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::naive_bytecount)]

use super::super::adapter::bases_match;
use super::trim;
use super::*;

fn make_record(seq: &[u8], qual: &[u8]) -> FastqRecord {
    FastqRecord {
        id: "test".to_string(),
        sequence: seq.to_vec(),
        quality: qual.to_vec(),
    }
}

fn qual_from_phred(scores: &[u8]) -> Vec<u8> {
    scores.iter().map(|&q| q + 33).collect()
}

#[test]
fn trim_leading_removes_low_quality() {
    let qual = qual_from_phred(&[2, 2, 2, 30, 30, 30]);
    assert_eq!(trim::trim_leading(&qual, 3, 33), 3);
}

#[test]
fn trim_leading_all_low() {
    let qual = qual_from_phred(&[2, 2, 2]);
    assert_eq!(trim::trim_leading(&qual, 3, 33), 3);
}

#[test]
fn trim_leading_all_high() {
    let qual = qual_from_phred(&[30, 30, 30]);
    assert_eq!(trim::trim_leading(&qual, 3, 33), 0);
}

#[test]
fn trim_trailing_removes_low_quality() {
    let qual = qual_from_phred(&[30, 30, 30, 2, 2, 2]);
    assert_eq!(trim::trim_trailing(&qual, 3, 33), 3);
}

#[test]
fn trim_trailing_all_low() {
    let qual = qual_from_phred(&[2, 2, 2]);
    assert_eq!(trim::trim_trailing(&qual, 3, 33), 0);
}

#[test]
fn trim_trailing_all_high() {
    let qual = qual_from_phred(&[30, 30, 30]);
    assert_eq!(trim::trim_trailing(&qual, 3, 33), 3);
}

#[test]
fn sliding_window_high_quality() {
    let qual = qual_from_phred(&[30, 30, 30, 30, 30, 30]);
    assert_eq!(trim::trim_sliding_window(&qual, 4, 20, 33), 6);
}

#[test]
fn sliding_window_drops_at_end() {
    let qual = qual_from_phred(&[30, 30, 30, 30, 5, 5, 5, 5]);
    let pos = trim::trim_sliding_window(&qual, 4, 20, 33);
    assert!((2..=5).contains(&pos), "pos={pos}");
}

#[test]
fn sliding_window_all_low() {
    let qual = qual_from_phred(&[5, 5, 5, 5, 5]);
    assert_eq!(trim::trim_sliding_window(&qual, 4, 20, 33), 0);
}

#[test]
fn trim_read_full_pipeline() {
    let qual = qual_from_phred(&[2, 2, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 2, 2, 2]);
    let seq: Vec<u8> = vec![b'A'; 15];
    let record = make_record(&seq, &qual);

    let params = QualityParams {
        min_length: 5,
        ..QualityParams::default()
    };

    let result = trim_read(&record, &params);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 2);
    assert_eq!(end, 12);
}

#[test]
fn trim_read_too_short() {
    let qual = qual_from_phred(&[30, 30, 30]);
    let seq: Vec<u8> = vec![b'A'; 3];
    let record = make_record(&seq, &qual);

    let params = QualityParams {
        min_length: 36,
        ..QualityParams::default()
    };

    assert!(trim_read(&record, &params).is_none());
}

#[test]
fn filter_reads_batch() {
    let records = vec![
        make_record(&[b'A'; 50], &qual_from_phred(&[30; 50])),
        make_record(&[b'A'; 50], &qual_from_phred(&[2; 50])),
        make_record(&[b'A'; 10], &qual_from_phred(&[30; 10])),
    ];

    let params = QualityParams {
        min_length: 36,
        ..QualityParams::default()
    };

    let (filtered, stats) = filter_reads(&records, &params);
    assert_eq!(stats.input_reads, 3);
    assert_eq!(stats.output_reads, 1);
    assert_eq!(stats.discarded_reads, 2);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].sequence.len(), 50);
}

#[test]
fn adapter_exact_match() {
    let seq = b"ACGTACGTAACTAGTCGA";
    let adapter = b"AACTAGTCGA";
    let pos = find_adapter_3prime(seq, adapter, 0, 5);
    assert_eq!(pos, Some(8));
}

#[test]
fn adapter_with_mismatches() {
    let seq = b"ACGTACGTAACTXGTCGA";
    let adapter = b"AACTAGTCGA";
    let pos = find_adapter_3prime(seq, adapter, 1, 5);
    assert_eq!(pos, Some(8));
}

#[test]
fn adapter_not_found() {
    let seq = b"ACGTACGTACGTACGT";
    let adapter = b"TTTTTTTTTT";
    let pos = find_adapter_3prime(seq, adapter, 0, 5);
    assert!(pos.is_none());
}

#[test]
fn adapter_partial_overlap() {
    let seq = b"ACGTACGTAACTA";
    let adapter = b"AACTAGTCGA";
    let pos = find_adapter_3prime(seq, adapter, 0, 5);
    assert_eq!(pos, Some(8));
}

#[test]
fn trim_adapter_3prime_found() {
    let record = make_record(b"ACGTACGTAACTAGTCGA", &[33 + 30; 18]);
    let trimmed = trim_adapter_3prime(&record, b"AACTAGTCGA", 0, 5);
    assert!(trimmed.is_some());
    assert_eq!(trimmed.unwrap().sequence.len(), 8);
}

#[test]
fn trim_adapter_3prime_not_found() {
    let record = make_record(b"ACGTACGT", &[33 + 30; 8]);
    let trimmed = trim_adapter_3prime(&record, b"TTTTTTTTTT", 0, 5);
    assert!(trimmed.is_none());
}

#[test]
fn n_bases_match_anything() {
    assert!(bases_match(b'N', b'A'));
    assert!(bases_match(b'A', b'N'));
    assert!(bases_match(b'N', b'N'));
}

#[test]
fn empty_record_returns_none() {
    let record = make_record(b"", &[]);
    assert!(trim_read(&record, &QualityParams::default()).is_none());
}

#[test]
fn gpu_params_from_quality_params() {
    let params = QualityParams::default();
    let gpu: QualityGpuParams = QualityGpuParams::from(&params);
    assert_eq!(gpu.window_size, 4);
    assert_eq!(gpu.window_min_quality, 20);
    assert_eq!(gpu.phred_offset, 33);
}

#[test]
fn filter_reads_flat_matches_structured() {
    let records = vec![
        make_record(&[b'A'; 50], &qual_from_phred(&[30; 50])),
        make_record(&[b'A'; 50], &qual_from_phred(&[2; 50])),
        make_record(&[b'A'; 10], &qual_from_phred(&[30; 10])),
    ];

    let params = QualityParams {
        min_length: 36,
        ..QualityParams::default()
    };

    let mut qualities = Vec::new();
    let mut offsets = Vec::new();
    let mut lengths = Vec::new();
    for r in &records {
        offsets.push(qualities.len());
        lengths.push(r.quality.len());
        qualities.extend_from_slice(&r.quality);
    }

    let flat = filter_reads_flat(&qualities, &offsets, &lengths, &params);
    let (structured, stats) = filter_reads(&records, &params);

    let flat_pass_count = flat.pass.iter().filter(|&&p| p == 1).count();
    assert_eq!(flat_pass_count, stats.output_reads);
    assert_eq!(flat_pass_count, structured.len());
}

#[test]
fn filter_reads_flat_empty() {
    let result = filter_reads_flat(&[], &[], &[], &QualityParams::default());
    assert!(result.starts.is_empty());
    assert!(result.ends.is_empty());
    assert!(result.pass.is_empty());
}

#[test]
fn filter_reads_flat_out_of_bounds_safe() {
    let result = filter_reads_flat(&[30; 5], &[10], &[5], &QualityParams::default());
    assert_eq!(result.pass[0], 0);
}
