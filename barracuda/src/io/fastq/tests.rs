// SPDX-License-Identifier: AGPL-3.0-or-later
//! FASTQ parser tests.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::redundant_closure_for_method_calls
)]

use super::*;
use std::io::Write;

/// Write a minimal FASTQ file and return its path.
fn write_fastq(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

#[test]
fn test_stats_from_file_single_record_quality() {
    let dir = tempfile::tempdir().unwrap();
    // Q=40 for 'I' (73-33=40)
    let content = "@s1\nACGT\n+\nIIII\n";
    let path = write_fastq(&dir, "q40.fastq", content);
    let stats = stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 1);
    assert!((stats.mean_quality - 40.0).abs() < 1e-10);
    assert!((stats.gc_content - 0.5).abs() < 1e-10);
    assert_eq!(stats.q30_count, 1);
}

#[test]
fn test_stats_from_file_low_quality() {
    let dir = tempfile::tempdir().unwrap();
    // Q=0 for '!' (33-33=0)
    let content = "@s1\nAAAA\n+\n!!!!\n";
    let path = write_fastq(&dir, "q0.fastq", content);
    let stats = stats_from_file(&path).unwrap();
    assert!((stats.mean_quality - 0.0).abs() < 1e-10);
    assert_eq!(stats.q30_count, 0);
}

#[test]
fn test_stats_gc_lowercase() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@s1\ngcgc\n+\nIIII\n";
    let path = write_fastq(&dir, "lc.fastq", content);
    let stats = stats_from_file(&path).unwrap();
    assert!((stats.gc_content - 1.0).abs() < 1e-10);
}

#[test]
fn test_parse_and_stats_agree() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@a\nACGTAC\n+\nIIIIII\n@b\nGGCC\n+\n!I!I\n";
    let path = write_fastq(&dir, "agree.fastq", content);
    let records: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let stats_from_records = compute_stats(&records);
    let stats_from_stream = stats_from_file(&path).unwrap();
    assert_eq!(
        stats_from_records.num_sequences,
        stats_from_stream.num_sequences
    );
    assert_eq!(
        stats_from_records.total_bases,
        stats_from_stream.total_bases
    );
    assert_eq!(stats_from_records.min_length, stats_from_stream.min_length);
    assert_eq!(stats_from_records.max_length, stats_from_stream.max_length);
    assert!((stats_from_records.gc_content - stats_from_stream.gc_content).abs() < 1e-12);
    assert!((stats_from_records.mean_quality - stats_from_stream.mean_quality).abs() < 1e-12);
}

#[test]
fn test_empty_stats() {
    let stats = compute_stats(&[]);
    assert_eq!(stats.num_sequences, 0);
    assert_eq!(stats.total_bases, 0);
    assert_eq!(stats.min_length, 0);
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

#[test]
fn test_gc_content_all_gc() {
    let rec = FastqRecord {
        id: "gc".to_string(),
        sequence: b"GGGGCCCC".to_vec(),
        quality: b"IIIIIIII".to_vec(),
    };
    let stats = compute_stats(&[rec]);
    assert!((stats.gc_content - 1.0).abs() < 1e-10);
}

#[test]
fn test_gc_content_no_gc() {
    let rec = FastqRecord {
        id: "at".to_string(),
        sequence: b"AAAATTTT".to_vec(),
        quality: b"IIIIIIII".to_vec(),
    };
    let stats = compute_stats(&[rec]);
    assert!(stats.gc_content.abs() < 1e-10);
}

#[test]
fn test_q30_threshold() {
    let high_q = FastqRecord {
        id: "high".to_string(),
        sequence: b"ACGT".to_vec(),
        quality: b"IIII".to_vec(), // Q=40
    };
    let low_q = FastqRecord {
        id: "low".to_string(),
        sequence: b"ACGT".to_vec(),
        quality: b"!!!!".to_vec(), // Q=0
    };
    let stats = compute_stats(&[high_q, low_q]);
    assert_eq!(stats.q30_count, 1);
}

/// Write a gzip-compressed FASTQ file and return its path.
fn write_fastq_gz(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    let path = dir.path().join(name);
    let file = std::fs::File::create(&path).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    gz.write_all(content.as_bytes()).unwrap();
    gz.finish().unwrap();
    path
}

#[test]
fn test_stats_from_file_trailing_blank_line() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@r1\nACGT\n+\nIIII\n\n";
    let path = write_fastq(&dir, "trailing.fastq", content);
    let stats = stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 1);
}

#[test]
fn test_stats_from_file_empty_quality_line() {
    let dir = tempfile::tempdir().unwrap();
    // quality line is present but empty — exercises qual_len == 0 path
    let content = "@r1\nACGT\n+\n\n";
    let path = write_fastq(&dir, "noq.fastq", content);
    let stats = stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 1);
    assert!((stats.mean_quality - 0.0).abs() < 1e-10);
}

#[test]
fn test_compute_stats_zero_gc_and_quality() {
    // All A/T, empty quality — exercises both gc_content==0 and mean_quality==0
    let rec = FastqRecord {
        id: "at".to_string(),
        sequence: b"AATTAATT".to_vec(),
        quality: vec![],
    };
    let stats = compute_stats(&[rec]);
    assert!(stats.gc_content.abs() < 1e-10);
    assert!(stats.mean_quality.abs() < 1e-10);
    assert_eq!(stats.q30_count, 0);
}

#[test]
fn test_compute_stats_empty_sequence_record() {
    // Non-empty records with zero-length sequence — exercises total_bases==0 fallback
    let rec = FastqRecord {
        id: "empty_seq".to_string(),
        sequence: vec![],
        quality: vec![],
    };
    let stats = compute_stats(&[rec]);
    assert_eq!(stats.num_sequences, 1);
    assert_eq!(stats.total_bases, 0);
    assert!(stats.gc_content.abs() < 1e-10);
    assert!(stats.mean_quality.abs() < 1e-10);
    assert_eq!(stats.min_length, 0);
    assert_eq!(stats.max_length, 0);
}

#[test]
fn test_length_distribution() {
    let r1 = FastqRecord {
        id: "a".into(),
        sequence: b"ACGT".to_vec(),
        quality: vec![],
    };
    let r2 = FastqRecord {
        id: "b".into(),
        sequence: b"ACGTAC".to_vec(),
        quality: vec![],
    };
    let r3 = FastqRecord {
        id: "c".into(),
        sequence: b"ACGT".to_vec(),
        quality: vec![],
    };
    let stats = compute_stats(&[r1, r2, r3]);
    assert_eq!(stats.length_distribution[&4], 2);
    assert_eq!(stats.length_distribution[&6], 1);
    assert_eq!(stats.min_length, 4);
    assert_eq!(stats.max_length, 6);
}

#[test]
fn fastq_iter_matches_parse() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@r1 desc\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCC\n+\n!!!!\n";
    let path = write_fastq(&dir, "iter.fastq", content);

    let buffered: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let streamed: Vec<FastqRecord> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();

    assert_eq!(buffered.len(), streamed.len());
    for (b, s) in buffered.iter().zip(streamed.iter()) {
        assert_eq!(b.id, s.id);
        assert_eq!(b.sequence, s.sequence);
        assert_eq!(b.quality, s.quality);
    }
}

#[test]
fn fastq_iter_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "empty.fastq", "");
    let records: Vec<FastqRecord> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert!(records.is_empty());
}

#[test]
fn for_each_record_matches_parse() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@r1 desc\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCC\n+\n!!!!\n";
    let path = write_fastq(&dir, "foreach.fastq", content);

    let mut collected: Vec<FastqRecord> = Vec::new();
    for_each_record(&path, |r: FastqRefRecord<'_>| {
        collected.push(FastqRecord {
            id: r.id.to_string(),
            sequence: r.sequence.to_vec(),
            quality: r.quality.to_vec(),
        });
        Ok(())
    })
    .unwrap();

    let parsed: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(collected.len(), parsed.len());
    for (c, p) in collected.iter().zip(parsed.iter()) {
        assert_eq!(c.id, p.id);
        assert_eq!(c.sequence, p.sequence);
        assert_eq!(c.quality, p.quality);
    }
}

#[test]
fn for_each_record_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "empty.fastq", "");
    let mut count = 0_usize;
    for_each_record(&path, |_| {
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn for_each_record_propagates_error() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@r1\nACGT\n+\nIIII\n";
    let path = write_fastq(&dir, "err.fastq", content);
    let result = for_each_record(&path, |r: FastqRefRecord<'_>| {
        if r.id == "r1" {
            Err(crate::error::Error::Fastq("stop".to_string()))
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
}

#[test]
fn for_each_record_bad_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "bad.fastq", "NOT_HEADER\nACGT\n+\nIIII\n");
    let result = for_each_record(&path, |_| Ok(()));
    assert!(result.is_err());
}

#[test]
fn for_each_record_gzip() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@gz1\nACGT\n+\nIIII\n@gz2\nTTTT\n+\n!!!!\n";
    let path = write_fastq_gz(&dir, "foreach.fastq.gz", content);
    let mut count = 0_usize;
    for_each_record(&path, |r: FastqRefRecord<'_>| {
        if count == 0 {
            assert_eq!(r.id, "gz1");
            assert_eq!(r.sequence, b"ACGT");
        } else {
            assert_eq!(r.id, "gz2");
            assert_eq!(r.sequence, b"TTTT");
        }
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn fastq_iter_gzip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("iter.fastq.gz");
    let file = std::fs::File::create(&path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    gz.write_all(b"@g1\nACGT\n+\nIIII\n").unwrap();
    gz.finish().unwrap();

    let records: Vec<FastqRecord> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "g1");
    assert_eq!(records[0].sequence, b"ACGT");
}

#[test]
fn parse_fastq_bad_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "bad.fastq", "NOPE\nACGT\n+\nIIII\n");
    assert!(
        FastqIter::open(&path)
            .and_then(|i| i.collect::<Result<Vec<_>>>())
            .is_err()
    );
}

#[test]
fn stats_from_file_bad_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "bad.fastq", "NOT_HEADER\nACGT\n+\nIIII\n");
    assert!(stats_from_file(&path).is_err());
}

#[test]
fn fastq_iter_bad_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "bad.fastq", "NOPE\nACGT\n+\nIIII\n");
    let mut iter = FastqIter::open(&path).unwrap();
    assert!(iter.next().unwrap().is_err());
    assert!(iter.next().is_none());
}

#[test]
fn parse_fastq_truncated_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "trunc.fastq", "@r1\nACGT\n+\n");
    let records: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(records.len(), 1);
    assert!(records[0].quality.is_empty());
}

#[test]
fn fastq_nonexistent_file() {
    let result = FastqIter::open(std::path::Path::new("/nonexistent/reads.fastq"));
    assert!(result.is_err());
}

#[test]
fn fastq_iter_malformed_header_after_valid_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(
        &dir,
        "malformed.fastq",
        "@r1\nACGT\n+\nIIII\nNOT_AT\nGGCC\n+\n!!!!\n",
    );
    let mut iter = FastqIter::open(&path).unwrap();
    let first = iter.next().unwrap();
    assert!(first.is_ok());
    assert_eq!(first.unwrap().id, "r1");
    let second = iter.next().unwrap();
    assert!(second.is_err());
    assert!(iter.next().is_none());
}

#[test]
fn fastq_iter_truncated_after_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "trunc2.fastq", "@r1\nACGT\n");
    let records: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "r1");
    assert_eq!(records[0].sequence, b"ACGT");
    assert!(records[0].quality.is_empty());
}

#[test]
fn fastq_iter_empty_line_terminates() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "empty_line.fastq", "@r1\nACGT\n+\nIIII\n\n");
    let records: Vec<_> = FastqIter::open(&path)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(records.len(), 1);
}

#[test]
fn for_each_record_processes_each_record() {
    let dir = tempfile::tempdir().unwrap();
    let content = "@a\nACGT\n+\nIIII\n@b\nGGCC\n+\n!!!!\n@c\nTTTT\n+\n!!!!\n";
    let path = write_fastq(&dir, "count.fastq", content);
    let mut count = 0_usize;
    for_each_record(&path, |r: FastqRefRecord<'_>| {
        count += 1;
        match count {
            1 => assert_eq!(r.id, "a"),
            2 => assert_eq!(r.id, "b"),
            3 => assert_eq!(r.id, "c"),
            _ => {}
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn for_each_record_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "trunc_foreach.fastq", "@r1\nACGT\n+\n");
    let mut count = 0_usize;
    for_each_record(&path, |r: FastqRefRecord<'_>| {
        count += 1;
        assert_eq!(r.id, "r1");
        assert_eq!(r.sequence, b"ACGT");
        assert!(r.quality.is_empty());
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn stats_from_file_truncated_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(&dir, "trunc_stats.fastq", "@r1\nACGT\n+\n");
    let stats = stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 1);
    assert_eq!(stats.total_bases, 4);
}

#[test]
fn parse_fastq_truncated_returns_partial() {
    let dir = tempfile::tempdir().unwrap();
    // First record complete, second truncated (no + or quality)
    let path = write_fastq(&dir, "partial.fastq", "@r1\nACGT\n+\nIIII\n@r2\nGG\n");
    let records = parse_fastq(&path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "r1");
    assert_eq!(records[0].sequence, b"ACGT");
    assert_eq!(records[0].quality, b"IIII");
    assert_eq!(records[1].id, "r2");
    assert_eq!(records[1].sequence, b"GG");
    assert!(records[1].quality.is_empty());
}

#[test]
fn for_each_record_stops_on_empty_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fastq(
        &dir,
        "empty_sep.fastq",
        "@r1\nACGT\n+\nIIII\n\n@r2\nGG\n+\nII\n",
    );
    let mut count = 0_usize;
    for_each_record(&path, |_record| {
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn for_each_record_nonexistent_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("__nonexistent__.fastq");
    let result = for_each_record(&path, |_| Ok(()));
    assert!(result.is_err());
}

#[test]
fn fastq_iter_nonexistent_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("__nonexistent__.fastq");
    let result = super::FastqIter::open(&path);
    assert!(result.is_err());
}
