// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapter sequence detection and trimming.
//!
//! Semi-global alignment of adapter sequences against read ends,
//! with IUPAC ambiguity support. Used by [`super::quality`] for 3'
//! adapter removal in quality-filtering pipelines.

use crate::io::fastq::FastqRecord;

/// Find adapter sequence at the 3' end of a read.
///
/// Uses semi-global alignment: the adapter can overlap the read end.
/// Returns the position where the adapter starts, or `None` if not found.
///
/// # Arguments
///
/// * `sequence` — Read sequence.
/// * `adapter` — Adapter sequence to search for.
/// * `max_mismatches` — Maximum allowed mismatches in the alignment.
/// * `min_overlap` — Minimum overlap between adapter and read end.
#[must_use]
pub fn find_adapter_3prime(
    sequence: &[u8],
    adapter: &[u8],
    max_mismatches: usize,
    min_overlap: usize,
) -> Option<usize> {
    if adapter.is_empty() || sequence.is_empty() {
        return None;
    }

    let seq_len = sequence.len();
    let adp_len = adapter.len();

    let earliest_start = seq_len.saturating_sub(adp_len);

    for start in earliest_start..seq_len {
        let overlap = seq_len - start;
        if overlap < min_overlap {
            break;
        }

        let mismatches = sequence[start..]
            .iter()
            .zip(adapter[..overlap].iter())
            .filter(|(a, b)| !bases_match(**a, **b))
            .count();

        if mismatches <= max_mismatches {
            return Some(start);
        }
    }

    if adp_len <= seq_len {
        for start in 0..=(seq_len - adp_len) {
            let mismatches = sequence[start..start + adp_len]
                .iter()
                .zip(adapter.iter())
                .filter(|(a, b)| !bases_match(**a, **b))
                .count();

            if mismatches <= max_mismatches {
                return Some(start);
            }
        }
    }

    None
}

/// Trim adapter from the 3' end of a record.
///
/// Returns `Some(trimmed_record)` if an adapter was found and removed,
/// or `None` if no adapter match — avoids cloning on the common path.
#[must_use]
pub fn trim_adapter_3prime(
    record: &FastqRecord,
    adapter: &[u8],
    max_mismatches: usize,
    min_overlap: usize,
) -> Option<FastqRecord> {
    let pos = find_adapter_3prime(&record.sequence, adapter, max_mismatches, min_overlap)?;
    Some(FastqRecord {
        id: record.id.clone(),
        sequence: record.sequence[..pos].to_vec(),
        quality: record.quality[..pos].to_vec(),
    })
}

/// Case-insensitive base comparison with IUPAC ambiguity support.
///
/// `N` matches any base on either side.
pub(crate) const fn bases_match(a: u8, b: u8) -> bool {
    let a = a.to_ascii_uppercase();
    let b = b.to_ascii_uppercase();
    if a == b {
        return true;
    }
    if a == b'N' || b == b'N' {
        return true;
    }
    false
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_record(seq: &[u8], qual: &[u8]) -> FastqRecord {
        FastqRecord {
            id: "test".to_string(),
            sequence: seq.to_vec(),
            quality: qual.to_vec(),
        }
    }

    #[test]
    fn find_adapter_exact_match_at_end() {
        let seq = b"ACGTACGTAATTCCGG";
        let adapter = b"AATTCCGG";
        let pos = find_adapter_3prime(seq, adapter, 0, 4);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn find_adapter_partial_overlap() {
        let seq = b"ACGTACGTAATT";
        let adapter = b"AATTCCGG";
        let pos = find_adapter_3prime(seq, adapter, 0, 4);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn find_adapter_with_mismatch() {
        let seq = b"ACGTACGTAATTCCGA";
        let adapter = b"AATTCCGG";
        assert!(find_adapter_3prime(seq, adapter, 0, 4).is_none());
        let pos = find_adapter_3prime(seq, adapter, 1, 4);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn find_adapter_n_ambiguity() {
        let seq = b"ACGTACGTNATTCCGG";
        let adapter = b"AATTCCGG";
        let pos = find_adapter_3prime(seq, adapter, 0, 4);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn find_adapter_empty_inputs() {
        assert!(find_adapter_3prime(b"", b"AATT", 0, 1).is_none());
        assert!(find_adapter_3prime(b"ACGT", b"", 0, 1).is_none());
    }

    #[test]
    fn trim_adapter_removes_suffix() {
        let record = make_record(b"ACGTACGTAATTCCGG", &[b'I'; 16]);
        let adapter = b"AATTCCGG";
        let trimmed = trim_adapter_3prime(&record, adapter, 0, 4).unwrap();
        assert_eq!(trimmed.sequence, b"ACGTACGT");
        assert_eq!(trimmed.quality.len(), 8);
    }

    #[test]
    fn trim_adapter_no_match() {
        let record = make_record(b"ACGTACGT", &[b'I'; 8]);
        assert!(trim_adapter_3prime(&record, b"GGGGGGGG", 0, 4).is_none());
    }

    #[test]
    fn bases_match_same() {
        assert!(bases_match(b'A', b'A'));
        assert!(bases_match(b'a', b'A'));
    }

    #[test]
    fn bases_match_n_wildcard() {
        assert!(bases_match(b'N', b'A'));
        assert!(bases_match(b'C', b'N'));
        assert!(bases_match(b'N', b'N'));
    }

    #[test]
    fn bases_match_different() {
        assert!(!bases_match(b'A', b'C'));
    }
}
