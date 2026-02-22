// SPDX-License-Identifier: AGPL-3.0-or-later
//! K-mer extraction for taxonomy classification.
//!
//! 2-bit encoding of DNA bases into k-mer integers for O(1) lookup
//! in the flat log-probability table. Reusable for any k-mer-based
//! analysis (also used by `bio::kmer` for counting).

/// Extract all k-mers from a DNA sequence (no canonicalization — direction matters for taxonomy).
#[inline]
#[must_use]
pub fn extract_kmers(seq: &[u8], k: usize) -> Vec<u64> {
    if seq.len() < k || k == 0 || k > 32 {
        return vec![];
    }

    let mut kmers = Vec::with_capacity(seq.len() - k + 1);
    let mask = if k == 32 {
        u64::MAX
    } else {
        (1_u64 << (2 * k)) - 1
    };
    let mut kmer = 0_u64;
    let mut valid_len = 0_usize;

    for &base in seq {
        if let Some(encoded) = encode_base(base) {
            kmer = ((kmer << 2) | encoded) & mask;
            valid_len += 1;
        } else {
            valid_len = 0;
            kmer = 0;
        }

        if valid_len >= k {
            kmers.push(kmer);
        }
    }

    kmers
}

/// Encode a single DNA base to 2-bit representation.
///
/// Returns `None` for ambiguous bases (N, IUPAC degenerate), which
/// resets the sliding window in [`extract_kmers`].
pub(crate) const fn encode_base(b: u8) -> Option<u64> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

/// Parse a FASTA reference database into [`super::types::ReferenceSeq`] records.
/// Expects SILVA-style headers: `>ID taxonomy_string`
#[must_use]
pub fn parse_reference_fasta(contents: &str) -> Vec<super::types::ReferenceSeq> {
    use super::types::{Lineage, ReferenceSeq};

    let mut refs = Vec::new();
    let mut current_id = String::new();
    let mut current_tax = String::new();
    let mut current_seq = Vec::new();

    for line in contents.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !current_id.is_empty() && !current_seq.is_empty() {
                refs.push(ReferenceSeq {
                    id: std::mem::take(&mut current_id),
                    sequence: std::mem::take(&mut current_seq),
                    lineage: Lineage::from_taxonomy_string(&current_tax),
                });
            }
            let parts: Vec<&str> = header.splitn(2, |c: char| c.is_whitespace()).collect();
            current_id = parts[0].to_string();
            current_tax = parts.get(1).unwrap_or(&"").to_string();
            current_seq.clear();
        } else {
            current_seq.extend(
                line.trim()
                    .bytes()
                    .filter(u8::is_ascii_alphabetic)
                    .map(|b| b.to_ascii_uppercase()),
            );
        }
    }

    if !current_id.is_empty() && !current_seq.is_empty() {
        refs.push(ReferenceSeq {
            id: current_id,
            sequence: current_seq,
            lineage: Lineage::from_taxonomy_string(&current_tax),
        });
    }

    refs
}
