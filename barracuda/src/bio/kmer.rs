// SPDX-License-Identifier: AGPL-3.0-or-later
//! K-mer counting engine with 2-bit DNA encoding.
//!
//! Encodes DNA sequences as packed u64 k-mers (k <= 32),
//! uses canonical form (min of forward/reverse complement),
//! and counts via `HashMap`.
//!
//! This is the core operation behind DADA2's dereplication step
//! and is a primary target for GPU acceleration (`hash_table_u64.wgsl`).

use std::collections::HashMap;

/// 2-bit DNA encoding: A=0, C=1, G=2, T=3.
/// Returns `None` for ambiguous bases (N, etc).
#[inline]
const fn encode_base(b: u8) -> Option<u64> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

/// Complement of a 2-bit encoded base.
#[inline]
const fn complement_2bit(b: u64) -> u64 {
    3 - b // A(0)<->T(3), C(1)<->G(2)
}

/// Compute reverse complement of a k-mer encoded as u64.
fn reverse_complement(kmer: u64, k: usize) -> u64 {
    let mut rc = 0_u64;
    let mut fwd = kmer;
    for _ in 0..k {
        let base = fwd & 3;
        rc = (rc << 2) | complement_2bit(base);
        fwd >>= 2;
    }
    rc
}

/// Canonical k-mer: min(forward, `reverse_complement`).
/// This ensures each k-mer and its reverse complement are counted together.
fn canonical(kmer: u64, k: usize) -> u64 {
    let rc = reverse_complement(kmer, k);
    kmer.min(rc)
}

/// K-mer counting result.
#[derive(Debug, Clone)]
pub struct KmerCounts {
    /// k-mer size.
    pub k: usize,
    /// Canonical k-mer -> count.
    pub counts: HashMap<u64, u32>,
    /// Total k-mers processed (excluding those with ambiguous bases).
    pub total_valid_kmers: u64,
    /// Number of k-mers skipped due to ambiguous bases.
    pub skipped_ambiguous: u64,
}

impl KmerCounts {
    /// Number of unique canonical k-mers.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.counts.len()
    }

    /// Total k-mer occurrences (sum of all counts).
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.counts.values().map(|&c| u64::from(c)).sum()
    }

    /// Top N most abundant k-mers.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(u64, u32)> {
        let mut entries: Vec<_> = self.counts.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Flat histogram of size 4^k, indexed by canonical k-mer value.
    ///
    /// For k <= 12 (4^12 = 16M entries), this produces a dense GPU buffer
    /// suitable for parallel histogram reduction or radix sort dispatch.
    #[inline]
    #[must_use]
    pub fn to_histogram(&self) -> Vec<u32> {
        let kmer_space = 1_usize << (2 * self.k);
        let mut hist = vec![0_u32; kmer_space];
        for (&kmer, &count) in &self.counts {
            hist[kmer as usize] = count;
        }
        hist
    }

    /// Reconstruct `KmerCounts` from a flat histogram produced by [`Self::to_histogram`].
    #[must_use]
    pub fn from_histogram(histogram: &[u32], k: usize) -> Self {
        let mut counts = HashMap::new();
        let mut total_valid = 0_u64;
        for (kmer, &count) in histogram.iter().enumerate() {
            if count > 0 {
                counts.insert(kmer as u64, count);
                total_valid += u64::from(count);
            }
        }
        Self {
            k,
            counts,
            total_valid_kmers: total_valid,
            skipped_ambiguous: 0,
        }
    }

    /// Sorted (kmer, count) pairs for GPU binary-search or merge operations.
    ///
    /// Sorted ascending by canonical k-mer value. Compact representation
    /// for large k where 4^k histogram would be too sparse.
    #[must_use]
    pub fn to_sorted_pairs(&self) -> Vec<(u64, u32)> {
        let mut pairs: Vec<_> = self.counts.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_unstable_by_key(|&(k, _)| k);
        pairs
    }

    /// Reconstruct `KmerCounts` from sorted pairs produced by [`Self::to_sorted_pairs`].
    #[must_use]
    pub fn from_sorted_pairs(pairs: &[(u64, u32)], k: usize) -> Self {
        let mut counts = HashMap::with_capacity(pairs.len());
        let mut total_valid = 0_u64;
        for &(kmer, count) in pairs {
            counts.insert(kmer, count);
            total_valid += u64::from(count);
        }
        Self {
            k,
            counts,
            total_valid_kmers: total_valid,
            skipped_ambiguous: 0,
        }
    }
}

/// Decode a 2-bit encoded k-mer back to a DNA string.
#[must_use]
pub fn decode_kmer(kmer: u64, k: usize) -> String {
    let mut result = Vec::with_capacity(k);
    for i in (0..k).rev() {
        let base = (kmer >> (2 * i)) & 3;
        result.push(match base {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => unreachable!(),
        });
    }
    String::from_utf8(result).unwrap_or_default()
}

/// Count canonical k-mers in a single sequence.
///
/// # Panics
///
/// Panics if `k` is 0 or greater than 32.
#[must_use]
pub fn count_kmers(sequence: &[u8], k: usize) -> KmerCounts {
    assert!(k > 0 && k <= 32, "k must be in [1, 32]");

    let mask = if k == 32 {
        u64::MAX
    } else {
        (1_u64 << (2 * k)) - 1
    };
    let mut counts: HashMap<u64, u32> = HashMap::new();
    let mut total_valid = 0_u64;
    let mut skipped = 0_u64;

    if sequence.len() < k {
        return KmerCounts {
            k,
            counts,
            total_valid_kmers: 0,
            skipped_ambiguous: 0,
        };
    }

    let mut kmer = 0_u64;
    let mut valid_bases = 0_usize;

    for &base in sequence {
        if let Some(encoded) = encode_base(base) {
            kmer = ((kmer << 2) | encoded) & mask;
            valid_bases += 1;
        } else {
            // Ambiguous base: reset window
            valid_bases = 0;
            skipped += 1;
            continue;
        }

        if valid_bases >= k {
            let canonical_kmer = canonical(kmer, k);
            *counts.entry(canonical_kmer).or_insert(0) += 1;
            total_valid += 1;
        }
    }

    KmerCounts {
        k,
        counts,
        total_valid_kmers: total_valid,
        skipped_ambiguous: skipped,
    }
}

/// Count canonical k-mers across multiple sequences.
///
/// # Panics
///
/// Panics if `k` is 0 or greater than 32.
#[must_use]
pub fn count_kmers_multi(sequences: &[&[u8]], k: usize) -> KmerCounts {
    let mut combined = KmerCounts {
        k,
        counts: HashMap::new(),
        total_valid_kmers: 0,
        skipped_ambiguous: 0,
    };

    for seq in sequences {
        let result = count_kmers(seq, k);
        combined.total_valid_kmers += result.total_valid_kmers;
        combined.skipped_ambiguous += result.skipped_ambiguous;
        for (kmer, count) in result.counts {
            *combined.counts.entry(kmer).or_insert(0) += count;
        }
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        assert_eq!(encode_base(b'A'), Some(0));
        assert_eq!(encode_base(b'C'), Some(1));
        assert_eq!(encode_base(b'G'), Some(2));
        assert_eq!(encode_base(b'T'), Some(3));
        assert_eq!(encode_base(b'N'), None);
    }

    #[test]
    fn test_reverse_complement() {
        // ACGT -> ACGT (palindrome)
        let k = 4;
        let acgt = 0b00_01_10_11_u64; // A=00, C=01, G=10, T=11
        let rc = reverse_complement(acgt, k);
        assert_eq!(rc, acgt, "ACGT should be its own reverse complement");
    }

    #[test]
    fn test_canonical_palindrome() {
        let k = 4;
        let acgt = 0b00_01_10_11_u64;
        assert_eq!(canonical(acgt, k), acgt);
    }

    #[test]
    fn test_count_simple() {
        let seq = b"ACGTACGT";
        let counts = count_kmers(seq, 4);
        assert!(counts.unique_count() > 0);
        assert_eq!(counts.total_valid_kmers, 5); // 8 - 4 + 1 = 5
    }

    #[test]
    fn test_count_with_n() {
        let seq = b"ACGTNACGT";
        let counts = count_kmers(seq, 4);
        assert_eq!(counts.skipped_ambiguous, 1);
        assert!(counts.total_valid_kmers > 0);
    }

    #[test]
    fn test_decode() {
        let kmer = 0b00_01_10_11_u64; // ACGT
        assert_eq!(decode_kmer(kmer, 4), "ACGT");
    }

    #[test]
    fn test_sequence_too_short() {
        let counts = count_kmers(b"AC", 4);
        assert_eq!(counts.total_valid_kmers, 0);
        assert_eq!(counts.unique_count(), 0);
    }

    #[test]
    fn test_multi_sequence() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGT"];
        let counts = count_kmers_multi(&seqs, 4);
        // Each sequence has 5 k-mers, so 10 total
        assert_eq!(counts.total_valid_kmers, 10);
    }

    #[test]
    fn test_decode_round_trip() {
        // Encode GCAT, then decode
        let encoded = (2 << 6) | (1 << 4) | 3; // G=10, C=01, A=00, T=11
        assert_eq!(decode_kmer(encoded, 4), "GCAT");
    }

    #[test]
    fn test_top_n() {
        let seq = b"AAAAAAAAAACCCCC";
        let counts = count_kmers(seq, 3);
        let top = counts.top_n(1);
        assert_eq!(top.len(), 1);
        // AAA or its canonical form should be the most abundant
        assert!(top[0].1 >= 2);
    }

    #[test]
    fn test_top_n_more_than_available() {
        let counts = count_kmers(b"ACGT", 4);
        let top = counts.top_n(100);
        assert!(top.len() <= 100);
        assert!(!top.is_empty());
    }

    #[test]
    fn test_lowercase_bases() {
        let upper = count_kmers(b"ACGTACGT", 4);
        let lower = count_kmers(b"acgtacgt", 4);
        assert_eq!(upper.total_valid_kmers, lower.total_valid_kmers);
        assert_eq!(upper.unique_count(), lower.unique_count());
    }

    #[test]
    fn test_k_equals_1() {
        let counts = count_kmers(b"ACGT", 1);
        assert_eq!(counts.total_valid_kmers, 4);
        // Canonical: A<->T (both map to min), C<->G (both map to min)
        // So we expect 2 unique canonical 1-mers
        assert_eq!(counts.unique_count(), 2);
    }

    #[test]
    fn test_all_ambiguous() {
        let counts = count_kmers(b"NNNNN", 3);
        assert_eq!(counts.total_valid_kmers, 0);
        assert_eq!(counts.skipped_ambiguous, 5);
    }

    #[test]
    fn test_complement_2bit_symmetry() {
        // A(0)<->T(3), C(1)<->G(2)
        assert_eq!(complement_2bit(0), 3);
        assert_eq!(complement_2bit(3), 0);
        assert_eq!(complement_2bit(1), 2);
        assert_eq!(complement_2bit(2), 1);
    }

    #[test]
    fn test_count_kmers_multi_empty() {
        let seqs: Vec<&[u8]> = vec![];
        let counts = count_kmers_multi(&seqs, 4);
        assert_eq!(counts.total_valid_kmers, 0);
        assert_eq!(counts.unique_count(), 0);
    }

    #[test]
    fn test_decode_kmer_single_base() {
        assert_eq!(decode_kmer(0, 1), "A");
        assert_eq!(decode_kmer(1, 1), "C");
        assert_eq!(decode_kmer(2, 1), "G");
        assert_eq!(decode_kmer(3, 1), "T");
    }

    #[test]
    fn test_total_count_method() {
        let counts = count_kmers(b"ACGTACGT", 4);
        assert!(counts.total_count() > 0);
        // total_count == total_valid_kmers for non-overlapping canonical k-mers
        assert_eq!(counts.total_count(), counts.total_valid_kmers);
    }

    #[test]
    fn test_k_equals_32() {
        // Exercises the u64::MAX mask path
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        assert!(seq.len() >= 32);
        let counts = count_kmers(seq, 32);
        // 40 - 32 + 1 = 9 k-mers
        assert_eq!(counts.total_valid_kmers, 9);
        assert!(counts.unique_count() > 0);
    }

    #[test]
    fn histogram_round_trip() {
        let counts = count_kmers(b"ACGTACGTACGTACGTAAACCCC", 4);
        let hist = counts.to_histogram();
        assert_eq!(hist.len(), 256); // 4^4
        let restored = KmerCounts::from_histogram(&hist, 4);
        assert_eq!(restored.unique_count(), counts.unique_count());
        for (&kmer, &count) in &counts.counts {
            assert_eq!(restored.counts.get(&kmer).copied(), Some(count));
        }
    }

    #[test]
    fn sorted_pairs_round_trip() {
        let counts = count_kmers(b"ACGTACGTACGTACGTAAACCCC", 4);
        let pairs = counts.to_sorted_pairs();
        for w in pairs.windows(2) {
            assert!(w[0].0 < w[1].0, "pairs must be sorted by kmer");
        }
        let restored = KmerCounts::from_sorted_pairs(&pairs, 4);
        assert_eq!(restored.unique_count(), counts.unique_count());
        for (&kmer, &count) in &counts.counts {
            assert_eq!(restored.counts.get(&kmer).copied(), Some(count));
        }
    }

    #[test]
    fn histogram_preserves_top_n() {
        let counts = count_kmers(b"AAAAAAAAAACCCCC", 3);
        let hist = counts.to_histogram();
        let restored = KmerCounts::from_histogram(&hist, 3);
        let top_orig = counts.top_n(1);
        let top_rest = restored.top_n(1);
        assert_eq!(top_orig.len(), 1);
        assert_eq!(top_rest.len(), 1);
        assert_eq!(top_orig[0].0, top_rest[0].0, "top kmer mismatch");
        assert_eq!(top_orig[0].1, top_rest[0].1, "top count mismatch");
    }

    #[test]
    fn histogram_gpu_buffer_size() {
        let counts = count_kmers(b"ACGT", 8);
        let hist = counts.to_histogram();
        assert_eq!(hist.len(), 65_536); // 4^8 = GPU-friendly
    }
}
