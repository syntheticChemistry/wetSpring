//! K-mer counting engine with 2-bit DNA encoding.
//!
//! Encodes DNA sequences as packed u64 k-mers (k <= 32),
//! uses canonical form (min of forward/reverse complement),
//! and counts via HashMap.
//!
//! This is the core operation behind DADA2's dereplication step
//! and is a primary target for GPU acceleration (hash_table_u64.wgsl).

use std::collections::HashMap;

/// 2-bit DNA encoding: A=0, C=1, G=2, T=3.
/// Returns None for ambiguous bases (N, etc).
#[inline]
fn encode_base(b: u8) -> Option<u64> {
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
fn complement_2bit(b: u64) -> u64 {
    3 - b // A(0)<->T(3), C(1)<->G(2)
}

/// Compute reverse complement of a k-mer encoded as u64.
fn reverse_complement(kmer: u64, k: usize) -> u64 {
    let mut rc = 0u64;
    let mut fwd = kmer;
    for _ in 0..k {
        let base = fwd & 3;
        rc = (rc << 2) | complement_2bit(base);
        fwd >>= 2;
    }
    rc
}

/// Canonical k-mer: min(forward, reverse_complement).
/// This ensures each k-mer and its reverse complement are counted together.
fn canonical(kmer: u64, k: usize) -> u64 {
    let rc = reverse_complement(kmer, k);
    kmer.min(rc)
}

/// K-mer counting result.
#[derive(Debug, Clone)]
pub struct KmerCounts {
    /// k-mer size
    pub k: usize,
    /// Canonical k-mer -> count
    pub counts: HashMap<u64, u32>,
    /// Total k-mers processed (including those with ambiguous bases)
    pub total_valid_kmers: u64,
    /// Number of k-mers skipped due to ambiguous bases
    pub skipped_ambiguous: u64,
}

impl KmerCounts {
    /// Number of unique canonical k-mers.
    pub fn unique_count(&self) -> usize {
        self.counts.len()
    }

    /// Total k-mer occurrences (sum of all counts).
    pub fn total_count(&self) -> u64 {
        self.counts.values().map(|&c| c as u64).sum()
    }

    /// Top N most abundant k-mers.
    pub fn top_n(&self, n: usize) -> Vec<(u64, u32)> {
        let mut entries: Vec<_> = self.counts.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }
}

/// Decode a 2-bit encoded k-mer back to a DNA string.
pub fn decode_kmer(kmer: u64, k: usize) -> String {
    let mut s = Vec::with_capacity(k);
    let mut val = kmer;
    for _ in 0..k {
        let base = match (val >> (2 * (k - 1 - s.len()))) & 3 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => b'N',
        };
        s.push(base);
        val &= !(3u64 << (2 * (k - 1 - (s.len() - 1))));
    }
    // Simpler approach: extract from MSB to LSB
    let mut result = Vec::with_capacity(k);
    for i in (0..k).rev() {
        let base = (kmer >> (2 * i)) & 3;
        result.push(match base {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => b'N',
        });
    }
    String::from_utf8(result).unwrap_or_default()
}

/// Count canonical k-mers in a single sequence.
pub fn count_kmers(sequence: &[u8], k: usize) -> KmerCounts {
    assert!(k > 0 && k <= 32, "k must be in [1, 32]");

    let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    let mut counts: HashMap<u64, u32> = HashMap::new();
    let mut total_valid = 0u64;
    let mut skipped = 0u64;

    if sequence.len() < k {
        return KmerCounts {
            k,
            counts,
            total_valid_kmers: 0,
            skipped_ambiguous: 0,
        };
    }

    let mut kmer = 0u64;
    let mut valid_bases = 0usize;

    for &base in sequence.iter() {
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
        let acgt = 0b00_01_10_11u64; // A=00, C=01, G=10, T=11
        let rc = reverse_complement(acgt, k);
        assert_eq!(rc, acgt, "ACGT should be its own reverse complement");
    }

    #[test]
    fn test_canonical_palindrome() {
        let k = 4;
        let acgt = 0b00_01_10_11u64;
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
        // Window resets at N, so we get kmers from ACGT (1) and NACGT after reset (1)
        assert!(counts.total_valid_kmers > 0);
    }

    #[test]
    fn test_decode() {
        let kmer = 0b00_01_10_11u64; // ACGT
        assert_eq!(decode_kmer(kmer, 4), "ACGT");
    }
}
