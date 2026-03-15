// SPDX-License-Identifier: AGPL-3.0-or-later
//! K-mer sketch acceleration for chimera detection.
//!
//! Builds 8-mer presence counts (sketches) for DNA sequences and computes
//! Jaccard-like similarity between sketches to rank parent candidates,
//! reducing the search space from O(N²) to O(N × K²) where K is small.

use std::collections::HashMap;

/// k-mer size for chimera sketch (8-mers).
pub(super) const SKETCH_K: usize = 8;

/// K-mer sketch: hash of k-mer → count.
pub(super) type KmerSketch = HashMap<u64, u16>;

/// Build a k-mer sketch (8-mer presence counts) for a DNA sequence.
#[must_use]
pub(super) fn build_sketch(seq: &[u8]) -> KmerSketch {
    let mut sketch = HashMap::new();
    if seq.len() < SKETCH_K {
        return sketch;
    }
    let mut kmer = 0_u64;
    let mask = (1_u64 << (2 * SKETCH_K)) - 1;
    let mut valid = 0_usize;

    for (i, &b) in seq.iter().enumerate() {
        let enc = match b {
            b'A' | b'a' => 0_u64,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => {
                valid = 0;
                continue;
            }
        };
        kmer = ((kmer << 2) | enc) & mask;
        valid += 1;
        if valid >= SKETCH_K {
            *sketch.entry(kmer).or_insert(0) += 1;
            if i >= seq.len().saturating_sub(1) {
                break;
            }
        }
    }
    sketch
}

/// Count shared k-mers between two sketches (Jaccard-like similarity).
#[must_use]
pub(super) fn sketch_similarity(a: &KmerSketch, b: &KmerSketch) -> u32 {
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    smaller
        .iter()
        .filter_map(|(k, &va)| larger.get(k).map(|&vb| u32::from(va.min(vb))))
        .sum()
}
