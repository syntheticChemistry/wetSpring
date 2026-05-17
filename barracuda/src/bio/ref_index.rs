// SPDX-License-Identifier: AGPL-3.0-or-later
//! FM-index for reference genome exact and seed matching.
//!
//! Builds a suffix array (via SA-IS), derives the BWT, and constructs an
//! FM-index with sampled occurrence tables for O(m) exact pattern matching
//! where m = pattern length. This is the one resequencing pipeline module
//! with no barraCuda GPU equivalent — it runs on CPU.
//!
//! # Algorithm
//!
//! 1. **SA-IS** (Nong, Zhang & Chan 2009): linear-time suffix array
//!    construction via induced sorting of LMS suffixes.
//! 2. **BWT**: derived from the suffix array in O(n).
//! 3. **FM-index**: occurrence table sampled every `OCC_INTERVAL` positions
//!    for O(m) backward search (exact matching).
//!
//! # Usage
//!
//! ```rust,no_run
//! use wetspring_barracuda::bio::ref_index::FmIndex;
//!
//! let reference = b"ACGTACGTACGT";
//! let index = FmIndex::build(reference);
//! let hits = index.exact_match(b"ACGT");
//! // hits contains 0-based positions where "ACGT" occurs in reference
//! ```

#[cfg(test)]
mod tests;

/// Sampling interval for the occurrence table.
const OCC_INTERVAL: usize = 32;

/// DNA alphabet: $ACGT (sentinel + 4 bases).
const ALPHA_SIZE: usize = 5;

/// Map a DNA byte to an alphabet index (0..5).
/// Sentinel ($) = 0, A = 1, C = 2, G = 3, T = 4.
/// Non-ACGT bases map to A (conservative).
#[inline]
const fn base_to_idx(b: u8) -> usize {
    match b {
        0 | b'$' => 0,
        b'A' | b'a' => 1,
        b'C' | b'c' => 2,
        b'G' | b'g' => 3,
        b'T' | b't' => 4,
        _ => 1,
    }
}

// ── SA-IS suffix array construction ──────────────────────────────

/// Build suffix array using SA-IS (Nong et al. 2009).
///
/// Input is an integer sequence with a sentinel (0) as the smallest element.
/// `alpha_size` is the number of distinct symbols (max value + 1).
/// Returns a suffix array of length `text.len()`.
#[expect(clippy::too_many_lines, reason = "SA-IS is a single algorithm, splitting would obscure flow")]
fn sais(text: &[usize], alpha_size: usize) -> Vec<usize> {
    let n = text.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    if n == 2 {
        return if text[0] <= text[1] {
            vec![0, 1]
        } else {
            vec![1, 0]
        };
    }

    // Step 1: classify each suffix as S-type or L-type
    let mut stype = vec![false; n]; // true = S-type, false = L-type
    stype[n - 1] = true; // sentinel is always S-type
    for i in (0..n - 1).rev() {
        stype[i] = if text[i] < text[i + 1] {
            true
        } else if text[i] > text[i + 1] {
            false
        } else {
            stype[i + 1]
        };
    }

    // Find LMS (Left-Most S-type) positions
    let mut is_lms = vec![false; n];
    let mut lms_positions = Vec::new();
    for i in 1..n {
        if stype[i] && !stype[i - 1] {
            is_lms[i] = true;
            lms_positions.push(i);
        }
    }

    // Bucket boundaries
    let mut bucket_sizes = vec![0usize; alpha_size];
    for &sym in text {
        bucket_sizes[sym] += 1;
    }
    let bucket_starts = {
        let mut starts = vec![0usize; alpha_size];
        let mut sum = 0;
        for (i, &sz) in bucket_sizes.iter().enumerate() {
            starts[i] = sum;
            sum += sz;
        }
        starts
    };
    let bucket_ends = {
        let mut ends = vec![0usize; alpha_size];
        let mut sum = 0;
        for (i, &sz) in bucket_sizes.iter().enumerate() {
            sum += sz;
            ends[i] = sum;
        }
        ends
    };

    let induced_sort = |sa: &mut Vec<usize>, text: &[usize], stype: &[bool], lms_order: &[usize]| {
        sa.fill(usize::MAX);

        // Place LMS suffixes at the end of their buckets
        let mut tails = bucket_ends.clone();
        for &pos in lms_order.iter().rev() {
            let c = text[pos];
            tails[c] -= 1;
            sa[tails[c]] = pos;
        }

        // Induce L-type from left to right
        let mut heads = bucket_starts.clone();
        for i in 0..n {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if !stype[j] {
                let c = text[j];
                sa[heads[c]] = j;
                heads[c] += 1;
            }
        }

        // Induce S-type from right to left
        let mut tails2 = bucket_ends.clone();
        for i in (0..n).rev() {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if stype[j] {
                let c = text[j];
                tails2[c] -= 1;
                sa[tails2[c]] = j;
            }
        }
    };

    // Step 2: initial induced sort with LMS positions in text order
    let mut sa = vec![usize::MAX; n];
    induced_sort(&mut sa, text, &stype, &lms_positions);

    // Step 3: name LMS substrings
    let num_lms = lms_positions.len();
    let mut name = vec![usize::MAX; n];
    let mut current_name = 0usize;
    let mut prev_lms = usize::MAX;

    for i in 0..n {
        if sa[i] == usize::MAX || !is_lms[sa[i]] {
            continue;
        }
        if prev_lms != usize::MAX {
            let mut different = false;
            let a = prev_lms;
            let b = sa[i];
            let mut d = 0;
            loop {
                if a + d >= n || b + d >= n {
                    different = true;
                    break;
                }
                if text[a + d] != text[b + d] || stype[a + d] != stype[b + d] {
                    different = true;
                    break;
                }
                if d > 0 && (is_lms[a + d] || is_lms[b + d]) {
                    break;
                }
                d += 1;
            }
            if different {
                current_name += 1;
            }
        }
        name[sa[i]] = current_name;
        prev_lms = sa[i];
    }

    // Compact the reduced string
    let mut reduced: Vec<usize> = Vec::with_capacity(num_lms);
    let mut lms_map: Vec<usize> = Vec::with_capacity(num_lms);
    for i in 0..n {
        if name[i] != usize::MAX {
            reduced.push(name[i]);
            lms_map.push(i);
        }
    }

    // Step 4: solve recursively if names are not unique
    let sorted_lms = if current_name + 1 < num_lms {
        let sub_sa = sais(&reduced, current_name + 1);
        sub_sa.iter().map(|&i| lms_map[i]).collect::<Vec<_>>()
    } else {
        // Names are unique — directly invert
        let mut order = vec![0usize; num_lms];
        for (i, &r) in reduced.iter().enumerate() {
            order[r] = lms_map[i];
        }
        order
    };

    // Step 5: final induced sort with correctly ordered LMS
    induced_sort(&mut sa, text, &stype, &sorted_lms);

    sa
}

// ── FM-Index ─────────────────────────────────────────────────────

/// FM-index for O(m) exact pattern matching on a reference genome.
///
/// Stores the BWT, sampled suffix array, and sampled occurrence counts.
/// Memory: ~5 bytes per reference base (BWT + sampled counts + sampled SA).
pub struct FmIndex {
    bwt: Vec<u8>,
    /// `C[c]` = number of characters in BWT that are lexicographically smaller than c.
    c_table: [usize; ALPHA_SIZE + 1],
    /// Sampled occurrence table: `occ[i * ALPHA_SIZE + c]` = count of character c
    /// in `bwt[0..i * OCC_INTERVAL]`.
    occ: Vec<u32>,
    /// Sampled suffix array (every `OCC_INTERVAL`-th position).
    sa_sample: Vec<u32>,
    /// Full length of the indexed text (including sentinel).
    len: usize,
}

impl FmIndex {
    /// Build an FM-index from a reference sequence.
    ///
    /// Appends a sentinel byte, builds the suffix array via SA-IS,
    /// derives the BWT, and constructs sampled occurrence tables.
    #[must_use]
    pub fn build(reference: &[u8]) -> Self {
        // Convert DNA bytes to integer alphabet and append sentinel
        let mut text: Vec<usize> = Vec::with_capacity(reference.len() + 1);
        for &b in reference {
            text.push(match b {
                b'A' | b'a' => 1,
                b'C' | b'c' => 2,
                b'G' | b'g' => 3,
                b'T' | b't' => 4,
                _ => 1, // N → A
            });
        }
        text.push(0); // sentinel

        let n = text.len();
        let sa = sais(&text, ALPHA_SIZE);

        // Derive BWT from suffix array (back to u8 for storage)
        let mut bwt = vec![0u8; n];
        for (i, &sa_val) in sa.iter().enumerate() {
            let sym = if sa_val == 0 { text[n - 1] } else { text[sa_val - 1] };
            bwt[i] = sym as u8;
        }

        // Build C table
        let mut c_table = [0usize; ALPHA_SIZE + 1];
        for &b in &bwt {
            c_table[b as usize + 1] += 1;
        }
        for i in 1..=ALPHA_SIZE {
            c_table[i] += c_table[i - 1];
        }

        // Build sampled occurrence table
        let num_samples = n / OCC_INTERVAL + 1;
        let mut occ = vec![0u32; num_samples * ALPHA_SIZE];
        let mut counts = [0u32; ALPHA_SIZE];
        for (i, &b) in bwt.iter().enumerate() {
            counts[b as usize] += 1;
            if (i + 1) % OCC_INTERVAL == 0 {
                let sample_idx = (i + 1) / OCC_INTERVAL;
                for c in 0..ALPHA_SIZE {
                    occ[sample_idx * ALPHA_SIZE + c] = counts[c];
                }
            }
        }

        // Sample suffix array
        #[expect(clippy::cast_possible_truncation, reason = "genome < 4GB")]
        let sa_sample: Vec<u32> = sa
            .iter()
            .enumerate()
            .filter(|(i, _)| i % OCC_INTERVAL == 0)
            .map(|(_, &v)| v as u32)
            .collect();

        Self {
            bwt,
            c_table,
            occ,
            sa_sample,
            len: n,
        }
    }

    /// Count occurrences of character `c` (as alphabet index) in `bwt[0..pos]`.
    fn occ_count(&self, c: usize, pos: usize) -> usize {
        let block = pos / OCC_INTERVAL;
        let base = if block == 0 {
            0
        } else {
            self.occ[block * ALPHA_SIZE + c] as usize
        };
        let start = block * OCC_INTERVAL;
        let mut count = base;
        for i in start..pos {
            if self.bwt[i] as usize == c {
                count += 1;
            }
        }
        count
    }

    /// Locate a suffix array position from a BWT index.
    ///
    /// Walks backward through the BWT using LF-mapping until hitting
    /// a sampled SA position, then adds the walk distance.
    fn locate(&self, mut idx: usize) -> usize {
        let mut steps = 0;
        loop {
            if idx % OCC_INTERVAL == 0 {
                let sa_val = self.sa_sample[idx / OCC_INTERVAL] as usize;
                let result = sa_val + steps;
                // Subtract 1 because we appended a sentinel
                return if result >= self.len { result - self.len } else { result };
            }
            let c = self.bwt[idx] as usize;
            idx = self.c_table[c] + self.occ_count(c, idx);
            steps += 1;
        }
    }

    /// Find all exact occurrences of `pattern` in the reference.
    ///
    /// Returns 0-based positions in the original reference (without sentinel).
    /// Runs in O(m + k) where m = pattern length, k = number of occurrences.
    #[must_use]
    pub fn exact_match(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() {
            return vec![];
        }

        let m = pattern.len();
        let c = base_to_idx(pattern[m - 1]);
        let mut lo = self.c_table[c];
        let mut hi = self.c_table[c + 1];

        for i in (0..m - 1).rev() {
            if lo >= hi {
                return vec![];
            }
            let c = base_to_idx(pattern[i]);
            lo = self.c_table[c] + self.occ_count(c, lo);
            hi = self.c_table[c] + self.occ_count(c, hi);
        }

        if lo >= hi {
            return vec![];
        }

        let mut positions: Vec<usize> = (lo..hi).map(|i| self.locate(i)).collect();
        // Filter out sentinel position and adjust for sentinel
        positions.retain(|&p| p < self.len - 1);
        positions.sort_unstable();
        positions
    }

    /// Count exact occurrences of `pattern` without locating them.
    ///
    /// Runs in O(m) — faster than [`exact_match`](Self::exact_match) when
    /// only the count is needed.
    #[must_use]
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() {
            return 0;
        }

        let m = pattern.len();
        let c = base_to_idx(pattern[m - 1]);
        let mut lo = self.c_table[c];
        let mut hi = self.c_table[c + 1];

        for i in (0..m - 1).rev() {
            if lo >= hi {
                return 0;
            }
            let c = base_to_idx(pattern[i]);
            lo = self.c_table[c] + self.occ_count(c, lo);
            hi = self.c_table[c] + self.occ_count(c, hi);
        }

        hi.saturating_sub(lo)
    }

    /// Reference length (without sentinel).
    #[must_use]
    pub fn reference_len(&self) -> usize {
        self.len - 1
    }

    /// Extract all k-mer seed positions from a query read.
    ///
    /// For each k-mer in the read, returns `(read_offset, reference_positions)`.
    /// Skips k-mers with more than `max_hits` occurrences (repetitive seeds).
    #[must_use]
    pub fn seed_kmers(&self, read: &[u8], k: usize, max_hits: usize) -> Vec<(usize, Vec<usize>)> {
        if read.len() < k {
            return vec![];
        }
        let mut seeds = Vec::new();
        for i in 0..=read.len() - k {
            let kmer = &read[i..i + k];
            let count = self.count(kmer);
            if count > 0 && count <= max_hits {
                let positions = self.exact_match(kmer);
                seeds.push((i, positions));
            }
        }
        seeds
    }
}
