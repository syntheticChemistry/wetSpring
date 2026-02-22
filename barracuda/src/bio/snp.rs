// SPDX-License-Identifier: AGPL-3.0-or-later
//! SNP (Single Nucleotide Polymorphism) calling from aligned sequences.
//!
//! Computes variant positions and allele frequencies from multiple
//! aligned sequences. Used in Anderson 2017 for population-level
//! genomic variation analysis at hydrothermal vents.

/// A detected variant at a specific position.
#[derive(Debug, Clone)]
pub struct Variant {
    /// 0-based position in the alignment.
    pub position: usize,
    /// Reference allele (most common base at this position).
    pub ref_allele: u8,
    /// Alternative allele(s) and their counts.
    pub alt_alleles: Vec<(u8, usize)>,
    /// Total depth (non-gap, non-N reads) at this position.
    pub depth: usize,
}

impl Variant {
    /// Allele frequency of the reference allele.
    #[must_use]
    pub fn ref_frequency(&self) -> f64 {
        if self.depth == 0 {
            return 0.0;
        }
        let alt_count: usize = self.alt_alleles.iter().map(|(_, c)| c).sum();
        (self.depth - alt_count) as f64 / self.depth as f64
    }

    /// Allele frequency of the major alternative allele.
    #[must_use]
    pub fn alt_frequency(&self) -> f64 {
        if self.depth == 0 {
            return 0.0;
        }
        let max_alt = self.alt_alleles.iter().map(|(_, c)| *c).max().unwrap_or(0);
        max_alt as f64 / self.depth as f64
    }
}

/// Result of SNP calling across an alignment.
#[derive(Debug, Clone)]
pub struct SnpResult {
    /// All detected variants.
    pub variants: Vec<Variant>,
    /// Alignment length.
    pub alignment_length: usize,
    /// Number of sequences in the alignment.
    pub n_sequences: usize,
}

impl SnpResult {
    /// SNP density: variants per base pair.
    #[must_use]
    pub fn snp_density(&self) -> f64 {
        if self.alignment_length == 0 {
            return 0.0;
        }
        self.variants.len() as f64 / self.alignment_length as f64
    }
}

const VALID_BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// GPU uniform parameters for batched SNP dispatch.
///
/// Maps directly to WGSL `var<uniform>` for future GPU absorption.
/// Each thread processes one alignment position independently.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SnpParams {
    /// Alignment length (number of positions to process).
    pub alignment_length: u32,
    /// Number of sequences in the alignment.
    pub n_sequences: u32,
    /// Minimum depth to call a variant.
    pub min_depth: u32,
}

/// Flat SNP result for GPU storage buffer binding.
///
/// Parallel arrays (`SoA`) instead of `Vec<Variant>` — each index
/// is one variant position. Maps directly to WGSL storage buffers.
#[derive(Debug, Clone)]
pub struct SnpFlatResult {
    /// Variant positions (0-based).
    pub positions: Vec<u32>,
    /// Reference allele at each position (encoded: A=0,C=1,G=2,T=3).
    pub ref_alleles: Vec<u8>,
    /// Major alt allele at each position.
    pub alt_alleles: Vec<u8>,
    /// Depth at each position.
    pub depths: Vec<u32>,
    /// Alt allele frequency at each position.
    pub alt_frequencies: Vec<f64>,
}

/// Call SNPs and return flat arrays (GPU-friendly API).
///
/// Equivalent to `call_snps` but returns `SoA` layout for direct
/// GPU buffer binding in future absorption.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn call_snps_flat(sequences: &[&[u8]]) -> SnpFlatResult {
    let result = call_snps(sequences);
    let n = result.variants.len();
    let mut positions = Vec::with_capacity(n);
    let mut ref_alleles = Vec::with_capacity(n);
    let mut alt_alleles = Vec::with_capacity(n);
    let mut depths = Vec::with_capacity(n);
    let mut alt_frequencies = Vec::with_capacity(n);

    for v in &result.variants {
        positions.push(v.position as u32);
        ref_alleles.push(v.ref_allele);
        let major_alt = v
            .alt_alleles
            .iter()
            .max_by_key(|(_, c)| *c)
            .map_or(b'N', |(b, _)| *b);
        alt_alleles.push(major_alt);
        depths.push(v.depth as u32);
        alt_frequencies.push(v.alt_frequency());
    }

    SnpFlatResult {
        positions,
        ref_alleles,
        alt_alleles,
        depths,
        alt_frequencies,
    }
}

/// Call SNPs across multiple independent alignments in batch.
///
/// Returns one `SnpFlatResult` per alignment. This is the entry point
/// for GPU dispatch: each alignment maps to one workgroup, with
/// positions distributed across threads.
#[must_use]
pub fn call_snps_batch(alignments: &[Vec<Vec<u8>>]) -> Vec<SnpFlatResult> {
    alignments
        .iter()
        .map(|aln| {
            let refs: Vec<&[u8]> = aln.iter().map(Vec::as_slice).collect();
            call_snps_flat(&refs)
        })
        .collect()
}

/// Call SNPs from a set of aligned sequences.
///
/// Each sequence must be the same length. Positions where all non-gap
/// bases are identical are invariant. Positions with 2+ different bases
/// are called as variants.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn call_snps(sequences: &[&[u8]]) -> SnpResult {
    if sequences.is_empty() {
        return SnpResult {
            variants: Vec::new(),
            alignment_length: 0,
            n_sequences: 0,
        };
    }

    let aln_len = sequences[0].len();
    let mut variants = Vec::new();

    for pos in 0..aln_len {
        let mut counts = [0usize; 4]; // A, C, G, T
        let mut depth = 0usize;

        for seq in sequences {
            if pos >= seq.len() {
                continue;
            }
            let base = seq[pos].to_ascii_uppercase();
            if base == b'-' || base == b'.' || base == b'N' {
                continue;
            }
            depth += 1;
            match base {
                b'A' => counts[0] += 1,
                b'C' => counts[1] += 1,
                b'G' => counts[2] += 1,
                b'T' => counts[3] += 1,
                _ => {}
            }
        }

        let n_alleles = counts.iter().filter(|&&c| c > 0).count();
        if n_alleles < 2 || depth < 2 {
            continue;
        }

        let ref_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map_or(0, |(i, _)| i);

        let alt_alleles: Vec<(u8, usize)> = counts
            .iter()
            .enumerate()
            .filter(|&(i, &c)| i != ref_idx && c > 0)
            .map(|(i, &c)| (VALID_BASES[i], c))
            .collect();

        variants.push(Variant {
            position: pos,
            ref_allele: VALID_BASES[ref_idx],
            alt_alleles,
            depth,
        });
    }

    SnpResult {
        variants,
        alignment_length: aln_len,
        n_sequences: sequences.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_variants_identical() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGATG"];
        let result = call_snps(&seqs);
        assert!(result.variants.is_empty());
    }

    #[test]
    fn single_snp() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGTTG"];
        let result = call_snps(&seqs);
        assert_eq!(result.variants.len(), 1);
        assert_eq!(result.variants[0].position, 3);
    }

    #[test]
    fn multiple_snps() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGATG", b"ATGTTG"];
        let result = call_snps(&seqs);
        assert_eq!(result.variants.len(), 2);
    }

    #[test]
    fn allele_frequency() {
        let seqs: Vec<&[u8]> = vec![b"A", b"A", b"A", b"T"];
        let result = call_snps(&seqs);
        assert_eq!(result.variants.len(), 1);
        let v = &result.variants[0];
        assert!((v.ref_frequency() - 0.75).abs() < 1e-10);
        assert!((v.alt_frequency() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn gaps_excluded() {
        let seqs: Vec<&[u8]> = vec![b"A-G", b"A-G", b"ATG"];
        let result = call_snps(&seqs);
        assert!(result.variants.is_empty() || result.variants.len() == 1);
    }

    #[test]
    fn snp_density() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGTTG"];
        let result = call_snps(&seqs);
        assert!(result.snp_density() > 0.0);
    }

    #[test]
    fn empty_input() {
        let seqs: Vec<&[u8]> = vec![];
        let result = call_snps(&seqs);
        assert!(result.variants.is_empty());
        assert_eq!(result.n_sequences, 0);
    }

    #[test]
    fn single_sequence_no_variants() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG"];
        let result = call_snps(&seqs);
        assert!(result.variants.is_empty());
    }

    #[test]
    fn deterministic() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGATG", b"ATGTTG"];
        let r1 = call_snps(&seqs);
        let r2 = call_snps(&seqs);
        assert_eq!(r1.variants.len(), r2.variants.len());
        for (v1, v2) in r1.variants.iter().zip(&r2.variants) {
            assert_eq!(v1.position, v2.position);
            assert_eq!(v1.ref_allele, v2.ref_allele);
        }
    }

    #[test]
    fn flat_matches_structured() {
        let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGATG", b"ATGTTG"];
        let structured = call_snps(&seqs);
        let flat = call_snps_flat(&seqs);
        assert_eq!(flat.positions.len(), structured.variants.len());
        for (i, v) in structured.variants.iter().enumerate() {
            assert_eq!(flat.positions[i], v.position as u32);
            assert_eq!(flat.ref_alleles[i], v.ref_allele);
            assert_eq!(flat.depths[i], v.depth as u32);
            assert!((flat.alt_frequencies[i] - v.alt_frequency()).abs() < 1e-15);
        }
    }

    #[test]
    fn variant_zero_depth() {
        let v = Variant {
            position: 0,
            ref_allele: b'A',
            alt_alleles: vec![],
            depth: 0,
        };
        assert!(v.ref_frequency().abs() < f64::EPSILON);
        assert!(v.alt_frequency().abs() < f64::EPSILON);
    }

    #[test]
    fn snp_density_empty_alignment() {
        let result = SnpResult {
            variants: vec![],
            alignment_length: 0,
            n_sequences: 0,
        };
        assert!(result.snp_density().abs() < f64::EPSILON);
    }

    #[test]
    fn batch_snps_multiple_alignments() {
        let aln1 = vec![b"ATGATG".to_vec(), b"CTGATG".to_vec()];
        let aln2 = vec![b"GGGG".to_vec(), b"GGGG".to_vec()];
        let aln3 = vec![b"ACGT".to_vec(), b"TCGT".to_vec(), b"ACGT".to_vec()];
        let results = call_snps_batch(&[aln1, aln2, aln3]);
        assert_eq!(results.len(), 3);
        assert!(!results[0].positions.is_empty(), "aln1 should have SNPs");
        assert!(
            results[1].positions.is_empty(),
            "aln2 (identical) should have no SNPs"
        );
        assert!(!results[2].positions.is_empty(), "aln3 should have SNPs");
    }

    #[test]
    fn batch_snps_empty() {
        let results = call_snps_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn batch_snps_single_alignment_matches_direct() {
        let aln = vec![b"ATGATG".to_vec(), b"CTGATG".to_vec(), b"ATGTTG".to_vec()];
        let batch = call_snps_batch(&[aln.clone()]);
        let refs: Vec<&[u8]> = aln.iter().map(Vec::as_slice).collect();
        let direct = call_snps_flat(&refs);
        assert_eq!(batch[0].positions, direct.positions);
        assert_eq!(batch[0].ref_alleles, direct.ref_alleles);
        assert_eq!(batch[0].alt_frequencies, direct.alt_frequencies);
    }
}
