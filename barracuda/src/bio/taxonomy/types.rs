// SPDX-License-Identifier: AGPL-3.0-or-later
//! Taxonomy domain types — ranks, lineages, reference sequences.

/// Taxonomic ranks used in 16S classification.
///
/// Follows the standard hierarchy from kingdom down to species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaxRank {
    /// Domain/kingdom (e.g., Bacteria, Archaea).
    Kingdom,
    /// Phylum-level classification.
    Phylum,
    /// Class-level classification.
    Class,
    /// Order-level classification.
    Order,
    /// Family-level classification.
    Family,
    /// Genus-level classification.
    Genus,
    /// Species-level classification.
    Species,
}

impl TaxRank {
    /// Return all ranks from kingdom to species, in order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Kingdom,
            Self::Phylum,
            Self::Class,
            Self::Order,
            Self::Family,
            Self::Genus,
            Self::Species,
        ]
    }

    /// Zero-based index for this rank (Kingdom=0, Species=6). Use with `ranks.get(depth)`.
    #[must_use]
    pub const fn depth(self) -> usize {
        match self {
            Self::Kingdom => 0,
            Self::Phylum => 1,
            Self::Class => 2,
            Self::Order => 3,
            Self::Family => 4,
            Self::Genus => 5,
            Self::Species => 6,
        }
    }
}

/// A taxonomic lineage (one entry per rank from kingdom to species).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lineage {
    /// Taxon names at each rank (index matches `TaxRank::depth()`).
    pub ranks: Vec<String>,
}

impl Lineage {
    /// Parse a semicolon-delimited taxonomy string (e.g., SILVA format).
    /// `d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;...`
    #[must_use]
    pub fn from_taxonomy_string(s: &str) -> Self {
        let ranks: Vec<String> = s
            .split(';')
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect();
        Self { ranks }
    }

    /// Lookup taxon name at a specific rank. Returns `None` if rank exceeds lineage depth.
    #[must_use]
    pub fn at_rank(&self, rank: TaxRank) -> Option<&str> {
        self.ranks.get(rank.depth()).map(String::as_str)
    }

    /// Format lineage up to (and including) the given rank, semicolon-separated.
    #[must_use]
    pub fn to_string_at_rank(&self, rank: TaxRank) -> String {
        let depth = rank.depth() + 1;
        self.ranks[..depth.min(self.ranks.len())].join(";")
    }
}

/// A reference sequence for training the classifier.
#[derive(Debug, Clone)]
pub struct ReferenceSeq {
    /// Accession or identifier from the FASTA header.
    pub id: String,
    /// DNA sequence as bytes (A,C,G,T in uppercase).
    pub sequence: Vec<u8>,
    /// Taxonomic lineage parsed from header or taxonomy file.
    pub lineage: Lineage,
}

/// Classification result for a single query.
#[derive(Debug, Clone)]
pub struct Classification {
    /// Assigned lineage.
    pub lineage: Lineage,
    /// Bootstrap confidence at each rank (0.0 to 1.0).
    pub confidence: Vec<f64>,
    /// Index of the matched taxon in the classifier.
    pub taxon_idx: usize,
}

/// Parameters for classification.
#[derive(Debug, Clone)]
pub struct ClassifyParams {
    /// K-mer size (must match training). Default: 8.
    pub k: usize,
    /// Number of bootstrap iterations. Default: 100.
    pub bootstrap_n: usize,
    /// Minimum bootstrap confidence to report a rank. Default: 0.8.
    pub min_confidence: f64,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            bootstrap_n: DEFAULT_BOOTSTRAP_N,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }
}

/// NPU-compatible int8 weight buffers for taxonomy classification.
///
/// Layout matches NPU FC layer requirements: weights as int8 with
/// affine quantization parameters for dequantization.
#[derive(Debug, Clone)]
pub struct NpuWeights {
    /// Quantized log-probability table: `n_taxa × kmer_space`, row-major.
    pub weights_i8: Vec<i8>,
    /// Quantized log-prior per taxon.
    pub priors_i8: Vec<i8>,
    /// Quantization scale: `real_value = quantized * scale + zero_point`.
    pub scale: f64,
    /// Quantization zero point.
    pub zero_point: f64,
    /// Number of taxa.
    pub n_taxa: usize,
    /// K-mer space size (4^k).
    pub kmer_space: usize,
}

pub(crate) const DEFAULT_K: usize = 8;
pub(crate) const DEFAULT_BOOTSTRAP_N: usize = 100;
pub(crate) const DEFAULT_MIN_CONFIDENCE: f64 = 0.8;
