// SPDX-License-Identifier: AGPL-3.0-or-later
//! Variant caller for resequencing pipelines.
//!
//! Identifies SNPs, insertions, and deletions from pileup columns by
//! comparing observed base frequencies against error models. The GPU
//! composition targets are `SnpCallingF64` (column-parallel SNP) and
//! `HmmBatchForwardF64` (error model / genotype likelihood).
//!
//! # Pipeline Position
//!
//! ```text
//! pileup → variant_caller → output.gd (breseq format)
//! ```
//!
//! # Variant Types
//!
//! - **SNP**: single nucleotide polymorphism (base substitution)
//! - **DEL**: deletion (missing bases relative to reference)
//! - **INS**: insertion (extra bases relative to reference)

#[cfg(test)]
mod tests;

use crate::bio::pileup::PileupColumn;
use crate::io::fasta::GenBankFeature;

#[cfg(feature = "gpu")]
use crate::bio::snp_gpu::SnpGpu;
#[cfg(feature = "gpu")]
use barracuda::device::WgpuDevice;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Variant type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantType {
    /// Single nucleotide polymorphism.
    Snp,
    /// Deletion relative to reference.
    Deletion,
    /// Insertion relative to reference.
    Insertion,
}

impl std::fmt::Display for VariantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Snp => write!(f, "SNP"),
            Self::Deletion => write!(f, "DEL"),
            Self::Insertion => write!(f, "INS"),
        }
    }
}

/// A called variant.
#[derive(Debug, Clone)]
pub struct CalledVariant {
    /// 1-based reference position.
    pub position: usize,
    /// Variant type.
    pub variant_type: VariantType,
    /// Reference allele.
    pub ref_allele: u8,
    /// Alternative allele (for SNPs).
    pub alt_allele: u8,
    /// Read depth at this position.
    pub depth: u32,
    /// Variant allele frequency.
    pub frequency: f64,
    /// Variant quality score (Phred-scaled).
    pub quality: f64,
    /// Gene annotation if overlapping a CDS.
    pub gene: Option<String>,
}

impl CalledVariant {
    /// Format as a breseq-style GenomeDiff (`.gd`) line.
    #[must_use]
    pub fn to_gd_line(&self, seq_id: &str) -> String {
        let type_str = match self.variant_type {
            VariantType::Snp => "SNP",
            VariantType::Deletion => "DEL",
            VariantType::Insertion => "INS",
        };
        let gene_str = self
            .gene
            .as_deref()
            .unwrap_or("intergenic");
        format!(
            "{type_str}\t.\t.\t{seq_id}\t{}\t{}\tgene={gene_str}\tfreq={:.4}",
            self.position,
            self.alt_allele as char,
            self.frequency
        )
    }
}

/// Variant calling configuration.
#[derive(Debug, Clone)]
pub struct CallerConfig {
    /// Minimum depth to call a variant.
    pub min_depth: u32,
    /// Minimum alternative allele frequency to call a variant.
    pub min_alt_frequency: f64,
    /// Minimum variant quality (Phred-scaled) to report.
    pub min_quality: f64,
    /// Minimum number of reads supporting the variant.
    pub min_alt_count: u32,
    /// Minimum insertion/deletion fraction at a position to call an indel.
    pub min_indel_fraction: f64,
    /// Minimum strand bias (fraction on minor strand) to accept a variant.
    /// A true variant should appear on both strands. Range [0, 0.5].
    /// 0.0 = no filter, 0.1 = at least 10% of variant reads on each strand.
    pub min_strand_balance: f64,
    /// Use quality-weighted base frequencies instead of raw counts.
    /// When true, each base contributes its Phred error probability weight
    /// rather than a flat count of 1.
    pub quality_weighted: bool,
}

impl Default for CallerConfig {
    fn default() -> Self {
        Self {
            min_depth: 8,
            min_alt_frequency: 0.75,
            min_quality: 20.0,
            min_alt_count: 3,
            min_indel_fraction: 0.3,
            min_strand_balance: 0.1,
            quality_weighted: true,
        }
    }
}

impl CallerConfig {
    /// Permissive configuration matching the original defaults.
    /// Useful for exploratory calling or low-coverage data.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            min_depth: 5,
            min_alt_frequency: 0.1,
            min_quality: 6.0,
            min_alt_count: 2,
            min_indel_fraction: 0.1,
            min_strand_balance: 0.0,
            quality_weighted: false,
        }
    }
}

/// Call variants from pileup columns.
///
/// # Arguments
///
/// * `pileup` - Pileup columns from [`generate_pileup`](crate::bio::pileup::generate_pileup)
/// * `reference` - Reference sequence (0-indexed)
/// * `features` - Optional GenBank features for gene annotation
/// * `config` - Calling parameters
#[must_use]
pub fn call_variants(
    pileup: &[PileupColumn],
    reference: &[u8],
    features: &[GenBankFeature],
    config: &CallerConfig,
) -> Vec<CalledVariant> {
    let mut variants = Vec::new();

    for col in pileup {
        if col.depth < config.min_depth {
            continue;
        }

        // SNP calling
        if let Some(snp) = call_snp(col, reference, features, config) {
            variants.push(snp);
        }

        // Deletion calling
        if let Some(del) = call_deletion(col, reference, features, config) {
            variants.push(del);
        }

        // Insertion calling
        if let Some(ins) = call_insertion(col, reference, features, config) {
            variants.push(ins);
        }
    }

    variants
}

// ── GPU-accelerated variant calling via SnpCallingF64 ────────────

/// Call variants using GPU-accelerated column-parallel SNP calling.
///
/// Uses `SnpCallingF64` (barraCuda) for the SNP pass — each alignment column
/// is processed in parallel on the GPU. Indels are still called on CPU from
/// pileup insertion/deletion counts (no GPU equivalent needed — sparse).
///
/// # Errors
///
/// Returns [`crate::error::Error::Gpu`] if GPU SNP dispatch fails.
/// Falls back to CPU variant calling on error.
#[cfg(feature = "gpu")]
pub fn call_variants_gpu(
    pileup: &[PileupColumn],
    reference: &[u8],
    features: &[GenBankFeature],
    config: &CallerConfig,
    device: &Arc<WgpuDevice>,
) -> crate::error::Result<Vec<CalledVariant>> {
    let snp_gpu = SnpGpu::new(device)?;

    // Build alignment tensor: each pileup column becomes a "position" in
    // a virtual MSA where the reads are "sequences". We encode the most
    // common bases at each position plus the reference.
    // For GPU SNP calling, we construct synthetic aligned sequences from pileup.
    let covered_positions: Vec<&PileupColumn> = pileup
        .iter()
        .filter(|c| c.depth >= config.min_depth && c.position < reference.len())
        .collect();

    if covered_positions.is_empty() {
        return Ok(Vec::new());
    }

    let aln_len = covered_positions.len();

    // Build synthetic sequences: for each covered position, generate "reads"
    // proportional to base counts. We cap at a synthetic depth to keep
    // GPU buffer size bounded.
    let max_synthetic_depth = 20usize;
    let mut sequences: Vec<Vec<u8>> = Vec::with_capacity(max_synthetic_depth);
    for _ in 0..max_synthetic_depth {
        sequences.push(vec![b'N'; aln_len]);
    }

    for (col_idx, col) in covered_positions.iter().enumerate() {
        let bases = [b'A', b'C', b'G', b'T'];
        let total: u32 = col.base_counts[..4].iter().sum();
        if total == 0 {
            continue;
        }
        let mut seq_idx = 0;
        for (base_idx, &count) in col.base_counts[..4].iter().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "scaled to max_synthetic_depth")]
            let scaled = ((u64::from(count) * max_synthetic_depth as u64) / u64::from(total)) as usize;
            for _ in 0..scaled {
                if seq_idx < max_synthetic_depth {
                    sequences[seq_idx][col_idx] = bases[base_idx];
                    seq_idx += 1;
                }
            }
        }
        // Fill remaining with reference
        let ref_base = reference[col.position].to_ascii_uppercase();
        for seq in sequences.iter_mut().skip(seq_idx) {
            seq[col_idx] = ref_base;
        }
    }

    let seq_refs: Vec<&[u8]> = sequences.iter().map(Vec::as_slice).collect();
    let gpu_result = snp_gpu.call_snps(&seq_refs)?;

    // Convert GPU results to CalledVariants with quality-weighted refinement
    let mut variants = Vec::new();
    let bases = [b'A', b'C', b'G', b'T'];

    for (col_idx, col) in covered_positions.iter().enumerate() {
        let position_1based = col.position + 1;
        let ref_base = reference[col.position].to_ascii_uppercase();

        if gpu_result.is_variant[col_idx] == 1 {
            let gpu_ref_idx = gpu_result.ref_alleles[col_idx] as usize;
            let (alt_base, alt_idx) = if gpu_ref_idx < 4 {
                col.base_counts[..4]
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != gpu_ref_idx)
                    .max_by_key(|(_, c)| *c)
                    .map_or((b'N', 4usize), |(i, _)| (bases[i], i))
            } else {
                (b'N', 4usize)
            };

            let alt_count = col.base_counts.get(alt_idx).copied().unwrap_or(0);
            if alt_count < config.min_alt_count {
                continue;
            }

            // Apply quality-weighted frequency from pileup (not GPU synthetic)
            let frequency = if config.quality_weighted && alt_idx < 4 {
                quality_weighted_freq(col, alt_idx)
            } else {
                gpu_result.alt_frequencies[col_idx]
            };

            if frequency < config.min_alt_frequency {
                continue;
            }

            // Strand bias filter
            if config.min_strand_balance > 0.0 {
                let minor_strand = col.strand_bias().min(1.0 - col.strand_bias());
                if minor_strand < config.min_strand_balance {
                    continue;
                }
            }

            let quality = if alt_idx < 4 {
                variant_quality_bq(col, alt_idx, alt_count)
            } else {
                variant_quality(alt_count, col.depth, frequency)
            };

            if quality >= config.min_quality {
                variants.push(CalledVariant {
                    position: position_1based,
                    variant_type: VariantType::Snp,
                    ref_allele: ref_base,
                    alt_allele: alt_base,
                    depth: col.depth,
                    frequency,
                    quality,
                    gene: find_gene(position_1based, features),
                });
            }
        }

        if let Some(del) = call_deletion(col, reference, features, config) {
            variants.push(del);
        }
        if let Some(ins) = call_insertion(col, reference, features, config) {
            variants.push(ins);
        }
    }

    Ok(variants)
}

/// Compute quality-weighted frequency for a base index at a pileup column.
///
/// Each base contributes `1 - 10^(-Q/10)` (probability of being correct)
/// instead of a flat 1.0. This down-weights low-quality bases that are
/// likely sequencing errors.
#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: quality sums bounded by coverage"
)]
fn quality_weighted_freq(col: &PileupColumn, base_idx: usize) -> f64 {
    let weights: Vec<f64> = (0..4)
        .map(|i| {
            let count = f64::from(col.base_counts[i]);
            if count == 0.0 {
                return 0.0;
            }
            let mean_q = if col.base_counts[i] > 0 {
                col.quality_sums[i] as f64 / count
            } else {
                0.0
            };
            // P(correct) = 1 - 10^(-Q/10)
            let p_correct = 1.0 - 10.0_f64.powf(-mean_q / 10.0);
            count * p_correct
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights[base_idx] / total
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: depth bounded by coverage"
)]
fn call_snp(
    col: &PileupColumn,
    reference: &[u8],
    features: &[GenBankFeature],
    config: &CallerConfig,
) -> Option<CalledVariant> {
    if col.position >= reference.len() {
        return None;
    }

    let ref_base = reference[col.position].to_ascii_uppercase();
    let ref_idx = base_to_idx(ref_base);
    let bases = [b'A', b'C', b'G', b'T'];

    let mut best_alt_idx = None;
    let mut best_alt_count = 0u32;

    for (i, &count) in col.base_counts[..4].iter().enumerate() {
        if i != ref_idx && count > best_alt_count {
            best_alt_count = count;
            best_alt_idx = Some(i);
        }
    }

    let alt_idx = best_alt_idx?;
    if best_alt_count < config.min_alt_count {
        return None;
    }

    let frequency = if config.quality_weighted {
        quality_weighted_freq(col, alt_idx)
    } else {
        f64::from(best_alt_count) / f64::from(col.depth)
    };

    if frequency < config.min_alt_frequency {
        return None;
    }

    // Strand bias filter: variant reads should appear on both strands
    if config.min_strand_balance > 0.0 && col.depth >= config.min_depth {
        let strand_ratio = col.strand_bias();
        let minor_strand = strand_ratio.min(1.0 - strand_ratio);
        if minor_strand < config.min_strand_balance {
            return None;
        }
    }

    let quality = variant_quality_bq(col, alt_idx, best_alt_count);
    if quality < config.min_quality {
        return None;
    }

    let gene = find_gene(col.position + 1, features);

    Some(CalledVariant {
        position: col.position + 1,
        variant_type: VariantType::Snp,
        ref_allele: ref_base,
        alt_allele: bases[alt_idx],
        depth: col.depth,
        frequency,
        quality,
        gene,
    })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: depth bounded by coverage"
)]
fn call_deletion(
    col: &PileupColumn,
    reference: &[u8],
    features: &[GenBankFeature],
    config: &CallerConfig,
) -> Option<CalledVariant> {
    if col.deletions == 0 || col.position >= reference.len() {
        return None;
    }

    let total = col.depth + col.deletions;
    let frequency = f64::from(col.deletions) / f64::from(total);

    if frequency < config.min_indel_fraction || col.deletions < config.min_alt_count {
        return None;
    }

    let quality = variant_quality(col.deletions, total, frequency);
    if quality < config.min_quality {
        return None;
    }

    let gene = find_gene(col.position + 1, features);

    Some(CalledVariant {
        position: col.position + 1,
        variant_type: VariantType::Deletion,
        ref_allele: reference[col.position].to_ascii_uppercase(),
        alt_allele: b'-',
        depth: total,
        frequency,
        quality,
        gene,
    })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: depth bounded by coverage"
)]
fn call_insertion(
    col: &PileupColumn,
    reference: &[u8],
    features: &[GenBankFeature],
    config: &CallerConfig,
) -> Option<CalledVariant> {
    if col.insertions == 0 || col.position >= reference.len() {
        return None;
    }

    let frequency = f64::from(col.insertions) / f64::from(col.depth);

    if frequency < config.min_indel_fraction || col.insertions < config.min_alt_count {
        return None;
    }

    let quality = variant_quality(col.insertions, col.depth, frequency);
    if quality < config.min_quality {
        return None;
    }

    let gene = find_gene(col.position + 1, features);

    Some(CalledVariant {
        position: col.position + 1,
        variant_type: VariantType::Insertion,
        ref_allele: reference[col.position].to_ascii_uppercase(),
        alt_allele: b'+',
        depth: col.depth,
        frequency,
        quality,
        gene,
    })
}

/// Compute Phred-scaled variant quality using per-base quality information.
///
/// Uses the actual mean quality of the alternative allele's supporting reads
/// to compute the expected error rate, rather than assuming a flat Q30.
/// Q = sum over alt reads of (-10 * log10(P_error_per_base)).
///
/// For low-quality alt bases, this naturally produces low variant quality,
/// suppressing calls driven by sequencing error.
#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: quality sums bounded by coverage"
)]
fn variant_quality_bq(col: &PileupColumn, alt_idx: usize, alt_count: u32) -> f64 {
    if col.depth == 0 || alt_count == 0 {
        return 0.0;
    }

    let alt_mean_q = if col.base_counts[alt_idx] > 0 {
        col.quality_sums[alt_idx] as f64 / f64::from(col.base_counts[alt_idx])
    } else {
        0.0
    };

    // P(error) for the alt base = 10^(-Q/10)
    let p_error = 10.0_f64.powf(-alt_mean_q / 10.0);
    if p_error >= 1.0 {
        return 0.0;
    }

    let alt = f64::from(alt_count);
    let total = f64::from(col.depth);
    let observed_freq = alt / total;

    if observed_freq <= p_error {
        return 0.0;
    }

    // Log-likelihood ratio: observed frequency vs per-base error rate
    let lr = (observed_freq / p_error).log10();
    (alt * lr * 10.0).min(999.0)
}

/// Legacy quality function (flat error rate assumption).
fn variant_quality(alt_count: u32, total_depth: u32, _frequency: f64) -> f64 {
    if total_depth == 0 {
        return 0.0;
    }
    let error_rate = 0.001;
    let alt = f64::from(alt_count);
    let total = f64::from(total_depth);
    let observed_freq = alt / total;
    if observed_freq <= error_rate {
        return 0.0;
    }
    let lr = (observed_freq / error_rate).log10();
    (alt * lr * 10.0).min(999.0)
}

fn base_to_idx(b: u8) -> usize {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

fn find_gene(position_1based: usize, features: &[GenBankFeature]) -> Option<String> {
    features
        .iter()
        .find(|f| f.feature_type == "CDS" && position_1based >= f.start && position_1based <= f.end)
        .and_then(|f| f.gene.clone())
}

/// Parse a breseq-style GenomeDiff (`.gd`) output file.
///
/// Returns a list of `(type, position, new_base)` tuples for comparison.
#[must_use]
pub fn parse_gd_file(contents: &str) -> Vec<(String, usize, String)> {
    let mut mutations = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            continue;
        }
        let variant_type = fields[0].to_string();
        if !matches!(variant_type.as_str(), "SNP" | "DEL" | "INS" | "SUB" | "MOB" | "AMP" | "CON" | "INV") {
            continue;
        }
        if let Ok(position) = fields[4].parse::<usize>() {
            let new_base = if fields.len() > 5 {
                fields[5].to_string()
            } else {
                String::new()
            };
            mutations.push((variant_type, position, new_base));
        }
    }
    mutations
}

/// Compare two sets of variant calls for parity.
///
/// Returns `(matches, only_in_a, only_in_b)` counts.
#[must_use]
pub fn compare_calls(
    sovereign: &[CalledVariant],
    baseline: &[(String, usize, String)],
) -> (usize, usize, usize) {
    let mut matches = 0;
    let mut only_sovereign = 0;

    for call in sovereign {
        let type_str = call.variant_type.to_string();
        let found = baseline
            .iter()
            .any(|(bt, bp, _)| bt == &type_str && *bp == call.position);
        if found {
            matches += 1;
        } else {
            only_sovereign += 1;
        }
    }

    let only_baseline = baseline
        .iter()
        .filter(|(bt, bp, _)| {
            !sovereign
                .iter()
                .any(|c| c.variant_type.to_string() == *bt && c.position == *bp)
        })
        .count();

    (matches, only_sovereign, only_baseline)
}
