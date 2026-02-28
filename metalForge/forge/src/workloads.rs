// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preset workloads for life science and analytical chemistry domains.
//!
//! Each workload declares its required capabilities and shader origin (local
//! WGSL or absorbed `ToadStool` primitive). The origin tracking enables:
//!
//! 1. Dispatch decisions — local shaders need `compile_shader_f64`; absorbed
//!    primitives use `ToadStool`'s pre-built pipelines.
//! 2. Absorption planning — `ToadStool` can see which domains still use local
//!    shaders and prioritize absorption accordingly.
//! 3. Validation routing — local shaders need CPU ↔ GPU parity checks;
//!    absorbed primitives are ToadStool-validated upstream.
//!
//! # Write → Absorb → Lean
//!
//! When `ToadStool` absorbs a local shader, we update the origin from
//! [`ShaderOrigin::Local`] to [`ShaderOrigin::Absorbed`] and rewire the
//! dispatch to use the upstream primitive. This is the Lean step.

use crate::dispatch::Workload;
use crate::substrate::{Capability, SubstrateKind};

/// Where the GPU shader for a workload lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderOrigin {
    /// Absorbed by `ToadStool` — uses `barracuda::ops::*` primitives.
    Absorbed,
    /// Local WGSL shader in `barracuda/src/shaders/` — pending absorption.
    Local,
    /// CPU-only domain — no GPU shader exists or is planned.
    CpuOnly,
}

/// A bio workload with shader provenance tracking.
#[derive(Debug)]
pub struct BioWorkload {
    /// The dispatch workload (name + capabilities).
    pub workload: Workload,
    /// Where the GPU implementation lives.
    pub origin: ShaderOrigin,
    /// `ToadStool` primitive name (if absorbed).
    pub primitive: Option<&'static str>,
    /// ODE system dimensions (if applicable).
    pub ode_dims: Option<OdeDims>,
}

/// ODE system dimensions for dispatch sizing.
#[derive(Debug, Clone, Copy)]
pub struct OdeDims {
    /// Number of state variables.
    pub n_vars: u32,
    /// Number of parameters per batch element.
    pub n_params: u32,
}

impl BioWorkload {
    const fn new_static(origin: ShaderOrigin) -> Self {
        Self {
            workload: Workload {
                name: String::new(),
                required: Vec::new(),
                preferred_substrate: None,
                data_bytes: None,
            },
            origin,
            primitive: None,
            ode_dims: None,
        }
    }

    fn named(mut self, name: &str, required: Vec<Capability>) -> Self {
        self.workload.name = name.to_string();
        self.workload.required = required;
        self
    }

    const fn with_primitive(mut self, primitive: &'static str) -> Self {
        self.primitive = Some(primitive);
        self
    }

    const fn with_ode(mut self, n_vars: u32, n_params: u32) -> Self {
        self.ode_dims = Some(OdeDims { n_vars, n_params });
        self
    }

    /// Whether this workload uses a local (non-absorbed) WGSL shader.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Local)
    }

    /// Whether this workload has been absorbed by `ToadStool`.
    #[must_use]
    pub const fn is_absorbed(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Absorbed)
    }

    /// Whether this workload is CPU-only (no GPU path).
    #[must_use]
    pub const fn is_cpu_only(&self) -> bool {
        matches!(self.origin, ShaderOrigin::CpuOnly)
    }
}

// ── Absorbed ToadStool domains ──────────────────────────────────────

/// Diversity metrics (Shannon, Simpson, Bray-Curtis).
#[must_use]
pub fn diversity() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "diversity",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// `PCoA` eigendecomposition.
#[must_use]
pub fn pcoa() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "pcoa",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedEighGpu")
}

/// K-mer histogram counting.
#[must_use]
pub fn kmer_histogram() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "kmer_histogram",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("KmerHistogramGpu")
}

/// `UniFrac` tree propagation.
#[must_use]
pub fn unifrac_propagate() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "unifrac_propagate",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("UniFracPropagateGpu")
}

/// QS/c-di-GMP ODE sweep (4 vars, 17 params — absorbed).
#[must_use]
pub fn qs_biofilm_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "qs_biofilm_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4F64")
        .with_ode(4, 17)
}

/// Smith-Waterman alignment.
#[must_use]
pub fn smith_waterman() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "smith_waterman",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("SmithWatermanGpu")
}

/// Felsenstein phylogenetic pruning.
#[must_use]
pub fn felsenstein() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "felsenstein",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

/// Taxonomy classification (NPU candidate via int8 quantization).
#[must_use]
pub fn taxonomy() -> BioWorkload {
    let mut w = BioWorkload::new_static(ShaderOrigin::Absorbed).named(
        "taxonomy",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w.with_primitive("TaxonomyFcF64")
}

// ── Absorbed ODE domains (trait-generated WGSL via BatchedOdeRK4) ───

/// Phage defense ODE (4 vars, 11 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn phage_defense_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "phage_defense_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<PhageDefenseOde>")
        .with_ode(4, 11)
}

/// Bistable QS ODE (5 vars, 21 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn bistable_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "bistable_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<BistableOde>")
        .with_ode(5, 21)
}

/// Multi-signal QS ODE (7 vars, 24 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn multi_signal_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "multi_signal_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<MultiSignalOde>")
        .with_ode(7, 24)
}

/// Cooperation game theory ODE (4 vars, 13 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn cooperation_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cooperation_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CooperationOde>")
        .with_ode(4, 13)
}

/// Capacitor phenotype ODE (6 vars, 16 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn capacitor_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "capacitor_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CapacitorOde>")
        .with_ode(6, 16)
}

// ── Write phase: new WGSL extensions for ToadStool absorption ───────

/// Fused diversity metrics (Shannon + Simpson + evenness in single dispatch).
///
/// Local WGSL extension following hotSpring's absorption pattern.
/// Computes all three diversity indices in one kernel pass, avoiding
/// three separate `FusedMapReduceF64` dispatches.
/// Absorbed by `ToadStool` S63 as `ops::bio::diversity_fusion`.
#[must_use]
pub fn diversity_fusion() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "diversity_fusion",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("DiversityFusionGpu")
}

// ── Composed GPU domains (ToadStool primitives) ─────────────────────

/// KMD (Kendrick mass defect) — element-wise via FMR.
#[must_use]
pub fn kmd() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "kmd",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// GBM batch inference — composes `TreeInferenceGpu`.
#[must_use]
pub fn gbm_inference() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "gbm_inference",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("TreeInferenceGpu")
}

/// Merge pairs — batch overlap scoring via FMR.
#[must_use]
pub fn merge_pairs() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "merge_pairs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Signal processing / peak detection — batch via FMR.
#[must_use]
pub fn signal_processing() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "signal_processing",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Feature table extraction — chains EIC + signal GPU.
#[must_use]
pub fn feature_table() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "feature_table",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + WeightedDotF64")
}

/// Robinson-Foulds tree distance — `PairwiseHammingGpu`.
#[must_use]
pub fn robinson_foulds() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "robinson_foulds",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("PairwiseHammingGpu")
}

/// Dereplication — parallel hashing via `KmerHistogramGpu` pattern.
#[must_use]
pub fn dereplication() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dereplication",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("KmerHistogramGpu")
}

/// Chimera detection — GEMM-based sketch scoring.
#[must_use]
pub fn chimera() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "chimera",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("GemmCachedF64")
}

/// Neighbor joining — GPU distance matrix + CPU NJ loop.
#[must_use]
pub fn neighbor_joining() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "neighbor_joining",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// DTL reconciliation — batch workgroup-per-family.
#[must_use]
pub fn reconciliation() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "reconciliation",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchReconcileGpu")
}

/// Molecular clock — element-wise relaxed rates via FMR.
#[must_use]
pub fn molecular_clock() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "molecular_clock",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

// ── Composed GPU domains (Felsenstein → compose) ────────────────────

/// DADA2 denoising — GPU E-step via `Dada2EStepGpu`.
#[must_use]
pub fn dada2() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dada2",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("Dada2EStepGpu")
}

/// Phylogenetic bootstrap — column resampling + `FelsensteinGpu` per replicate.
#[must_use]
pub fn bootstrap() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "bootstrap",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

/// Metagenomic placement — edge-parallel `FelsensteinGpu` for reads.
#[must_use]
pub fn placement() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "placement",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

// ── NUCLEUS data-driven domains (Tower → Nest → Node) ───────────────

/// Assembly statistics (N50, GC, genome size) — CPU f64 compute.
///
/// Processes NCBI genome assemblies resolved via the Nest data chain.
/// GPU promotion via `FusedMapReduceF64` for large collections.
#[must_use]
pub fn assembly_statistics() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "assembly_statistics",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// GC content analysis across assembly collections.
///
/// Computes per-assembly GC fractions and collection-level diversity
/// (Shannon entropy of GC distribution).
#[must_use]
pub fn gc_analysis() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "gc_analysis",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Genome diversity metrics across assembly collections.
///
/// Shannon/Simpson entropy on genome size distributions and GC profiles.
#[must_use]
pub fn genome_diversity() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "genome_diversity",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("DiversityFusionGpu")
}

/// PFAS spectral matching against reference libraries.
///
/// Cosine similarity scoring between measured and reference spectra.
#[must_use]
pub fn pfas_spectral_match() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "pfas_spectral_match",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("WeightedDotF64")
}

/// Vibrio landscape analysis — cross-assembly comparative genomics.
///
/// K-mer profiling + diversity across Vibrio assembly collections.
#[must_use]
pub fn vibrio_landscape() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "vibrio_landscape",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + KmerHistogramGpu")
}

/// Campylobacterota comparative genomics — pan-genome statistics.
#[must_use]
pub fn campylobacterota_comparative() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "campylobacterota_comparative",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// NCBI assembly ingestion pipeline — I/O + compute coordination.
///
/// CPU-only I/O phase (FASTA parse, gzip decompress) followed by
/// GPU-eligible statistics computation. Dispatched as CPU because
/// the I/O phase dominates.
#[must_use]
pub fn ncbi_assembly_ingest() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("ncbi_assembly_ingest", vec![Capability::CpuCompute])
}

// ── CPU-only domains (I/O-bound, no GPU benefit) ────────────────────

/// FASTQ parsing (CPU-only, I/O-bound).
#[must_use]
pub fn fastq_parsing() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("fastq_parsing", vec![Capability::CpuCompute])
}

// ── Extension Papers (Exp144-156) — Anderson-QS Three-Tier ──────────

/// Cold seep QS catalog (Exp144): diversity + Anderson + ODE.
#[must_use]
pub fn cold_seep_catalog() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cold_seep_catalog",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4<QsBiofilm>")
}

/// Cold seep QS geometry (Exp145): diversity + Anderson localization.
#[must_use]
pub fn cold_seep_geometry() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cold_seep_geometry",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// `LuxR` phylogeny geometry (Exp146): diversity + phylogenetics + Anderson.
#[must_use]
pub fn luxr_phylogeny() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "luxr_phylogeny",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FusedMapReduceF64 + RobinsonFouldsF64")
}

/// Mechanical wave Anderson (Exp147): Anderson localization + wave model.
#[must_use]
pub fn mechanical_wave() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "mechanical_wave_anderson",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4 + FusedMapReduceF64")
}

/// QS wave localization (Exp148): Anderson + QS ODE.
#[must_use]
pub fn qs_wave_localization() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "qs_wave_localization",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<QsBiofilm> + FusedMapReduceF64")
        .with_ode(4, 17)
}

/// Burst statistics Anderson (Exp149): stochastic + Anderson.
#[must_use]
pub fn burst_statistics() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "burst_statistics_anderson",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Physical communication Anderson (Exp152): comm + Anderson disorder.
#[must_use]
pub fn physical_comm() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "physical_comm_anderson",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4")
}

/// Nitrifying QS (Exp153): QS biofilm + diversity + Anderson.
#[must_use]
pub fn nitrifying_qs() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "nitrifying_qs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4<QsBiofilm>")
        .with_ode(4, 17)
}

/// Marine interkingdom QS (Exp154): cross-domain QS + diversity.
#[must_use]
pub fn marine_interkingdom() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "marine_interkingdom_qs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// Myxococcus critical density (Exp155): cooperation ODE + Anderson.
#[must_use]
pub fn myxococcus_critical_density() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "myxococcus_critical_density",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CooperationOde>")
        .with_ode(4, 13)
}

/// Dictyostelium relay (Exp156): signal relay ODE + Anderson.
#[must_use]
pub fn dictyostelium_relay() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dictyostelium_relay",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<MultiSignalOde>")
        .with_ode(7, 24)
}

// ── Inventory ───────────────────────────────────────────────────────

/// All known bio domain workloads.
///
/// Returns the full catalog for dispatch planning and absorption tracking.
#[must_use]
pub fn all_workloads() -> Vec<BioWorkload> {
    vec![
        // Absorbed ToadStool domains
        diversity(),
        pcoa(),
        kmer_histogram(),
        unifrac_propagate(),
        qs_biofilm_ode(),
        smith_waterman(),
        felsenstein(),
        taxonomy(),
        // Pure GPU promotion: composed domains
        kmd(),
        gbm_inference(),
        merge_pairs(),
        signal_processing(),
        feature_table(),
        robinson_foulds(),
        dereplication(),
        chimera(),
        neighbor_joining(),
        reconciliation(),
        molecular_clock(),
        // Absorbed ODE domains (trait-generated WGSL)
        phage_defense_ode(),
        bistable_ode(),
        multi_signal_ode(),
        cooperation_ode(),
        capacitor_ode(),
        // Felsenstein-composed domains
        dada2(),
        bootstrap(),
        placement(),
        // Absorbed S63: diversity fusion
        diversity_fusion(),
        // Extension papers (Exp144-156): Anderson-QS three-tier
        cold_seep_catalog(),
        cold_seep_geometry(),
        luxr_phylogeny(),
        mechanical_wave(),
        qs_wave_localization(),
        burst_statistics(),
        physical_comm(),
        nitrifying_qs(),
        marine_interkingdom(),
        myxococcus_critical_density(),
        dictyostelium_relay(),
        // NUCLEUS data-driven domains (Tower → Nest → Node)
        assembly_statistics(),
        gc_analysis(),
        genome_diversity(),
        pfas_spectral_match(),
        vibrio_landscape(),
        campylobacterota_comparative(),
        // CPU-only (I/O-bound)
        ncbi_assembly_ingest(),
        fastq_parsing(),
    ]
}

/// Count workloads by shader origin.
#[must_use]
pub fn origin_summary() -> (usize, usize, usize) {
    let all = all_workloads();
    let absorbed = all.iter().filter(|w| w.is_absorbed()).count();
    let local = all.iter().filter(|w| w.is_local()).count();
    let cpu_only = all
        .iter()
        .filter(|w| matches!(w.origin, ShaderOrigin::CpuOnly))
        .count();
    (absorbed, local, cpu_only)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn all_workloads_has_entries() {
        let all = all_workloads();
        assert!(all.len() >= 47, "expected at least 47 workloads");
    }

    #[test]
    fn origin_counts_match() {
        let (absorbed, local, cpu_only) = origin_summary();
        assert_eq!(
            absorbed, 45,
            "45 absorbed domains (28 base + 11 extension + 6 NUCLEUS data-driven)"
        );
        assert_eq!(local, 0, "0 local WGSL extensions (all absorbed)");
        assert_eq!(cpu_only, 2, "2 CPU-only domains (fastq_parsing + ncbi_assembly_ingest)");
    }

    #[test]
    fn absorbed_ode_workloads_have_dims() {
        for w in [
            phage_defense_ode(),
            bistable_ode(),
            multi_signal_ode(),
            cooperation_ode(),
            capacitor_ode(),
        ] {
            assert!(w.is_absorbed());
            assert!(
                w.ode_dims.is_some(),
                "{} should have ODE dims",
                w.workload.name
            );
        }
    }

    #[test]
    fn absorbed_workloads_have_primitive() {
        for w in all_workloads() {
            if !w.is_absorbed() {
                continue;
            }
            assert!(
                w.primitive.is_some(),
                "{} should have primitive name",
                w.workload.name
            );
        }
    }

    #[test]
    fn qs_biofilm_is_absorbed_ode() {
        let w = qs_biofilm_ode();
        assert!(w.is_absorbed());
        let dims = w.ode_dims.expect("should have dims");
        assert_eq!(dims.n_vars, 4);
        assert_eq!(dims.n_params, 17);
    }

    #[test]
    fn taxonomy_prefers_npu() {
        let w = taxonomy();
        assert_eq!(w.workload.preferred_substrate, Some(SubstrateKind::Npu));
    }

    #[test]
    fn fastq_parsing_is_cpu_only() {
        let w = fastq_parsing();
        assert!(w.is_cpu_only());
        assert!(!w.is_local());
        assert!(!w.is_absorbed());
    }

    #[test]
    fn diversity_is_absorbed() {
        let w = diversity();
        assert!(w.is_absorbed());
        assert!(!w.is_local());
        assert!(!w.is_cpu_only());
    }

    #[test]
    fn all_workloads_no_duplicate_names() {
        let all = all_workloads();
        let mut names: Vec<&str> = all.iter().map(|w| w.workload.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate workload names found");
    }
}
