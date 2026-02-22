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

// ── Local WGSL domains (Write phase) ───────────────────────────────

/// Phage defense ODE (4 vars, 11 params — local WGSL).
#[must_use]
pub fn phage_defense_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Local)
        .named(
            "phage_defense_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_ode(4, 11)
}

/// Bistable QS ODE (5 vars, 21 params — local WGSL).
#[must_use]
pub fn bistable_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Local)
        .named(
            "bistable_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_ode(5, 21)
}

/// Multi-signal QS ODE (7 vars, 24 params — local WGSL).
#[must_use]
pub fn multi_signal_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Local)
        .named(
            "multi_signal_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_ode(7, 24)
}

// ── CPU-only domains ────────────────────────────────────────────────

/// Chimera detection (CPU-only, sequential branching).
#[must_use]
pub fn chimera() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly).named(
        "chimera",
        vec![Capability::CpuCompute],
    )
}

/// FASTQ parsing (CPU-only, I/O-bound).
#[must_use]
pub fn fastq_parsing() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly).named(
        "fastq_parsing",
        vec![Capability::CpuCompute],
    )
}

// ── Inventory ───────────────────────────────────────────────────────

/// All known bio domain workloads.
///
/// Returns the full catalog for dispatch planning and absorption tracking.
#[must_use]
pub fn all_workloads() -> Vec<BioWorkload> {
    vec![
        diversity(),
        pcoa(),
        kmer_histogram(),
        unifrac_propagate(),
        qs_biofilm_ode(),
        smith_waterman(),
        felsenstein(),
        taxonomy(),
        phage_defense_ode(),
        bistable_ode(),
        multi_signal_ode(),
        chimera(),
        fastq_parsing(),
    ]
}

/// Count workloads by shader origin.
#[must_use]
pub fn origin_summary() -> (usize, usize, usize) {
    let all = all_workloads();
    let absorbed = all.iter().filter(|w| w.is_absorbed()).count();
    let local = all.iter().filter(|w| w.is_local()).count();
    let cpu_only = all.iter().filter(|w| matches!(w.origin, ShaderOrigin::CpuOnly)).count();
    (absorbed, local, cpu_only)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn all_workloads_has_entries() {
        let all = all_workloads();
        assert!(all.len() >= 13, "expected at least 13 workloads");
    }

    #[test]
    fn origin_counts_match() {
        let (absorbed, local, cpu_only) = origin_summary();
        assert_eq!(absorbed, 8, "8 absorbed domains");
        assert_eq!(local, 3, "3 local WGSL domains");
        assert_eq!(cpu_only, 2, "2 CPU-only domains");
    }

    #[test]
    fn local_ode_workloads_have_dims() {
        for w in [phage_defense_ode(), bistable_ode(), multi_signal_ode()] {
            assert!(w.is_local());
            assert!(w.ode_dims.is_some(), "{} should have ODE dims", w.workload.name);
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
}
