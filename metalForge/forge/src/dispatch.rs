// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dispatch routing — route life science workloads to capable substrates.
//!
//! The dispatcher examines the inventory and routes each workload to the
//! substrate that best matches its requirements. This is capability-based:
//! we ask "who can do f64 + reduce?" not "send to GPU #0".
//!
//! # Life Science Dispatch Patterns
//!
//! - **Felsenstein pruning**: f64 + shader → GPU (site-parallel)
//! - **Diversity metrics**: f64 + reduce → GPU (fused map-reduce)
//! - **HMM forward**: f64 + shader → GPU (observation-parallel)
//! - **Taxonomy classify**: quant(8) → NPU (low-power FC inference)
//! - **FASTQ parsing**: CPU (I/O-bound, sequential)
//! - **Chimera detection**: CPU (branching, hash-heavy)

use crate::substrate::{Capability, Substrate, SubstrateKind};
use barracuda::unified_hardware::BandwidthTier;

/// A workload that needs to be dispatched to a substrate.
#[derive(Debug)]
pub struct Workload {
    /// Human-readable workload name.
    pub name: String,
    /// Capabilities required for this workload.
    pub required: Vec<Capability>,
    /// Preferred substrate kind (if any).
    pub preferred_substrate: Option<SubstrateKind>,
    /// Data transfer size in bytes (for bandwidth-aware routing).
    pub data_bytes: Option<usize>,
}

/// Dispatch decision — which substrate was chosen and why.
#[derive(Debug)]
pub struct Decision<'a> {
    /// The chosen substrate.
    pub substrate: &'a Substrate,
    /// Why this substrate was chosen.
    pub reason: Reason,
}

/// Why a particular substrate was chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum Reason {
    /// The workload's preferred substrate had all capabilities.
    Preferred,
    /// Best capable substrate by priority (GPU > NPU > CPU).
    BestAvailable,
    /// GPU was capable but transfer cost exceeded compute benefit.
    BandwidthFallback,
}

impl Workload {
    /// Create a workload with name and required capabilities.
    #[must_use]
    pub fn new(name: impl Into<String>, required: Vec<Capability>) -> Self {
        Self {
            name: name.into(),
            required,
            preferred_substrate: None,
            data_bytes: None,
        }
    }

    /// Set the preferred substrate kind.
    #[must_use]
    pub const fn prefer(mut self, kind: SubstrateKind) -> Self {
        self.preferred_substrate = Some(kind);
        self
    }

    /// Set the data transfer size for bandwidth-aware routing.
    #[must_use]
    pub const fn with_data_bytes(mut self, bytes: usize) -> Self {
        self.data_bytes = Some(bytes);
        self
    }
}

/// Route a workload to the best matching substrate.
///
/// Selection priority:
/// 1. Preferred substrate (if specified and capable)
/// 2. GPU (for compute-heavy work)
/// 3. NPU (for inference)
/// 4. CPU (fallback, always available)
#[must_use]
pub fn route<'a>(workload: &Workload, substrates: &'a [Substrate]) -> Option<Decision<'a>> {
    let capable: Vec<&Substrate> = substrates
        .iter()
        .filter(|s| workload.required.iter().all(|req| s.has(req)))
        .collect();

    if capable.is_empty() {
        return None;
    }

    if let Some(pref) = workload.preferred_substrate {
        if let Some(s) = capable.iter().find(|s| s.kind == pref) {
            return Some(Decision {
                substrate: s,
                reason: Reason::Preferred,
            });
        }
    }

    let best = capable
        .iter()
        .find(|s| s.kind == SubstrateKind::Gpu)
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Npu))
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Cpu))?;

    Some(Decision {
        substrate: best,
        reason: Reason::BestAvailable,
    })
}

/// Route a workload with bandwidth-aware GPU/CPU decisions.
///
/// Like [`route`], but when a GPU substrate is chosen and the workload
/// specifies `data_bytes`, the estimated `PCIe` transfer cost is compared
/// against the GPU dispatch overhead. If the transfer cost dominates
/// (small workloads over slow `PCIe` links), falls back to CPU with
/// `Reason::BandwidthFallback`.
///
/// The threshold heuristic: if estimated transfer microseconds exceed
/// the barracuda GPU dispatch overhead constant, prefer CPU for
/// non-preferred workloads.
#[must_use]
pub fn route_bandwidth_aware<'a>(
    workload: &Workload,
    substrates: &'a [Substrate],
) -> Option<Decision<'a>> {
    let decision = route(workload, substrates)?;

    if decision.substrate.kind != SubstrateKind::Gpu {
        return Some(decision);
    }
    if workload.preferred_substrate == Some(SubstrateKind::Gpu) {
        return Some(decision);
    }

    let data_bytes = match workload.data_bytes {
        Some(b) if b > 0 => b,
        _ => return Some(decision),
    };

    let tier = BandwidthTier::detect_from_adapter_name(&decision.substrate.identity.name);
    let cost = tier.transfer_cost();
    let transfer_us = cost.estimated_us(data_bytes);

    if transfer_us > barracuda::unified_hardware::GPU_DISPATCH_OVERHEAD_US {
        let cpu_fallback = substrates
            .iter()
            .filter(|s| s.kind == SubstrateKind::Cpu)
            .find(|s| workload.required.iter().all(|req| s.has(req)));

        if let Some(cpu) = cpu_fallback {
            return Some(Decision {
                substrate: cpu,
                reason: Reason::BandwidthFallback,
            });
        }
    }

    Some(decision)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::substrate::{Identity, Properties, SubstrateOrigin};

    fn make_gpu(name: &str, caps: Vec<Capability>) -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named(name),
            properties: Properties::default(),
            capabilities: caps,
            origin: SubstrateOrigin::Local,
        }
    }

    fn make_cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("CPU"),
            properties: Properties::default(),
            capabilities: vec![Capability::F64Compute, Capability::F32Compute],
            origin: SubstrateOrigin::Local,
        }
    }

    fn make_npu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity::named("AKD1000"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::F32Compute,
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
            ],
            origin: SubstrateOrigin::Local,
        }
    }

    #[test]
    fn routes_felsenstein_to_gpu() {
        let gpu = make_gpu(
            "RTX 4070",
            vec![
                Capability::F64Compute,
                Capability::ScalarReduce,
                Capability::ShaderDispatch,
            ],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = Workload::new(
            "Felsenstein pruning",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
        assert_eq!(d.reason, Reason::BestAvailable);
    }

    #[test]
    fn routes_taxonomy_to_npu() {
        let npu = make_npu();
        let cpu = make_cpu();
        let subs = [cpu, npu];
        let work = Workload::new(
            "Taxonomy classify",
            vec![Capability::QuantizedInference { bits: 8 }],
        );

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Npu);
    }

    #[test]
    fn falls_back_to_cpu() {
        let subs = [make_cpu()];
        let work = Workload::new("FASTQ parsing", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route to CPU");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
    }

    #[test]
    fn no_route_if_incapable() {
        let subs = [make_cpu()];
        let work = Workload::new(
            "NPU inference",
            vec![Capability::QuantizedInference { bits: 4 }],
        );

        assert!(route(&work, &subs).is_none());
    }

    #[test]
    fn respects_cpu_preference() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute]);
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work =
            Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
        assert_eq!(d.reason, Reason::Preferred);
    }

    #[test]
    fn preference_ignored_if_incapable() {
        let gpu = make_gpu(
            "GPU",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = Workload::new(
            "diversity map-reduce",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .prefer(SubstrateKind::Npu);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
        assert_eq!(d.reason, Reason::BestAvailable);
    }

    #[test]
    fn empty_substrate_list() {
        let work = Workload::new("anything", vec![Capability::F64Compute]);
        assert!(route(&work, &[]).is_none());
    }

    #[test]
    fn gpu_over_npu_for_f64() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute, Capability::F32Compute]);
        let npu = make_npu();
        let cpu = make_cpu();
        let subs = [cpu, npu, gpu];
        let work = Workload::new("diversity", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn workload_new_sets_name_and_caps() {
        let w = Workload::new("test_workload", vec![Capability::F64Compute]);
        assert_eq!(w.name, "test_workload");
        assert_eq!(w.required.len(), 1);
        assert!(w.preferred_substrate.is_none());
    }

    #[test]
    fn workload_prefer_chain() {
        let w = Workload::new("inference", vec![Capability::F32Compute]).prefer(SubstrateKind::Npu);
        assert_eq!(w.preferred_substrate, Some(SubstrateKind::Npu));
    }

    #[test]
    fn workload_with_data_bytes() {
        let w = Workload::new("matmul", vec![Capability::F64Compute]).with_data_bytes(1_048_576);
        assert_eq!(w.data_bytes, Some(1_048_576));
    }

    #[test]
    fn bandwidth_aware_respects_preference() {
        let gpu = make_gpu("NVIDIA GeForce RTX 4070", vec![Capability::F64Compute]);
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let w = Workload::new("forced_gpu", vec![Capability::F64Compute])
            .prefer(SubstrateKind::Gpu)
            .with_data_bytes(1_000_000_000);

        let d = route_bandwidth_aware(&w, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn bandwidth_aware_no_data_bytes_uses_standard() {
        let gpu = make_gpu(
            "NVIDIA GeForce RTX 4070",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let w = Workload::new(
            "diversity",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );

        let d = route_bandwidth_aware(&w, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
        assert_eq!(d.reason, Reason::BestAvailable);
    }

    #[test]
    fn bandwidth_fallback_reason() {
        assert_ne!(Reason::BandwidthFallback, Reason::BestAvailable);
        assert_ne!(Reason::BandwidthFallback, Reason::Preferred);
    }
}
