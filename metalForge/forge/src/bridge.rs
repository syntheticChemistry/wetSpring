// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between forge substrates and barracuda device creation.
//!
//! Connects the capability-based substrate model (forge) to barracuda's
//! GPU device management ([`barracuda::device::WgpuDevice`]). When
//! `ToadStool` absorbs forge, this bridge becomes the integration point —
//! substrate discovery feeds directly into device creation.
//!
//! # Absorption path
//!
//! Forge [`Substrate`] → barracuda [`WgpuDevice`]. The capability model
//! (`F64Compute`, `ShaderDispatch`, etc.) maps directly to barracuda's
//! feature requirements. The bridge provides:
//!
//! 1. [`create_device`] — forge substrate → barracuda `WgpuDevice`
//! 2. [`best_f64_gpu`] — find the best f64-capable GPU from inventory
//! 3. [`substrate_from_device`] — wrap existing barracuda device as substrate

use crate::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use barracuda::device::WgpuDevice;
use barracuda::unified_hardware::BandwidthTier;

/// Create a barracuda [`WgpuDevice`] from a forge GPU substrate.
///
/// Uses the adapter index from the substrate identity to select the
/// correct wgpu adapter via barracuda's standard creation path.
///
/// Returns `None` if the substrate is not a GPU, has no adapter index,
/// or if device creation fails.
#[must_use]
pub fn create_device(substrate: &Substrate) -> Option<WgpuDevice> {
    if substrate.kind != SubstrateKind::Gpu {
        return None;
    }

    let adapter_index = substrate.identity.adapter_index?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    rt.block_on(async { WgpuDevice::from_adapter_index(adapter_index).await.ok() })
}

/// Find the best f64-capable GPU from a substrate inventory.
///
/// Prefers discrete GPUs with `F64Compute` capability. Among multiple
/// f64-capable GPUs, returns the first found.
#[must_use]
pub fn best_f64_gpu(substrates: &[Substrate]) -> Option<&Substrate> {
    substrates
        .iter()
        .find(|s| s.kind == SubstrateKind::Gpu && s.has(&Capability::F64Compute))
}

/// Wrap an existing barracuda [`WgpuDevice`] as a forge substrate.
///
/// Enables code that already has a barracuda device to participate
/// in forge's capability-based dispatch. The adapter info populates
/// identity and capabilities.
#[must_use]
pub fn substrate_from_device(device: &WgpuDevice) -> Substrate {
    let info = device.adapter_info();
    let has_f64 = device.has_f64_shaders();

    let mut capabilities = vec![Capability::F32Compute, Capability::ShaderDispatch];
    if has_f64 {
        capabilities.push(Capability::F64Compute);
        capabilities.push(Capability::ScalarReduce);
    }

    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity {
            name: info.name.clone(),
            driver: Some(format!("{} ({})", info.driver, info.driver_info)),
            backend: Some(format!("{:?}", info.backend)),
            adapter_index: None,
            device_node: None,
            pci_id: None,
        },
        properties: Properties {
            has_f64,
            has_timestamps: false,
            ..Properties::default()
        },
        capabilities,
        origin: crate::substrate::SubstrateOrigin::Local,
    }
}

/// Detect the `BandwidthTier` for a GPU substrate from its adapter name.
///
/// Uses barracuda's `BandwidthTier::detect_from_adapter_name` to map GPU
/// model strings to `PCIe` generation + lane width. Falls back to
/// `BandwidthTier::Unknown` for unrecognised adapters.
///
/// Returns `None` for non-GPU substrates.
#[must_use]
pub fn detect_bandwidth_tier(substrate: &Substrate) -> Option<BandwidthTier> {
    if substrate.kind != SubstrateKind::Gpu {
        return None;
    }
    Some(BandwidthTier::detect_from_adapter_name(
        &substrate.identity.name,
    ))
}

/// Estimate the data transfer cost in microseconds for moving `data_bytes`
/// to a GPU substrate.
///
/// Combines the `BandwidthTier`'s latency and throughput model. Returns
/// `None` for non-GPU substrates.
#[must_use]
pub fn estimated_transfer_us(substrate: &Substrate, data_bytes: usize) -> Option<f64> {
    let tier = detect_bandwidth_tier(substrate)?;
    let cost = tier.transfer_cost();
    Some(cost.estimated_us(data_bytes))
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::inventory;
    use crate::substrate::SubstrateOrigin;

    #[test]
    fn best_f64_gpu_returns_gpu_if_available() {
        let subs = inventory::discover();
        let gpu = best_f64_gpu(&subs);
        if let Some(g) = gpu {
            assert_eq!(g.kind, SubstrateKind::Gpu);
            assert!(g.has(&Capability::F64Compute));
        }
    }

    #[test]
    fn best_f64_gpu_returns_none_for_cpu_only() {
        let subs = vec![crate::probe::probe_cpu()];
        assert!(best_f64_gpu(&subs).is_none());
    }

    #[test]
    fn best_f64_finds_first_capable() {
        let gpu_f32 = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("f32-only GPU"),
            properties: Properties::default(),
            capabilities: vec![Capability::F32Compute, Capability::ShaderDispatch],
            origin: SubstrateOrigin::Local,
        };
        let gpu_f64 = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("f64 GPU"),
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::ShaderDispatch,
            ],
            origin: SubstrateOrigin::Local,
        };
        let cpu = crate::probe::probe_cpu();

        let subs = vec![gpu_f32, gpu_f64, cpu];
        let best = best_f64_gpu(&subs).expect("should find f64 GPU");
        assert_eq!(best.identity.name, "f64 GPU");
    }

    #[test]
    fn create_device_rejects_cpu() {
        let cpu = crate::probe::probe_cpu();
        assert!(create_device(&cpu).is_none());
    }

    #[test]
    fn create_device_rejects_no_adapter_index() {
        let gpu = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("no-index GPU"),
            properties: Properties::default(),
            capabilities: vec![Capability::ShaderDispatch],
            origin: SubstrateOrigin::Local,
        };
        assert!(create_device(&gpu).is_none());
    }

    #[test]
    fn detect_bandwidth_tier_for_rtx4070() {
        let gpu = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("NVIDIA GeForce RTX 4070"),
            properties: Properties::default(),
            capabilities: vec![Capability::F64Compute],
            origin: SubstrateOrigin::Local,
        };
        let tier = detect_bandwidth_tier(&gpu).expect("GPU should have tier");
        assert_eq!(tier, BandwidthTier::PciE4x16);
    }

    #[test]
    fn detect_bandwidth_tier_none_for_cpu() {
        let cpu = crate::probe::probe_cpu();
        assert!(detect_bandwidth_tier(&cpu).is_none());
    }

    #[test]
    fn estimated_transfer_us_positive() {
        let gpu = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("NVIDIA GeForce RTX 4070"),
            properties: Properties::default(),
            capabilities: vec![Capability::F64Compute],
            origin: SubstrateOrigin::Local,
        };
        let us = estimated_transfer_us(&gpu, 1_048_576).expect("should have cost");
        assert!(us > 0.0, "transfer cost should be positive: {us}");
    }

    #[test]
    fn estimated_transfer_us_none_for_cpu() {
        let cpu = crate::probe::probe_cpu();
        assert!(estimated_transfer_us(&cpu, 1024).is_none());
    }
}
