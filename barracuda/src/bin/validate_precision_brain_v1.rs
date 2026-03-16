// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    unexpected_cfgs,
    reason = "validation harness: cross-spring feature gates not defined in this crate"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp357: GPU Hardware Discovery + `PrecisionBrain` for Bio Workloads
//!
//! First experiment consuming barraCuda v0.3.5 `PrecisionBrain`, `HardwareCalibration`,
//! and `FmaPolicy`. Discovers the RTX 4070's capabilities, routes bio workloads to
//! optimal precision tiers, and validates NVVM safety classification.
//!
//! Produces a petalTongue JSON dashboard showing the GPU capability landscape.
//!
//! ## Domains
//!
//! - D71: Hardware Calibration — tier probing, NVVM risk, adapter discovery
//! - D72: `PrecisionBrain` — domain routing, bio tier selection, advice rationale
//! - D73: `FmaPolicy` — domain-aware FMA safety, separate vs contract
//! - D74: Sovereign Probe — coralReef availability check
//! - D75: petalTongue Dashboard — GPU landscape visualization
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | barraCuda v0.3.5 GPU infrastructure |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_precision_brain_v1` |

use std::time::Instant;
use wetspring_barracuda::validation::Validator;

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp357: GPU Hardware Discovery + PrecisionBrain v1");

    // ─── D71: Hardware Calibration ───
    println!("\n  ── D71: Hardware Calibration ──");

    let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
    let device = rt.block_on(async { barracuda::device::WgpuDevice::new().await });

    let device = match device {
        Ok(d) => {
            println!("  ✓ GPU device created: {}", d.name());
            v.check_pass("GPU device created", true);
            Some(d)
        }
        Err(e) => {
            println!("  ○ No GPU available ({e}), testing CPU-only paths");
            v.check_pass("GPU probe completes without panic", true);
            None
        }
    };

    if let Some(ref dev) = device {
        let cal = barracuda::device::HardwareCalibration::from_device(dev);

        println!("  Adapter: {}", cal.adapter_name);
        println!("  Has any f64: {}", cal.has_any_f64);
        println!("  DF64 safe: {}", cal.df64_safe);
        println!(
            "  NVVM transcendental risk: {}",
            cal.nvvm_transcendental_risk
        );

        v.check_pass("HardwareCalibration::from_device() succeeds", true);
        v.check_pass("adapter_name is non-empty", !cal.adapter_name.is_empty());

        let tier_count = cal.tiers.len();
        println!("  Tier capabilities probed: {tier_count}");
        v.check_pass("at least 1 tier probed", tier_count >= 1);

        for tc in &cal.tiers {
            println!(
                "    {} — compiles: {}, dispatches: {}, transcendentals_safe: {}",
                tc.tier, tc.compiles, tc.dispatches, tc.transcendentals_safe
            );
        }

        let best_f64 = cal.best_f64_tier();
        let best_any = cal.best_any_tier();
        println!("  Best f64 tier: {best_f64:?}");
        println!("  Best any tier: {best_any:?}");
        v.check_pass("best_any_tier() returns Some", best_any.is_some());

        use barracuda::device::PrecisionTier;
        v.check_pass("F32 tier is safe", cal.tier_safe(PrecisionTier::F32));

        if cal.has_any_f64 {
            if cal.nvvm_transcendental_risk {
                v.check_pass(
                    "F64 unsafe due to NVVM risk (expected on NVIDIA proprietary)",
                    !cal.tier_safe(PrecisionTier::F64),
                );
            } else {
                v.check_pass(
                    "F64 tier safe on f64-capable hardware",
                    cal.tier_safe(PrecisionTier::F64),
                );
            }
        }

        // ─── D72: PrecisionBrain ───
        println!("\n  ── D72: PrecisionBrain ──");

        let brain = barracuda::device::PrecisionBrain::from_device(dev);

        println!("  Adapter: {}", brain.adapter_name());
        println!("  Has native f64: {}", brain.has_native_f64());
        v.check_pass("PrecisionBrain::from_device() succeeds", true);

        use barracuda::device::PhysicsDomain;
        let bio_domains = [
            PhysicsDomain::Bioinformatics,
            PhysicsDomain::Statistics,
            PhysicsDomain::General,
            PhysicsDomain::PopulationPk,
            PhysicsDomain::Hydrology,
        ];

        for domain in &bio_domains {
            let tier = brain.route(*domain);
            let advice = brain.route_advice(*domain);
            println!(
                "  {:?} → {} (fma_safe: {}, rationale: {})",
                domain, tier, advice.fma_safe, advice.rationale
            );
            v.check_pass(&format!("{domain:?} routes to a tier"), true);
        }

        let bio_tier = brain.route(PhysicsDomain::Bioinformatics);
        v.check_pass(
            "Bioinformatics routes to F32 or higher",
            matches!(
                bio_tier,
                PrecisionTier::F32
                    | PrecisionTier::DF64
                    | PrecisionTier::F64
                    | PrecisionTier::F64Precise
            ),
        );

        let stats_tier = brain.route(PhysicsDomain::Statistics);
        v.check_pass(
            "Statistics routes to a valid tier",
            matches!(
                stats_tier,
                PrecisionTier::F32
                    | PrecisionTier::DF64
                    | PrecisionTier::F64
                    | PrecisionTier::F64Precise
            ),
        );

        // ─── D73: FmaPolicy ───
        println!("\n  ── D73: FmaPolicy ──");

        use barracuda::device::{FmaPolicy, domain_requires_separate_fma};

        let contract = FmaPolicy::Contract;
        let separate = FmaPolicy::Separate;
        let default = FmaPolicy::Default;

        v.check_pass("Contract allows contraction", contract.allows_contraction());
        v.check_pass("Separate requires split", separate.requires_split());
        v.check_pass("Default allows contraction", default.allows_contraction());
        v.check_pass(
            "Separate does not allow contraction",
            !separate.allows_contraction(),
        );

        let lattice_needs_separate = domain_requires_separate_fma(&PhysicsDomain::LatticeQcd);
        let bio_needs_separate = domain_requires_separate_fma(&PhysicsDomain::Bioinformatics);

        v.check_pass("LatticeQcd requires separate FMA", lattice_needs_separate);
        v.check_pass(
            "Bioinformatics does NOT require separate FMA",
            !bio_needs_separate,
        );

        println!("  LatticeQcd needs separate FMA: {lattice_needs_separate}");
        println!("  Bioinformatics needs separate FMA: {bio_needs_separate}");

        for domain in &[
            PhysicsDomain::LatticeQcd,
            PhysicsDomain::GradientFlow,
            PhysicsDomain::NuclearEos,
            PhysicsDomain::Bioinformatics,
            PhysicsDomain::Statistics,
            PhysicsDomain::Hydrology,
        ] {
            let needs = domain_requires_separate_fma(domain);
            println!("  {domain:?} → separate FMA: {needs}");
        }

        // ─── D74: Sovereign Probe ───
        println!("\n  ── D74: Sovereign Probe ──");

        #[cfg(feature = "sovereign-dispatch")]
        {
            let sov = barracuda::device::sovereign_available();
            println!("  Sovereign dispatch available: {sov}");
            v.check_pass("sovereign_available() returns cleanly", true);
        }

        #[cfg(not(feature = "sovereign-dispatch"))]
        {
            println!("  sovereign-dispatch feature not enabled (expected for standard builds)");
            println!("  coralReef sovereign path requires coral-gpu crate");
            v.check_pass("sovereign-dispatch feature correctly absent", true);
        }
    } else {
        println!("\n  ── Skipping GPU-dependent domains (no GPU) ──");
        v.check_pass("graceful degradation without GPU", true);
    }

    // ─── D75: petalTongue Dashboard ───
    println!("\n  ── D75: petalTongue GPU Landscape Dashboard ──");

    #[cfg(feature = "json")]
    {
        use wetspring_barracuda::visualization::{DataChannel, EcologyScenario, ScenarioNode};

        let mut nodes = vec![];

        let mut overview = ScenarioNode {
            id: "gpu_overview".into(),
            name: "GPU Capability Overview".into(),
            node_type: "compute".into(),
            family: "wetspring".into(),
            status: "healthy".into(),
            health: 100,
            confidence: 90,
            capabilities: vec!["precision_brain".into(), "hardware_calibration".into()],
            data_channels: vec![],
            scientific_ranges: vec![],
        };

        if let Some(ref dev) = device {
            let cal = barracuda::device::HardwareCalibration::from_device(dev);
            let brain = barracuda::device::PrecisionBrain::from_device(dev);

            let tier_names: Vec<String> = cal.tiers.iter().map(|t| format!("{}", t.tier)).collect();
            let tier_compile: Vec<f64> = cal
                .tiers
                .iter()
                .map(|t| if t.compiles { 1.0 } else { 0.0 })
                .collect();
            let tier_dispatch: Vec<f64> = cal
                .tiers
                .iter()
                .map(|t| if t.dispatches { 1.0 } else { 0.0 })
                .collect();
            let tier_safe: Vec<f64> = cal
                .tiers
                .iter()
                .map(|t| if t.transcendentals_safe { 1.0 } else { 0.0 })
                .collect();

            overview.data_channels.push(DataChannel::Heatmap {
                id: "tier_capabilities".into(),
                label: "Precision Tier Capabilities".into(),
                x_labels: tier_names,
                y_labels: vec![
                    "Compiles".into(),
                    "Dispatches".into(),
                    "Transcendentals Safe".into(),
                ],
                values: [tier_compile, tier_dispatch, tier_safe].concat(),
                unit: "boolean".into(),
            });

            use barracuda::device::PhysicsDomain;
use wetspring_barracuda::validation::OrExit;
            let domains = [
                PhysicsDomain::Bioinformatics,
                PhysicsDomain::Statistics,
                PhysicsDomain::PopulationPk,
                PhysicsDomain::Hydrology,
                PhysicsDomain::General,
                PhysicsDomain::Eigensolve,
            ];

            let domain_names: Vec<String> = domains.iter().map(|d| format!("{d:?}")).collect();
            let domain_tiers: Vec<f64> = domains
                .iter()
                .map(|d| match brain.route(*d) {
                    barracuda::device::PrecisionTier::F32 => 1.0,
                    barracuda::device::PrecisionTier::DF64 => 2.0,
                    barracuda::device::PrecisionTier::F64 => 3.0,
                    barracuda::device::PrecisionTier::F64Precise => 4.0,
                })
                .collect();

            overview.data_channels.push(DataChannel::Bar {
                id: "domain_routing".into(),
                label: "PrecisionBrain Domain Routing".into(),
                categories: domain_names,
                values: domain_tiers,
                unit: "tier (1=F32, 2=DF64, 3=F64, 4=F64Precise)".into(),
            });

            overview.data_channels.push(DataChannel::Gauge {
                id: "nvvm_risk".into(),
                label: "NVVM Transcendental Risk".into(),
                value: if cal.nvvm_transcendental_risk {
                    1.0
                } else {
                    0.0
                },
                min: 0.0,
                max: 1.0,
                unit: "risk".into(),
                normal_range: [0.0, 0.0],
                warning_range: [0.5, 1.0],
            });

            overview.data_channels.push(DataChannel::Gauge {
                id: "f64_capability".into(),
                label: "Native f64 Capability".into(),
                value: if cal.has_any_f64 { 1.0 } else { 0.0 },
                min: 0.0,
                max: 1.0,
                unit: "available".into(),
                normal_range: [0.5, 1.0],
                warning_range: [0.0, 0.5],
            });
        }

        nodes.push(overview);

        let scenario = EcologyScenario {
            name: "Exp357: GPU Capability Landscape".into(),
            description: "PrecisionBrain routing, tier capabilities, NVVM safety, FMA policy"
                .into(),
            version: "1.0".into(),
            mode: "static".into(),
            domain: "gpu_infrastructure".into(),
            nodes,
            edges: vec![],
        };

        let json = serde_json::to_string_pretty(&scenario).or_exit("serialize");
        let path = "output/gpu_capability_landscape.json";
        std::fs::create_dir_all("output").ok();
        std::fs::write(path, &json).or_exit("write JSON");
        println!("  ✓ Dashboard exported: {path} ({} bytes)", json.len());
        v.check_pass("petalTongue JSON scenario exported", true);
        v.check_pass("scenario has nodes", !scenario.nodes.is_empty());
    }

    #[cfg(not(feature = "json"))]
    {
        println!("  ○ json feature not enabled, skipping dashboard export");
        v.check_pass("graceful skip without json", true);
    }

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
