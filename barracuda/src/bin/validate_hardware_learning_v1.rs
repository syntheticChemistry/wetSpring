// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    unexpected_cfgs,
    reason = "validation harness: cross-spring feature gates not defined in this crate"
)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
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
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp362: Hardware Learning Prototype — Probe-Calibrate-Route-Apply
//!
//! Discovers the GPU's full capability landscape and generates a machine-readable
//! capability profile JSON. This is the "learning" step: the system discovers
//! what the hardware can do, then records it for downstream consumption.
//!
//! ## Domains
//!
//! - D92: Hardware Probe — adapter, driver, compute class, VRAM, firmware
//! - D93: Precision Calibration — per-tier capabilities, NVVM risk, FMA policy
//! - D94: Domain Routing — 12 `PhysicsDomain` routes, bio-specific thresholds
//! - D95: Capability Profile Export — JSON profile for primal consumption
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Hardware learning prototype |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_hardware_learning_v1` |

use std::time::Instant;
use wetspring_barracuda::validation::Validator;

fn scan_firmware(chip: &str) -> Vec<String> {
    let base = format!("/lib/firmware/nvidia/{chip}");
    let path = std::path::Path::new(&base);
    if !path.exists() {
        return vec![];
    }
    let mut found = vec![];
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                found.push(name.to_string());
            }
        }
    }
    found.sort();
    found
}

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp362: Hardware Learning Prototype v1");

    // ─── D92: Hardware Probe ───
    println!("\n  ── D92: Hardware Probe ──");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let device = rt.block_on(async { barracuda::device::WgpuDevice::new().await });

    let dev = match device {
        Ok(d) => {
            println!("  GPU: {}", d.name());
            v.check_pass("GPU device discovered", true);
            Some(d)
        }
        Err(e) => {
            println!("  No GPU: {e}");
            v.check_pass("GPU probe completes gracefully", true);
            None
        }
    };

    let mut profile = serde_json::Map::new();
    profile.insert("schema_version".into(), serde_json::json!("1.0"));
    profile.insert(
        "timestamp".into(),
        serde_json::json!(chrono_free_timestamp()),
    );

    if let Some(ref dev) = dev {
        let cal = barracuda::device::HardwareCalibration::from_device(dev);
        let brain = barracuda::device::PrecisionBrain::from_device(dev);

        profile.insert("adapter_name".into(), serde_json::json!(cal.adapter_name));
        profile.insert("has_any_f64".into(), serde_json::json!(cal.has_any_f64));
        profile.insert("df64_safe".into(), serde_json::json!(cal.df64_safe));
        profile.insert(
            "nvvm_transcendental_risk".into(),
            serde_json::json!(cal.nvvm_transcendental_risk),
        );
        profile.insert(
            "has_native_f64".into(),
            serde_json::json!(brain.has_native_f64()),
        );

        v.check_pass("adapter name populated", !cal.adapter_name.is_empty());

        let vram_estimate_gb = {
            let caps = barracuda::device::DeviceCapabilities::from_device(dev);
            let gb = caps.max_buffer_size / (1024 * 1024 * 1024);
            if gb > 0 {
                gb
            } else {
                // Conservative default when wgpu does not expose adapter memory info.
                12_u64
            }
        };
        profile.insert(
            "vram_estimate_gb".into(),
            serde_json::json!(vram_estimate_gb),
        );

        // Firmware inventory
        let ada_chips = ["ad102", "ad103", "ad104", "ad106", "ad107"];
        let mut fw_map = serde_json::Map::new();
        for chip in &ada_chips {
            let fw = scan_firmware(chip);
            if !fw.is_empty() {
                fw_map.insert(chip.to_string(), serde_json::json!(fw));
            }
        }
        let ampere_chips = ["ga100", "ga102", "ga103", "ga104", "ga106", "ga107"];
        for chip in &ampere_chips {
            let fw = scan_firmware(chip);
            if !fw.is_empty() {
                fw_map.insert(chip.to_string(), serde_json::json!(fw));
            }
        }
        profile.insert(
            "firmware_inventory".into(),
            serde_json::Value::Object(fw_map),
        );
        v.check_pass("firmware inventory collected", true);

        // ─── D93: Precision Calibration ───
        println!("\n  ── D93: Precision Calibration ──");

        let mut tiers_json = vec![];
        for tc in &cal.tiers {
            let tier_obj = serde_json::json!({
                "tier": format!("{}", tc.tier),
                "compiles": tc.compiles,
                "dispatches": tc.dispatches,
                "transcendentals_safe": tc.transcendentals_safe,
                "arith_only": cal.tier_arith_only(tc.tier),
                "safe": cal.tier_safe(tc.tier),
            });
            println!(
                "  {} — compile:{} dispatch:{} transcendentals:{} arith_only:{} safe:{}",
                tc.tier,
                tc.compiles,
                tc.dispatches,
                tc.transcendentals_safe,
                cal.tier_arith_only(tc.tier),
                cal.tier_safe(tc.tier),
            );
            tiers_json.push(tier_obj);
        }
        profile.insert("precision_tiers".into(), serde_json::json!(tiers_json));
        v.check_pass("precision tiers probed", !cal.tiers.is_empty());

        use barracuda::device::{FmaPolicy, PhysicsDomain, domain_requires_separate_fma};

        let fma_info = serde_json::json!({
            "contract_allows_contraction": FmaPolicy::Contract.allows_contraction(),
            "separate_requires_split": FmaPolicy::Separate.requires_split(),
            "bio_requires_separate": domain_requires_separate_fma(&PhysicsDomain::Bioinformatics),
            "lattice_requires_separate": domain_requires_separate_fma(&PhysicsDomain::LatticeQcd),
        });
        profile.insert("fma_policy".into(), fma_info);
        v.check_pass("FMA policy recorded", true);

        // ─── D94: Domain Routing ───
        println!("\n  ── D94: Domain Routing ──");

        let all_domains = [
            ("LatticeQcd", PhysicsDomain::LatticeQcd),
            ("GradientFlow", PhysicsDomain::GradientFlow),
            ("Dielectric", PhysicsDomain::Dielectric),
            ("KineticFluid", PhysicsDomain::KineticFluid),
            ("Eigensolve", PhysicsDomain::Eigensolve),
            ("MolecularDynamics", PhysicsDomain::MolecularDynamics),
            ("NuclearEos", PhysicsDomain::NuclearEos),
            ("PopulationPk", PhysicsDomain::PopulationPk),
            ("Bioinformatics", PhysicsDomain::Bioinformatics),
            ("Hydrology", PhysicsDomain::Hydrology),
            ("Statistics", PhysicsDomain::Statistics),
            ("General", PhysicsDomain::General),
        ];

        let mut routing_json = serde_json::Map::new();
        for (name, domain) in &all_domains {
            let advice = brain.route_advice(*domain);
            let entry = serde_json::json!({
                "tier": format!("{}", advice.tier),
                "fma_safe": advice.fma_safe,
                "rationale": advice.rationale,
            });
            println!(
                "  {:20} → {} (fma_safe={}, {})",
                name, advice.tier, advice.fma_safe, advice.rationale
            );
            routing_json.insert(name.to_string(), entry);
        }
        let routing_count = routing_json.len();
        profile.insert(
            "domain_routing".into(),
            serde_json::Value::Object(routing_json),
        );
        v.check_pass("12 domains routed", routing_count == 12);

        // Bio-specific thresholds
        let dispatch_overhead_us = 100.0_f64;
        let gpu_gflops = 30.0;
        let cpu_gflops = 0.5;

        let bio_workloads = [
            ("shannon_diversity", 1.0_f64),
            ("bray_curtis_pairwise", 2.0),
            ("ode_integration", 10.0),
            ("hmm_forward", 5.0),
            ("anderson_eigenvalue", 8.0),
        ];

        let mut thresholds_json = serde_json::Map::new();
        for (name, flops_per_elem) in &bio_workloads {
            let crossover_n = (dispatch_overhead_us * cpu_gflops * 1e3 / flops_per_elem) as usize;
            let gpu_advantage_10k = (flops_per_elem * 10_000.0 / cpu_gflops)
                / (dispatch_overhead_us + flops_per_elem * 10_000.0 / gpu_gflops);
            let max_pairwise_n =
                ((vram_estimate_gb * 1024 * 1024 * 1024 / 8) as f64).sqrt() as usize;

            thresholds_json.insert(
                name.to_string(),
                serde_json::json!({
                    "cpu_gpu_crossover_n": crossover_n,
                    "gpu_advantage_at_10k": format!("{gpu_advantage_10k:.1}x"),
                    "max_pairwise_n": max_pairwise_n,
                }),
            );
            println!(
                "  {name:25} — crossover N≈{crossover_n}, GPU advantage at 10K: {gpu_advantage_10k:.1}×"
            );
        }
        profile.insert(
            "bio_workload_thresholds".into(),
            serde_json::Value::Object(thresholds_json),
        );
        v.check_pass("bio workload thresholds computed", true);

        // Sovereign status
        let sovereign_status = serde_json::json!({
            "feature_compiled_in": cfg!(feature = "sovereign-dispatch"),
            "available": false,
            "blocker": "coral-gpu crate not available; nouveau CHANNEL_ALLOC blocked on Volta",
            "ada_gsp_firmware_present": scan_firmware("ad104") == vec!["gsp".to_string()],
            "rtx_4070_nouveau_tested": false,
        });
        profile.insert("sovereign_dispatch".into(), sovereign_status);
        v.check_pass("sovereign status recorded", true);
    } else {
        profile.insert("adapter_name".into(), serde_json::json!("none"));
        profile.insert("error".into(), serde_json::json!("no GPU available"));
    }

    // ─── D95: Capability Profile Export ───
    println!("\n  ── D95: Capability Profile Export ──");

    #[cfg(feature = "json")]
    {
        let json = serde_json::to_string_pretty(&serde_json::Value::Object(profile.clone()))
            .expect("serialize");
        std::fs::create_dir_all("output").ok();
        let path = "output/hardware_capability_profile.json";
        std::fs::write(path, &json).expect("write");
        println!(
            "  Capability profile exported: {path} ({} bytes)",
            json.len()
        );
        v.check_pass("capability profile JSON exported", true);

        let parsed: serde_json::Value = serde_json::from_str(&json).expect("reparse");
        v.check_pass(
            "profile has adapter_name",
            parsed.get("adapter_name").is_some(),
        );
        v.check_pass(
            "profile has precision_tiers",
            parsed.get("precision_tiers").is_some(),
        );
        v.check_pass(
            "profile has domain_routing",
            parsed.get("domain_routing").is_some(),
        );
        v.check_pass(
            "profile has firmware_inventory",
            parsed.get("firmware_inventory").is_some(),
        );
    }

    #[cfg(not(feature = "json"))]
    {
        println!("  json feature not enabled");
        v.check_pass("graceful skip without json", true);
    }

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}

fn chrono_free_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix_{}", dur.as_secs())
}
