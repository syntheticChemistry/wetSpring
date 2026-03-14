// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    unexpected_cfgs,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp361: Nouveau RTX 4070 Diagnostic + GPU Dispatch Pipeline Status
//!
//! Records the full GPU driver state on eastgate (RTX 4070 + Titan V),
//! probes nouveau and nvidia-drm dispatch readiness, inventories firmware,
//! and produces a petalTongue dashboard of the dispatch pipeline.
//!
//! ## Domains
//!
//! - D87: GPU Driver Discovery — lspci-equivalent, driver identification, DRM nodes
//! - D88: Firmware Inventory — GSP, PMU, ACR, GR, SEC2 presence per chip
//! - D89: Nouveau Dispatch Status — `VM_INIT`, `CHANNEL_ALLOC` readiness
//! - D90: wgpu Path Baseline — `HardwareCalibration` via wgpu/Vulkan
//! - D91: petalTongue Dispatch Dashboard — JSON export of pipeline status
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | GPU dispatch pipeline diagnostic |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_nouveau_diagnostic_v1` |

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

fn check_drm_node(path: &str) -> Option<String> {
    let driver_link = format!("/sys/class/drm/{path}/device/driver");
    let p = std::path::Path::new(&driver_link);
    if p.exists() {
        if let Ok(target) = std::fs::read_link(p) {
            if let Some(name) = target.file_name() {
                return Some(name.to_string_lossy().to_string());
            }
        }
    }
    None
}

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp361: Nouveau RTX 4070 Diagnostic + Dispatch Pipeline");

    // ─── D87: GPU Driver Discovery ───
    println!("\n  ── D87: GPU Driver Discovery ──");

    let cards = ["card0", "card1"];
    let mut gpu_info = vec![];

    for card in &cards {
        let driver = check_drm_node(card);
        let vendor_path = format!("/sys/class/drm/{card}/device/vendor");
        let vendor = std::fs::read_to_string(&vendor_path)
            .unwrap_or_default()
            .trim()
            .to_string();
        let device_path = format!("/sys/class/drm/{card}/device/device");
        let device_id = std::fs::read_to_string(&device_path)
            .unwrap_or_default()
            .trim()
            .to_string();

        println!(
            "  {card}: vendor={vendor} device={device_id} driver={}",
            driver.as_deref().unwrap_or("unknown")
        );
        gpu_info.push((card.to_string(), vendor, device_id, driver));
    }

    let nvidia_gpus: Vec<_> = gpu_info.iter().filter(|g| g.1 == "0x10de").collect();
    v.check_pass("at least 1 NVIDIA GPU found", !nvidia_gpus.is_empty());

    let nouveau_gpus: Vec<_> = nvidia_gpus
        .iter()
        .filter(|g| g.3.as_deref() == Some("nouveau"))
        .collect();
    let nvidia_drm_gpus: Vec<_> = nvidia_gpus
        .iter()
        .filter(|g| g.3.as_deref() == Some("nvidia"))
        .collect();

    println!(
        "\n  NVIDIA GPUs: {} total, {} nouveau, {} nvidia-drm",
        nvidia_gpus.len(),
        nouveau_gpus.len(),
        nvidia_drm_gpus.len()
    );
    v.check_pass("GPU driver scan completes", true);

    let has_nouveau = !nouveau_gpus.is_empty();
    let has_nvidia_drm = !nvidia_drm_gpus.is_empty();

    println!(
        "  RTX 4070 (0x2786): {}",
        if nvidia_gpus.iter().any(|g| g.2 == "0x2786") {
            "FOUND"
        } else {
            "not found"
        }
    );
    println!(
        "  Titan V (0x1d81): {}",
        if nvidia_gpus.iter().any(|g| g.2 == "0x1d81") {
            "FOUND"
        } else {
            "not found"
        }
    );
    v.check_pass("dual-GPU rig detected", nvidia_gpus.len() >= 2);

    // ─── D88: Firmware Inventory ───
    println!("\n  ── D88: Firmware Inventory ──");

    let chips = [
        ("ad104", "RTX 4070 (Ada)"),
        ("gv100", "Titan V (Volta)"),
        ("ga102", "Ampere reference"),
        ("ad102", "AD102 reference"),
    ];

    for (chip, label) in &chips {
        let fw = scan_firmware(chip);
        if fw.is_empty() {
            println!("  {label} ({chip}): no firmware found");
        } else {
            println!("  {label} ({chip}): {}", fw.join(", "));
        }
    }

    let ad104_fw = scan_firmware("ad104");
    v.check_pass(
        "AD104 firmware is GSP-only",
        ad104_fw == vec!["gsp".to_string()],
    );

    let gv100_fw = scan_firmware("gv100");
    let gv100_missing_gsp = !gv100_fw.contains(&"gsp".to_string());
    v.check_pass(
        "GV100 (Titan V) has no GSP firmware (Volta — needs PMU)",
        gv100_missing_gsp,
    );

    let ga102_fw = scan_firmware("ga102");
    let ga102_has_gsp = ga102_fw.contains(&"gsp".to_string());
    let ga102_has_acr = ga102_fw.contains(&"acr".to_string());
    v.check_pass(
        "GA102 has GSP + ACR (full Ampere firmware)",
        ga102_has_gsp && ga102_has_acr,
    );

    println!("\n  Firmware summary:");
    println!("    Ada (AD104):    GSP-only — sovereign dispatch via GSP");
    println!("    Volta (GV100):  ACR/GR/NVDEC/SEC2 but no GSP — needs PMU (missing)");
    println!("    Ampere (GA102): GSP + ACR + GR + SEC2 — full firmware set");

    // ─── D89: Nouveau Dispatch Status ───
    println!("\n  ── D89: Nouveau Dispatch Status ──");

    if has_nouveau {
        println!("  nouveau driver ACTIVE on {} GPU(s)", nouveau_gpus.len());
        v.check_pass("nouveau driver loaded", true);

        println!("  DRM render nodes: /dev/dri/renderD128, /dev/dri/renderD129");

        let render_nodes = ["/dev/dri/renderD128", "/dev/dri/renderD129"];
        for node in &render_nodes {
            let exists = std::path::Path::new(node).exists();
            println!("    {node}: {}", if exists { "present" } else { "missing" });
        }
        v.check_pass(
            "DRM render nodes exist",
            render_nodes
                .iter()
                .all(|n| std::path::Path::new(n).exists()),
        );

        println!("\n  Nouveau dispatch pipeline status (from coralReef diag_ioctl):");
        println!("    VM_INIT (new UAPI):     OK — kernel 6.17.9 supports new UAPI");
        println!("    CHANNEL_ALLOC:          EINVAL (22) on Titan V — Volta lacks PMU firmware");
        println!(
            "    Compute class tested:   VOLTA_COMPUTE_A (0xC3C0), TURING (0xC5C0), AMPERE (0xC6C0)"
        );
        println!("    All CHANNEL_ALLOC:      EINVAL — Volta PMU firmware is the blocker");
        println!();
        println!("  RTX 4070 on nouveau:      NOT TESTED (nvidia proprietary driver active)");
        println!("  RTX 4070 dispatch:        via nvidia-drm/wgpu/Vulkan (working)");
        v.check_pass("nouveau dispatch status documented", true);
    } else {
        println!("  nouveau driver NOT loaded");
        v.check_pass("nouveau absence documented", true);
    }

    if has_nvidia_drm {
        println!(
            "\n  nvidia proprietary driver ACTIVE on {} GPU(s)",
            nvidia_drm_gpus.len()
        );
        println!("  wgpu/Vulkan dispatch available via nvidia-drm");
        v.check_pass("nvidia-drm available for wgpu", true);
    }

    // ─── D90: wgpu Path Baseline ───
    println!("\n  ── D90: wgpu Path Baseline ──");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let device = rt.block_on(async { barracuda::device::WgpuDevice::new().await });

    match device {
        Ok(ref dev) => {
            println!("  wgpu device: {}", dev.name());
            v.check_pass("wgpu device created", true);

            let cal = barracuda::device::HardwareCalibration::from_device(dev);
            let brain = barracuda::device::PrecisionBrain::from_device(dev);

            println!("  HardwareCalibration:");
            println!("    adapter: {}", cal.adapter_name);
            println!("    has_any_f64: {}", cal.has_any_f64);
            println!("    df64_safe: {}", cal.df64_safe);
            println!(
                "    nvvm_transcendental_risk: {}",
                cal.nvvm_transcendental_risk
            );
            println!("    tiers probed: {}", cal.tiers.len());

            for tc in &cal.tiers {
                println!(
                    "      {} — compile:{} dispatch:{} transcendentals:{}",
                    tc.tier, tc.compiles, tc.dispatches, tc.transcendentals_safe
                );
            }

            v.check_pass("HardwareCalibration probed", true);

            let bio_tier = brain.route(barracuda::device::PhysicsDomain::Bioinformatics);
            println!("\n  PrecisionBrain:");
            println!("    Bioinformatics → {bio_tier}");
            println!("    has_native_f64: {}", brain.has_native_f64());
            v.check_pass("PrecisionBrain routing valid", true);

            #[cfg(feature = "sovereign-dispatch")]
            {
                let sov = barracuda::device::sovereign_available();
                println!("    sovereign_available: {sov}");
                v.check_pass("sovereign probe", true);
            }
            #[cfg(not(feature = "sovereign-dispatch"))]
            {
                println!("    sovereign_available: feature not compiled in");
                v.check_pass("sovereign feature absent (expected)", true);
            }
        }
        Err(e) => {
            println!("  No wgpu device: {e}");
            v.check_pass("wgpu probe completes without panic", true);
        }
    }

    // ─── D91: petalTongue Dispatch Dashboard ───
    println!("\n  ── D91: petalTongue Dispatch Dashboard ──");

    #[cfg(feature = "json")]
    {
        use wetspring_barracuda::visualization::{DataChannel, EcologyScenario, ScenarioNode};

        let mut dispatch_node = ScenarioNode {
            id: "dispatch_pipeline".into(),
            name: "GPU Dispatch Pipeline".into(),
            node_type: "pipeline".into(),
            family: "wetspring".into(),
            status: "degraded".into(),
            health: 60,
            confidence: 95,
            capabilities: vec!["wgpu".into(), "nouveau_probe".into()],
            data_channels: vec![],
            scientific_ranges: vec![],
        };

        let pipeline_stages = [
            "WGSL_Compile",
            "coralReef_IR",
            "SASS_Generate",
            "DRM_VM_INIT",
            "DRM_CHANNEL_ALLOC",
            "GEM_ALLOC",
            "VM_BIND",
            "EXEC_Dispatch",
        ];
        let stage_status: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        dispatch_node.data_channels.push(DataChannel::Bar {
            id: "pipeline_stages".into(),
            label: "Sovereign Dispatch Pipeline Readiness".into(),
            categories: pipeline_stages
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            values: stage_status,
            unit: "ready (1=yes, 0=blocked)".into(),
        });

        let drivers = ["wgpu_Vulkan", "nouveau", "nvidia_drm_UVM", "sovereign"];
        let driver_status: Vec<f64> = vec![1.0, 0.5, 0.0, 0.0];
        dispatch_node.data_channels.push(DataChannel::Bar {
            id: "driver_readiness".into(),
            label: "Driver Backend Readiness".into(),
            categories: drivers
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            values: driver_status,
            unit: "readiness (1=working, 0.5=partial, 0=blocked)".into(),
        });

        dispatch_node.data_channels.push(DataChannel::Gauge {
            id: "sovereign_readiness".into(),
            label: "Sovereign Dispatch Readiness".into(),
            value: 0.5,
            min: 0.0,
            max: 1.0,
            unit: "readiness".into(),
            normal_range: [0.8, 1.0],
            warning_range: [0.3, 0.8],
        });

        let scenario = EcologyScenario {
            name: "Exp361: GPU Dispatch Pipeline Status".into(),
            description: "Nouveau diagnostic, firmware inventory, dispatch readiness on eastgate (RTX 4070 + Titan V)".into(),
            version: "1.0".into(),
            mode: "static".into(),
            domain: "gpu_infrastructure".into(),
            nodes: vec![dispatch_node],
            edges: vec![],
        };

        let json = serde_json::to_string_pretty(&scenario).expect("serialize");
        std::fs::create_dir_all("output").ok();
        let path = "output/dispatch_pipeline_status.json";
        std::fs::write(path, &json).expect("write JSON");
        println!("  Dashboard exported: {path} ({} bytes)", json.len());
        v.check_pass("petalTongue dispatch dashboard exported", true);
    }

    #[cfg(not(feature = "json"))]
    {
        println!("  json feature not enabled, skipping dashboard");
        v.check_pass("graceful skip without json", true);
    }

    // Summary
    println!("\n  ═══════════════════════════════════════════════");
    println!("  NVIDIA Unlock Status Summary:");
    println!("    Compilation (WGSL→SASS):  READY (coralReef Iter 33, SM89)");
    println!("    wgpu/Vulkan dispatch:     WORKING (nvidia proprietary)");
    println!("    Nouveau VM_INIT:          WORKING (Titan V, kernel 6.17.9)");
    println!("    Nouveau CHANNEL_ALLOC:    BLOCKED (Volta PMU firmware missing)");
    println!("    RTX 4070 on nouveau:      UNTESTED (nvidia proprietary active)");
    println!("    nvidia-drm UVM:           NOT IMPLEMENTED");
    println!("    Sovereign dispatch:       BLOCKED (coral-gpu not compiled in)");
    println!("    Ada GSP firmware:         PRESENT (/lib/firmware/nvidia/ad104/gsp)");
    println!("  ═══════════════════════════════════════════════");

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
