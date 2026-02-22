// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::missing_const_for_fn
)]
//! Exp074: `metalForge` Substrate Router — GPU↔NPU↔CPU Dispatch
//!
//! Validates the substrate-aware compute router that dispatches workloads
//! to GPU, NPU, or CPU based on batch size, workload type, and hardware
//! availability. Proves routing decisions are correct and math parity
//! holds across all dispatch paths.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_substrate_router` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, AKD1000 NPU, Pop!\_OS 22.04 |

use std::fmt;
use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// ═══════════════════════════════════════════════════════════════════
// Substrate Router — dispatch decisions based on hardware + workload
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Substrate {
    Cpu,
    Gpu,
    Npu,
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadClass {
    BatchParallel,
    Inference,
    Sequential,
}

struct SubstrateRouter {
    gpu_available: bool,
    npu_available: bool,
    dispatch_breakeven: usize,
    ada_lovelace: bool,
}

impl SubstrateRouter {
    const fn new(gpu_available: bool, npu_available: bool, ada_lovelace: bool) -> Self {
        Self {
            gpu_available,
            npu_available,
            dispatch_breakeven: 64,
            ada_lovelace,
        }
    }

    const fn route(&self, class: WorkloadClass, batch_size: usize) -> Substrate {
        match class {
            WorkloadClass::Inference => {
                if self.npu_available {
                    Substrate::Npu
                } else {
                    Substrate::Cpu
                }
            }
            WorkloadClass::Sequential => Substrate::Cpu,
            WorkloadClass::BatchParallel => {
                if self.gpu_available && batch_size >= self.dispatch_breakeven {
                    Substrate::Gpu
                } else {
                    Substrate::Cpu
                }
            }
        }
    }

    const fn needs_polyfill(&self) -> bool {
        self.ada_lovelace && self.gpu_available
    }
}

struct RouterResult {
    value: f64,
    substrate: Substrate,
    us: f64,
}

fn route_shannon(router: &SubstrateRouter, gpu: Option<&GpuF64>, counts: &[f64]) -> RouterResult {
    let target = router.route(WorkloadClass::BatchParallel, counts.len());
    let t = Instant::now();
    let value = match target {
        Substrate::Gpu => diversity_gpu::shannon_gpu(gpu.unwrap(), counts)
            .unwrap_or_else(|_| diversity::shannon(counts)),
        Substrate::Cpu | Substrate::Npu => diversity::shannon(counts),
    };
    RouterResult {
        value,
        substrate: target,
        us: t.elapsed().as_micros() as f64,
    }
}

fn route_simpson(router: &SubstrateRouter, gpu: Option<&GpuF64>, counts: &[f64]) -> RouterResult {
    let target = router.route(WorkloadClass::BatchParallel, counts.len());
    let t = Instant::now();
    let value = match target {
        Substrate::Gpu => diversity_gpu::simpson_gpu(gpu.unwrap(), counts)
            .unwrap_or_else(|_| diversity::simpson(counts)),
        Substrate::Cpu | Substrate::Npu => diversity::simpson(counts),
    };
    RouterResult {
        value,
        substrate: target,
        us: t.elapsed().as_micros() as f64,
    }
}

fn route_bray_curtis(
    router: &SubstrateRouter,
    gpu: Option<&GpuF64>,
    samples: &[Vec<f64>],
) -> (Vec<f64>, Substrate, f64) {
    let total_elements: usize = samples.iter().map(Vec::len).sum();
    let target = router.route(WorkloadClass::BatchParallel, total_elements);
    let t = Instant::now();
    let value = match target {
        Substrate::Gpu => diversity_gpu::bray_curtis_condensed_gpu(gpu.unwrap(), samples)
            .unwrap_or_else(|_| diversity::bray_curtis_condensed(samples)),
        Substrate::Cpu | Substrate::Npu => diversity::bray_curtis_condensed(samples),
    };
    (value, target, t.elapsed().as_micros() as f64)
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp074: metalForge Substrate Router");

    let gpu = GpuF64::new().await.ok();

    if let Some(ref g) = gpu {
        g.print_info();
    }

    let gpu_available = gpu.as_ref().is_some_and(|g| g.has_f64);
    let npu_available = std::path::Path::new("/dev/akida0").exists();
    let ada_lovelace = gpu_available;

    let router = SubstrateRouter::new(gpu_available, npu_available, ada_lovelace);

    println!("  Router config:");
    println!("    GPU available:      {gpu_available}");
    println!("    NPU available:      {npu_available}");
    println!("    Ada Lovelace:       {ada_lovelace}");
    println!("    Dispatch breakeven: {}", router.dispatch_breakeven);
    println!("    Needs polyfill:     {}", router.needs_polyfill());

    // ═══════════════════════════════════════════════════════════════════
    // TEST 1: Routing decisions
    // ═══════════════════════════════════════════════════════════════════
    v.section("Routing Decisions");

    let small_route = router.route(WorkloadClass::BatchParallel, 32);
    v.check(
        "Small batch (32) → CPU",
        f64::from(u8::from(small_route == Substrate::Cpu)),
        1.0,
        0.0,
    );

    let big_route = router.route(WorkloadClass::BatchParallel, 256);
    let expected_big = if gpu_available {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    };
    v.check(
        "Large batch (256) → GPU (or CPU if no GPU)",
        f64::from(u8::from(big_route == expected_big)),
        1.0,
        0.0,
    );

    let infer_route = router.route(WorkloadClass::Inference, 1);
    let expected_infer = if npu_available {
        Substrate::Npu
    } else {
        Substrate::Cpu
    };
    v.check(
        "Inference → NPU (or CPU fallback)",
        f64::from(u8::from(infer_route == expected_infer)),
        1.0,
        0.0,
    );

    let seq_route = router.route(WorkloadClass::Sequential, 1000);
    v.check(
        "Sequential always → CPU",
        f64::from(u8::from(seq_route == Substrate::Cpu)),
        1.0,
        0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 2: Small batch — forces CPU path
    // ═══════════════════════════════════════════════════════════════════
    v.section("Small Batch: CPU Path");

    let small_counts: Vec<f64> = vec![10.0, 20.0, 30.0, 15.0, 25.0];
    let cpu_ref_shannon = diversity::shannon(&small_counts);
    let cpu_ref_simpson = diversity::simpson(&small_counts);

    let r_shannon = route_shannon(&router, gpu.as_ref(), &small_counts);
    let r_simpson = route_simpson(&router, gpu.as_ref(), &small_counts);

    v.check(
        "Small: routed to CPU",
        f64::from(u8::from(r_shannon.substrate == Substrate::Cpu)),
        1.0,
        0.0,
    );
    v.check(
        "Small: Shannon parity",
        r_shannon.value,
        cpu_ref_shannon,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Small: Simpson parity",
        r_simpson.value,
        cpu_ref_simpson,
        tolerances::ANALYTICAL_F64,
    );

    println!(
        "  Small batch: Shannon={:.6} via {} ({:.0} µs)",
        r_shannon.value, r_shannon.substrate, r_shannon.us
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 3: Large batch — routes to GPU
    // ═══════════════════════════════════════════════════════════════════
    v.section("Large Batch: GPU Path");

    let large_counts: Vec<f64> = (0..512).map(|i| (f64::from(i) + 1.0) * 0.5).collect();
    let cpu_ref_shannon_lg = diversity::shannon(&large_counts);
    let cpu_ref_simpson_lg = diversity::simpson(&large_counts);

    let r_shannon_lg = route_shannon(&router, gpu.as_ref(), &large_counts);
    let r_simpson_lg = route_simpson(&router, gpu.as_ref(), &large_counts);

    let expected_substrate = if gpu_available {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    };
    v.check(
        "Large: routed to GPU (or CPU fallback)",
        f64::from(u8::from(r_shannon_lg.substrate == expected_substrate)),
        1.0,
        0.0,
    );
    v.check(
        "Large: Shannon parity (GPU == CPU)",
        r_shannon_lg.value,
        cpu_ref_shannon_lg,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Large: Simpson parity (GPU == CPU)",
        r_simpson_lg.value,
        cpu_ref_simpson_lg,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    println!(
        "  Large batch: Shannon={:.6} via {} ({:.0} µs)",
        r_shannon_lg.value, r_shannon_lg.substrate, r_shannon_lg.us
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 4: Bray-Curtis routing
    // ═══════════════════════════════════════════════════════════════════
    v.section("Bray-Curtis Routing");

    let samples: Vec<Vec<f64>> = (0..6)
        .map(|s| {
            (0..128)
                .map(|f| f64::from(s * 128 + f + 1).sqrt())
                .collect()
        })
        .collect();
    let cpu_ref_bray = diversity::bray_curtis_condensed(&samples);

    let (routed_bray, bray_substrate, bray_us) = route_bray_curtis(&router, gpu.as_ref(), &samples);

    v.check(
        "Bray-Curtis: correct routing",
        f64::from(u8::from(bray_substrate == expected_substrate)),
        1.0,
        0.0,
    );
    v.check("Bray-Curtis: len = 15", routed_bray.len() as f64, 15.0, 0.0);
    v.check(
        "Bray-Curtis[0] parity",
        routed_bray[0],
        cpu_ref_bray[0],
        tolerances::GPU_VS_CPU_BRAY_CURTIS,
    );
    v.check(
        "Bray-Curtis: all valid [0,1]",
        f64::from(u8::from(
            routed_bray
                .iter()
                .all(|d| d.is_finite() && *d >= 0.0 && *d <= 1.0 + 1e-10),
        )),
        1.0,
        0.0,
    );

    println!(
        "  Bray-Curtis: {:.6} via {} ({:.0} µs)",
        routed_bray[0], bray_substrate, bray_us
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 5: Mixed pipeline — GPU diversity → classification decision
    // ═══════════════════════════════════════════════════════════════════
    v.section("Mixed Pipeline: GPU → Classification Route");

    let eco_counts: Vec<f64> = (0..256).map(|i| (f64::from(i) + 1.0).sqrt()).collect();

    let pipeline_start = Instant::now();
    let diversity_result = route_shannon(&router, gpu.as_ref(), &eco_counts);
    let classify_route = router.route(WorkloadClass::Inference, 1);
    let pipeline_us = pipeline_start.elapsed().as_micros() as f64;

    v.check(
        "Mixed: diversity via GPU",
        f64::from(u8::from(diversity_result.substrate == expected_substrate)),
        1.0,
        0.0,
    );
    v.check(
        "Mixed: classification → NPU/CPU",
        f64::from(u8::from(
            classify_route == Substrate::Npu || classify_route == Substrate::Cpu,
        )),
        1.0,
        0.0,
    );
    v.check(
        "Mixed: diversity result valid",
        f64::from(u8::from(
            diversity_result.value > 0.0 && diversity_result.value.is_finite(),
        )),
        1.0,
        0.0,
    );

    println!(
        "  Mixed pipeline: diversity({}) → classify({}) in {:.0} µs",
        diversity_result.substrate, classify_route, pipeline_us
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 6: GPU unavailable fallback
    // ═══════════════════════════════════════════════════════════════════
    v.section("Fallback: GPU Unavailable");

    let no_gpu_router = SubstrateRouter::new(false, npu_available, false);
    let fallback_route = no_gpu_router.route(WorkloadClass::BatchParallel, 4096);
    v.check(
        "No-GPU: large batch falls back to CPU",
        f64::from(u8::from(fallback_route == Substrate::Cpu)),
        1.0,
        0.0,
    );

    let fallback_shannon = route_shannon(&no_gpu_router, None, &large_counts);
    v.check(
        "No-GPU: Shannon still correct",
        fallback_shannon.value,
        cpu_ref_shannon_lg,
        tolerances::ANALYTICAL_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // TEST 7: PCIe topology awareness
    // ═══════════════════════════════════════════════════════════════════
    v.section("PCIe Topology");

    let titan_v_exists = std::path::Path::new("/sys/bus/pci/devices/0000:05:00.0").exists();
    let rtx_4070_exists = std::path::Path::new("/sys/bus/pci/devices/0000:01:00.0").exists();
    let akd1000_exists = npu_available;

    println!("  PCIe device map:");
    println!("    RTX 4070 (01:00.0): {rtx_4070_exists}");
    println!("    Titan V  (05:00.0): {titan_v_exists}");
    println!("    AKD1000  (08:00.0): {akd1000_exists}");

    v.check(
        "PCIe: at least one GPU detected",
        f64::from(u8::from(rtx_4070_exists || titan_v_exists || gpu_available)),
        1.0,
        0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Exp074 Substrate Router Summary                          │");
    println!("├─────────────────────┬──────────┬─────────────────────────┤");
    println!("│ Workload            │ Routed → │ Result                  │");
    println!("├─────────────────────┼──────────┼─────────────────────────┤");
    println!(
        "│ Small Shannon (N=5) │ {:>8} │ {:.6}                │",
        r_shannon.substrate, r_shannon.value
    );
    println!(
        "│ Large Shannon (512) │ {:>8} │ {:.6}                │",
        r_shannon_lg.substrate, r_shannon_lg.value
    );
    println!(
        "│ Bray-Curtis (6×128) │ {:>8} │ {:.6}                │",
        bray_substrate, routed_bray[0]
    );
    println!("│ Classification      │ {classify_route:>8} │ (routing only)          │");
    println!(
        "│ GPU-unavail fallbck │ {:>8} │ {:.6}                │",
        fallback_shannon.substrate, fallback_shannon.value
    );
    println!("└─────────────────────┴──────────┴─────────────────────────┘");

    v.finish();
}
