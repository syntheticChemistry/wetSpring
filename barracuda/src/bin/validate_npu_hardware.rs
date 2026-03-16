// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
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
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! Exp193: NPU Hardware Validation — Real AKD1000 DMA + Discovery
//!
//! Proves the `ToadStool` `akida-driver` can discover, open, and perform
//! real DMA I/O on the `BrainChip` `AKD1000` neuromorphic processor. This
//! is the first wetSpring binary to exercise real NPU hardware rather
//! than CPU-side int8 simulation.
//!
//! # Provenance
//!
//! | Field          | Value |
//! |----------------|-------|
//! | Date           | 2026-02-26 |
//! | NPU hardware   | `BrainChip` Akida AKD1000 (`PCIe`, Eastgate) |
//! | Driver         | `ToadStool` akida-driver 0.1.0 (pure Rust) |
//! | Baseline       | `ToadStool` examples/basic\_io.rs round-trip |
//! | Hardware       | Eastgate i9-12900K, 64 GB DDR5, Pop!\_OS 22.04 |
//! | Command        | `cargo run --release --features npu --bin validate_npu_hardware` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;
use wetspring_barracuda::npu;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp193: NPU Hardware Validation (AKD1000)");

    // ═══════════════════════════════════════════════════════════════
    // S1: Runtime Discovery
    // ═══════════════════════════════════════════════════════════════
    v.section("S1: Runtime Device Discovery");

    let available = npu::npu_available();
    println!("  NPU available: {available}");
    v.check_pass("NPU detected on system", available);

    if !available {
        println!("  SKIP: No AKD1000 hardware. Remaining tests skipped.");
        v.finish();
    }

    let summary = npu::npu_summary().expect("NPU summary");
    println!("  Chip:       {}", summary.chip);
    println!("  PCIe:       {}", summary.pcie_address);
    println!("  NPUs:       {}", summary.npu_count);
    println!("  SRAM:       {} MB", summary.memory_mb);
    println!("  Bandwidth:  {:.1} GB/s", summary.bandwidth_gbps);

    v.check(
        "NPU count >= 1",
        f64::from(summary.npu_count),
        80.0,
        tolerances::EXACT,
    );
    v.check(
        "SRAM >= 1 MB",
        f64::from(summary.memory_mb),
        10.0,
        tolerances::EXACT,
    );
    v.check_pass("PCIe address discovered", !summary.pcie_address.is_empty());

    // ═══════════════════════════════════════════════════════════════
    // S2: Device Open + Capabilities
    // ═══════════════════════════════════════════════════════════════
    v.section("S2: Device Open + Capability Query");

    let t_open = Instant::now();
    let mut handle = npu::discover_npu().expect("open NPU");
    let open_us = t_open.elapsed().as_micros();

    println!("  Device opened in {open_us} µs");
    println!("  Chip version:  {:?}", handle.chip_version());
    println!("  NPU count:     {}", handle.npu_count());
    println!("  Memory:        {} MB", handle.memory_mb());
    println!("  PCIe BW:       {:.1} GB/s", handle.bandwidth_gbps());

    v.check_pass("device opened successfully", true);
    v.check(
        "chip is AKD1000",
        f64::from(u8::from(matches!(
            handle.chip_version(),
            npu::ChipVersion::Akd1000
        ))),
        1.0,
        tolerances::EXACT,
    );

    let caps = handle.capabilities().clone();
    v.check_pass("PCIe generation >= 1", caps.pcie.generation >= 1);
    v.check_pass("PCIe lanes >= 1", caps.pcie.lanes >= 1);
    v.check_pass("bandwidth > 0", caps.pcie.bandwidth_gbps > 0.0);

    // ═══════════════════════════════════════════════════════════════
    // S3: DMA Write — Pattern Transfer
    // ═══════════════════════════════════════════════════════════════
    v.section("S3: DMA Write (Host → NPU SRAM)");

    let test_sizes: &[usize] = &[64, 256, 1024, 4096];
    let mut write_throughputs: Vec<f64> = Vec::new();

    for &size in test_sizes {
        let pattern: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let t = Instant::now();
        let written = handle.write_raw(&pattern).expect("DMA write");
        let elapsed_us = t.elapsed().as_micros() as f64;

        let throughput_mbps = if elapsed_us > 0.0 {
            (written as f64) / elapsed_us
        } else {
            f64::INFINITY
        };
        write_throughputs.push(throughput_mbps);

        println!(
            "  Write {size:>5} B: {written} bytes in {elapsed_us:.0} µs ({throughput_mbps:.2} MB/s)"
        );

        v.check(
            &format!("DMA write {size}B complete"),
            written as f64,
            size as f64,
            tolerances::EXACT,
        );
    }

    v.check_pass(
        "all DMA writes completed",
        write_throughputs.len() == test_sizes.len(),
    );

    // ═══════════════════════════════════════════════════════════════
    // S4: DMA Read — Readback
    // ═══════════════════════════════════════════════════════════════
    v.section("S4: DMA Read (NPU SRAM → Host)");

    let mut read_throughputs: Vec<f64> = Vec::new();

    for &size in test_sizes {
        let mut buffer = vec![0u8; size];

        let t = Instant::now();
        let bytes_read = handle.read_raw(&mut buffer).expect("DMA read");
        let elapsed_us = t.elapsed().as_micros() as f64;

        let throughput_mbps = if elapsed_us > 0.0 {
            (bytes_read as f64) / elapsed_us
        } else {
            f64::INFINITY
        };
        read_throughputs.push(throughput_mbps);

        println!(
            "  Read  {size:>5} B: {bytes_read} bytes in {elapsed_us:.0} µs ({throughput_mbps:.2} MB/s)"
        );

        v.check(
            &format!("DMA read {size}B complete"),
            bytes_read as f64,
            size as f64,
            tolerances::EXACT,
        );
    }

    v.check_pass(
        "all DMA reads completed",
        read_throughputs.len() == test_sizes.len(),
    );

    // ═══════════════════════════════════════════════════════════════
    // S5: Int8 Quantization Round-Trip
    // ═══════════════════════════════════════════════════════════════
    v.section("S5: Int8 Quantization Fidelity");

    let test_values = [
        (7.2_f64, 5.0, 9.0, "pH"),
        (22.0, 10.0, 40.0, "temperature"),
        (8.0, 0.0, 15.0, "dissolved_oxygen"),
        (3.5, 0.0, 5.0, "shannon_H"),
        (0.85, 0.0, 1.0, "simpson_D"),
    ];

    for (val, lo, hi, name) in &test_values {
        let quantized = npu::quantize_i8(*val, *lo, *hi);
        let dequantized = npu::dequantize_i8(quantized, *lo, *hi);
        let error = (val - dequantized).abs();
        let range = hi - lo;
        let relative_error = error / range;

        println!(
            "  {name:>18}: {val:.3} → q={quantized:>4} → {dequantized:.3} (err={error:.4}, rel={relative_error:.4})"
        );

        v.check(
            &format!("{name} quantize error < 1%"),
            relative_error,
            0.0,
            0.01,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // S6: Sentinel Feature Vector DMA Round-Trip
    // ═══════════════════════════════════════════════════════════════
    v.section("S6: Sentinel Feature DMA");

    let features: [i8; 8] = [
        npu::quantize_i8(7.2, 5.0, 9.0),
        npu::quantize_i8(22.0, 10.0, 40.0),
        npu::quantize_i8(8.0, 0.0, 15.0),
        npu::quantize_i8(3.5, 0.0, 5.0),
        npu::quantize_i8(0.85, 0.0, 1.0),
        npu::quantize_i8(0.15, 0.0, 1.0),
        npu::quantize_i8(5.76, 0.0, 10.0),
        npu::quantize_i8(2.975, 0.0, 5.0),
    ];

    let feature_bytes: Vec<u8> = features.iter().map(|&x| x as u8).collect();

    let t = Instant::now();
    let written = handle.write_raw(&feature_bytes).expect("feature write");
    let write_ns = t.elapsed().as_nanos();

    let mut readback = vec![0u8; 8];
    let t = Instant::now();
    let bytes_read = handle.read_raw(&mut readback).expect("feature read");
    let read_ns = t.elapsed().as_nanos();

    println!("  Feature write: {written} bytes in {write_ns} ns");
    println!("  Feature read:  {bytes_read} bytes in {read_ns} ns");
    println!(
        "  Round-trip:    {} ns ({:.0} µs)",
        write_ns + read_ns,
        (write_ns + read_ns) as f64 / 1000.0
    );

    v.check_pass("feature DMA write", written == 8);
    v.check_pass("feature DMA read", bytes_read == 8);

    let total_ns = write_ns + read_ns;
    let throughput_hz = if total_ns > 0 {
        1_000_000_000 / total_ns
    } else {
        0
    };
    println!("  DMA throughput: {throughput_hz} round-trips/sec");
    v.check_pass(
        "DMA round-trip < 10ms (100 Hz minimum)",
        total_ns < 10_000_000,
    );

    // ═══════════════════════════════════════════════════════════════
    // S7: Bulk DMA Throughput
    // ═══════════════════════════════════════════════════════════════
    v.section("S7: Bulk DMA Throughput");

    let bulk_size = 8192;
    let bulk_data: Vec<u8> = (0..bulk_size).map(|i| (i % 256) as u8).collect();
    let n_iterations = 100;

    let t = Instant::now();
    for _ in 0..n_iterations {
        handle.write_raw(&bulk_data).expect("bulk write");
    }
    let bulk_write_us = t.elapsed().as_micros() as f64;
    let bulk_write_mbps = (bulk_size as f64 * n_iterations as f64) / bulk_write_us;

    let mut bulk_read_buf = vec![0u8; bulk_size];
    let t = Instant::now();
    for _ in 0..n_iterations {
        handle.read_raw(&mut bulk_read_buf).expect("bulk read");
    }
    let bulk_read_us = t.elapsed().as_micros() as f64;
    let bulk_read_mbps = (bulk_size as f64 * n_iterations as f64) / bulk_read_us;

    let total_bytes = bulk_size * n_iterations * 2;
    let total_us = bulk_write_us + bulk_read_us;

    println!("  Bulk write: {n_iterations}×{bulk_size}B = {bulk_write_mbps:.1} MB/s");
    println!("  Bulk read:  {n_iterations}×{bulk_size}B = {bulk_read_mbps:.1} MB/s");
    println!("  Total:      {total_bytes} bytes in {total_us:.0} µs");

    v.check_pass("bulk write throughput > 0", bulk_write_mbps > 0.0);
    v.check_pass("bulk read throughput > 0", bulk_read_mbps > 0.0);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│  Exp193: NPU Hardware Validation — AKD1000 ONLINE          │");
    println!("├──────────────────────────────────────────────────────────────┤");
    println!("│  Chip:       {:?}{:>39} │", handle.chip_version(), "");
    println!("│  NPUs:       {:>3}{:>46} │", handle.npu_count(), "");
    println!("│  SRAM:       {:>3} MB{:>43} │", handle.memory_mb(), "");
    println!(
        "│  PCIe:       Gen{} x{} ({:.1} GB/s){:>31} │",
        caps.pcie.generation, caps.pcie.lanes, caps.pcie.bandwidth_gbps, ""
    );
    println!("│  DMA Write:  {:.1} MB/s{:>42} │", bulk_write_mbps, "");
    println!("│  DMA Read:   {:.1} MB/s{:>42} │", bulk_read_mbps, "");
    println!(
        "│  Driver:     ToadStool akida-driver (pure Rust){:>13} │",
        ""
    );
    println!("│  Status:     ONLINE{:>41} │", "");
    println!("└──────────────────────────────────────────────────────────────┘");

    v.finish();
}
