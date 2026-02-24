// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark report formatting and printing.

use super::{BenchReport, PhaseResult, format_duration, format_eval_time};

/// Print the benchmark summary table to stdout.
///
/// # Panics
///
/// Panics if `matching` has fewer than 2 elements when computing speedup (internal logic error).
#[allow(
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]
pub fn print_bench_report(report: &BenchReport) {
    println!();
    println!(
        "══════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  SUBSTRATE BENCHMARK REPORT — {} ({} / {})",
        report.hardware.gate_name, report.hardware.cpu_model, report.hardware.gpu_name
    );
    println!(
        "══════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!();

    println!(
        "  {:<24} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10}",
        "Phase", "Substrate", "Wall Time", "per-eval", "Energy J", "J/eval", "W (avg)", "W (peak)"
    );
    println!("  {}", "─".repeat(100));

    for p in &report.phases {
        let wall_str = format_duration(p.wall_time_s);
        let eval_str = if p.per_eval_us > 0.0 {
            format_eval_time(p.per_eval_us)
        } else {
            "—".to_string()
        };

        let is_gpu = p.substrate.contains("GPU") || p.substrate.contains("gpu");
        let primary_joules = if is_gpu {
            p.energy.gpu_joules
        } else {
            p.energy.cpu_joules
        };
        let primary_watts = if is_gpu {
            p.energy.gpu_watts_avg
        } else if p.energy.cpu_joules > 0.0 && p.wall_time_s > 0.0 {
            p.energy.cpu_joules / p.wall_time_s
        } else {
            0.0
        };

        let energy_str = if primary_joules > 0.01 {
            format!("{primary_joules:.2}")
        } else if primary_joules > 0.0 {
            format!("{primary_joules:.4}")
        } else {
            "—".to_string()
        };

        let j_per_eval = if primary_joules > 0.0 && p.n_evals > 0 {
            let j = primary_joules / p.n_evals as f64;
            if j > 0.01 {
                format!("{j:.3}")
            } else if j > 0.0001 {
                format!("{j:.1e}")
            } else {
                format!("{j:.2e}")
            }
        } else {
            "—".to_string()
        };

        let watts_str = if primary_watts > 0.1 {
            format!("{primary_watts:.0} W")
        } else {
            "—".to_string()
        };

        let peak_watts = if is_gpu {
            p.energy.gpu_watts_peak
        } else {
            primary_watts
        };
        let peak_watts_str = if peak_watts > 0.1 {
            format!("{peak_watts:.0} W")
        } else {
            "—".to_string()
        };

        let substrate = &p.substrate;
        let sub_label = if is_gpu {
            format!("{substrate} [G]")
        } else {
            format!("{substrate} [C]")
        };

        println!(
            "  {:<24} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10}",
            p.phase,
            sub_label,
            wall_str,
            eval_str,
            energy_str,
            j_per_eval,
            watts_str,
            peak_watts_str
        );
    }
    println!("  {}", "─".repeat(100));
    println!(
        "  [C] = CPU energy (RAPL)  [G] = GPU energy (nvidia-smi, {}ms polling)",
        100
    );

    let gpu_phases: Vec<&PhaseResult> = report
        .phases
        .iter()
        .filter(|p| p.substrate.contains("GPU") || p.substrate.contains("gpu"))
        .collect();
    if !gpu_phases.is_empty() {
        println!();
        println!("  GPU Power Detail:");
        println!(
            "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Phase", "W (avg)", "W (peak)", "Temp °C", "VRAM MB", "Samples", "Total J"
        );
        println!("  {}", "─".repeat(72));
        for p in &gpu_phases {
            println!(
                "  {:<22} {:>7.1} {:>7.1} {:>7.0} {:>7.0} {:>8} {:>8.0}",
                p.phase,
                p.energy.gpu_watts_avg,
                p.energy.gpu_watts_peak,
                p.energy.gpu_temp_peak_c,
                p.energy.gpu_vram_peak_mib,
                p.energy.gpu_samples,
                p.energy.gpu_joules
            );
        }
        println!("  {}", "─".repeat(72));
    }

    // Pairwise speedup comparisons for matching phases across substrates
    println!();
    let mut seen = std::collections::HashSet::new();
    for p in &report.phases {
        if seen.contains(&p.phase) {
            continue;
        }
        seen.insert(p.phase.clone());

        let matching: Vec<&PhaseResult> = report
            .phases
            .iter()
            .filter(|q| q.phase == p.phase)
            .collect();
        if matching.len() < 2 {
            continue;
        }

        let (Some(fastest), Some(slowest)) = (
            matching
                .iter()
                .min_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s)),
            matching
                .iter()
                .max_by(|a, b| a.wall_time_s.total_cmp(&b.wall_time_s)),
        ) else {
            continue; // matching non-empty, so min/max are Some; defensive fallback
        };

        if fastest.wall_time_s > 0.0 && slowest.wall_time_s > fastest.wall_time_s {
            let speedup = slowest.wall_time_s / fastest.wall_time_s;
            println!(
                "  {} : {} is {:.1}x faster than {} ({} vs {})",
                fastest.phase,
                fastest.substrate,
                speedup,
                slowest.substrate,
                format_duration(fastest.wall_time_s),
                format_duration(slowest.wall_time_s)
            );

            let fast_gpu = fastest.substrate.contains("GPU") || fastest.substrate.contains("gpu");
            let slow_gpu = slowest.substrate.contains("GPU") || slowest.substrate.contains("gpu");
            let fast_j = if fast_gpu {
                fastest.energy.gpu_joules
            } else {
                fastest.energy.cpu_joules
            };
            let slow_j = if slow_gpu {
                slowest.energy.gpu_joules
            } else {
                slowest.energy.cpu_joules
            };
            if fast_j > 0.0 && slow_j > 0.0 {
                let ratio = slow_j / fast_j;
                println!(
                    "           energy: {:.2}J ({}) vs {:.2}J ({}) — {:.1}x less",
                    fast_j,
                    if fast_gpu { "GPU" } else { "CPU" },
                    slow_j,
                    if slow_gpu { "GPU" } else { "CPU" },
                    ratio
                );
            }
        }
    }
    println!();
}
