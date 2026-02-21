// SPDX-License-Identifier: AGPL-3.0-or-later

//! Benchmark harness for wetSpring validation runs.
//!
//! Captures hardware inventory, wall-clock time, CPU energy (Intel RAPL),
//! GPU power/temperature/VRAM (nvidia-smi), and process memory for every
//! validation phase.  Produces machine-readable JSON and human-readable
//! summary tables so that identical pipelines can be compared across
//! substrates (Python, `BarraCUDA` CPU, `BarraCUDA` GPU) and gates.
//!
//! See `benchmarks/PROTOCOL.md` for the full measurement specification.
//!
//! JSON serialization is hand-rolled to avoid a serde dependency.

mod hardware;
mod power;

pub use hardware::HardwareInventory;
pub use power::{EnergyReport, PowerMonitor};

// ═══════════════════════════════════════════════════════════════════
//  Phase Result
// ═══════════════════════════════════════════════════════════════════

/// Result from a single benchmark phase.
#[derive(Debug, Clone)]
pub struct PhaseResult {
    /// Phase name (e.g., `"shannon_1m"`, `"felsenstein"`).
    pub phase: String,
    /// Substrate label (e.g., `"Python"`, `"BarraCUDA CPU"`, `"BarraCUDA GPU"`).
    pub substrate: String,
    /// Total wall-clock time in seconds.
    pub wall_time_s: f64,
    /// Microseconds per evaluation (`wall_time_s` / `n_evals` * 1e6). Zero if no evals.
    pub per_eval_us: f64,
    /// Number of evaluations/computations performed.
    pub n_evals: usize,
    /// CPU (RAPL) and GPU (nvidia-smi) energy/power measurements.
    pub energy: EnergyReport,
    /// Peak resident set size in MB (`VmHWM` from `/proc/self/status`).
    pub peak_rss_mb: f64,
    /// Optional free-form notes (e.g., `"smoke test"`).
    pub notes: String,
}

impl PhaseResult {
    fn to_json(&self) -> String {
        format!(
            r#"    {{
      "phase": "{}",
      "substrate": "{}",
      "wall_time_s": {:.6},
      "per_eval_us": {:.3},
      "n_evals": {},
      "energy": {{
{}
      }},
      "peak_rss_mb": {:.1},
      "notes": "{}"
    }}"#,
            json_escape(&self.phase),
            json_escape(&self.substrate),
            self.wall_time_s,
            self.per_eval_us,
            self.n_evals,
            self.energy.to_json(),
            self.peak_rss_mb,
            json_escape(&self.notes),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Bench Report (top-level container)
// ═══════════════════════════════════════════════════════════════════

/// Full benchmark report for a validation run.
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// ISO8601 timestamp when the report was created.
    pub timestamp: String,
    /// Hardware snapshot (CPU, GPU, RAM) at run start.
    pub hardware: HardwareInventory,
    /// Per-phase results. Multiple phases may share the same name across substrates.
    pub phases: Vec<PhaseResult>,
}

impl BenchReport {
    /// Create a new report with hardware inventory.
    #[must_use]
    pub fn new(hw: HardwareInventory) -> Self {
        Self {
            timestamp: now_iso8601(),
            hardware: hw,
            phases: Vec::new(),
        }
    }

    /// Add a phase result.
    pub fn add_phase(&mut self, phase: PhaseResult) {
        self.phases.push(phase);
    }

    /// Serialize to JSON string (hand-rolled, no serde).
    pub fn to_json(&self) -> String {
        let phases_json: Vec<String> = self.phases.iter().map(PhaseResult::to_json).collect();
        format!(
            r#"{{
  "timestamp": "{}",
  "hardware": {{
{}
  }},
  "phases": [
{}
  ]
}}"#,
            json_escape(&self.timestamp),
            self.hardware.to_json(),
            phases_json.join(",\n"),
        )
    }

    /// Save to JSON file.  Returns the path written.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation or file writing fails.
    pub fn save_json(&self, dir: &str) -> std::io::Result<String> {
        std::fs::create_dir_all(dir)?;
        let filename = format!(
            "{}_{}.json",
            self.hardware
                .gate_name
                .to_lowercase()
                .replace([' ', '/'], "_"),
            self.timestamp.replace(':', "-").replace(' ', "_"),
        );
        let path = format!("{dir}/{filename}");
        std::fs::write(&path, self.to_json())?;
        Ok(path)
    }

    /// Print summary table to stdout.
    ///
    /// # Panics
    ///
    /// Panics if `matching` has fewer than 2 elements when computing speedup (internal logic error).
    #[allow(
        clippy::cast_precision_loss,
        clippy::missing_panics_doc,
        clippy::too_many_lines
    )]
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!(
            "  SUBSTRATE BENCHMARK REPORT — {} ({} / {})",
            self.hardware.gate_name, self.hardware.cpu_model, self.hardware.gpu_name
        );
        println!("══════════════════════════════════════════════════════════════════════════════════════════");
        println!();

        println!(
            "  {:<24} {:<14} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10}",
            "Phase",
            "Substrate",
            "Wall Time",
            "per-eval",
            "Energy J",
            "J/eval",
            "W (avg)",
            "W (peak)"
        );
        println!("  {}", "─".repeat(100));

        for p in &self.phases {
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

            let sub_label = if is_gpu {
                format!("{} [G]", p.substrate)
            } else {
                format!("{} [C]", p.substrate)
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

        let gpu_phases: Vec<&PhaseResult> = self
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
        for p in &self.phases {
            if seen.contains(&p.phase) {
                continue;
            }
            seen.insert(p.phase.clone());

            let matching: Vec<&PhaseResult> =
                self.phases.iter().filter(|q| q.phase == p.phase).collect();
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

                let fast_gpu =
                    fastest.substrate.contains("GPU") || fastest.substrate.contains("gpu");
                let slow_gpu =
                    slowest.substrate.contains("GPU") || slowest.substrate.contains("gpu");
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
}

// ═══════════════════════════════════════════════════════════════════
//  Utility: read process RSS from /proc/self/status
// ═══════════════════════════════════════════════════════════════════

/// Read peak resident set size (`VmHWM`) in MB.
#[must_use]
pub fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

// ═══════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Current time as ISO8601 string (e.g., `2025-02-21T14:30:00`). No timezone.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let day_secs = (secs % 86400) as u32;
    let (hour, minute, second) = (day_secs / 3600, (day_secs % 3600) / 60, day_secs % 60);
    let z = (secs / 86400) as i64 + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}")
}

/// Format seconds as human-readable string (us, ms, s, min).
#[must_use]
pub fn format_duration(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.1} us", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.1} ms", secs * 1e3)
    } else if secs < 60.0 {
        format!("{secs:.2} s")
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

/// Format per-eval time in microseconds as human-readable string (us, ms, s).
#[must_use]
pub fn format_eval_time(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.1} us")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn energy_report_default_values() {
        let r = EnergyReport::default();
        assert_eq!(r.cpu_joules, 0.0);
        assert_eq!(r.gpu_joules, 0.0);
        assert_eq!(r.gpu_watts_avg, 0.0);
        assert_eq!(r.gpu_watts_peak, 0.0);
        assert_eq!(r.gpu_temp_peak_c, 0.0);
        assert_eq!(r.gpu_vram_peak_mib, 0.0);
        assert_eq!(r.gpu_samples, 0);
    }

    #[test]
    fn phase_result_creation_and_fields() {
        let energy = EnergyReport {
            cpu_joules: 1.5,
            gpu_joules: 0.0,
            ..Default::default()
        };
        let pr = PhaseResult {
            phase: "shannon_1m".to_string(),
            substrate: "BarraCUDA GPU".to_string(),
            wall_time_s: 2.5,
            per_eval_us: 0.42,
            n_evals: 10_000,
            energy,
            peak_rss_mb: 128.0,
            notes: "smoke test".to_string(),
        };
        assert_eq!(pr.phase, "shannon_1m");
        assert_eq!(pr.substrate, "BarraCUDA GPU");
        assert!((pr.wall_time_s - 2.5).abs() < 1e-9);
        assert_eq!(pr.n_evals, 10_000);
        assert!((pr.energy.cpu_joules - 1.5).abs() < f64::EPSILON);
        assert!((pr.peak_rss_mb - 128.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bench_report_json_roundtrip() {
        let hw = HardwareInventory {
            gate_name: "test".to_string(),
            cpu_model: "i9-12900K".to_string(),
            cpu_cores: 16,
            cpu_threads: 24,
            cpu_cache_kb: 30720,
            ram_total_mb: 65536,
            gpu_name: "RTX 4070".to_string(),
            gpu_vram_mb: 12288,
            gpu_driver: "550.120".to_string(),
            gpu_compute_cap: "8.9".to_string(),
            os_kernel: "6.17.4".to_string(),
            rust_version: "1.82".to_string(),
        };
        let mut report = BenchReport::new(hw);
        report.add_phase(PhaseResult {
            phase: "shannon".to_string(),
            substrate: "CPU".to_string(),
            wall_time_s: 0.005,
            per_eval_us: 5.0,
            n_evals: 1000,
            energy: EnergyReport::default(),
            peak_rss_mb: 32.0,
            notes: String::new(),
        });
        let json = report.to_json();
        assert!(json.contains("\"phase\": \"shannon\""));
        assert!(json.contains("\"substrate\": \"CPU\""));
        assert!(json.contains("\"cpu_model\": \"i9-12900K\""));
    }

    #[test]
    fn format_duration_sub_millisecond() {
        assert!(format_duration(0.0001).contains("us"));
    }

    #[test]
    fn format_duration_milliseconds() {
        let s = format_duration(0.05);
        assert!(s.contains("ms"));
    }

    #[test]
    fn format_duration_seconds() {
        let s = format_duration(1.5);
        assert!(s.contains('s'));
        assert!(!s.contains("min"));
    }

    #[test]
    fn format_duration_minutes() {
        let s = format_duration(90.0);
        assert!(s.contains("min"));
    }

    #[test]
    fn format_eval_time_microseconds() {
        let s = format_eval_time(500.0);
        assert!(s.contains("us"));
    }

    #[test]
    fn format_eval_time_milliseconds() {
        let s = format_eval_time(5_000.0);
        assert!(s.contains("ms"));
    }

    #[test]
    fn format_eval_time_seconds() {
        let s = format_eval_time(2_000_000.0);
        assert!(s.contains('s'));
    }

    #[test]
    fn now_iso8601_format() {
        let s = now_iso8601();
        let parts: Vec<&str> = s.split('T').collect();
        assert_eq!(parts.len(), 2, "expected YYYY-MM-DDTHH:MM:SS format");
        let date: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date.len(), 3);
        assert_eq!(date[0].len(), 4);
        assert_eq!(date[1].len(), 2);
        assert_eq!(date[2].len(), 2);
    }

    #[test]
    fn peak_rss_mb_non_negative() {
        let rss = peak_rss_mb();
        assert!(rss >= 0.0, "peak_rss_mb should be non-negative");
    }

    #[test]
    fn json_escape_special_chars() {
        assert_eq!(json_escape(r#"foo"bar"#), r#"foo\"bar"#);
        assert_eq!(json_escape("foo\\bar"), "foo\\\\bar");
        assert_eq!(json_escape("foo\nbar"), "foo\\nbar");
    }

    #[test]
    #[ignore = "requires nvidia-smi and RAPL"]
    fn power_monitor_start_stop() {
        let monitor = PowerMonitor::start();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = monitor.stop();
    }
}
