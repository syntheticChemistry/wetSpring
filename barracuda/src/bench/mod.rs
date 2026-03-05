// SPDX-License-Identifier: AGPL-3.0-or-later

//! Benchmark harness for wetSpring validation runs.
//!
//! Captures hardware inventory, wall-clock time, CPU energy (Intel RAPL),
//! GPU power/temperature/VRAM (nvidia-smi), and process memory for every
//! validation phase.  Produces machine-readable JSON and human-readable
//! summary tables so that identical pipelines can be compared across
//! substrates (Python, `BarraCuda` CPU, `BarraCuda` GPU) and gates.
//!
//! See `benchmarks/PROTOCOL.md` for the full measurement specification.
//!
//! JSON serialization is hand-rolled to avoid a serde dependency.

mod hardware;
mod power;
mod report;
#[cfg(test)]
mod tests;

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
    /// Substrate label (e.g., `"Python"`, `"BarraCuda CPU"`, `"BarraCuda GPU"`).
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
    pub fn print_summary(&self) {
        report::print_bench_report(self);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Utility: read process RSS from /proc/self/status
// ═══════════════════════════════════════════════════════════════════

/// Parse peak RSS from `/proc/self/status` content.
///
/// Extracted from `peak_rss_mb` for testability without filesystem access.
#[must_use]
pub fn parse_peak_rss_mb(status_content: &str) -> f64 {
    for line in status_content.lines() {
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

/// Read peak resident set size (`VmHWM`) in MB from the live system.
///
/// Uses `/proc/self/status` on Linux. Returns `0.0` on non-Linux platforms
/// where this procfs file is unavailable, rather than panicking.
#[must_use]
pub fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        parse_peak_rss_mb(&status)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0.0
    }
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
