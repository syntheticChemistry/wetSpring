// SPDX-License-Identifier: AGPL-3.0-or-later
//! Power monitoring: Intel RAPL and nvidia-smi sampling.
//!
//! Parsing functions ([`parse_nvidia_smi_sample`], [`compute_gpu_energy`])
//! are separated from I/O for testability. The [`PowerMonitor`] type
//! orchestrates live system sampling.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct GpuSample {
    pub watts: f64,
    pub temp_c: f64,
    pub vram_mib: f64,
    pub timestamp: Instant,
}

/// Energy and power measurements for a single benchmark phase.
#[derive(Debug, Clone, Default)]
pub struct EnergyReport {
    /// CPU energy consumed (Joules) from Intel RAPL.
    pub cpu_joules: f64,
    /// GPU energy consumed (Joules) — integrated from nvidia-smi power samples.
    pub gpu_joules: f64,
    /// Average GPU power draw during the phase (Watts).
    pub gpu_watts_avg: f64,
    /// Peak GPU power draw (Watts).
    pub gpu_watts_peak: f64,
    /// Peak GPU temperature (Celsius).
    pub gpu_temp_peak_c: f64,
    /// Peak GPU VRAM usage (MiB).
    pub gpu_vram_peak_mib: f64,
    /// Number of nvidia-smi samples collected.
    pub gpu_samples: usize,
}

impl EnergyReport {
    pub(crate) fn to_json(&self) -> String {
        format!(
            r#"      "cpu_joules": {:.4},
      "gpu_joules": {:.4},
      "gpu_watts_avg": {:.2},
      "gpu_watts_peak": {:.2},
      "gpu_temp_peak_c": {:.1},
      "gpu_vram_peak_mib": {:.1},
      "gpu_samples": {}"#,
            self.cpu_joules,
            self.gpu_joules,
            self.gpu_watts_avg,
            self.gpu_watts_peak,
            self.gpu_temp_peak_c,
            self.gpu_vram_peak_mib,
            self.gpu_samples,
        )
    }
}

// ── Pure computation (testable without hardware) ────────────────

/// Parse a single nvidia-smi CSV sample line into (watts, `temp_c`, `vram_mib`).
///
/// Returns `None` if the line is malformed.
#[must_use]
pub fn parse_nvidia_smi_sample(line: &str) -> Option<(f64, f64, f64)> {
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() >= 3 {
        let watts = parts[0].trim().parse().ok()?;
        let temp = parts[1].trim().parse().ok()?;
        let vram = parts[2].trim().parse().ok()?;
        Some((watts, temp, vram))
    } else {
        None
    }
}

/// Compute RAPL energy delta in Joules, handling counter wraparound.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn rapl_delta_joules(start_uj: u64, end_uj: u64, max_uj: u64) -> f64 {
    let delta = if end_uj >= start_uj {
        end_uj - start_uj
    } else {
        max_uj - start_uj + end_uj
    };
    delta as f64 / 1_000_000.0
}

/// Compute GPU energy report from a slice of samples and wall-clock time.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_gpu_energy(samples: &[GpuSample], wall_elapsed_s: f64) -> EnergyReport {
    let n = samples.len();
    if n == 0 {
        return EnergyReport::default();
    }

    let mut gpu_joules = 0.0_f64;
    let mut watts_sum = 0.0_f64;
    let mut watts_peak = 0.0_f64;
    let mut temp_peak = 0.0_f64;
    let mut vram_peak = 0.0_f64;

    for (i, s) in samples.iter().enumerate() {
        watts_sum += s.watts;
        if s.watts > watts_peak {
            watts_peak = s.watts;
        }
        if s.temp_c > temp_peak {
            temp_peak = s.temp_c;
        }
        if s.vram_mib > vram_peak {
            vram_peak = s.vram_mib;
        }
        if i > 0 {
            let dt = s
                .timestamp
                .duration_since(samples[i - 1].timestamp)
                .as_secs_f64();
            let avg_w = (s.watts + samples[i - 1].watts) * 0.5;
            gpu_joules += avg_w * dt;
        }
    }

    let gpu_joules = if n == 1 {
        samples[0].watts * wall_elapsed_s
    } else {
        gpu_joules
    };

    EnergyReport {
        cpu_joules: 0.0,
        gpu_joules,
        gpu_watts_avg: watts_sum / n as f64,
        gpu_watts_peak: watts_peak,
        gpu_temp_peak_c: temp_peak,
        gpu_vram_peak_mib: vram_peak,
        gpu_samples: n,
    }
}

// ── Live system I/O ─────────────────────────────────────────────

/// Background monitor that samples RAPL + nvidia-smi.
#[derive(Debug)]
pub struct PowerMonitor {
    rapl_start_uj: Option<u64>,
    wall_start: Instant,
    smi_child: Option<std::process::Child>,
    gpu_samples: Arc<Mutex<Vec<GpuSample>>>,
    reader_handle: Option<std::thread::JoinHandle<()>>,
}

impl PowerMonitor {
    /// Begin monitoring.  Reads RAPL baseline and spawns nvidia-smi.
    #[must_use]
    pub fn start() -> Self {
        let rapl_start_uj = read_rapl_energy_uj();
        let wall_start = Instant::now();
        let gpu_samples: Arc<Mutex<Vec<GpuSample>>> = Arc::new(Mutex::new(Vec::new()));

        let (smi_child, reader_handle) = spawn_nvidia_smi_poller(gpu_samples.clone());

        Self {
            rapl_start_uj,
            wall_start,
            smi_child,
            gpu_samples,
            reader_handle,
        }
    }

    /// Stop monitoring and return the energy report.
    #[allow(clippy::cast_precision_loss, clippy::significant_drop_tightening)]
    pub fn stop(mut self) -> EnergyReport {
        let wall_elapsed = self.wall_start.elapsed().as_secs_f64();

        if let Some(ref mut child) = self.smi_child {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }

        let cpu_joules = match (self.rapl_start_uj, read_rapl_energy_uj()) {
            (Some(start), Some(end)) => {
                let max = read_rapl_max_energy_uj().unwrap_or(u64::MAX);
                rapl_delta_joules(start, end, max)
            }
            _ => 0.0,
        };

        let samples = self
            .gpu_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut report = compute_gpu_energy(&samples, wall_elapsed);
        report.cpu_joules = cpu_joules;
        report
    }
}

pub fn read_rapl_energy_uj() -> Option<u64> {
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

pub fn read_rapl_max_energy_uj() -> Option<u64> {
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

pub fn spawn_nvidia_smi_poller(
    samples: Arc<Mutex<Vec<GpuSample>>>,
) -> (
    Option<std::process::Child>,
    Option<std::thread::JoinHandle<()>>,
) {
    let child = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(mut child) => {
            let Some(stdout) = child.stdout.take() else {
                return (None, None);
            };
            let handle = std::thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut line_buf = String::new();
                loop {
                    line_buf.clear();
                    match reader.read_line(&mut line_buf) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let line = line_buf.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some((watts, temp, vram)) = parse_nvidia_smi_sample(line) {
                        if let Ok(mut v) = samples.lock() {
                            v.push(GpuSample {
                                watts,
                                temp_c: temp,
                                vram_mib: vram,
                                timestamp: Instant::now(),
                            });
                        }
                    }
                }
            });
            (Some(child), Some(handle))
        }
        Err(_) => (None, None),
    }
}

#[cfg(test)]
#[path = "power_tests.rs"]
mod tests;
