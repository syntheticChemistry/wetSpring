// SPDX-License-Identifier: AGPL-3.0-or-later
//! Power monitoring: Intel RAPL and nvidia-smi sampling.

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
                let delta = if end >= start {
                    end - start
                } else {
                    let max = read_rapl_max_energy_uj().unwrap_or(u64::MAX);
                    max - start + end
                };
                delta as f64 / 1_000_000.0
            }
            _ => 0.0,
        };

        let samples = self
            .gpu_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = samples.len();
        if n == 0 {
            return EnergyReport {
                cpu_joules,
                ..Default::default()
            };
        }

        let mut gpu_joules = 0.0_f64;
        let mut gpu_watts_sum = 0.0_f64;
        let mut gpu_watts_peak = 0.0_f64;
        let mut gpu_temp_peak = 0.0_f64;
        let mut gpu_vram_peak = 0.0_f64;

        for i in 0..n {
            gpu_watts_sum += samples[i].watts;
            if samples[i].watts > gpu_watts_peak {
                gpu_watts_peak = samples[i].watts;
            }
            if samples[i].temp_c > gpu_temp_peak {
                gpu_temp_peak = samples[i].temp_c;
            }
            if samples[i].vram_mib > gpu_vram_peak {
                gpu_vram_peak = samples[i].vram_mib;
            }

            if i > 0 {
                let dt = samples[i]
                    .timestamp
                    .duration_since(samples[i - 1].timestamp)
                    .as_secs_f64();
                let avg_w = (samples[i].watts + samples[i - 1].watts) * 0.5;
                gpu_joules += avg_w * dt;
            }
        }
        let gpu_joules = if n == 1 {
            samples[0].watts * wall_elapsed
        } else {
            gpu_joules
        };

        let gpu_watts_avg = gpu_watts_sum / n as f64;

        EnergyReport {
            cpu_joules,
            gpu_joules,
            gpu_watts_avg,
            gpu_watts_peak,
            gpu_temp_peak_c: gpu_temp_peak,
            gpu_vram_peak_mib: gpu_vram_peak,
            gpu_samples: n,
        }
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
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() >= 3 {
                        let watts = parts[0].trim().parse().unwrap_or(0.0);
                        let temp = parts[1].trim().parse().unwrap_or(0.0);
                        let vram = parts[2].trim().parse().unwrap_or(0.0);
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
