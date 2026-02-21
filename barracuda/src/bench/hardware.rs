// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware inventory detection (CPU, GPU, RAM, etc.)

use std::collections::HashSet;
use std::process::Command;

/// Complete hardware description captured once at the start of a run.
#[derive(Debug, Clone)]
pub struct HardwareInventory {
    /// Machine or gate identifier (e.g., `"eastgate"`, `"ci-linux"`).
    pub gate_name: String,
    /// CPU model string from `/proc/cpuinfo` (e.g., `"Intel(R) Core(TM) i9-12900K"`).
    pub cpu_model: String,
    /// Number of physical cores (from `core id`).
    pub cpu_cores: usize,
    /// Number of logical threads (from `processor` entries).
    pub cpu_threads: usize,
    /// L3 cache size in KB (from `cache size`).
    pub cpu_cache_kb: usize,
    /// Total RAM in MB (from `MemTotal` in `/proc/meminfo`).
    pub ram_total_mb: usize,
    /// GPU product name from nvidia-smi (or `"N/A"` if none).
    pub gpu_name: String,
    /// GPU VRAM in MB.
    pub gpu_vram_mb: usize,
    /// NVIDIA driver version.
    pub gpu_driver: String,
    /// Compute capability (e.g., `"8.9"`).
    pub gpu_compute_cap: String,
    /// OS kernel version from `uname -r`.
    pub os_kernel: String,
    /// Rust toolchain version (if populated).
    pub rust_version: String,
}

impl HardwareInventory {
    /// Auto-detect hardware from Linux sysfs / nvidia-smi.
    #[must_use]
    pub fn detect(gate_name: &str) -> Self {
        let (cpu_model, cpu_cores, cpu_threads, cpu_cache_kb) = read_cpuinfo();
        let ram_total_mb = read_meminfo();
        let (gpu_name, gpu_vram_mb, gpu_driver, gpu_compute_cap) = read_nvidia_smi_inventory();
        let os_kernel = read_stdout("uname", &["-r"]);
        let rust_version = String::new();

        Self {
            gate_name: gate_name.to_string(),
            cpu_model,
            cpu_cores,
            cpu_threads,
            cpu_cache_kb,
            ram_total_mb,
            gpu_name,
            gpu_vram_mb,
            gpu_driver,
            gpu_compute_cap,
            os_kernel,
            rust_version,
        }
    }

    /// Pretty-print the inventory block.
    pub fn print(&self) {
        let w = 52;
        println!("  ┌── Hardware ─{}┐", "─".repeat(w - 14));
        println!(
            "  │ {:<width$}│",
            format!("Gate:   {}", self.gate_name),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("CPU:    {}", self.cpu_model),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!(
                "Cores:  {} ({} threads), L3 {} KB",
                self.cpu_cores, self.cpu_threads, self.cpu_cache_kb
            ),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("RAM:    {} MB", self.ram_total_mb),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("GPU:    {}", self.gpu_name),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!(
                "VRAM:   {} MB, Driver {}, CC {}",
                self.gpu_vram_mb, self.gpu_driver, self.gpu_compute_cap
            ),
            width = w
        );
        println!(
            "  │ {:<width$}│",
            format!("Kernel: {}", self.os_kernel),
            width = w
        );
        println!("  └─{}┘", "─".repeat(w + 1));
    }

    pub(crate) fn to_json(&self) -> String {
        format!(
            r#"    "gate_name": "{}",
    "cpu_model": "{}",
    "cpu_cores": {},
    "cpu_threads": {},
    "cpu_cache_kb": {},
    "ram_total_mb": {},
    "gpu_name": "{}",
    "gpu_vram_mb": {},
    "gpu_driver": "{}",
    "gpu_compute_cap": "{}",
    "os_kernel": "{}",
    "rust_version": "{}""#,
            super::json_escape(&self.gate_name),
            super::json_escape(&self.cpu_model),
            self.cpu_cores,
            self.cpu_threads,
            self.cpu_cache_kb,
            self.ram_total_mb,
            super::json_escape(&self.gpu_name),
            self.gpu_vram_mb,
            super::json_escape(&self.gpu_driver),
            super::json_escape(&self.gpu_compute_cap),
            super::json_escape(&self.os_kernel),
            super::json_escape(&self.rust_version),
        )
    }
}

pub fn read_cpuinfo() -> (String, usize, usize, usize) {
    let content = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut model = String::from("unknown");
    let mut physical_ids = HashSet::new();
    let mut core_ids = HashSet::new();
    let mut thread_count = 0_usize;
    let mut cache_kb = 0_usize;

    for line in content.lines() {
        if line.starts_with("model name") {
            if let Some(v) = line.split(':').nth(1) {
                model = v.trim().to_string();
            }
        } else if line.starts_with("physical id") {
            if let Some(v) = line.split(':').nth(1) {
                physical_ids.insert(v.trim().to_string());
            }
        } else if line.starts_with("core id") {
            if let Some(v) = line.split(':').nth(1) {
                core_ids.insert(v.trim().to_string());
            }
        } else if line.starts_with("processor") {
            thread_count += 1;
        } else if line.starts_with("cache size") {
            if let Some(v) = line.split(':').nth(1) {
                let v = v.trim().replace(" KB", "");
                cache_kb = v.parse().unwrap_or(0);
            }
        }
    }

    let _ = physical_ids; // may use later for NUMA
    let cores = core_ids.len().max(1);
    (model, cores, thread_count, cache_kb)
}

pub fn read_meminfo() -> usize {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

pub fn read_nvidia_smi_inventory() -> (String, usize, String, String) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let parts: Vec<&str> = s.split(", ").collect();
            if parts.len() >= 4 {
                let name = parts[0].trim().to_string();
                let vram_mb = parts[1].trim().parse().unwrap_or(0);
                let driver = parts[2].trim().to_string();
                let cc = parts[3].trim().to_string();
                return (name, vram_mb, driver, cc);
            }
            (s, 0, String::new(), String::new())
        }
        _ => ("N/A".into(), 0, "N/A".into(), "N/A".into()),
    }
}

fn read_stdout(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or_else(
        |_| "unknown".to_string(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}
