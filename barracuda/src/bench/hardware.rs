// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware inventory detection (CPU, GPU, RAM, etc.)
//!
//! Detection logic is separated from I/O: [`parse_cpuinfo`] and
//! [`parse_meminfo`] accept string content (capability-based), while
//! [`HardwareInventory::detect`] reads from the live system.

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
    /// Build a hardware inventory from pre-read system content.
    ///
    /// Accepts the raw content strings rather than reading filesystem
    /// directly. This allows metalForge or other callers to provide
    /// the hardware data, keeping barracuda independent of filesystem paths.
    #[must_use]
    pub fn from_content(
        gate_name: &str,
        cpuinfo: &str,
        meminfo: &str,
        nvidia_csv: &str,
        os_kernel: &str,
    ) -> Self {
        let (cpu_model, cpu_cores, cpu_threads, cpu_cache_kb) = parse_cpuinfo(cpuinfo);
        let ram_total_mb = parse_meminfo(meminfo);
        let (gpu_name, gpu_vram_mb, gpu_driver, gpu_compute_cap) =
            parse_nvidia_smi_output(nvidia_csv);
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
            os_kernel: os_kernel.to_string(),
            rust_version: String::new(),
        }
    }

    /// Auto-detect hardware from Linux sysfs / nvidia-smi.
    #[must_use]
    pub fn detect(gate_name: &str) -> Self {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let nvidia_csv = query_nvidia_smi_csv();
        let os_kernel = read_stdout("uname", &["-r"]);
        Self::from_content(gate_name, &cpuinfo, &meminfo, &nvidia_csv, &os_kernel)
    }

    /// Write the hardware inventory block to the given writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let col = 52;
        writeln!(w, "  ┌── Hardware ─{}┐", "─".repeat(col - 14))?;
        let Self {
            gate_name,
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
            ..
        } = self;
        writeln!(w, "  │ {:<col$}│", format!("Gate:   {gate_name}"))?;
        writeln!(w, "  │ {:<col$}│", format!("CPU:    {cpu_model}"))?;
        writeln!(
            w,
            "  │ {:<col$}│",
            format!("Cores:  {cpu_cores} ({cpu_threads} threads), L3 {cpu_cache_kb} KB"),
        )?;
        writeln!(w, "  │ {:<col$}│", format!("RAM:    {ram_total_mb} MB"))?;
        writeln!(w, "  │ {:<col$}│", format!("GPU:    {gpu_name}"))?;
        writeln!(
            w,
            "  │ {:<col$}│",
            format!("VRAM:   {gpu_vram_mb} MB, Driver {gpu_driver}, CC {gpu_compute_cap}"),
        )?;
        writeln!(w, "  │ {:<col$}│", format!("Kernel: {os_kernel}"))?;
        writeln!(w, "  └─{}┘", "─".repeat(col + 1))?;
        Ok(())
    }

    /// Pretty-print the inventory block to stdout.
    pub fn print(&self) {
        let mut stdout = std::io::stdout().lock();
        let _ = self.write_to(&mut stdout);
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

// ── Pure parsing functions (testable without hardware) ──────────

/// Parse `/proc/cpuinfo` content into (model, cores, threads, `cache_kb`).
#[must_use]
pub fn parse_cpuinfo(content: &str) -> (String, usize, usize, usize) {
    let mut model = String::from("unknown");
    let mut core_ids = HashSet::new();
    let mut thread_count = 0_usize;
    let mut cache_kb = 0_usize;

    for line in content.lines() {
        if line.starts_with("model name") {
            if let Some(v) = line.split(':').nth(1) {
                model = v.trim().to_string();
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

    let cores = core_ids.len().max(1);
    (model, cores, thread_count, cache_kb)
}

/// Parse `/proc/meminfo` content into total RAM in MB.
#[must_use]
pub fn parse_meminfo(content: &str) -> usize {
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

/// Parse nvidia-smi CSV inventory output.
#[must_use]
pub fn parse_nvidia_smi_output(csv_line: &str) -> (String, usize, String, String) {
    let parts: Vec<&str> = csv_line.split(", ").collect();
    if parts.len() >= 4 {
        let name = parts[0].trim().to_string();
        let vram_mb = parts[1].trim().parse().unwrap_or(0);
        let driver = parts[2].trim().to_string();
        let cc = parts[3].trim().to_string();
        (name, vram_mb, driver, cc)
    } else if csv_line.is_empty() {
        ("N/A".into(), 0, "N/A".into(), "N/A".into())
    } else {
        (csv_line.to_string(), 0, String::new(), String::new())
    }
}

// ── System I/O (thin wrappers for live detection) ───────────────

/// Query nvidia-smi for GPU inventory. Returns raw CSV string for use with
/// [`parse_nvidia_smi_output`] or [`HardwareInventory::from_content`].
#[must_use]
pub fn query_nvidia_smi_csv() -> String {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => String::new(),
    }
}

fn read_stdout(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map_or_else(
        |_| "unknown".to_string(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}

#[cfg(test)]
#[path = "hardware_tests.rs"]
mod tests;
