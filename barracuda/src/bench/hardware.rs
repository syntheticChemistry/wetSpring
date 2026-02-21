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
    /// Auto-detect hardware from Linux sysfs / nvidia-smi.
    #[must_use]
    pub fn detect(gate_name: &str) -> Self {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let (cpu_model, cpu_cores, cpu_threads, cpu_cache_kb) = parse_cpuinfo(&cpuinfo);
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let ram_total_mb = parse_meminfo(&meminfo);
        let (gpu_name, gpu_vram_mb, gpu_driver, gpu_compute_cap) = query_nvidia_smi_inventory();
        let os_kernel = read_stdout("uname", &["-r"]);

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
            rust_version: String::new(),
        }
    }

    /// Pretty-print the inventory block.
    pub fn print(&self) {
        let w = 52;
        println!("  ┌── Hardware ─{}┐", "─".repeat(w - 14));
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
        println!("  │ {:<w$}│", format!("Gate:   {gate_name}"));
        println!("  │ {:<w$}│", format!("CPU:    {cpu_model}"));
        println!(
            "  │ {:<w$}│",
            format!("Cores:  {cpu_cores} ({cpu_threads} threads), L3 {cpu_cache_kb} KB"),
        );
        println!("  │ {:<w$}│", format!("RAM:    {ram_total_mb} MB"));
        println!("  │ {:<w$}│", format!("GPU:    {gpu_name}"));
        println!(
            "  │ {:<w$}│",
            format!("VRAM:   {gpu_vram_mb} MB, Driver {gpu_driver}, CC {gpu_compute_cap}"),
        );
        println!("  │ {:<w$}│", format!("Kernel: {os_kernel}"));
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

/// Query nvidia-smi for GPU inventory. Returns fallback on failure.
pub fn query_nvidia_smi_inventory() -> (String, usize, String, String) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            parse_nvidia_smi_output(&s)
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CPUINFO: &str = "\
processor\t: 0
model name\t: Intel(R) Core(TM) i9-12900K
core id\t\t: 0
cache size\t: 30720 KB

processor\t: 1
model name\t: Intel(R) Core(TM) i9-12900K
core id\t\t: 1
cache size\t: 30720 KB

processor\t: 2
model name\t: Intel(R) Core(TM) i9-12900K
core id\t\t: 0
cache size\t: 30720 KB
";

    #[test]
    fn parse_cpuinfo_extracts_fields() {
        let (model, cores, threads, cache_kb) = parse_cpuinfo(SAMPLE_CPUINFO);
        assert_eq!(model, "Intel(R) Core(TM) i9-12900K");
        assert_eq!(cores, 2);
        assert_eq!(threads, 3);
        assert_eq!(cache_kb, 30720);
    }

    #[test]
    fn parse_cpuinfo_empty() {
        let (model, cores, threads, cache_kb) = parse_cpuinfo("");
        assert_eq!(model, "unknown");
        assert_eq!(cores, 1);
        assert_eq!(threads, 0);
        assert_eq!(cache_kb, 0);
    }

    #[test]
    fn parse_cpuinfo_single_core() {
        let content = "processor\t: 0\nmodel name\t: ARM\ncore id\t\t: 0\n";
        let (model, cores, threads, _) = parse_cpuinfo(content);
        assert_eq!(model, "ARM");
        assert_eq!(cores, 1);
        assert_eq!(threads, 1);
    }

    #[test]
    fn parse_meminfo_standard() {
        let content = "MemTotal:       65536000 kB\nMemFree:        32000000 kB\n";
        let mb = parse_meminfo(content);
        assert_eq!(mb, 64000);
    }

    #[test]
    fn parse_meminfo_empty() {
        assert_eq!(parse_meminfo(""), 0);
    }

    #[test]
    fn parse_meminfo_malformed() {
        let content = "MemTotal: not_a_number kB\n";
        assert_eq!(parse_meminfo(content), 0);
    }

    #[test]
    fn parse_nvidia_smi_full() {
        let csv = "NVIDIA GeForce RTX 4070, 12282, 550.120, 8.9";
        let (name, vram, driver, cc) = parse_nvidia_smi_output(csv);
        assert_eq!(name, "NVIDIA GeForce RTX 4070");
        assert_eq!(vram, 12282);
        assert_eq!(driver, "550.120");
        assert_eq!(cc, "8.9");
    }

    #[test]
    fn parse_nvidia_smi_empty() {
        let (name, vram, driver, cc) = parse_nvidia_smi_output("");
        assert_eq!(name, "N/A");
        assert_eq!(vram, 0);
        assert_eq!(driver, "N/A");
        assert_eq!(cc, "N/A");
    }

    #[test]
    fn parse_nvidia_smi_partial() {
        let csv = "Some GPU Only";
        let (name, vram, _, _) = parse_nvidia_smi_output(csv);
        assert_eq!(name, "Some GPU Only");
        assert_eq!(vram, 0);
    }

    #[test]
    fn hardware_inventory_to_json_format() {
        let hw = HardwareInventory {
            gate_name: "test-gate".to_string(),
            cpu_model: "Test CPU".to_string(),
            cpu_cores: 4,
            cpu_threads: 8,
            cpu_cache_kb: 8192,
            ram_total_mb: 16384,
            gpu_name: "Test GPU".to_string(),
            gpu_vram_mb: 8192,
            gpu_driver: "500.0".to_string(),
            gpu_compute_cap: "8.0".to_string(),
            os_kernel: "6.0.0".to_string(),
            rust_version: "1.82".to_string(),
        };
        let json = hw.to_json();
        assert!(json.contains("\"gate_name\": \"test-gate\""));
        assert!(json.contains("\"cpu_cores\": 4"));
        assert!(json.contains("\"gpu_vram_mb\": 8192"));
    }

    #[test]
    fn hardware_inventory_to_json_escapes_special_chars() {
        let hw = HardwareInventory {
            gate_name: "gate\"special".to_string(),
            cpu_model: "CPU\\model".to_string(),
            cpu_cores: 1,
            cpu_threads: 1,
            cpu_cache_kb: 0,
            ram_total_mb: 0,
            gpu_name: "N/A".to_string(),
            gpu_vram_mb: 0,
            gpu_driver: "N/A".to_string(),
            gpu_compute_cap: "N/A".to_string(),
            os_kernel: "test".to_string(),
            rust_version: String::new(),
        };
        let json = hw.to_json();
        assert!(json.contains(r#"gate\"special"#));
        assert!(json.contains("CPU\\\\model"));
    }
}
