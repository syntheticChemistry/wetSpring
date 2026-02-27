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
    fn from_content_constructs_inventory() {
        let hw = HardwareInventory::from_content(
            "test",
            "processor\t: 0\nmodel name\t: Test\ncore id\t\t: 0\n",
            "MemTotal:       8192000 kB\n",
            "Test GPU, 8192, 500.0, 8.0",
            "6.0.0",
        );
        assert_eq!(hw.gate_name, "test");
        assert_eq!(hw.cpu_model, "Test");
        assert_eq!(hw.gpu_name, "Test GPU");
        assert_eq!(hw.os_kernel, "6.0.0");
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

    #[test]
    fn parse_cpuinfo_no_core_id() {
        let content = "processor\t: 0\nmodel name\t: ARM CPU\n";
        let (model, cores, threads, cache_kb) = parse_cpuinfo(content);
        assert_eq!(model, "ARM CPU");
        assert_eq!(cores, 1);
        assert_eq!(threads, 1);
        assert_eq!(cache_kb, 0);
    }

    #[test]
    fn parse_cpuinfo_multiple_cores_many_threads() {
        let content = "\
processor\t: 0
model name\t: Intel Xeon
core id\t\t: 0
cache size\t: 16384 KB

processor\t: 1
core id\t\t: 1

processor\t: 2
core id\t\t: 0

processor\t: 3
core id\t\t: 1

processor\t: 4
core id\t\t: 2

processor\t: 5
core id\t\t: 3

processor\t: 6
core id\t\t: 4

processor\t: 7
core id\t\t: 5

processor\t: 8
core id\t\t: 6

processor\t: 9
core id\t\t: 7

processor\t: 10
core id\t\t: 0

processor\t: 11
core id\t\t: 1

processor\t: 12
core id\t\t: 2

processor\t: 13
core id\t\t: 3

processor\t: 14
core id\t\t: 4

processor\t: 15
core id\t\t: 5
";
        let (_, cores, threads, _) = parse_cpuinfo(content);
        assert_eq!(cores, 8);
        assert_eq!(threads, 16);
    }

    #[test]
    fn parse_cpuinfo_no_cache_size() {
        let content = "processor\t: 0\nmodel name\t: Test\ncore id\t\t: 0\n";
        let (_, _, _, cache_kb) = parse_cpuinfo(content);
        assert_eq!(cache_kb, 0);
    }

    #[test]
    fn parse_meminfo_memtotal_only() {
        let content = "MemTotal:       16384000 kB\n";
        let mb = parse_meminfo(content);
        assert_eq!(mb, 16000);
    }

    #[test]
    fn hardware_inventory_to_json_roundtrip_fields() {
        let hw = HardwareInventory {
            gate_name: "gate1".to_string(),
            cpu_model: "CPU Model".to_string(),
            cpu_cores: 4,
            cpu_threads: 8,
            cpu_cache_kb: 8192,
            ram_total_mb: 16384,
            gpu_name: "GPU Name".to_string(),
            gpu_vram_mb: 8192,
            gpu_driver: "550.0".to_string(),
            gpu_compute_cap: "8.9".to_string(),
            os_kernel: "6.0.0".to_string(),
            rust_version: "1.82".to_string(),
        };
        let json = hw.to_json();
        assert!(json.contains("\"gate_name\""));
        assert!(json.contains("\"cpu_model\""));
        assert!(json.contains("\"cpu_cores\""));
        assert!(json.contains("\"cpu_threads\""));
        assert!(json.contains("\"cpu_cache_kb\""));
        assert!(json.contains("\"ram_total_mb\""));
        assert!(json.contains("\"gpu_name\""));
        assert!(json.contains("\"gpu_vram_mb\""));
        assert!(json.contains("\"gpu_driver\""));
        assert!(json.contains("\"gpu_compute_cap\""));
        assert!(json.contains("\"os_kernel\""));
        assert!(json.contains("\"rust_version\""));
    }

    #[test]
    fn parse_nvidia_smi_output_invalid_vram() {
        let csv = "NVIDIA GPU, not_a_number, 550.0, 8.9";
        let (name, vram, driver, cc) = parse_nvidia_smi_output(csv);
        assert_eq!(name, "NVIDIA GPU");
        assert_eq!(vram, 0);
        assert_eq!(driver, "550.0");
        assert_eq!(cc, "8.9");
    }

    #[test]
    fn write_to_produces_valid_output() {
        let hw = HardwareInventory::from_content(
            "test-gate",
            SAMPLE_CPUINFO,
            "MemTotal:       8192000 kB\n",
            "Test GPU, 4096, 500.0, 7.5",
            "6.1.0",
        );
        let mut buf = Vec::new();
        hw.write_to(&mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("test-gate"));
        assert!(output.contains("i9-12900K"));
        assert!(output.contains("Test GPU"));
        assert!(output.contains("Hardware"));
    }

    #[test]
    fn detect_does_not_panic() {
        let _ = HardwareInventory::detect("unit-test");
    }

    #[test]
    fn query_nvidia_smi_csv_does_not_panic() {
        let _ = query_nvidia_smi_csv();
    }

    #[test]
    fn from_content_empty_strings() {
        let hw = HardwareInventory::from_content("", "", "", "", "");
        assert_eq!(hw.gate_name, "");
        assert_eq!(hw.cpu_model, "unknown");
        assert_eq!(hw.gpu_name, "N/A");
        assert_eq!(hw.ram_total_mb, 0);
    }
}
