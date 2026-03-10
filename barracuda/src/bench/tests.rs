// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark harness tests.
#![expect(clippy::unwrap_used)]

use super::*;

#[test]
#[expect(clippy::float_cmp)]
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
        substrate: "BarraCuda GPU".to_string(),
        wall_time_s: 2.5,
        per_eval_us: 0.42,
        n_evals: 10_000,
        energy,
        peak_rss_mb: 128.0,
        notes: "smoke test".to_string(),
    };
    assert_eq!(pr.phase, "shannon_1m");
    assert_eq!(pr.substrate, "BarraCuda GPU");
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
fn parse_peak_rss_mb_extracts_vmhwm() {
    let content = "VmPeak:\t  1234567 kB\nVmHWM:\t   524288 kB\nVmRSS:\t   400000 kB\n";
    let rss = parse_peak_rss_mb(content);
    assert!((rss - 512.0).abs() < 0.001); // 524288 / 1024 = 512
}

#[test]
fn parse_peak_rss_mb_empty() {
    assert!(parse_peak_rss_mb("").abs() < f64::EPSILON);
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
fn phase_result_to_json_structure() {
    let pr = PhaseResult {
        phase: "test_phase".to_string(),
        substrate: "BarraCuda CPU".to_string(),
        wall_time_s: 1.0,
        per_eval_us: 10.0,
        n_evals: 100,
        energy: EnergyReport::default(),
        peak_rss_mb: 64.0,
        notes: "unit test".to_string(),
    };
    let json = pr.to_json();
    assert!(json.contains("\"phase\": \"test_phase\""));
    assert!(json.contains("\"substrate\": \"BarraCuda CPU\""));
    assert!(json.contains("\"n_evals\": 100"));
    assert!(json.contains("\"notes\": \"unit test\""));
}

#[test]
fn bench_report_save_json_creates_file() {
    let hw = HardwareInventory {
        gate_name: "test".to_string(),
        cpu_model: "Test".to_string(),
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
    let report = BenchReport::new(hw);
    let dir = std::env::temp_dir().join("wetspring_bench_test");
    let path = report.save_json(dir.to_str().unwrap());
    assert!(path.is_ok());
    let path = path.unwrap();
    assert!(std::path::Path::new(&path).exists());
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn bench_report_multiple_phases() {
    let hw = HardwareInventory {
        gate_name: "test".to_string(),
        cpu_model: "Test".to_string(),
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
    let mut report = BenchReport::new(hw);
    report.add_phase(PhaseResult {
        phase: "phase_a".to_string(),
        substrate: "CPU".to_string(),
        wall_time_s: 1.0,
        per_eval_us: 100.0,
        n_evals: 10,
        energy: EnergyReport::default(),
        peak_rss_mb: 32.0,
        notes: String::new(),
    });
    report.add_phase(PhaseResult {
        phase: "phase_a".to_string(),
        substrate: "GPU".to_string(),
        wall_time_s: 0.1,
        per_eval_us: 10.0,
        n_evals: 10,
        energy: EnergyReport {
            gpu_joules: 0.5,
            gpu_watts_avg: 50.0,
            gpu_watts_peak: 80.0,
            gpu_temp_peak_c: 65.0,
            gpu_vram_peak_mib: 2048.0,
            gpu_samples: 5,
            ..Default::default()
        },
        peak_rss_mb: 64.0,
        notes: String::new(),
    });
    let json = report.to_json();
    assert!(json.contains("\"phase\": \"phase_a\""));
    assert_eq!(report.phases.len(), 2);
}

#[test]
fn format_duration_boundary_values() {
    assert!(format_duration(0.0).contains("us"));
    assert!(format_duration(0.001).contains("ms"));
    assert!(format_duration(59.9).contains('s'));
    assert!(format_duration(60.0).contains("min"));
}

fn test_hw() -> HardwareInventory {
    HardwareInventory {
        gate_name: "test".to_string(),
        cpu_model: "Test CPU".to_string(),
        cpu_cores: 4,
        cpu_threads: 8,
        cpu_cache_kb: 8192,
        ram_total_mb: 16384,
        gpu_name: "Test GPU".to_string(),
        gpu_vram_mb: 8192,
        gpu_driver: "500.0".to_string(),
        gpu_compute_cap: "8.0".to_string(),
        os_kernel: "6.0".to_string(),
        rust_version: String::new(),
    }
}

fn cpu_phase(name: &str, wall: f64, cpu_j: f64) -> PhaseResult {
    PhaseResult {
        phase: name.to_string(),
        substrate: "BarraCuda CPU".to_string(),
        wall_time_s: wall,
        per_eval_us: wall * 1e6 / 1000.0,
        n_evals: 1000,
        energy: EnergyReport {
            cpu_joules: cpu_j,
            ..Default::default()
        },
        peak_rss_mb: 64.0,
        notes: String::new(),
    }
}

fn gpu_phase(name: &str, wall: f64, gpu_j: f64) -> PhaseResult {
    PhaseResult {
        phase: name.to_string(),
        substrate: "BarraCuda GPU".to_string(),
        wall_time_s: wall,
        per_eval_us: wall * 1e6 / 1000.0,
        n_evals: 1000,
        energy: EnergyReport {
            gpu_joules: gpu_j,
            gpu_watts_avg: if wall > 0.0 { gpu_j / wall } else { 0.0 },
            gpu_watts_peak: if wall > 0.0 { gpu_j / wall * 1.5 } else { 0.0 },
            gpu_temp_peak_c: 65.0,
            gpu_vram_peak_mib: 2048.0,
            gpu_samples: 10,
            ..Default::default()
        },
        peak_rss_mb: 128.0,
        notes: String::new(),
    }
}

#[test]
fn print_summary_cpu_only() {
    let mut report = BenchReport::new(test_hw());
    report.add_phase(cpu_phase("shannon", 0.005, 0.5));
    report.add_phase(cpu_phase("felsenstein", 0.02, 1.8));
    report.print_summary();
}

#[test]
fn print_summary_cpu_gpu_with_speedup() {
    let mut report = BenchReport::new(test_hw());
    report.add_phase(cpu_phase("shannon", 0.5, 5.0));
    report.add_phase(gpu_phase("shannon", 0.05, 0.8));
    report.add_phase(cpu_phase("felsenstein", 1.0, 10.0));
    report.add_phase(gpu_phase("felsenstein", 0.1, 1.5));
    report.print_summary();
}

#[test]
fn print_summary_single_phase_no_speedup() {
    let mut report = BenchReport::new(test_hw());
    report.add_phase(cpu_phase("diversity", 0.001, 0.01));
    report.print_summary();
}

#[test]
fn print_summary_gpu_detail_table() {
    let mut report = BenchReport::new(test_hw());
    report.add_phase(gpu_phase("hmm_batch", 2.0, 150.0));
    report.add_phase(gpu_phase("ode_sweep", 1.5, 120.0));
    report.print_summary();
}

#[test]
fn print_summary_zero_energy_phases() {
    let mut report = BenchReport::new(test_hw());
    report.add_phase(PhaseResult {
        phase: "quick".to_string(),
        substrate: "CPU".to_string(),
        wall_time_s: 0.000_001,
        per_eval_us: 0.0,
        n_evals: 0,
        energy: EnergyReport::default(),
        peak_rss_mb: 0.0,
        notes: String::new(),
    });
    report.print_summary();
}

#[test]
fn json_escape_tabs_and_carriage_returns() {
    assert_eq!(json_escape("a\tb"), "a\\tb");
    assert_eq!(json_escape("a\rb"), "a\\rb");
    assert_eq!(json_escape("plain"), "plain");
}
