// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

#[test]
fn parse_nvidia_smi_sample_valid() {
    let (w, t, v) = parse_nvidia_smi_sample("45.23, 62, 1024").unwrap();
    assert!((w - 45.23).abs() < f64::EPSILON);
    assert!((t - 62.0).abs() < f64::EPSILON);
    assert!((v - 1024.0).abs() < f64::EPSILON);
}

#[test]
fn parse_nvidia_smi_sample_empty() {
    assert!(parse_nvidia_smi_sample("").is_none());
}

#[test]
fn parse_nvidia_smi_sample_malformed() {
    assert!(parse_nvidia_smi_sample("not, a, number").is_none());
}

#[test]
fn parse_nvidia_smi_sample_too_few_fields() {
    assert!(parse_nvidia_smi_sample("45.0, 62").is_none());
}

#[test]
fn rapl_delta_normal() {
    let j = rapl_delta_joules(1_000_000, 3_000_000, u64::MAX);
    assert!((j - 2.0).abs() < f64::EPSILON);
}

#[test]
fn rapl_delta_wraparound() {
    let max = 10_000_000;
    let j = rapl_delta_joules(9_000_000, 1_000_000, max);
    assert!((j - 2.0).abs() < f64::EPSILON);
}

#[test]
fn rapl_delta_zero() {
    let j = rapl_delta_joules(5_000_000, 5_000_000, u64::MAX);
    assert!(j.abs() < f64::EPSILON);
}

#[test]
fn compute_gpu_energy_empty() {
    let report = compute_gpu_energy(&[], 1.0);
    assert_eq!(report.gpu_samples, 0);
    assert!(report.gpu_joules.abs() < f64::EPSILON);
}

#[test]
fn compute_gpu_energy_single_sample() {
    let now = Instant::now();
    let samples = vec![GpuSample {
        watts: 100.0,
        temp_c: 60.0,
        vram_mib: 4096.0,
        timestamp: now,
    }];
    let report = compute_gpu_energy(&samples, 2.0);
    assert_eq!(report.gpu_samples, 1);
    assert!((report.gpu_joules - 200.0).abs() < f64::EPSILON);
    assert!((report.gpu_watts_avg - 100.0).abs() < f64::EPSILON);
    assert!((report.gpu_watts_peak - 100.0).abs() < f64::EPSILON);
    assert!((report.gpu_temp_peak_c - 60.0).abs() < f64::EPSILON);
    assert!((report.gpu_vram_peak_mib - 4096.0).abs() < f64::EPSILON);
}

#[test]
fn compute_gpu_energy_two_samples() {
    let t0 = Instant::now();
    let t1 = t0 + std::time::Duration::from_secs(1);
    let samples = vec![
        GpuSample {
            watts: 100.0,
            temp_c: 55.0,
            vram_mib: 2048.0,
            timestamp: t0,
        },
        GpuSample {
            watts: 200.0,
            temp_c: 70.0,
            vram_mib: 4096.0,
            timestamp: t1,
        },
    ];
    let report = compute_gpu_energy(&samples, 1.0);
    assert_eq!(report.gpu_samples, 2);
    assert!((report.gpu_joules - 150.0).abs() < f64::EPSILON);
    assert!((report.gpu_watts_avg - 150.0).abs() < f64::EPSILON);
    assert!((report.gpu_watts_peak - 200.0).abs() < f64::EPSILON);
    assert!((report.gpu_temp_peak_c - 70.0).abs() < f64::EPSILON);
    assert!((report.gpu_vram_peak_mib - 4096.0).abs() < f64::EPSILON);
}

#[test]
fn energy_report_to_json_format() {
    let r = EnergyReport {
        cpu_joules: 1.5,
        gpu_joules: 3.25,
        gpu_watts_avg: 150.0,
        gpu_watts_peak: 200.0,
        gpu_temp_peak_c: 72.0,
        gpu_vram_peak_mib: 4096.0,
        gpu_samples: 10,
    };
    let json = r.to_json();
    assert!(json.contains("\"cpu_joules\": 1.5000"));
    assert!(json.contains("\"gpu_joules\": 3.2500"));
    assert!(json.contains("\"gpu_samples\": 10"));
}

#[test]
#[ignore = "requires nvidia-smi and RAPL"]
fn power_monitor_start_stop() {
    let monitor = PowerMonitor::start();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let _ = monitor.stop();
}

/// Runs on any system — nvidia-smi and RAPL may be absent; exercises fallback paths.
#[test]
fn power_monitor_start_stop_no_hardware_required() {
    let monitor = PowerMonitor::start();
    let report = monitor.stop();
    // When nvidia-smi is unavailable, gpu_samples is 0; when RAPL is absent, cpu_joules is 0.
    assert!(report.gpu_watts_avg >= 0.0);
    assert!(report.gpu_watts_peak >= 0.0);
    assert!(report.gpu_temp_peak_c >= 0.0);
    assert!(report.gpu_vram_peak_mib >= 0.0);
}

#[test]
fn power_monitor_stop_with_zero_gpu_samples() {
    let monitor = PowerMonitor::start();
    let report = monitor.stop();
    // On CI / systems without nvidia-smi, gpu_samples will be 0; report should be valid.
    assert!(report.cpu_joules >= 0.0);
    assert!(report.gpu_joules >= 0.0);
}

#[test]
fn spawn_nvidia_smi_handles_missing_binary() {
    let samples: Arc<Mutex<Vec<GpuSample>>> = Arc::new(Mutex::new(Vec::new()));
    let (smi_child, reader_handle) = spawn_nvidia_smi_poller(samples);
    // When nvidia-smi is unavailable: (None, None). When available: (Some, Some). No panic.
    let _ = (smi_child, reader_handle);
}

#[test]
fn compute_gpu_energy_three_samples() {
    let t0 = Instant::now();
    let t1 = t0 + std::time::Duration::from_millis(500);
    let t2 = t1 + std::time::Duration::from_millis(500);
    let samples = vec![
        GpuSample {
            watts: 100.0,
            temp_c: 50.0,
            vram_mib: 1024.0,
            timestamp: t0,
        },
        GpuSample {
            watts: 150.0,
            temp_c: 60.0,
            vram_mib: 2048.0,
            timestamp: t1,
        },
        GpuSample {
            watts: 200.0,
            temp_c: 70.0,
            vram_mib: 4096.0,
            timestamp: t2,
        },
    ];
    let report = compute_gpu_energy(&samples, 1.0);
    assert_eq!(report.gpu_samples, 3);
    assert!((report.gpu_watts_avg - 150.0).abs() < f64::EPSILON);
    assert!((report.gpu_watts_peak - 200.0).abs() < f64::EPSILON);
    assert!((report.gpu_temp_peak_c - 70.0).abs() < f64::EPSILON);
    assert!((report.gpu_vram_peak_mib - 4096.0).abs() < f64::EPSILON);
    let expected_joules = 125.0_f64.mul_add(0.5, 175.0 * 0.5);
    assert!((report.gpu_joules - expected_joules).abs() < 0.01);
}

#[test]
fn compute_gpu_energy_peak_tracking() {
    let t0 = Instant::now();
    let t1 = t0 + std::time::Duration::from_secs(1);
    let t2 = t1 + std::time::Duration::from_secs(1);
    let samples = vec![
        GpuSample {
            watts: 50.0,
            temp_c: 40.0,
            vram_mib: 512.0,
            timestamp: t0,
        },
        GpuSample {
            watts: 250.0,
            temp_c: 85.0,
            vram_mib: 8192.0,
            timestamp: t1,
        },
        GpuSample {
            watts: 100.0,
            temp_c: 60.0,
            vram_mib: 2048.0,
            timestamp: t2,
        },
    ];
    let report = compute_gpu_energy(&samples, 2.0);
    assert!((report.gpu_watts_peak - 250.0).abs() < f64::EPSILON);
    assert!((report.gpu_temp_peak_c - 85.0).abs() < f64::EPSILON);
    assert!((report.gpu_vram_peak_mib - 8192.0).abs() < f64::EPSILON);
}

#[test]
fn energy_report_default() {
    let r = EnergyReport::default();
    assert!(r.cpu_joules.abs() < f64::EPSILON);
    assert!(r.gpu_joules.abs() < f64::EPSILON);
    assert!(r.gpu_watts_avg.abs() < f64::EPSILON);
    assert!(r.gpu_watts_peak.abs() < f64::EPSILON);
    assert!(r.gpu_temp_peak_c.abs() < f64::EPSILON);
    assert!(r.gpu_vram_peak_mib.abs() < f64::EPSILON);
    assert_eq!(r.gpu_samples, 0);
}

#[test]
fn energy_report_to_json_contains_all_fields() {
    let r = EnergyReport {
        cpu_joules: 1.5,
        gpu_joules: 3.25,
        gpu_watts_avg: 150.0,
        gpu_watts_peak: 200.0,
        gpu_temp_peak_c: 72.0,
        gpu_vram_peak_mib: 4096.0,
        gpu_samples: 10,
    };
    let json = r.to_json();
    assert!(json.contains("\"cpu_joules\": 1.5000"));
    assert!(json.contains("\"gpu_joules\": 3.2500"));
    assert!(json.contains("\"gpu_watts_avg\": 150.00"));
    assert!(json.contains("\"gpu_watts_peak\": 200.00"));
    assert!(json.contains("\"gpu_temp_peak_c\": 72.0"));
    assert!(json.contains("\"gpu_vram_peak_mib\": 4096.0"));
    assert!(json.contains("\"gpu_samples\": 10"));
}

#[test]
fn parse_nvidia_smi_sample_with_whitespace() {
    let (w, t, v) = parse_nvidia_smi_sample("  45.23 ,  62 ,  1024  ").unwrap();
    assert!((w - 45.23).abs() < f64::EPSILON);
    assert!((t - 62.0).abs() < f64::EPSILON);
    assert!((v - 1024.0).abs() < f64::EPSILON);
}

#[test]
fn parse_nvidia_smi_sample_negative_values() {
    let (w, t, v) = parse_nvidia_smi_sample("-5.0, -10, 512").unwrap();
    assert!((w - (-5.0)).abs() < f64::EPSILON);
    assert!((t - (-10.0)).abs() < f64::EPSILON);
    assert!((v - 512.0).abs() < f64::EPSILON);
}

#[test]
fn rapl_read_does_not_panic() {
    let _ = read_rapl_energy_uj();
    let _ = read_rapl_max_energy_uj();
}

#[test]
fn parse_nvidia_smi_sample_extra_fields() {
    let (w, t, v) = parse_nvidia_smi_sample("100.0, 70, 2048, extra").unwrap();
    assert!((w - 100.0).abs() < f64::EPSILON);
    assert!((t - 70.0).abs() < f64::EPSILON);
    assert!((v - 2048.0).abs() < f64::EPSILON);
}

#[test]
fn rapl_delta_large_values() {
    let j = rapl_delta_joules(999_999_000_000, 1_000_001_000_000, u64::MAX);
    assert!((j - 2.0).abs() < f64::EPSILON);
}

#[test]
fn energy_report_default_zeroes() {
    let r = EnergyReport::default();
    assert!(r.gpu_watts_avg.abs() < f64::EPSILON);
    assert_eq!(r.gpu_samples, 0);
}
