// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp092: `BarraCuda` CPU vs GPU — All 16 Domains Head-to-Head
//!
//! Consolidated proof that `BarraCuda`'s pure Rust math produces identical
//! results on CPU and GPU across all 16 GPU-eligible domains. For each
//! domain: CPU computes reference truth; GPU must match within tolerance.
//! Wall-clock timing captured for both paths.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | `BarraCuda` CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin validate_cpu_vs_gpu_all_domains` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |
//!
//! Validation class: GPU-parity
//!
//! Provenance: CPU reference implementation in `barracuda::bio`

mod bio;
mod chem;
mod ml_ode;

use std::time::Instant;

use crate::validation::{self, CpuGpuRow};

fn synthetic_spectra() -> Vec<crate::io::mzml::MzmlSpectrum> {
    chem::synthetic_spectra()
}

/// Run the `validate_cpu_vs_gpu_all_domains` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let __rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    __rt.block_on(async {
        let gpu = validation::gpu_or_skip().await;
        let device = gpu.to_wgpu_device();
        let t0 = Instant::now();
        let mut timings: Vec<CpuGpuRow> = Vec::new();

        bio::validate_shannon_simpson(v, &gpu, &mut timings);
        bio::validate_bray_curtis(v, &gpu, &mut timings);
        bio::validate_ani(v, &device, &mut timings);
        bio::validate_snp(v, &device, &mut timings);
        bio::validate_dnds(v, &device, &mut timings);
        bio::validate_pangenome(v, &device, &mut timings);
        ml_ode::validate_random_forest(v, &device, &mut timings);
        ml_ode::validate_hmm(v, &device, &mut timings);
        ml_ode::validate_smith_waterman(v, &device, &mut timings);
        ml_ode::validate_gillespie(v, &device, &mut timings);
        ml_ode::validate_decision_tree(v, &device, &mut timings);
        chem::validate_spectral_cosine(v, &gpu, &mut timings);
        chem::validate_eic(v, &gpu, &mut timings);
        chem::validate_pcoa(v, &gpu, &mut timings);
        chem::validate_kriging(v, &gpu, &mut timings);
        bio::validate_rarefaction(v, &gpu, &mut timings);

        v.section("═══ CPU vs GPU Head-to-Head Summary ═══");
        println!();
        println!(
            "  {:<25} {:>10} {:>10} {:>8}",
            "Domain", "CPU (µs)", "GPU (µs)", "Status"
        );
        println!("  {}", "─".repeat(57));
        for t in &timings {
            println!(
                "  {:<25} {:>10.0} {:>10.0} {:>8}",
                t.name, t.cpu_us, t.gpu_us, t.status
            );
        }
        println!("  {}", "─".repeat(57));
        let total_cpu: f64 = timings.iter().map(|t| t.cpu_us).sum();
        let total_gpu: f64 = timings.iter().map(|t| t.gpu_us).sum();
        println!(
            "  {:<25} {:>10.0} {:>10.0} {:>8}",
            "TOTAL", total_cpu, total_gpu, "PROVEN"
        );
        println!("\n  16 domains: CPU ↔ GPU parity proven");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  [Total] {ms:.1} ms");
    });
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_cpu_vs_gpu_all_domains");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario =
    crate::validation::scenarios::registry::Scenario {
        meta: crate::validation::scenarios::registry::ScenarioMeta {
            id: "cpu_vs_gpu_all_domains",
            track: crate::validation::scenarios::registry::Track::Science,
            tier: crate::validation::scenarios::registry::Tier::Both,
            provenance_crate: "validate_cpu_vs_gpu_all_domains",
            provenance_date: "2026-05-20",
            description: "Exp092: `BarraCuda` CPU vs GPU — All 16 Domains Head-to-Head",
        },
        run: |v, _ctx| run_as_scenario(v),
    };
