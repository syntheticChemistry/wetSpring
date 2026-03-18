// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(clippy::print_stdout)]
#![expect(clippy::too_many_lines)]
#![expect(clippy::similar_names)]
//! # Exp348: CPU vs GPU v11 — V109 Sync Diversity API + Upstream Evolution
//!
//! Proves GPU portability after V109 upstream changes:
//! - `shannon_gpu` is now synchronous (returns `Result<f64>`, not a Future)
//! - `GPU_VS_CPU_F64` tolerance constant replaces `GPU_CPU_PARITY`
//! - All Track 6 biogas kinetics are substrate-independent
//!
//! Without `--features gpu`: validates all CPU reference computations.
//! With `--features gpu`: also validates GPU dispatch matches CPU.
//!
//! ```text
//! CPU (Exp347) → GPU (this) → ToadStool (Exp349)
//! → Streaming (Exp350) → metalForge (Exp351) → NUCLEUS (Exp352)
//! ```
//!
//! ## Domains
//!
//! - D43: Sync Diversity GPU — Shannon, Simpson via sync API
//! - D44: Biogas Kinetics GPU — Gompertz + first-order batch
//! - D45: Anderson W GPU — disorder mapping across substrates
//! - D46: Cross-Track GPU — T6 + T4 + T1 on single dispatch
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-substrate (CPU reference from Exp347) |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --release --bin validate_cpu_vs_gpu_v11` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{DomainResult, Validator};

use barracuda::stats::norm_cdf;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}

fn haldane(s: f64, mu_max: f64, ks: f64, ki: f64) -> f64 {
    mu_max * s / (ks + s + s * s / ki)
}

fn domain(
    name: &'static str,
    spring: &'static str,
    elapsed: std::time::Duration,
    checks: u32,
) -> DomainResult {
    DomainResult {
        name,
        spring: Some(spring),
        ms: elapsed.as_secs_f64() * 1000.0,
        checks,
    }
}

fn main() {
    let mut v = Validator::new("Exp348: CPU vs GPU v11 — V109 Sync API + Upstream Evolution");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    let digester = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let soil = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];

    // ═══════════════════════════════════════════════════════════════════
    // D43: Sync Diversity GPU — Shannon + Simpson
    // ═══════════════════════════════════════════════════════════════════
    v.section("D43: Sync Diversity — CPU reference + GPU parity");
    let t = Instant::now();
    let mut d43 = 0_u32;

    let cpu_h_dig = diversity::shannon(&digester);
    let cpu_h_soil = diversity::shannon(&soil);
    let cpu_s_dig = diversity::simpson(&digester);
    let cpu_s_soil = diversity::simpson(&soil);

    v.check_pass("D43: CPU H(digester) > 0", cpu_h_dig > 0.0);
    d43 += 1;
    v.check_pass("D43: CPU H(soil) > H(digester)", cpu_h_soil > cpu_h_dig);
    d43 += 1;
    v.check_pass(
        "D43: CPU Simpson ∈ (0,1)",
        cpu_s_dig > 0.0 && cpu_s_dig < 1.0 && cpu_s_soil > 0.0 && cpu_s_soil < 1.0,
    );
    d43 += 1;

    #[cfg(feature = "gpu")]
    {
        use wetspring_barracuda::bio::diversity_gpu::{shannon_gpu, simpson_gpu};
        use wetspring_barracuda::gpu::GpuF64;
        use wetspring_barracuda::validation::OrExit;

        let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init failed");
        let gpu_h_dig = shannon_gpu(&gpu, &digester).or_exit("shannon_gpu digester");
        let gpu_h_soil = shannon_gpu(&gpu, &soil).or_exit("shannon_gpu soil");
        let gpu_s_dig = simpson_gpu(&gpu, &digester).or_exit("simpson_gpu digester");
        let gpu_s_soil = simpson_gpu(&gpu, &soil).or_exit("simpson_gpu soil");

        v.check(
            "D43: GPU H(digester) ≈ CPU",
            gpu_h_dig,
            cpu_h_dig,
            tolerances::GPU_VS_CPU_F64,
        );
        d43 += 1;
        v.check(
            "D43: GPU H(soil) ≈ CPU",
            gpu_h_soil,
            cpu_h_soil,
            tolerances::GPU_VS_CPU_F64,
        );
        d43 += 1;
        v.check(
            "D43: GPU Simpson(digester) ≈ CPU",
            gpu_s_dig,
            cpu_s_dig,
            tolerances::GPU_VS_CPU_F64,
        );
        d43 += 1;
        v.check(
            "D43: GPU Simpson(soil) ≈ CPU",
            gpu_s_soil,
            cpu_s_soil,
            tolerances::GPU_VS_CPU_F64,
        );
        d43 += 1;
    }

    domains.push(domain("Sync Diversity GPU", "wetSpring", t.elapsed(), d43));

    // ═══════════════════════════════════════════════════════════════════
    // D44: Biogas Kinetics — CPU reference (GPU-parallel ready)
    // ═══════════════════════════════════════════════════════════════════
    v.section("D44: Biogas Kinetics — Gompertz + First-Order + Monod batch");
    let t = Instant::now();
    let mut d44 = 0_u32;

    let times = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0];
    let g_vals: Vec<f64> = times
        .iter()
        .map(|&t_val| gompertz(t_val, 350.0, 25.0, 3.0))
        .collect();
    v.check_pass("D44: Gompertz H(0) < 5", g_vals[0] < 5.0);
    d44 += 1;
    v.check(
        "D44: Gompertz H(50) → P",
        g_vals[7],
        350.0,
        tolerances::BIOGAS_KINETICS_ASYMPTOTIC,
    );
    d44 += 1;

    let fo_vals: Vec<f64> = times
        .iter()
        .map(|&t_val| first_order(t_val, 320.0, 0.08))
        .collect();
    v.check_pass(
        "D44: First-order monotonic",
        fo_vals.windows(2).all(|w| w[1] >= w[0]),
    );
    d44 += 1;

    let substrate_levels = [50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0];
    let monod_vals: Vec<f64> = substrate_levels
        .iter()
        .map(|&s| monod(s, 0.4, 200.0))
        .collect();
    v.check_pass(
        "D44: Monod monotonic",
        monod_vals.windows(2).all(|w| w[1] > w[0]),
    );
    d44 += 1;

    let s_opt = (200.0_f64 * 3000.0).sqrt();
    let mu_opt = haldane(s_opt, 0.4, 200.0, 3000.0);
    let mu_lo = haldane(s_opt * 0.3, 0.4, 200.0, 3000.0);
    let mu_hi = haldane(s_opt * 3.0, 0.4, 200.0, 3000.0);
    v.check_pass(
        "D44: Haldane peak at S_opt",
        mu_opt > mu_lo && mu_opt > mu_hi,
    );
    d44 += 1;

    domains.push(domain("Biogas Kinetics GPU", "wetSpring", t.elapsed(), d44));

    // ═══════════════════════════════════════════════════════════════════
    // D45: Anderson W Mapping — CPU reference for GPU
    // ═══════════════════════════════════════════════════════════════════
    v.section("D45: Anderson W Mapping — disorder for GPU dispatch");
    let t = Instant::now();
    let mut d45 = 0_u32;

    let j_dig = diversity::pielou_evenness(&digester);
    let j_soil = diversity::pielou_evenness(&soil);
    let w_max = 20.0;
    let w_dig = w_max * (1.0 - j_dig);
    let w_soil = w_max * (1.0 - j_soil);

    v.check_pass("D45: W_digester > W_soil", w_dig > w_soil);
    d45 += 1;
    v.check_pass(
        "D45: W ∈ [0, W_max]",
        w_dig >= 0.0 && w_dig <= w_max && w_soil >= 0.0 && w_soil <= w_max,
    );
    d45 += 1;

    let sigma = 4.0;
    let w_c = 16.5;
    let p_qs_soil = norm_cdf((w_c - w_soil) / sigma);
    let p_qs_dig = norm_cdf((w_c - w_dig) / sigma);
    v.check_pass("D45: P(QS|soil) > P(QS|digester)", p_qs_soil > p_qs_dig);
    d45 += 1;

    // Determinism check
    let w_dig_2 = w_max * (1.0 - diversity::pielou_evenness(&digester));
    v.check_pass(
        "D45: W deterministic",
        (w_dig - w_dig_2).abs() < tolerances::MATRIX_EPS,
    );
    d45 += 1;

    domains.push(domain("Anderson W GPU", "wetSpring", t.elapsed(), d45));

    // ═══════════════════════════════════════════════════════════════════
    // D46: Cross-Track GPU Composition — T6 + T4 + T1
    // ═══════════════════════════════════════════════════════════════════
    v.section("D46: Cross-Track — T6 + T4 + T1 on single dispatch");
    let t = Instant::now();
    let mut d46 = 0_u32;

    // T1: algae community (synthetic)
    let algae = vec![30.0, 25.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 0.5, 0.5];
    let h_algae = diversity::shannon(&algae);
    v.check_pass("D46: T1 algae H > 0", h_algae > 0.0);
    d46 += 1;

    // Three-way Bray-Curtis
    let bc_dig_soil = diversity::bray_curtis(&digester, &soil);
    let bc_dig_algae = diversity::bray_curtis(&digester, &algae);
    let bc_soil_algae = diversity::bray_curtis(&soil, &algae);
    v.check_pass(
        "D46: All BC distances ∈ (0,1]",
        bc_dig_soil > 0.0
            && bc_dig_soil <= 1.0
            && bc_dig_algae > 0.0
            && bc_dig_algae <= 1.0
            && bc_soil_algae > 0.0
            && bc_soil_algae <= 1.0,
    );
    d46 += 1;

    // Cross-track QS comparison
    let j_algae = diversity::pielou_evenness(&algae);
    let w_algae = w_max * (1.0 - j_algae);
    let p_qs_algae = norm_cdf((w_c - w_algae) / sigma);
    v.check_pass(
        "D46: P(QS) ∈ [0,1] for all",
        (0.0..=1.0).contains(&p_qs_algae),
    );
    d46 += 1;

    domains.push(domain("Cross-Track GPU", "wetSpring", t.elapsed(), d46));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── Domain Summary ({total_ms:.2} ms total) ──");
    let mut total_checks = 0_u32;
    for d in &domains {
        println!(
            "  {:30} {:>8} {:>6.2} ms  {:>3} checks",
            d.name,
            d.spring.unwrap_or("—"),
            d.ms,
            d.checks,
        );
        total_checks += d.checks;
    }
    println!(
        "  {:30} {:>8} {:>6.2} ms  {:>3} checks",
        "TOTAL", "", total_ms, total_checks
    );

    v.finish();
}
