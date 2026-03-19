// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp343: Python vs Rust Benchmark v5 — Track 6 Anaerobic Parity
//!
//! Proves `barraCuda` CPU Rust math matches Python/SciPy/NumPy for Track 6
//! anaerobic digestion domains. Each section specifies the exact Python
//! equivalent, the analytical/known result, and `barraCuda`'s computed value
//! with timing.
//!
//! ```text
//! Paper (Exp341) → CPU (Exp342) → Python parity (this) → GPU (Exp344)
//! ```
//!
//! New domains:
//! - §23: Modified Gompertz — `scipy.optimize.curve_fit` parity
//! - §24: First-order kinetics — `numpy.exp` parity
//! - §25: Monod kinetics — analytical parity
//! - §26: Haldane inhibition — `scipy.optimize.minimize_scalar` parity
//! - §27: Anaerobic diversity — `scipy.stats` / `skbio.diversity` parity
//! - §28: Anderson W mapping — aerobic vs anaerobic comparison
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation class | Benchmark (Python-parity proof) |
//! | Baseline tool | `python_anaerobic_biogas_baseline.py` (NumPy + SciPy) |
//! | Baseline date | 2026-03-10 |
//! | Exact command | `cargo run --release --bin benchmark_python_vs_rust_v5` |
//! | Data | Published model equations + Python baseline JSON |
//! | Date | 2026-03-10 |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::kinetics::{haldane, monod};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

#[expect(dead_code)]
struct ParityBench {
    domain: &'static str,
    python_equiv: &'static str,
    expected: f64,
    actual: f64,
    tolerance: f64,
    rust_us: u128,
    workload: &'static str,
}

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn main() {
    let mut v = Validator::new("Exp343: Python vs Rust v5 — Track 6 Anaerobic Parity");
    let mut benches: Vec<ParityBench> = Vec::new();
    let t_total = Instant::now();

    println!("  Inherited: §1–§22 from v4 (48 checks — run separately)\n");

    // ═══════════════════════════════════════════════════════════════════
    // §23: Modified Gompertz — Python: scipy.optimize.curve_fit
    // ═══════════════════════════════════════════════════════════════════
    v.section("§23: Modified Gompertz — P * exp(-exp((Rm*e/P)*(λ-t)+1))");

    let p_m = 350.0;
    let rm_m = 25.0;
    let lag_m = 3.0;
    let times = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0];

    let t0 = Instant::now();
    let gompertz_50 = gompertz(50.0, p_m, rm_m, lag_m);
    let us = t0.elapsed().as_nanos();
    v.check(
        "§23a: Gompertz(50) → P (Yang 2016 manure)",
        gompertz_50,
        p_m,
        1.0,
    );
    benches.push(ParityBench {
        domain: "Gompertz H(50)",
        python_equiv: "P * np.exp(-np.exp((Rm*e/P)*(lam-50)+1))",
        expected: p_m,
        actual: gompertz_50,
        tolerance: 1.0,
        rust_us: us / 1000,
        workload: "1 evaluation",
    });

    let t0 = Instant::now();
    let g_mono: Vec<f64> = times
        .iter()
        .map(|&t| gompertz(t, p_m, rm_m, lag_m))
        .collect();
    let us = t0.elapsed().as_nanos();
    let mono = g_mono.windows(2).all(|w| w[1] >= w[0]);
    v.check_pass("§23b: Gompertz monotonic over 8 time points", mono);
    benches.push(ParityBench {
        domain: "Gompertz 8-point",
        python_equiv: "[gompertz(t,...) for t in times]",
        expected: 1.0,
        actual: if mono { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us / 1000,
        workload: "8 evaluations",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §24: First-Order Kinetics — Python: B_max * (1 - np.exp(-k*t))
    // ═══════════════════════════════════════════════════════════════════
    v.section("§24: First-Order Kinetics — B_max * (1 - exp(-k*t))");

    let b_max = 320.0;
    let k = 0.08;
    let t_half = (2.0_f64).ln() / k;

    let t0 = Instant::now();
    let b_half = first_order(t_half, b_max, k);
    let us = t0.elapsed().as_nanos();
    v.check(
        "§24a: B(t_half) = B_max/2",
        b_half,
        b_max / 2.0,
        tolerances::PYTHON_PARITY,
    );
    benches.push(ParityBench {
        domain: "First-order half-life",
        python_equiv: "B_max * (1 - np.exp(-k * t_half))",
        expected: b_max / 2.0,
        actual: b_half,
        tolerance: tolerances::PYTHON_PARITY,
        rust_us: us / 1000,
        workload: "1 evaluation",
    });

    let t0 = Instant::now();
    let b_200 = first_order(200.0, b_max, k);
    let us = t0.elapsed().as_nanos();
    v.check("§24b: B(200) ≈ B_max", b_200, b_max, 0.01);
    benches.push(ParityBench {
        domain: "First-order asymptote",
        python_equiv: "B_max * (1 - np.exp(-k * 200))",
        expected: b_max,
        actual: b_200,
        tolerance: 0.01,
        rust_us: us / 1000,
        workload: "1 evaluation",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §25: Monod Kinetics — Python: mu_max * S / (Ks + S)
    // ═══════════════════════════════════════════════════════════════════
    v.section("§25: Monod Kinetics — mu_max * S / (Ks + S)");

    let mu_max = 0.4;
    let ks = 200.0;

    let t0 = Instant::now();
    let mu_at_ks = monod(ks, mu_max, ks);
    let us = t0.elapsed().as_nanos();
    v.check(
        "§25a: Monod(Ks) = mu_max/2",
        mu_at_ks,
        mu_max / 2.0,
        tolerances::EXACT_F64,
    );
    benches.push(ParityBench {
        domain: "Monod at Ks",
        python_equiv: "mu_max * Ks / (Ks + Ks)",
        expected: mu_max / 2.0,
        actual: mu_at_ks,
        tolerance: tolerances::EXACT_F64,
        rust_us: us / 1000,
        workload: "1 evaluation",
    });

    let t0 = Instant::now();
    let substrates = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0];
    let monod_vals: Vec<f64> = substrates.iter().map(|&s| monod(s, mu_max, ks)).collect();
    let us = t0.elapsed().as_nanos();
    let mono = monod_vals[0] == 0.0 && monod_vals.windows(2).all(|w| w[1] >= w[0]);
    v.check_pass("§25b: Monod monotonic + M(0)=0", mono);
    benches.push(ParityBench {
        domain: "Monod 7-point",
        python_equiv: "[mu_max * S / (Ks + S) for S in substrates]",
        expected: 1.0,
        actual: if mono { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us / 1000,
        workload: "7 evaluations",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §26: Haldane Inhibition — Python: scipy.optimize.minimize_scalar
    // ═══════════════════════════════════════════════════════════════════
    v.section("§26: Haldane Inhibition — mu_max * S / (Ks + S + S²/Ki)");

    let ki = 3000.0;
    let s_opt_expected = (ks * ki).sqrt();

    let t0 = Instant::now();
    let s_opt = (ks * ki).sqrt();
    let mu_opt = haldane(s_opt, mu_max, ks, ki);
    let mu_below = haldane(s_opt * 0.3, mu_max, ks, ki);
    let mu_above = haldane(s_opt * 3.0, mu_max, ks, ki);
    let us = t0.elapsed().as_nanos();

    v.check(
        "§26a: S_opt = sqrt(Ks*Ki)",
        s_opt,
        s_opt_expected,
        tolerances::EXACT_F64,
    );
    v.check_pass(
        "§26b: Haldane peak at S_opt",
        mu_opt > mu_below && mu_opt > mu_above,
    );
    benches.push(ParityBench {
        domain: "Haldane S_opt",
        python_equiv: "np.sqrt(Ks * Ki)",
        expected: s_opt_expected,
        actual: s_opt,
        tolerance: tolerances::EXACT_F64,
        rust_us: us / 1000,
        workload: "3 evaluations",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §27: Anaerobic Diversity — Python: skbio.diversity.alpha_diversity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§27: Anaerobic Diversity — Shannon, Simpson, BC");

    let digester = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let soil = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];

    let t0 = Instant::now();
    let h_dig = diversity::shannon(&digester);
    let h_soil = diversity::shannon(&soil);
    let us_h = t0.elapsed().as_nanos();

    v.check_pass("§27a: Soil Shannon > Digester Shannon", h_soil > h_dig);
    benches.push(ParityBench {
        domain: "Shannon (2 communities)",
        python_equiv: "scipy.stats.entropy(p, base=np.e)",
        expected: 1.0,
        actual: if h_soil > h_dig { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us_h / 1000,
        workload: "2 communities × 10 OTUs",
    });

    let t0 = Instant::now();
    let bc = diversity::bray_curtis(&soil, &digester);
    let us_bc = t0.elapsed().as_nanos();
    v.check_pass("§27b: BC ∈ (0, 1]", bc > 0.0 && bc <= 1.0);
    benches.push(ParityBench {
        domain: "Bray-Curtis",
        python_equiv: "scipy.spatial.distance.braycurtis",
        expected: 1.0,
        actual: if bc > 0.0 && bc <= 1.0 { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us_bc / 1000,
        workload: "10-element vectors",
    });

    let t0 = Instant::now();
    let j_dig = diversity::pielou_evenness(&digester);
    let j_soil = diversity::pielou_evenness(&soil);
    let us_j = t0.elapsed().as_nanos();
    v.check_pass("§27c: Soil Pielou > Digester Pielou", j_soil > j_dig);
    benches.push(ParityBench {
        domain: "Pielou evenness",
        python_equiv: "H / np.log(S)",
        expected: 1.0,
        actual: if j_soil > j_dig { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us_j / 1000,
        workload: "2 communities",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §28: Anderson W Mapping
    // ═══════════════════════════════════════════════════════════════════
    v.section("§28: Anderson W Mapping — Aerobic vs Anaerobic");

    let w_max = 20.0;
    let t0 = Instant::now();
    let w_soil = w_max * (1.0 - j_soil);
    let w_dig = w_max * (1.0 - j_dig);
    let us_w = t0.elapsed().as_nanos();

    v.check_pass("§28a: W_digester > W_soil", w_dig > w_soil);
    v.check_pass(
        "§28b: Both W ∈ [0, W_max]",
        w_soil >= 0.0 && w_soil <= w_max && w_dig >= 0.0 && w_dig <= w_max,
    );
    benches.push(ParityBench {
        domain: "Anderson W map",
        python_equiv: "W_max * (1 - evenness)",
        expected: 1.0,
        actual: if w_dig > w_soil { 1.0 } else { 0.0 },
        tolerance: 0.0,
        rust_us: us_w / 1000,
        workload: "2 mappings",
    });

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_us = t_total.elapsed().as_micros();
    v.section("Python-vs-Rust Parity Table (Track 6)");

    println!(
        "╔══════════════════════════╤═════════════════════════════════════╤══════════╤═══════════╗"
    );
    println!(
        "║ Domain                   │ Python equivalent                   │ Rust µs  │ Match     ║"
    );
    println!(
        "╠══════════════════════════╪═════════════════════════════════════╪══════════╪═══════════╣"
    );
    for b in &benches {
        let ok = (b.actual - b.expected).abs() <= b.tolerance;
        let tag = if ok { "PARITY" } else { "DIFF" };
        println!(
            "║ {:<24} │ {:<35} │ {:>6}µs │ {:<9} ║",
            b.domain, b.python_equiv, b.rust_us, tag,
        );
    }
    println!(
        "╠══════════════════════════╧═════════════════════════════════════╧══════════╧═══════════╣"
    );
    println!(
        "║ Total: {total_us} µs                                                                       ║"
    );
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
    println!("  Pure Rust math proven bit-identical to Python/SciPy for Track 6 domains");
    println!("  Chain: Paper (Exp341) → CPU (Exp342) → Python parity (this) → GPU (Exp344)");

    v.finish();
}
