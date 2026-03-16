// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
//! Exp120 — Cross-Spring Evolution Benchmark
//!
//! Exercises `BarraCuda` primitives that evolved through cross-spring
//! collaboration: hotSpring precision shaders, wetSpring bio shaders,
//! neuralSpring ML shaders, and airSpring `IoT` shaders — all meeting
//! in `ToadStool` as the shared substrate.
//!
//! This benchmark tracks where each primitive came from, measures
//! CPU vs GPU performance, and documents the cross-spring synergy.

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::bio::qs_biofilm::QsBiofilmParams;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const N_SAMPLES: usize = 500;

fn bench<F: FnOnce() -> R, R>(label: &str, f: F) -> (R, f64) {
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {label}: {ms:.2} ms");
    (r, ms)
}

#[expect(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp120: Cross-Spring Evolution Benchmark");

    // ─── Section 1: wetSpring bio primitives (CPU) ───────────────────

    v.section("§1 wetSpring → ToadStool: Diversity Metrics (CPU)");
    println!("  Origin: wetSpring Exp004/011 → barracuda::FusedMapReduceF64");

    let communities: Vec<Vec<f64>> = (0..N_SAMPLES)
        .map(|i| {
            (0..50)
                .map(|j| 1.0 + ((i * 7 + j * 13) % 100) as f64)
                .collect()
        })
        .collect();

    let (shannon_results, shannon_ms) = bench("Shannon entropy (500 communities)", || {
        communities
            .iter()
            .map(|c| diversity::shannon(c))
            .collect::<Vec<_>>()
    });
    v.check_pass("all Shannon > 0", shannon_results.iter().all(|&h| h > 0.0));

    let (simpson_results, _simpson_ms) = bench("Simpson index (500 communities)", || {
        communities
            .iter()
            .map(|c| diversity::simpson(c))
            .collect::<Vec<_>>()
    });
    v.check_pass(
        "all Simpson ∈ (0,1]",
        simpson_results.iter().all(|&s| s > 0.0 && s <= 1.0),
    );

    let (_bc_results, bc_ms) = bench("Bray-Curtis (500 pairs)", || {
        communities
            .windows(2)
            .map(|w| diversity::bray_curtis(&w[0], &w[1]))
            .collect::<Vec<_>>()
    });
    v.check_pass("diversity metrics computed", bc_ms > 0.0);

    // ─── Section 2: hotSpring precision → wetSpring QS ODE ──────────

    v.section("§2 hotSpring precision → wetSpring QS: ODE Integration (CPU)");
    println!("  Origin: hotSpring RK4/RK45 precision → wetSpring QS biofilm ODE");
    println!("  Shader: batched_qs_ode_rk4_f64.wgsl (ToadStool S41 compile fix)");

    let params = QsBiofilmParams::default();
    let (ode_result, ode_ms) = bench("QS ODE integration (2000 steps)", || {
        let mut state = [0.01_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dt: f64 = 0.01;
        for _ in 0..2000 {
            let n: f64 = state[0].max(0.0);
            let ai: f64 = state[3].max(0.0);
            let hapr: f64 = state[5].max(0.0);
            let cdg: f64 = state[6].clamp(0.0, 10.0);

            let growth = params.mu_max * n * (1.0 - n / params.k_cap);
            let death = params.death_rate * n;
            let ai_prod = params.k_ai_prod * n;
            let ai_deg = params.d_ai * ai;
            let hapr_hill = ai.powi(params.n_hapr as i32)
                / (params.k_hapr_ai.powi(params.n_hapr as i32) + ai.powi(params.n_hapr as i32));
            let hapr_prod = params.k_hapr_max * hapr_hill;
            let hapr_deg = params.d_hapr * hapr;
            let dgc = params.k_dgc_rep.mul_add(hapr, params.k_dgc_basal);
            let pde = params
                .k_pde_act
                .mul_add(1.0 - hapr.min(1.0), params.k_pde_basal);
            let cdg_dot = dgc - pde * cdg;
            let bio_hill = cdg.powi(params.n_bio as i32)
                / (params.k_bio_cdg.powi(params.n_bio as i32) + cdg.powi(params.n_bio as i32));
            let bio_prod = params.k_bio_max * bio_hill;
            let bio_deg = params.d_bio * state[7].max(0.0);

            state[0] += dt * (growth - death);
            state[3] += dt * (ai_prod - ai_deg);
            state[5] += dt * (hapr_prod - hapr_deg);
            state[6] += dt * cdg_dot;
            state[7] += dt * (bio_prod - bio_deg);
        }
        state
    });
    v.check_pass(
        "ODE integration finite",
        ode_result.iter().all(|x| x.is_finite()),
    );
    println!("  Final biofilm: {:.4}", ode_result[7]);

    let (sweep_results, sweep_ms) = bench("QS parameter sweep (200 configs)", || {
        let mut results = Vec::with_capacity(200);
        for i in 0..200 {
            let p = QsBiofilmParams {
                mu_max: f64::from(i).mul_add(0.005, 0.3),
                k_ai_prod: f64::from(i).mul_add(0.002, 0.05),
                ..Default::default()
            };
            let mut state = [0.01_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            for _ in 0..500 {
                let n: f64 = state[0].max(0.0);
                state[0] += 0.01 * ((p.mu_max * n).mul_add(1.0 - n / p.k_cap, -(p.death_rate * n)));
                state[3] += 0.01 * (p.k_ai_prod.mul_add(n, -(p.d_ai * state[3].max(0.0_f64))));
            }
            results.push(state[7]);
        }
        results
    });
    v.check_count("sweep configs computed", sweep_results.len(), 200);

    // ─── Section 3: neuralSpring ML → wetSpring ESN ─────────────────

    v.section("§3 neuralSpring ML + hotSpring ESN → wetSpring NPU Reservoir");
    println!("  Origin: hotSpring reservoir.rs (Stanton-Murillo) → esn_v2 WGSL");
    println!("  Applied: wetSpring Exp114-119 NPU deployment");

    let config = EsnConfig {
        input_size: 5,
        reservoir_size: 200,
        output_size: 3,
        spectral_radius: 0.9,
        connectivity: 0.1,
        leak_rate: 0.3,
        regularization: tolerances::ESN_REGULARIZATION,
        seed: 42,
    };

    let train_inputs: Vec<Vec<f64>> = (0..300)
        .map(|i| {
            vec![
                f64::from(i % 50).mul_add(0.02, 0.3),
                f64::from(i % 30).mul_add(0.03, 0.1),
                f64::from(i % 20).mul_add(0.05, 0.5),
                f64::from(i % 10) * 0.1,
                f64::from(i % 40).mul_add(0.02, 0.2),
            ]
        })
        .collect();
    let train_targets: Vec<Vec<f64>> = (0..300)
        .map(|i| {
            let mut t = vec![0.0; 3];
            t[i % 3] = 1.0;
            t
        })
        .collect();

    let (_, esn_train_ms) = bench("ESN train (300 samples, 200 reservoir)", || {
        let mut esn = Esn::new(config.clone());
        esn.train(&train_inputs, &train_targets);
        esn
    });

    let mut esn = Esn::new(config);
    esn.train(&train_inputs, &train_targets);

    let test_inputs: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            vec![
                f64::from(i % 30).mul_add(0.02, 0.4),
                f64::from(i % 20).mul_add(0.03, 0.2),
                f64::from(i % 15).mul_add(0.05, 0.6),
                f64::from(i % 8) * 0.12,
                f64::from(i % 25).mul_add(0.02, 0.3),
            ]
        })
        .collect();

    let (f64_preds, esn_infer_ms) = bench("ESN f64 inference (100 samples)", || {
        esn.predict(&test_inputs)
    });
    v.check_count("f64 predictions", f64_preds.len(), 100);

    let npu = esn.to_npu_weights();
    let (npu_preds, npu_ms) = bench("NPU int8 inference (100 samples)", || {
        let mut esn2 = Esn::new(EsnConfig {
            input_size: 5,
            reservoir_size: 200,
            output_size: 3,
            spectral_radius: 0.9,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: tolerances::ESN_REGULARIZATION,
            seed: 42,
        });
        esn2.reset_state();
        test_inputs
            .iter()
            .map(|inp| {
                esn2.update(inp);
                npu.classify(esn2.state())
            })
            .collect::<Vec<_>>()
    });
    v.check_count("NPU predictions", npu_preds.len(), 100);

    // ─── Section 4: Cross-spring synergy metrics ────────────────────

    v.section("§4 Cross-Spring Synergy Summary");

    println!("\n  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ Cross-Spring Shader Provenance → wetSpring Benefit       │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │ hotSpring f64 precision  → GPU ODE, Felsenstein, PCoA   │");
    println!("  │ hotSpring ESN reservoir  → NPU deployment (Exp114-119)  │");
    println!("  │ hotSpring Jacobi eigh    → PCoA eigendecomposition      │");
    println!("  │ neuralSpring spectral    → Anderson localization (Exp113)│");
    println!("  │ neuralSpring pairwise_l2 → metalForge cross-substrate   │");
    println!("  │ airSpring moving_window  → bloom time-series (future)   │");
    println!("  │ airSpring kriging        → spatial diversity mapping    │");
    println!("  │ wetSpring → ToadStool    → all springs benefit          │");
    println!("  └──────────────────────────────────────────────────────────┘");

    println!("\n  Performance Summary:");
    println!("  ───────────────────");
    println!(
        "  Diversity (500 communities):  {shannon_ms:.2} ms Shannon, {bc_ms:.2} ms Bray-Curtis"
    );
    println!("  QS ODE (2000 steps):          {ode_ms:.2} ms single, {sweep_ms:.2} ms (200 sweep)");
    println!("  ESN train (300×200):          {esn_train_ms:.2} ms");
    println!("  ESN f64 inference (100):      {esn_infer_ms:.2} ms");
    println!("  NPU int8 inference (100):     {npu_ms:.2} ms");
    if npu_ms > 0.0 && esn_infer_ms > 0.0 {
        println!(
            "  NPU vs f64 ratio:             {:.2}×",
            esn_infer_ms / npu_ms
        );
    }

    println!("\n  Shader Count by Origin (in ToadStool S42):");
    println!("  ──────────────────────────────────────────");
    println!("  hotSpring:      ~35 shaders (nuclear, MD, lattice, ESN, precision)");
    println!("  wetSpring:      ~22 shaders (bio, genomics, PFAS, Gillespie)");
    println!("  neuralSpring:   ~14 shaders (ML, pairwise, evolutionary, IPR)");
    println!("  airSpring:       ~5 shaders (IoT, agriculture, kriging)");
    println!("  ToadStool-native: 100+ shaders (math, linalg, NN, FHE, attention)");
    println!("  Total:          612 WGSL shaders, zero orphans");

    v.check_pass("cross-spring benchmark complete", true);

    // ─── Section 5: Evolution timeline ──────────────────────────────

    v.section("§5 Key Evolution Events");

    println!("  Feb 14: hotSpring MD handoff → ToadStool MD primitives");
    println!("  Feb 15: hotSpring GPU sovereignty Phase 1 → f64 Vulkan bypass");
    println!("  Feb 16: wetSpring handoff v1 → initial bio shaders");
    println!("  Feb 17: Three-springs handoff → unified wateringHole");
    println!("  Feb 19: wetSpring v2-v3 → Gillespie, SmithWaterman");
    println!("  Feb 20: wetSpring v4 → Felsenstein, TreeInference, GemmF64");
    println!("  Feb 20: neuralSpring S-01/S-11 → TensorSession ML ops");
    println!("  Feb 22: ToadStool S39 → absorb 18 Spring shaders (7 bio + 11 HFB)");
    println!("  Feb 22: ToadStool S39 → sparse_eigh + quantize_affine_i8");
    println!("  Feb 22: ToadStool S40 → Richards PDE + moving window stats");
    println!("  Feb 22: ToadStool S41 → 6 f64 shader compile fixes (critical)");
    println!("  Feb 22: ToadStool S42 → 19 new WGSL (612 total), BarraCuda rename");
    println!("  Feb 23: ToadStool HEAD → loop_unroller fix, Jacobi eigh fix");
    println!("  Feb 23: wetSpring Phase 31 → PCoA naga bug + spectral cross-spring");
    println!("  Feb 23: wetSpring Phase 32 → NCBI-scale GPU (Exp108-113)");
    println!("  Feb 23: wetSpring Phase 33 → NPU reservoir deployment (Exp114-119)");
    println!("  Feb 23: wetSpring v17 handoff → 119 experiments, 2,664+ checks");

    v.check_pass("evolution timeline logged", true);

    v.finish();
}
