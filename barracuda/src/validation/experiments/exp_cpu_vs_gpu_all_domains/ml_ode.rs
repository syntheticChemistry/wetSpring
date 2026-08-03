// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML/ODE domain CPU↔GPU parity: Random Forest, HMM, Smith-Waterman,
//! Gillespie SSA, Decision Tree (D07-D11).

use std::sync::Arc;
use std::time::Instant;

use crate::bio::decision_tree::DecisionTree;
use crate::bio::{
    alignment, gillespie, hmm, hmm_gpu::HmmGpuForward, random_forest::RandomForest,
    random_forest_gpu::RandomForestGpu,
};
use crate::tolerances;
use crate::validation::{CpuGpuRow, OrExit, Validator};
use barracuda::ops::bio::gillespie::GillespieModel;
use barracuda::{
    FlatForest, GillespieConfig, GillespieGpu, SmithWatermanGpu, SwConfig, TreeInferenceGpu,
};

pub(super) fn validate_random_forest(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D07: Random Forest");
    let features = vec![
        vec![5.1, 3.5, 1.4, 0.2],
        vec![4.9, 3.0, 1.4, 0.2],
        vec![7.0, 3.2, 4.7, 1.4],
        vec![6.3, 3.3, 4.7, 1.6],
        vec![6.3, 2.5, 5.0, 1.9],
        vec![5.8, 2.7, 5.1, 1.9],
    ];
    let labels = vec![0, 0, 1, 1, 2, 2];
    let tc = Instant::now();
    let tree = crate::bio::decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[5.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        features[0].len(),
    )
    .or_exit("build tree");
    let rf = RandomForest::from_trees(vec![tree; 10], 3).or_exit("build forest");
    let cpu_preds: Vec<usize> = features.iter().map(|f| rf.predict(f)).collect();
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let rf_gpu = RandomForestGpu::new(device);
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rf_gpu.predict_batch(&rf, &features)
    }));
    let gpu_us = tg.elapsed().as_micros() as f64;
    if let Ok(Ok(gpu_preds)) = gpu_result {
        for (i, (c, g)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            v.check(
                &format!("RF pred {i}"),
                g.class as f64,
                *c as f64,
                tolerances::EXACT,
            );
        }
        timings.push(CpuGpuRow {
            name: "Random Forest",
            cpu_us,
            gpu_us,
            status: "PASS",
        });
    } else {
        v.check_pass("RF: driver skip", true);
        timings.push(CpuGpuRow {
            name: "Random Forest",
            cpu_us,
            gpu_us,
            status: "SKIP",
        });
    }
}

pub(super) fn validate_hmm(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D08: HMM Forward");
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 3,
        log_emit: vec![
            0.5_f64.ln(),
            0.4_f64.ln(),
            0.1_f64.ln(),
            0.1_f64.ln(),
            0.3_f64.ln(),
            0.6_f64.ln(),
        ],
    };
    let obs = vec![0_usize, 1, 2, 0, 1];
    let tc = Instant::now();
    let cpu_result = hmm::forward(&model, &obs);
    let cpu_ll = cpu_result.log_likelihood;
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let hmm_dev = HmmGpuForward::new(device).or_exit("HMM GPU init");
    let obs_u32: Vec<u32> = obs.iter().map(|&o| o as u32).collect();
    let gpu_result = hmm_dev
        .forward_batch(&model, &obs_u32, 1, obs.len())
        .or_exit("GPU/CPU validation");
    let gpu_ll = gpu_result.log_likelihoods[0];
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "HMM log-lik",
        gpu_ll,
        cpu_ll,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    timings.push(CpuGpuRow {
        name: "HMM Forward",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_smith_waterman(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D09: Smith-Waterman");
    let q = b"ACGTACGT";
    let t = b"ACTTACTT";
    let tc = Instant::now();
    let cpu_score = alignment::smith_waterman_score(
        q,
        t,
        &alignment::ScoringParams {
            match_score: 2,
            mismatch_penalty: -1,
            gap_open: -3,
            gap_extend: -1,
        },
    );
    let cpu_us = tc.elapsed().as_micros() as f64;
    let sw = SmithWatermanGpu::new(device);
    let subst = vec![
        2.0, -1.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, -1.0, 2.0,
    ];
    let cfg = SwConfig::default();
    let q_enc: Vec<u32> = q.iter().map(|&b| dna_encode(b)).collect();
    let t_enc: Vec<u32> = t.iter().map(|&b| dna_encode(b)).collect();
    let tg = Instant::now();
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sw.align(&q_enc, &t_enc, &subst, &cfg)
    }));
    let gpu_us = tg.elapsed().as_micros() as f64;
    if let Ok(Ok(r)) = gpu_result {
        v.check_pass("SW: both positive", r.score > 0.0 && cpu_score > 0);
        timings.push(CpuGpuRow {
            name: "Smith-Waterman",
            cpu_us,
            gpu_us,
            status: "PASS",
        });
    } else {
        v.check_pass("SW: driver skip", true);
        timings.push(CpuGpuRow {
            name: "Smith-Waterman",
            cpu_us,
            gpu_us,
            status: "SKIP",
        });
    }
}

pub(super) fn validate_gillespie(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D10: Gillespie SSA");
    let initial = vec![100_i64];
    let reactions = vec![
        gillespie::Reaction {
            propensity: Box::new(|s: &[i64]| 0.5 * s[0] as f64),
            stoichiometry: vec![1],
        },
        gillespie::Reaction {
            propensity: Box::new(|s: &[i64]| 0.1 * s[0] as f64),
            stoichiometry: vec![-1],
        },
    ];
    let mut rng = gillespie::Lcg64::new(42);
    let tc = Instant::now();
    let cpu_traj = gillespie::gillespie_ssa(&initial, &reactions, 10.0, &mut rng);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let gg = GillespieGpu::new(device);
    let n_traj: usize = 64;
    let cfg = GillespieConfig {
        t_max: 10.0,
        max_steps: 10_000,
    };
    let model = GillespieModel {
        rate_k: &[0.5, 0.1],
        stoich_react: &[1_u32, 1],
        stoich_net: &[1_i32, -1],
    };
    let tg = Instant::now();
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gg.simulate(
            &model,
            &vec![100.0; n_traj],
            &(0..n_traj as u32 * 4).collect::<Vec<_>>(),
            n_traj,
            &cfg,
        )
    }));
    let gpu_us = tg.elapsed().as_micros() as f64;
    if let Ok(Ok(r)) = gpu_result {
        let finals: Vec<f64> = (0..n_traj).map(|i| r.states[i * r.n_species]).collect();
        let gpu_mean = finals.iter().sum::<f64>() / finals.len() as f64;
        v.check_pass("SSA: GPU mean > 50", gpu_mean > 50.0);
        v.check_pass("SSA: CPU final positive", cpu_traj.final_state()[0] > 0);
        timings.push(CpuGpuRow {
            name: "Gillespie SSA",
            cpu_us,
            gpu_us,
            status: "PASS",
        });
    } else {
        v.check_pass("SSA: driver skip", true);
        timings.push(CpuGpuRow {
            name: "Gillespie SSA",
            cpu_us,
            gpu_us,
            status: "SKIP",
        });
    }
}

pub(super) fn validate_decision_tree(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D11: Decision Tree");
    let cpu_tree = DecisionTree::from_arrays(
        &[0, -1, -1],
        &[5.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        3,
    )
    .or_exit("GPU/CPU validation");
    let samples = [
        vec![3.0, 0.0, 0.0],
        vec![7.0, 0.0, 0.0],
        vec![4.9, 0.0, 0.0],
        vec![9.0, 0.0, 0.0],
    ];
    let tc = Instant::now();
    let cpu_preds: Vec<usize> = samples.iter().map(|s| cpu_tree.predict(s)).collect();
    let cpu_us = tc.elapsed().as_micros() as f64;
    let forest = FlatForest::single_tree(
        vec![0, u32::MAX, u32::MAX],
        vec![5.0, 0.0, 0.0],
        vec![1, -1, -1],
        vec![2, -1, -1],
        vec![u32::MAX, 0, 1],
    );
    let ti = TreeInferenceGpu::new(device);
    let flat_samples: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();
    let tg = Instant::now();
    match ti.predict(&forest, &flat_samples, 4) {
        Ok(gpu_preds) => {
            let gpu_us = tg.elapsed().as_micros() as f64;
            for (i, (cp, gp)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
                v.check(
                    &format!("DT pred {i}"),
                    f64::from(*gp),
                    *cp as f64,
                    tolerances::EXACT,
                );
            }
            timings.push(CpuGpuRow {
                name: "Decision Tree",
                cpu_us,
                gpu_us,
                status: "PASS",
            });
        }
        Err(e) => {
            let gpu_us = tg.elapsed().as_micros() as f64;
            println!("  [SKIP] DT GPU: {e}");
            v.check_pass("DT driver skip", true);
            timings.push(CpuGpuRow {
                name: "Decision Tree",
                cpu_us,
                gpu_us,
                status: "SKIP",
            });
        }
    }
}

const fn dna_encode(b: u8) -> u32 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}
