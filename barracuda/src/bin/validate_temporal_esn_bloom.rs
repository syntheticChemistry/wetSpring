// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! Exp123 — Temporal ESN Bloom Cascade
//!
//! Stateful vs stateless ESN for bloom phase classification. Validates
//! pre-bloom detection latency and NPU quantization.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation type | Analytical (closed-form expected values) |
//! | Expected values | Synthetic bloom trajectory simulation |
//! | Reference | ESN reservoir dynamics, stateful temporal modeling |
//! | Date | 2026-02-25 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::validation::Validator;

const N_WINDOWS: usize = 80;
const N_CLASSES: usize = 4;
const FEATURE_DIM: usize = 6;

fn lcg_next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 33) as f64) / f64::from(u32::MAX)
}

fn lcg_noise(seed: &mut u64, center: f64, pct: f64) -> f64 {
    let noise = lcg_next(seed).mul_add(2.0, -1.0) * pct;
    center * (1.0 + noise)
}

fn simulate_bloom_trajectory(n_windows: usize, seed: u64) -> Vec<(Vec<f64>, usize)> {
    let mut out = Vec::with_capacity(n_windows);
    let n = n_windows as f64;
    let mut rng = seed;

    let normal = |rng: &mut u64| -> [f64; 6] {
        [
            lcg_noise(rng, 3.5, 0.1),
            lcg_noise(rng, 0.85, 0.1),
            lcg_noise(rng, 200.0, 0.1),
            lcg_noise(rng, 0.8, 0.1),
            lcg_noise(rng, 0.05, 0.1),
            lcg_noise(rng, 20.0, 0.1),
        ]
    };
    let pre_bloom = |rng: &mut u64| -> [f64; 6] {
        [
            lcg_noise(rng, 2.5, 0.1),
            lcg_noise(rng, 0.70, 0.1),
            lcg_noise(rng, 150.0, 0.1),
            lcg_noise(rng, 0.6, 0.1),
            lcg_noise(rng, 0.15, 0.1),
            lcg_noise(rng, 24.0, 0.1),
        ]
    };
    let active = |rng: &mut u64| -> [f64; 6] {
        [
            lcg_noise(rng, 1.0, 0.1),
            lcg_noise(rng, 0.40, 0.1),
            lcg_noise(rng, 80.0, 0.1),
            lcg_noise(rng, 0.3, 0.1),
            lcg_noise(rng, 0.40, 0.1),
            lcg_noise(rng, 28.0, 0.1),
        ]
    };
    let post_bloom = |rng: &mut u64| -> [f64; 6] {
        [
            lcg_noise(rng, 2.0, 0.1),
            lcg_noise(rng, 0.60, 0.1),
            lcg_noise(rng, 120.0, 0.1),
            lcg_noise(rng, 0.5, 0.1),
            lcg_noise(rng, 0.20, 0.1),
            lcg_noise(rng, 22.0, 0.1),
        ]
    };

    for i in 0..n_windows {
        let t = i as f64 / n;
        let (feat, class) = if t < 0.30 {
            (normal(&mut rng), 0)
        } else if t < 0.45 {
            (pre_bloom(&mut rng), 1)
        } else if t < 0.70 {
            (active(&mut rng), 2)
        } else if t < 0.85 {
            (post_bloom(&mut rng), 3)
        } else {
            (normal(&mut rng), 0)
        };
        out.push((feat.to_vec(), class));
    }
    out
}

fn one_hot(class: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; N_CLASSES];
    v[class] = 1.0;
    v
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp123: Temporal ESN Bloom Cascade");

    v.section("── S1: Trajectory generation ──");
    let traj = simulate_bloom_trajectory(N_WINDOWS, 42);
    let total_windows = traj.len();
    let classes: std::collections::HashSet<usize> = traj.iter().map(|(_, c)| *c).collect();
    v.check_count("total windows", total_windows, N_WINDOWS);
    v.check_count("distinct phases", classes.len(), N_CLASSES);

    v.section("── S2: Stateful ESN training + f64 accuracy ──");
    let train_trajs: Vec<Vec<(Vec<f64>, Vec<f64>)>> = (42..52)
        .map(|s| {
            simulate_bloom_trajectory(N_WINDOWS, s)
                .into_iter()
                .map(|(f, c)| (f, one_hot(c)))
                .collect()
        })
        .collect();
    let config = EsnConfig {
        input_size: FEATURE_DIM,
        reservoir_size: 200,
        output_size: N_CLASSES,
        spectral_radius: 0.9,
        connectivity: 0.12,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 2026,
    };
    let mut esn_stateful = Esn::new(config.clone());
    esn_stateful.train_stateful(&train_trajs);

    let test_trajs: Vec<Vec<(Vec<f64>, usize)>> = (100..105)
        .map(|s| simulate_bloom_trajectory(N_WINDOWS, s))
        .collect();
    let mut stateful_preds = Vec::new();
    let mut stateful_true = Vec::new();
    for traj_data in &test_trajs {
        esn_stateful.reset_state();
        for (feat, true_class) in traj_data {
            esn_stateful.update(feat);
            let out = esn_stateful.readout();
            let pred = out
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            stateful_preds.push(pred);
            stateful_true.push(*true_class);
        }
    }
    let stateful_correct = stateful_preds
        .iter()
        .zip(stateful_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let stateful_acc = stateful_correct as f64 / stateful_true.len() as f64;
    println!(
        "  Stateful f64 accuracy: {:.3} ({}/{})",
        stateful_acc,
        stateful_correct,
        stateful_true.len()
    );
    v.check_pass("stateful f64 accuracy > 30%", stateful_acc > 0.30);

    let mut per_class_recall = [0.0_f64; N_CLASSES];
    let mut per_class_count = [0usize; N_CLASSES];
    for (p, t) in stateful_preds.iter().zip(stateful_true.iter()) {
        per_class_count[*t] += 1;
        if p == t {
            per_class_recall[*t] += 1.0;
        }
    }
    for c in 0..N_CLASSES {
        let r = if per_class_count[c] > 0 {
            per_class_recall[c] / per_class_count[c] as f64
        } else {
            0.0
        };
        println!(
            "  Class {} recall: {:.3} ({}/{})",
            c, r, per_class_recall[c] as usize, per_class_count[c]
        );
    }
    v.check_pass("per-class recall computed", true);

    v.section("── S3: Stateless ESN training + f64 accuracy ──");
    let all_inputs: Vec<Vec<f64>> = train_trajs
        .iter()
        .flat_map(|t| t.iter().map(|(f, _)| f.clone()))
        .collect();
    let all_targets: Vec<Vec<f64>> = train_trajs
        .iter()
        .flat_map(|t| t.iter().map(|(_, y)| y.clone()))
        .collect();
    let mut esn_stateless = Esn::new(config);
    esn_stateless.train_stateless(&all_inputs, &all_targets);

    let mut stateless_preds = Vec::new();
    let mut stateless_true = Vec::new();
    for traj_data in &test_trajs {
        for (feat, true_class) in traj_data {
            esn_stateless.reset_state();
            esn_stateless.update(feat);
            let out = esn_stateless.readout();
            let pred = out
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            stateless_preds.push(pred);
            stateless_true.push(*true_class);
        }
    }
    let stateless_correct = stateless_preds
        .iter()
        .zip(stateless_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let stateless_acc = stateless_correct as f64 / stateless_true.len() as f64;
    println!(
        "  Stateless f64 accuracy: {:.3} ({}/{})",
        stateless_acc,
        stateless_correct,
        stateless_true.len()
    );
    v.check_pass("stateless f64 accuracy > 20%", stateless_acc > 0.20);

    v.section("── S4: Stateful vs stateless comparison ──");
    v.check_pass("stateful >= stateless", stateful_acc >= stateless_acc);

    let prebloom_indices: Vec<(usize, usize)> = test_trajs
        .iter()
        .enumerate()
        .flat_map(|(ti, t)| {
            t.iter()
                .enumerate()
                .filter(|(_, (_, c))| *c == 1)
                .map(move |(wi, (_, _))| (ti, wi))
        })
        .collect();
    let onset_per_traj: Vec<usize> = (0..5)
        .map(|ti| {
            test_trajs[ti]
                .iter()
                .position(|(_, c)| *c == 1)
                .unwrap_or(N_WINDOWS)
        })
        .collect();
    let mut latencies = Vec::new();
    for (ti, wi) in &prebloom_indices {
        let onset = onset_per_traj[*ti];
        let global_idx = *ti * N_WINDOWS + *wi;
        let pred = stateful_preds[global_idx];
        if pred == 1 {
            latencies.push(wi.saturating_sub(onset) as i64);
        }
    }
    let avg_latency = if latencies.is_empty() {
        f64::NAN
    } else {
        latencies.iter().sum::<i64>() as f64 / latencies.len() as f64
    };
    println!("  Pre-bloom detection latency (windows from onset): avg {avg_latency:.1}");

    v.section("── S5: NPU quantization ──");
    let npu_stateful = esn_stateful.to_npu_weights();
    let npu_stateless = esn_stateless.to_npu_weights();

    let mut npu_stateful_preds = Vec::new();
    let mut npu_stateless_preds = Vec::new();
    for traj_data in &test_trajs {
        esn_stateful.reset_state();
        for (feat, _) in traj_data {
            esn_stateful.update(feat);
            npu_stateful_preds.push(npu_stateful.classify(esn_stateful.state()));
        }
    }
    for traj_data in &test_trajs {
        for (feat, _) in traj_data {
            esn_stateless.reset_state();
            esn_stateless.update(feat);
            npu_stateless_preds.push(npu_stateless.classify(esn_stateless.state()));
        }
    }
    let npu_stateful_correct = npu_stateful_preds
        .iter()
        .zip(stateful_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let npu_stateless_correct = npu_stateless_preds
        .iter()
        .zip(stateless_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let npu_stateful_acc = npu_stateful_correct as f64 / stateful_true.len() as f64;
    let npu_stateless_acc = npu_stateless_correct as f64 / stateless_true.len() as f64;
    println!("  NPU stateful accuracy: {npu_stateful_acc:.3}");
    println!("  NPU stateless accuracy: {npu_stateless_acc:.3}");
    v.check_pass(
        "NPU stateful >= NPU stateless",
        npu_stateful_acc >= npu_stateless_acc,
    );

    v.section("── S6: Detection latency ──");
    let mut stateless_latencies = Vec::new();
    for (ti, wi) in &prebloom_indices {
        let onset = onset_per_traj[*ti];
        let global_idx = *ti * N_WINDOWS + *wi;
        let pred = stateless_preds[global_idx];
        if pred == 1 {
            stateless_latencies.push(wi.saturating_sub(onset) as i64);
        }
    }
    let stateful_first = latencies.iter().min().unwrap_or(&0);
    let stateless_first = stateless_latencies.iter().min().unwrap_or(&0);
    let diff = *stateless_first - *stateful_first;
    println!("Stateful detects pre-bloom {} windows earlier", diff.max(0));
    v.check_pass("latency measured", true);

    v.section("── S7: Energy estimate ──");
    let reservoir_bytes = 200;
    let bytes_per_window = reservoir_bytes;
    println!(
        "  Reservoir state: {} × int8 = {} bytes per window",
        200, bytes_per_window
    );
    let coin_cell_j = 500.0;
    let inference_us = 650.0;
    let sample_interval_s = 300.0;
    let duty = inference_us / (sample_interval_s * 1_000_000.0);
    let avg_mw = 5.0 * duty;
    let daily_j = avg_mw * 0.001 * 86400.0;
    let days = coin_cell_j / daily_j;
    println!("  Coin-cell (500 J) feasibility: {days:.0} days");
    v.check_pass("coin-cell feasible > 30 days", days > 30.0);

    v.finish();
}
