// SPDX-License-Identifier: AGPL-3.0-or-later
//! S56 primitives: disordered_laplacian, belief_propagation_chain,
//! boltzmann_sampling (neuralSpring → ToadStool → wetSpring).

use std::time::Instant;

use barracuda::linalg::{
    belief_propagation_chain, disordered_laplacian, effective_rank, graph_laplacian,
};
use barracuda::sample::{BoltzmannResult, boltzmann_sampling};
use barracuda::spectral::level_spacing_ratio;
use crate::tolerances;
use crate::validation::{CrossSpringEntry, Validator};

pub(super) fn validate(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    println!("\n══ S56: neuralSpring Final Absorption Primitives ══════════════\n");

    validate_disordered_laplacian(v, bench_results);
    validate_belief_propagation(v, bench_results);
    validate_boltzmann_sampling(v, bench_results);
}

fn validate_disordered_laplacian(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("disordered_laplacian (neuralSpring → S56 → wetSpring Anderson-QS)");
    let n = 6;
    #[rustfmt::skip]
    let adjacency: Vec<f64> = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];

    let laplacian = graph_laplacian(&adjacency, n);
    let heterogeneity: Vec<f64> = vec![0.3, 0.7, 0.5, 0.9, 0.2, 0.4];

    let t0 = Instant::now();
    let weak_disorder = disordered_laplacian(&laplacian, n, &heterogeneity, 1.0);
    let strong_disorder = disordered_laplacian(&laplacian, n, &heterogeneity, 10.0);
    let disorder_us = t0.elapsed().as_micros() as f64;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                v.check(
                    &format!("disordered off-diag [{i},{j}] preserved"),
                    weak_disorder[i * n + j],
                    laplacian[i * n + j],
                    tolerances::PYTHON_PARITY_TIGHT,
                );
            }
        }
    }

    let diag_additions: f64 = (0..n)
        .map(|i| weak_disorder[i * n + i] - laplacian[i * n + i])
        .sum();
    v.check(
        "disorder centered (mean shift = 0)",
        diag_additions,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    let eigs_clean = super::eigenvalues_from_dense(&laplacian, n);
    let eigs_weak = super::eigenvalues_from_dense(&weak_disorder, n);
    let eigs_strong = super::eigenvalues_from_dense(&strong_disorder, n);

    let r_clean = level_spacing_ratio(&eigs_clean);
    let r_weak = level_spacing_ratio(&eigs_weak);
    let r_strong = level_spacing_ratio(&eigs_strong);

    v.check(
        "clean r closer to GOE than strong",
        if (r_clean - 0.530).abs() < (r_strong - 0.530).abs() {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::EXACT,
    );

    let rank_clean = effective_rank(&eigs_clean);
    let rank_strong = effective_rank(&eigs_strong);
    println!("    Clean spectrum r={r_clean:.4}, effective_rank={rank_clean:.2}");
    println!("    Weak disorder  r={r_weak:.4}");
    println!("    Strong disorder r={r_strong:.4}, effective_rank={rank_strong:.2}");
    println!("    → Disorder drives Poisson statistics (Anderson localization)");

    bench_results.push(CrossSpringEntry {
        primitive: "disordered_laplacian",
        evolved_by: "neuralSpring",
        session: "S56",
        cpu_us: disorder_us,
        problem: "6-node ring + disorder",
        checks: 32,
    });
}

fn validate_belief_propagation(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("belief_propagation_chain (neuralSpring → S56 → wetSpring taxonomy)");
    let genus_dist = vec![0.4, 0.3, 0.2, 0.1];

    #[rustfmt::skip]
    let genus_to_family: Vec<f64> = vec![
        0.9, 0.1, 0.0,
        0.1, 0.8, 0.1,
        0.0, 0.2, 0.8,
        0.0, 0.1, 0.9,
    ];

    #[rustfmt::skip]
    let family_to_order: Vec<f64> = vec![
        0.95, 0.05,
        0.50, 0.50,
        0.10, 0.90,
    ];

    let transitions = [genus_to_family.as_slice(), family_to_order.as_slice()];
    let layer_dims = [3, 2];

    let t0 = Instant::now();
    let distributions = belief_propagation_chain(&genus_dist, &transitions, &layer_dims);
    let bp_us = t0.elapsed().as_micros() as f64;

    for (k, dist) in distributions.iter().enumerate() {
        let sum: f64 = dist.iter().sum();
        v.check(
            &format!("BP layer {k} sums to 1"),
            sum,
            1.0,
            tolerances::PYTHON_PARITY_TIGHT,
        );
    }

    v.check(
        "BP produces 3 distributions",
        distributions.len() as f64,
        3.0,
        tolerances::EXACT,
    );

    v.check(
        "order-level has 2 categories",
        distributions[2].len() as f64,
        2.0,
        tolerances::EXACT,
    );

    println!(
        "    Genus:  {:?}",
        distributions[0].iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
    );
    println!(
        "    Family: {:?}",
        distributions[1].iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
    );
    println!(
        "    Order:  {:?}",
        distributions[2].iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
    );

    bench_results.push(CrossSpringEntry {
        primitive: "belief_propagation_chain",
        evolved_by: "neuralSpring",
        session: "S56",
        cpu_us: bp_us,
        problem: "4→3→2 taxonomy",
        checks: 5,
    });
}

fn validate_boltzmann_sampling(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("boltzmann_sampling (neuralSpring → S56 → wetSpring optimization)");
    let loss = |p: &[f64]| -> f64 {
        let x = p[0];
        let y = p[1];
        let dx = 1.0 - x;
        let dy = y - x.mul_add(x, 0.0);
        100.0f64.mul_add(dy.powi(2), dx.powi(2))
    };

    let initial = vec![0.0, 0.0];

    let t0 = Instant::now();
    let result: BoltzmannResult = boltzmann_sampling(&loss, &initial, 0.1, 0.01, 5_000, 42);
    let mcmc_us = t0.elapsed().as_micros() as f64;

    let final_loss = loss(&result.final_params);
    let best_loss = result.losses.iter().copied().fold(f64::INFINITY, f64::min);
    v.check(
        "MCMC finds loss < initial",
        if best_loss < loss(&initial) { 1.0 } else { 0.0 },
        1.0,
        tolerances::EXACT,
    );
    v.check(
        "MCMC acceptance rate reasonable (5-95%)",
        if result.acceptance_rate > 0.05 && result.acceptance_rate < 0.95 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::EXACT,
    );
    v.check(
        "MCMC loss trajectory exists",
        if result.losses.len() > 1 { 1.0 } else { 0.0 },
        1.0,
        tolerances::EXACT,
    );

    println!(
        "    Final params: ({:.4}, {:.4}), final_loss: {final_loss:.4}, best_loss: {best_loss:.4}",
        result.final_params[0], result.final_params[1]
    );
    println!(
        "    Acceptance rate: {:.1}%, trajectory length: {}",
        result.acceptance_rate * 100.0,
        result.losses.len()
    );

    bench_results.push(CrossSpringEntry {
        primitive: "boltzmann_sampling",
        evolved_by: "neuralSpring",
        session: "S56",
        cpu_us: mcmc_us,
        problem: "Rosenbrock 5K steps",
        checks: 3,
    });
}
