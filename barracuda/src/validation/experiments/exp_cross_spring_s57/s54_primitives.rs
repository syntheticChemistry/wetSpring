// SPDX-License-Identifier: AGPL-3.0-or-later
//! S54 primitives: graph_laplacian, effective_rank, numerical_hessian
//! (neuralSpring baseCamp → ToadStool → wetSpring).

use std::time::Instant;

use crate::tolerances;
use crate::validation::{CrossSpringEntry, Validator};
use barracuda::linalg::{effective_rank, graph_laplacian};
use barracuda::numerical::numerical_hessian;

pub(super) fn validate(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    println!("\n══ S54: neuralSpring baseCamp Primitives ════════════════════════\n");

    validate_graph_laplacian(v, bench_results);
    validate_effective_rank(v, bench_results);
    validate_numerical_hessian(v, bench_results);
}

fn validate_graph_laplacian(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("graph_laplacian (neuralSpring → S54 → wetSpring bio)");
    #[rustfmt::skip]
    let adjacency: Vec<f64> = vec![
        0.0, 0.8, 0.2, 0.0, 0.1,
        0.8, 0.0, 0.6, 0.3, 0.0,
        0.2, 0.6, 0.0, 0.9, 0.4,
        0.0, 0.3, 0.9, 0.0, 0.7,
        0.1, 0.0, 0.4, 0.7, 0.0,
    ];
    let n = 5;

    let t0 = Instant::now();
    let laplacian = graph_laplacian(&adjacency, n);
    let lap_us = t0.elapsed().as_micros() as f64;

    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| laplacian[i * n + j]).sum();
        v.check(
            &format!("graph_laplacian row {i} sum"),
            row_sum,
            0.0,
            tolerances::PYTHON_PARITY_TIGHT,
        );
    }

    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adjacency[i * n + j]).sum();
        v.check(
            &format!("graph_laplacian diag[{i}] = degree"),
            laplacian[i * n + i],
            degree,
            tolerances::PYTHON_PARITY_TIGHT,
        );
    }

    let mut sorted_eigs = super::eigenvalues_from_dense(&laplacian, n);
    sorted_eigs.sort_by(f64::total_cmp);
    let fiedler = sorted_eigs[1];
    v.check(
        "Fiedler value > 0 (connected graph)",
        if fiedler > 0.0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::EXACT,
    );
    println!("    Fiedler value: {fiedler:.4} (community connectivity)");

    bench_results.push(CrossSpringEntry {
        primitive: "graph_laplacian",
        evolved_by: "neuralSpring baseCamp",
        session: "S54",
        cpu_us: lap_us,
        problem: "5×5 community",
        checks: 11,
    });
}

fn validate_effective_rank(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("effective_rank (neuralSpring → S54 → wetSpring diversity)");
    let diverse_spectrum: Vec<f64> = vec![4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5];
    let dominated_spectrum: Vec<f64> = vec![10.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    let t0 = Instant::now();
    let rank_diverse = effective_rank(&diverse_spectrum);
    let rank_dominated = effective_rank(&dominated_spectrum);
    let rank_us = t0.elapsed().as_micros() as f64;

    v.check(
        "diverse community: high effective rank",
        if rank_diverse > rank_dominated {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::EXACT,
    );

    let uniform: Vec<f64> = vec![1.0; 8];
    let rank_uniform = effective_rank(&uniform);
    v.check(
        "uniform spectrum: rank = n",
        rank_uniform,
        8.0,
        tolerances::PYTHON_PARITY,
    );

    let single: Vec<f64> = vec![5.0, 0.0, 0.0, 0.0];
    let rank_single = effective_rank(&single);
    v.check(
        "single eigenvalue: rank = 1",
        rank_single,
        1.0,
        tolerances::PYTHON_PARITY,
    );

    println!("    Diverse community effective rank: {rank_diverse:.2} / 8");
    println!("    Dominated community effective rank: {rank_dominated:.2} / 8");

    bench_results.push(CrossSpringEntry {
        primitive: "effective_rank",
        evolved_by: "neuralSpring baseCamp",
        session: "S54",
        cpu_us: rank_us,
        problem: "8-eigenvalue spectrum",
        checks: 3,
    });
}

fn validate_numerical_hessian(v: &mut Validator, bench_results: &mut Vec<CrossSpringEntry>) {
    v.section("numerical_hessian (neuralSpring → S54 → wetSpring ML)");
    let params = vec![1.0, 2.0];
    let loss = |p: &[f64]| -> f64 { p[0].mul_add(p[1], p[0].mul_add(p[0], 2.0 * p[1] * p[1])) };

    let t0 = Instant::now();
    let hessian = numerical_hessian(&loss, &params, tolerances::NUMERICAL_HESSIAN_EPSILON);
    let hess_us = t0.elapsed().as_micros() as f64;

    v.check(
        "Hessian[0,0] = 2",
        hessian[0],
        2.0,
        tolerances::PYTHON_PVALUE,
    );
    v.check(
        "Hessian[0,1] = 1",
        hessian[1],
        1.0,
        tolerances::PYTHON_PVALUE,
    );
    v.check(
        "Hessian[1,0] = 1",
        hessian[2],
        1.0,
        tolerances::PYTHON_PVALUE,
    );
    v.check(
        "Hessian[1,1] = 4",
        hessian[3],
        4.0,
        tolerances::PYTHON_PVALUE,
    );

    let trace = hessian[0] + hessian[3];
    let det = hessian[0].mul_add(hessian[3], -(hessian[1] * hessian[2]));
    v.check(
        "Hessian PD: trace > 0",
        if trace > 0.0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::EXACT,
    );
    v.check(
        "Hessian PD: det > 0",
        if det > 0.0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::EXACT,
    );

    let sqrt_disc = (trace.mul_add(trace, -4.0 * det)).sqrt();
    println!(
        "    Hessian eigenvalues: {:.4}, {:.4} (both positive → convex)",
        f64::midpoint(trace, sqrt_disc),
        f64::midpoint(trace, -sqrt_disc)
    );

    bench_results.push(CrossSpringEntry {
        primitive: "numerical_hessian",
        evolved_by: "neuralSpring baseCamp",
        session: "S54",
        cpu_us: hess_us,
        problem: "2-param quadratic",
        checks: 6,
    });
}
