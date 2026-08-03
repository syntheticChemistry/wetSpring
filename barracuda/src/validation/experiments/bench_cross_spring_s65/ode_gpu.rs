// SPDX-License-Identifier: AGPL-3.0-or-later
//! §1 GPU ODE: 5 bio systems benchmarked via ToadStool BatchedOdeRK4.

use std::sync::Arc;

use crate::bio::bistable::BistableParams;
use crate::bio::bistable_gpu::{BistableGpu, N_VARS as BIST_VARS};
use crate::bio::capacitor_gpu::{CapacitorGpu, CapacitorOdeConfig};
use crate::bio::cooperation::CooperationParams;
use crate::bio::cooperation_gpu::{CooperationGpu, CooperationOdeConfig};
use crate::bio::multi_signal_gpu::{MultiSignalGpu, MultiSignalOdeConfig};
use crate::bio::phage_defense::PhageDefenseParams;
use crate::bio::phage_defense_gpu::{PhageDefenseGpu, PhageDefenseOdeConfig};
use crate::validation::{BenchRow, OrExit, Validator, bench_print};

pub(super) fn validate(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRow>,
) {
    v.section("§1 GPU ODE: 5 bio systems (wetSpring → ToadStool S58, BGL S62)");

    let n_batches: u32 = 128;
    let nb = n_batches as usize;

    let (bist_res, bist_ms) = bench_print("Bistable GPU (128 batches)", || {
        let gpu_ode = BistableGpu::new(Arc::clone(device)).or_exit("BistableGpu");
        let params: Vec<BistableParams> = (0..nb)
            .map(|i| BistableParams {
                alpha_fb: (i as f64).mul_add(0.01, 2.0),
                ..BistableParams::default()
            })
            .collect();
        let initial: Vec<[f64; BIST_VARS]> = vec![[0.01, 0.0, 0.0, 0.0, 0.5]; nb];
        gpu_ode
            .integrate_params(&params, &initial, 500, 0.01)
            .or_exit("integrate")
    });
    v.check_pass(
        "Bistable: 128 batches finite",
        bist_res.iter().all(|r| r.iter().all(|x| x.is_finite())),
    );
    timings.push(BenchRow {
        label: "Bistable GPU 128×",
        origin: "wetSpring→S58",
        ms: bist_ms,
    });

    let (coop_res, coop_ms) = bench_print("Cooperation GPU (128 batches)", || {
        let gpu_ode = CooperationGpu::new(Arc::clone(device)).or_exit("CooperationGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| [0.01, 0.0, 0.0, 0.0].iter().copied())
            .collect();
        let params: Vec<CooperationParams> = (0..nb)
            .map(|i| CooperationParams {
                mu_coop: (i as f64).mul_add(0.002, 0.5),
                ..CooperationParams::default()
            })
            .collect();
        let flat_p: Vec<f64> = params.iter().flat_map(CooperationParams::to_flat).collect();
        let config = CooperationOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .or_exit("integrate")
    });
    v.check_pass(
        "Cooperation: 128 batches finite",
        coop_res.iter().all(|x| x.is_finite()),
    );
    timings.push(BenchRow {
        label: "Cooperation GPU 128×",
        origin: "wetSpring→S58",
        ms: coop_ms,
    });

    let (phage_res, phage_ms) = bench_print("PhageDefense GPU (128 batches)", || {
        let gpu_ode = PhageDefenseGpu::new(Arc::clone(device)).or_exit("PhageDefenseGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| [1.0, 0.001, 0.01, 10.0].iter().copied())
            .collect();
        let params: Vec<PhageDefenseParams> = (0..nb)
            .map(|i| PhageDefenseParams {
                burst_size: (i as f64).mul_add(0.5, 50.0),
                ..PhageDefenseParams::default()
            })
            .collect();
        let flat_p: Vec<f64> = params
            .iter()
            .flat_map(PhageDefenseParams::to_flat)
            .collect();
        let config = PhageDefenseOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.001,
            t0: 0.0,
            clamp_max: 1e8,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .or_exit("integrate")
    });
    v.check_pass(
        "PhageDefense: 128 batches finite",
        phage_res.iter().all(|x| x.is_finite()),
    );
    timings.push(BenchRow {
        label: "PhageDefense GPU 128×",
        origin: "wetSpring→S58",
        ms: phage_ms,
    });

    let (cap_res, cap_ms) = bench_print("Capacitor GPU (128 batches)", || {
        use crate::bio::capacitor::{CapacitorParams, N_VARS as CAP_V};
        let gpu_ode = CapacitorGpu::new(Arc::clone(device)).or_exit("CapacitorGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| {
                let mut y = [0.0f64; CAP_V];
                y[0] = 0.01;
                y.into_iter()
            })
            .collect();
        let flat_p: Vec<f64> = (0..nb)
            .flat_map(|i| {
                let mut p = CapacitorParams::default();
                p.mu_max += (i as f64) * 0.005;
                p.to_flat().into_iter()
            })
            .collect();
        let config = CapacitorOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .or_exit("integrate")
    });
    v.check_pass(
        "Capacitor: 128 batches finite",
        cap_res.iter().all(|x| x.is_finite()),
    );
    timings.push(BenchRow {
        label: "Capacitor GPU 128×",
        origin: "wetSpring→S58",
        ms: cap_ms,
    });

    let (multi_res, multi_ms) = bench_print("MultiSignal GPU (128 batches)", || {
        use crate::bio::multi_signal::{MultiSignalParams, N_VARS as MS_V};
        let gpu_ode = MultiSignalGpu::new(Arc::clone(device)).or_exit("MultiSignalGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| {
                let mut y = [0.0f64; MS_V];
                y[0] = 0.01;
                y.into_iter()
            })
            .collect();
        let flat_p: Vec<f64> = (0..nb)
            .flat_map(|i| {
                let mut p = MultiSignalParams::default();
                p.mu_max += (i as f64) * 0.002;
                p.to_flat().into_iter()
            })
            .collect();
        let config = MultiSignalOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .or_exit("integrate")
    });
    v.check_pass(
        "MultiSignal: 128 batches finite",
        multi_res.iter().all(|x| x.is_finite()),
    );
    timings.push(BenchRow {
        label: "MultiSignal GPU 128×",
        origin: "wetSpring→S58",
        ms: multi_ms,
    });
}
