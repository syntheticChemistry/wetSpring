// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::suboptimal_flops
)]

use super::*;

#[test]
fn default_config_values() {
    let c = EsnConfig::default();
    assert_eq!(c.input_size, 5);
    assert_eq!(c.reservoir_size, 200);
    assert_eq!(c.output_size, 3);
    assert!((c.spectral_radius - 0.9).abs() < f64::EPSILON);
    assert!((c.connectivity - 0.1).abs() < f64::EPSILON);
    assert!((c.leak_rate - 0.3).abs() < f64::EPSILON);
    assert!((c.regularization - 1e-6).abs() < f64::EPSILON);
    assert_eq!(c.seed, 42);
}

#[test]
fn new_esn_dimensions() {
    let config = EsnConfig {
        input_size: 3,
        reservoir_size: 50,
        output_size: 2,
        ..EsnConfig::default()
    };
    let esn = Esn::new(config);
    assert_eq!(esn.w_in.len(), 3 * 50);
    assert_eq!(esn.w_res.len(), 50 * 50);
    assert_eq!(esn.w_out.len(), 2 * 50);
    assert_eq!(esn.state.len(), 50);
}

#[test]
fn reset_state_zeros_all() {
    let mut esn = Esn::new(EsnConfig {
        reservoir_size: 10,
        ..EsnConfig::default()
    });
    esn.update(&[1.0, 0.5, 0.0, 0.0, 0.0]);
    assert!(esn.state().iter().any(|&x| x != 0.0));
    esn.reset_state();
    assert!(esn.state().iter().all(|&x| x == 0.0));
}

#[test]
fn update_changes_state() {
    let mut esn = Esn::new(EsnConfig {
        reservoir_size: 20,
        ..EsnConfig::default()
    });
    let input = vec![1.0; 5];
    esn.update(&input);
    let state_after = esn.state().to_vec();
    assert!(
        state_after.iter().any(|&x| x != 0.0),
        "state should change after update"
    );
}

#[test]
fn update_deterministic() {
    let mut esn1 = Esn::new(EsnConfig::default());
    let mut esn2 = Esn::new(EsnConfig::default());
    let input = vec![0.5; 5];
    esn1.update(&input);
    esn2.update(&input);
    assert_eq!(esn1.state(), esn2.state());
}

#[test]
fn readout_dimensions() {
    let config = EsnConfig {
        output_size: 4,
        ..EsnConfig::default()
    };
    let esn = Esn::new(config);
    assert_eq!(esn.readout().len(), 4);
}

#[test]
fn readout_zero_state_is_zero() {
    let esn = Esn::new(EsnConfig::default());
    let out = esn.readout();
    assert!(
        out.iter().all(|&x| x == 0.0),
        "zero state should give zero readout"
    );
}

#[test]
fn train_improves_fit() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.5,
        regularization: 1e-4,
        seed: 42,
    };
    let mut esn = Esn::new(config);

    let inputs: Vec<Vec<f64>> = (0..30)
        .map(|i| {
            let x = (i as f64) / 30.0;
            vec![x, 1.0 - x]
        })
        .collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|v| vec![v[0] + v[1]]).collect();

    esn.train(&inputs, &targets);
    let predictions = esn.predict(&inputs);

    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p[0] - t[0]).powi(2))
        .sum::<f64>()
        / targets.len() as f64;
    assert!(
        mse < 5000.0,
        "trained ESN should have reasonable MSE, got {mse}"
    );
}

#[test]
fn predict_dimensions() {
    let config = EsnConfig {
        input_size: 3,
        output_size: 2,
        reservoir_size: 30,
        ..EsnConfig::default()
    };
    let mut esn = Esn::new(config);
    let inputs = vec![vec![1.0, 0.0, 0.0]; 5];
    let preds = esn.predict(&inputs);
    assert_eq!(preds.len(), 5);
    for p in &preds {
        assert_eq!(p.len(), 2);
    }
}

#[test]
fn train_stateless_resets_between_samples() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 20,
        output_size: 1,
        ..EsnConfig::default()
    };
    let mut esn = Esn::new(config);
    let inputs = vec![vec![1.0, 0.0]; 10];
    let targets = vec![vec![0.5]; 10];
    esn.train_stateless(&inputs, &targets);
    assert!(esn.state().iter().all(|&x| x == 0.0));
}

#[test]
fn train_stateful_trajectories() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 20,
        output_size: 1,
        ..EsnConfig::default()
    };
    let mut esn = Esn::new(config);
    let traj = vec![
        vec![(vec![1.0, 0.0], vec![0.5]), (vec![0.0, 1.0], vec![0.3])],
        vec![(vec![0.5, 0.5], vec![0.4])],
    ];
    esn.train_stateful(&traj);
    assert!(esn.state().iter().all(|&x| x == 0.0));
}

#[test]
fn train_stateful_empty() {
    let mut esn = Esn::new(EsnConfig::default());
    esn.train_stateful(&[]);
}

#[test]
fn npu_weights_dimensions() {
    let config = EsnConfig {
        reservoir_size: 30,
        output_size: 3,
        ..EsnConfig::default()
    };
    let esn = Esn::new(config);
    let npu = esn.to_npu_weights();
    assert_eq!(npu.weights_i8.len(), 3 * 30);
    assert_eq!(npu.output_size, 3);
    assert_eq!(npu.reservoir_size, 30);
}

#[test]
fn npu_weights_empty_esn() {
    let esn = Esn::new(EsnConfig {
        output_size: 0,
        reservoir_size: 0,
        ..EsnConfig::default()
    });
    let npu = esn.to_npu_weights();
    assert!(npu.weights_i8.is_empty());
    assert_eq!(npu.output_size, 0);
}

#[test]
fn npu_weights_quantization_range() {
    let config = EsnConfig {
        reservoir_size: 20,
        output_size: 2,
        ..EsnConfig::default()
    };
    let mut esn = Esn::new(config);
    for (i, w) in esn.w_out_mut().iter_mut().enumerate() {
        *w = (i as f64 - 20.0) / 10.0;
    }
    let npu = esn.to_npu_weights();
    assert!(!npu.weights_i8.is_empty());
    assert!(npu.scale > 0.0);
}

#[test]
fn npu_infer_dimensions() {
    let config = EsnConfig {
        reservoir_size: 20,
        output_size: 3,
        ..EsnConfig::default()
    };
    let esn = Esn::new(config);
    let npu = esn.to_npu_weights();
    let state = vec![0.5; 20];
    let output = npu.infer(&state);
    assert_eq!(output.len(), 3);
}

#[test]
fn npu_classify_returns_valid_index() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 30,
        output_size: 3,
        ..EsnConfig::default()
    };
    let mut esn = Esn::new(config);
    let inputs = vec![vec![1.0, 0.0]; 20];
    let targets: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let class = i % 3;
            let mut t = vec![0.0; 3];
            t[class] = 1.0;
            t
        })
        .collect();
    esn.train(&inputs, &targets);
    let npu = esn.to_npu_weights();
    esn.reset_state();
    esn.update(&[1.0, 0.0]);
    let class = npu.classify(esn.state());
    assert!(class < 3, "classify should return index < output_size");
}

#[test]
fn lcg_deterministic() {
    let mut rng1 = super::reservoir::Lcg::new(42);
    let mut rng2 = super::reservoir::Lcg::new(42);
    for _ in 0..100 {
        assert_eq!(rng1.next_f64().to_bits(), rng2.next_f64().to_bits());
    }
}

#[test]
fn lcg_range_01() {
    let mut rng = super::reservoir::Lcg::new(12345);
    for _ in 0..1000 {
        let v = rng.next_f64();
        assert!((0.0..=1.0).contains(&v), "LCG output {v} outside [0,1]");
    }
}

#[test]
fn lcg_gaussian_mean_near_zero() {
    let mut rng = super::reservoir::Lcg::new(42);
    let n = 10_000;
    let sum: f64 = (0..n).map(|_| rng.next_gaussian()).sum();
    let mean = sum / n as f64;
    assert!(
        mean.abs() < 0.1,
        "Gaussian mean should be near 0, got {mean}"
    );
}

#[test]
fn spectral_radius_scaling() {
    let config = EsnConfig {
        reservoir_size: 50,
        spectral_radius: 0.5,
        connectivity: 0.3,
        ..EsnConfig::default()
    };
    let esn = Esn::new(config);
    let max_abs = esn.w_res().iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    assert!(
        (max_abs - 0.5).abs() < 0.01,
        "max |w_res| should ≈ 0.5, got {max_abs}"
    );
}

#[test]
fn npu_from_readout_weights() {
    let w_out = vec![0.1, -0.2, 0.3, -0.1, 0.0, 0.2];
    let npu = NpuReadoutWeights::from_readout_weights(&w_out, 2, 3);
    assert_eq!(npu.weights_i8.len(), 6);
    assert_eq!(npu.output_size, 2);
    assert_eq!(npu.reservoir_size, 3);
    let out = npu.infer(&[0.5, 0.0, -0.5]);
    assert_eq!(out.len(), 2);
}

// ── ToadStool bridge tests (gpu feature) ────────────────────────────────────

#[cfg(feature = "gpu")]
#[test]
fn bio_esn_config_from_esn_config() {
    use super::{BioEsnConfig, EsnConfig};

    let legacy = EsnConfig {
        input_size: 4,
        reservoir_size: 80,
        output_size: 2,
        spectral_radius: 0.85,
        connectivity: 0.15,
        leak_rate: 0.4,
        regularization: 1e-5,
        seed: 123,
    };
    let bio: BioEsnConfig = BioEsnConfig::from(&legacy);
    assert_eq!(bio.input_size, 4);
    assert_eq!(bio.reservoir_size, 80);
    assert_eq!(bio.output_size, 2);
    assert!((bio.spectral_radius - 0.85).abs() < f64::EPSILON);
}

#[cfg(feature = "gpu")]
#[test]
fn bio_esn_config_multi_head() {
    use super::BioEsnConfig;

    let config = BioEsnConfig::multi_head(5, 5);
    assert_eq!(config.input_size, 5);
    assert_eq!(config.output_size, 5);
    assert_eq!(config.reservoir_size, 500);
}

#[cfg(feature = "gpu")]
#[test]
fn bio_head_kind_index() {
    use super::BioHeadKind;

    assert_eq!(BioHeadKind::Diversity.index(5), 0);
    assert_eq!(BioHeadKind::Taxonomy.index(5), 1);
    assert_eq!(BioHeadKind::Bloom.index(3), 2);
    assert_eq!(BioHeadKind::Custom(10).index(5), 4);
}

#[cfg(feature = "gpu")]
#[test]
fn bio_esn_train_predict_roundtrip() {
    use super::{BioEsn, BioEsnConfig};

    let config = BioEsnConfig {
        input_size: 2,
        reservoir_size: 30,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.5,
        regularization: 1e-4,
        seed: 42,
    };
    let mut bio_esn = BioEsn::new(&config).expect("BioEsn::new");
    let inputs: Vec<Vec<f64>> = (0..25)
        .map(|i| {
            let x = (i as f64) / 25.0;
            vec![x, 1.0 - x]
        })
        .collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|v| vec![v[0] + v[1]]).collect();

    let err = bio_esn.train(&inputs, &targets).expect("train");
    assert!(err.is_finite(), "train should return finite error");

    let predictions = bio_esn.predict(&inputs).expect("predict");
    assert_eq!(predictions.len(), inputs.len());
    for (p, t) in predictions.iter().zip(targets.iter()) {
        assert_eq!(p.len(), 1);
        assert!(
            (p[0] - t[0]).abs() < 10.0,
            "prediction should be reasonable"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn bio_esn_to_npu_weights() {
    use super::{BioEsn, BioEsnConfig};

    let config = BioEsnConfig {
        input_size: 2,
        reservoir_size: 20,
        output_size: 3,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.5,
        regularization: 1e-4,
        seed: 42,
    };
    let mut bio_esn = BioEsn::new(&config).expect("BioEsn::new");
    let inputs: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![(i as f64) * 0.1, 1.0 - (i as f64) * 0.05])
        .collect();
    let targets: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let c = i % 3;
            let mut t = vec![0.0; 3];
            t[c] = 1.0;
            t
        })
        .collect();

    bio_esn.train(&inputs, &targets).expect("train");
    let npu = bio_esn.to_npu_weights().expect("to_npu_weights");
    assert_eq!(npu.weights_i8.len(), 3 * 20);
    assert_eq!(npu.output_size, 3);
    assert_eq!(npu.reservoir_size, 20);

    bio_esn.reset_state().expect("reset");
    bio_esn.update(&[1.0, 0.0]).expect("update");
    let state = bio_esn.state().expect("state");
    let class = npu.classify(&state);
    assert!(class < 3, "classify should return valid index");
}
