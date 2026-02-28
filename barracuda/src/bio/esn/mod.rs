// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::needless_range_loop)] // ESN uses 2D matrix indexing (i*n+j)
//! Minimal Echo State Network (ESN) for NPU deployment validation.
//!
//! Implements reservoir computing: a fixed random recurrent network (reservoir)
//! projects input into a high-dimensional state space. Only the readout layer
//! (`W_out`) is trained via ridge regression. The reservoir weights (`W_in`, `W_res`)
//! are generated once and never updated.
//!
//! This is the wetSpring-local ESN used for training data → NPU pipeline
//! validation. For production GPU/NPU ESN, use `barracuda::esn_v2::ESN`
//! from `ToadStool` (hardware-agnostic, WGSL fused reservoir update).
//!
//! # NPU Deployment Path
//!
//! ```text
//! GPU/CPU training data (Exp108-113)
//!   → ESN train (ridge regression readout)
//!     → export W_out as f64
//!       → quantize to int8 (NpuReadoutWeights)
//!         → Akida AKD1000 FC layer inference
//! ```

mod npu;
pub use npu::NpuReadoutWeights;

/// Simple LCG PRNG for deterministic reservoir generation.
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        f64::from((self.0 >> 33) as u32) / f64::from(u32::MAX)
    }
    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(crate::tolerances::BOX_MULLER_U1_FLOOR);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Echo State Network configuration.
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Number of input dimensions per time step.
    pub input_size: usize,
    /// Number of reservoir neurons (state dimensionality).
    pub reservoir_size: usize,
    /// Number of output classes or regression targets.
    pub output_size: usize,
    /// Spectral radius of `W_res` (controls echo state property, typically < 1.0).
    pub spectral_radius: f64,
    /// Fraction of non-zero reservoir connections (sparsity = 1 − connectivity).
    pub connectivity: f64,
    /// Leaky integration rate: `state = (1-α)·old + α·tanh(...)`.
    pub leak_rate: f64,
    /// Ridge regression regularization λ (Tikhonov).
    pub regularization: f64,
    /// PRNG seed for deterministic reservoir generation.
    pub seed: u64,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            input_size: 5,
            reservoir_size: 200,
            output_size: 3,
            spectral_radius: 0.9,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: crate::tolerances::ESN_REGULARIZATION,
            seed: 42,
        }
    }
}

/// Trained Echo State Network.
pub struct Esn {
    w_in: Vec<f64>,
    w_res: Vec<f64>,
    w_out: Vec<f64>,
    state: Vec<f64>,
    config: EsnConfig,
}

/// Solve `(SᵀS + λI)·w = Sᵀy` via `ToadStool`'s Cholesky-based ridge regression.
///
/// `barracuda` is always available (default-features = false for CPU-only builds).
/// Falls back to zero weights if the solve fails.
fn solve_ridge(
    w_out: &mut [f64],
    flat_states: &[f64],
    flat_targets: &[f64],
    n_samples: usize,
    n_res: usize,
    n_out: usize,
    regularization: f64,
) {
    match barracuda::linalg::ridge_regression(
        flat_states,
        flat_targets,
        n_samples,
        n_res,
        n_out,
        regularization,
    ) {
        Ok(result) => {
            w_out[..result.weights.len()].copy_from_slice(&result.weights);
        }
        Err(_) => {
            w_out.fill(0.0);
        }
    }
}

impl Esn {
    /// Create a new ESN with random reservoir.
    #[must_use]
    pub fn new(config: EsnConfig) -> Self {
        let mut rng = Lcg::new(config.seed);
        let n_in = config.input_size;
        let n_res = config.reservoir_size;

        // W_in: input_size × reservoir_size, uniform [-1, 1]
        let w_in: Vec<f64> = (0..n_in * n_res)
            .map(|_| rng.next_f64().mul_add(2.0, -1.0))
            .collect();

        // W_res: reservoir_size × reservoir_size, sparse Gaussian
        let mut w_res = vec![0.0_f64; n_res * n_res];
        for i in 0..n_res {
            for j in 0..n_res {
                if rng.next_f64() < config.connectivity {
                    w_res[i * n_res + j] = rng.next_gaussian();
                }
            }
        }

        // Scale to desired spectral radius (approximate: scale by sr / max_abs)
        let max_abs = w_res.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_abs > 0.0 {
            let scale = config.spectral_radius / max_abs;
            for w in &mut w_res {
                *w *= scale;
            }
        }

        Self {
            w_in,
            w_res,
            w_out: vec![0.0; config.output_size * n_res],
            state: vec![0.0; n_res],
            config,
        }
    }

    /// Reset reservoir state to zero.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Run one reservoir update step.
    pub fn update(&mut self, input: &[f64]) {
        let n_res = self.config.reservoir_size;
        let n_in = self.config.input_size;
        let leak = self.config.leak_rate;

        let mut new_state = vec![0.0_f64; n_res];

        for j in 0..n_res {
            let mut val = 0.0;
            // W_in * input
            for i in 0..n_in.min(input.len()) {
                val += self.w_in[i * n_res + j] * input[i];
            }
            // W_res * state
            for i in 0..n_res {
                val += self.w_res[i * n_res + j] * self.state[i];
            }
            new_state[j] = (1.0 - leak).mul_add(self.state[j], leak * val.tanh());
        }

        self.state = new_state;
    }

    /// Get a reference to the current reservoir state (for NPU inference).
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Input weight matrix `W_in` (flat, `input_size × reservoir_size`).
    #[must_use]
    pub fn w_in(&self) -> &[f64] {
        &self.w_in
    }

    /// Reservoir weight matrix `W_res` (flat, `reservoir_size × reservoir_size`).
    #[must_use]
    pub fn w_res(&self) -> &[f64] {
        &self.w_res
    }

    /// Readout weight matrix `W_out` (flat, `output_size × reservoir_size`).
    #[must_use]
    pub fn w_out(&self) -> &[f64] {
        &self.w_out
    }

    /// Mutable access to readout weights for online learning / evolution.
    pub fn w_out_mut(&mut self) -> &mut [f64] {
        &mut self.w_out
    }

    /// Configuration used to build this ESN.
    #[must_use]
    pub const fn config(&self) -> &EsnConfig {
        &self.config
    }

    /// Compute readout from current state.
    #[must_use]
    pub fn readout(&self) -> Vec<f64> {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let mut output = vec![0.0_f64; n_out];
        for o in 0..n_out {
            for r in 0..n_res {
                output[o] += self.w_out[o * n_res + r] * self.state[r];
            }
        }
        output
    }

    /// Train readout weights via ridge regression on collected states.
    ///
    /// `inputs`: sequence of input vectors, `targets`: corresponding outputs.
    /// Runs reservoir on all inputs, collects states, solves
    /// `(SᵀS + λI)·W_out = Sᵀ·Y` via Cholesky decomposition.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        self.reset_state();

        let mut flat_states = Vec::with_capacity(n_samples * n_res);
        for input in inputs.iter().take(n_samples) {
            self.update(input);
            flat_states.extend_from_slice(&self.state);
        }

        let mut flat_targets = vec![0.0_f64; n_samples * n_out];
        for (t, target) in targets.iter().take(n_samples).enumerate() {
            for (o, &val) in target.iter().take(n_out).enumerate() {
                flat_targets[t * n_out + o] = val;
            }
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Train from sequences with reset between each trajectory.
    /// State carries across windows within a trajectory; resets between trajectories.
    pub fn train_stateful(&mut self, trajectories: &[Vec<(Vec<f64>, Vec<f64>)>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let mut flat_states = Vec::new();
        let mut flat_targets = Vec::new();
        let mut n_samples = 0;

        for traj in trajectories {
            self.reset_state();
            for (input, target) in traj {
                self.update(input);
                flat_states.extend_from_slice(&self.state);
                for o in 0..n_out {
                    flat_targets.push(if o < target.len() { target[o] } else { 0.0 });
                }
                n_samples += 1;
            }
        }

        if n_samples == 0 {
            return;
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Train with reset before each sample (stateless: each window independent).
    pub fn train_stateless(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        let mut flat_states = Vec::with_capacity(n_samples * n_res);
        for input in inputs.iter().take(n_samples) {
            self.reset_state();
            self.update(input);
            flat_states.extend_from_slice(&self.state);
        }

        let mut flat_targets = vec![0.0_f64; n_samples * n_out];
        for (t, target) in targets.iter().take(n_samples).enumerate() {
            for (o, &val) in target.iter().take(n_out).enumerate() {
                flat_targets[t * n_out + o] = val;
            }
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Predict on a sequence of inputs (full drive + readout).
    #[must_use]
    pub fn predict(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.reset_state();
        inputs
            .iter()
            .map(|input| {
                self.update(input);
                self.readout()
            })
            .collect()
    }

    /// Export readout weights as int8 for NPU deployment.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn to_npu_weights(&self) -> NpuReadoutWeights {
        if self.w_out.is_empty() {
            return NpuReadoutWeights {
                weights_i8: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
                output_size: 0,
                reservoir_size: 0,
            };
        }

        let min_val = self.w_out.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = self.w_out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = min_val;

        let weights_i8: Vec<i8> = self
            .w_out
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i64 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();

        NpuReadoutWeights {
            weights_i8,
            scale,
            zero_point,
            output_size: self.config.output_size,
            reservoir_size: self.config.reservoir_size,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
mod tests {
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
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_f64().to_bits(), rng2.next_f64().to_bits());
        }
    }

    #[test]
    fn lcg_range_01() {
        let mut rng = Lcg::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..=1.0).contains(&v), "LCG output {v} outside [0,1]");
        }
    }

    #[test]
    fn lcg_gaussian_mean_near_zero() {
        let mut rng = Lcg::new(42);
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
}
