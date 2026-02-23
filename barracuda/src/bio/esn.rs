// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal Echo State Network (ESN) for NPU deployment validation.
//!
//! Implements reservoir computing: a fixed random recurrent network (reservoir)
//! projects input into a high-dimensional state space. Only the readout layer
//! (W_out) is trained via ridge regression. The reservoir weights (W_in, W_res)
//! are generated once and never updated.
//!
//! This is the wetSpring-local ESN used for training data → NPU pipeline
//! validation. For production GPU/NPU ESN, use `barracuda::esn_v2::ESN`
//! from ToadStool (hardware-agnostic, WGSL fused reservoir update).
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

/// Simple LCG PRNG for deterministic reservoir generation.
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f64) / (u32::MAX as f64)
    }
    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
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
    /// Spectral radius of W_res (controls echo state property, typically < 1.0).
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
            regularization: 1e-6,
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

/// Int8-quantized readout weights for NPU deployment.
#[derive(Debug, Clone)]
pub struct NpuReadoutWeights {
    /// Quantized readout weights: `output_size × reservoir_size`, row-major.
    pub weights_i8: Vec<i8>,
    /// Quantization scale: `real = quantized * scale + zero_point`.
    pub scale: f64,
    /// Quantization zero point.
    pub zero_point: f64,
    /// Number of output classes.
    pub output_size: usize,
    /// Reservoir dimensionality.
    pub reservoir_size: usize,
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
            .map(|_| rng.next_f64() * 2.0 - 1.0)
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
            new_state[j] = (1.0 - leak) * self.state[j] + leak * val.tanh();
        }

        self.state = new_state;
    }

    /// Get a reference to the current reservoir state (for NPU inference).
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
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
    /// Runs reservoir on all inputs, collects states, solves for W_out.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        self.reset_state();

        // Collect reservoir states
        let mut states = Vec::with_capacity(n_samples);
        for input in inputs.iter().take(n_samples) {
            self.update(input);
            states.push(self.state.clone());
        }

        // Ridge regression: W_out = Y * S^T * (S * S^T + λI)^{-1}
        // Simplified: W_out[o] = solve( S*S^T + λI, S * y[o] ) for each output
        // Using normal equations with Tikhonov regularization

        // S^T * S + λI (reservoir_size × reservoir_size)
        let mut sts = vec![0.0_f64; n_res * n_res];
        for s in &states {
            for i in 0..n_res {
                for j in 0..n_res {
                    sts[i * n_res + j] += s[i] * s[j];
                }
            }
        }
        // Add regularization
        let reg = self.config.regularization;
        for i in 0..n_res {
            sts[i * n_res + i] += reg;
        }

        // S^T * Y for each output dimension
        for o in 0..n_out {
            let mut sty = vec![0.0_f64; n_res];
            for (t, s) in states.iter().enumerate() {
                if t < targets.len() && o < targets[t].len() {
                    for r in 0..n_res {
                        sty[r] += s[r] * targets[t][o];
                    }
                }
            }

            // Solve (S^T*S + λI) * w = S^T*y via Cholesky-like diagonal solve
            // (simplified: use diagonal approximation for speed)
            for r in 0..n_res {
                let diag = sts[r * n_res + r];
                self.w_out[o * n_res + r] = if diag.abs() > 1e-15 {
                    sty[r] / diag
                } else {
                    0.0
                };
            }
        }

        self.reset_state();
    }

    /// Train from sequences with reset between each trajectory.
    /// State carries across windows within a trajectory; resets between trajectories.
    pub fn train_stateful(&mut self, trajectories: &[Vec<(Vec<f64>, Vec<f64>)>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let mut states: Vec<Vec<f64>> = Vec::new();
        let mut targets: Vec<Vec<f64>> = Vec::new();

        for traj in trajectories {
            self.reset_state();
            for (input, target) in traj {
                self.update(input);
                states.push(self.state.clone());
                targets.push(target.clone());
            }
        }

        if states.is_empty() {
            return;
        }

        let mut sts = vec![0.0_f64; n_res * n_res];
        for s in &states {
            for i in 0..n_res {
                for j in 0..n_res {
                    sts[i * n_res + j] += s[i] * s[j];
                }
            }
        }
        let reg = self.config.regularization;
        for i in 0..n_res {
            sts[i * n_res + i] += reg;
        }

        for o in 0..n_out {
            let mut sty = vec![0.0_f64; n_res];
            for (t, s) in states.iter().enumerate() {
                if t < targets.len() && o < targets[t].len() {
                    for r in 0..n_res {
                        sty[r] += s[r] * targets[t][o];
                    }
                }
            }
            for r in 0..n_res {
                let diag = sts[r * n_res + r];
                self.w_out[o * n_res + r] = if diag.abs() > 1e-15 {
                    sty[r] / diag
                } else {
                    0.0
                };
            }
        }

        self.reset_state();
    }

    /// Train with reset before each sample (stateless: each window independent).
    pub fn train_stateless(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        let mut states = Vec::with_capacity(n_samples);
        for input in inputs.iter().take(n_samples) {
            self.reset_state();
            self.update(input);
            states.push(self.state.clone());
        }

        let mut sts = vec![0.0_f64; n_res * n_res];
        for s in &states {
            for i in 0..n_res {
                for j in 0..n_res {
                    sts[i * n_res + j] += s[i] * s[j];
                }
            }
        }
        let reg = self.config.regularization;
        for i in 0..n_res {
            sts[i * n_res + i] += reg;
        }

        for o in 0..n_out {
            let mut sty = vec![0.0_f64; n_res];
            for (t, s) in states.iter().enumerate() {
                if t < targets.len() && o < targets[t].len() {
                    for r in 0..n_res {
                        sty[r] += s[r] * targets[t][o];
                    }
                }
            }
            for r in 0..n_res {
                let diag = sts[r * n_res + r];
                self.w_out[o * n_res + r] = if diag.abs() > 1e-15 {
                    sty[r] / diag
                } else {
                    0.0
                };
            }
        }

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

        let min_val = self.w_out.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = self.w_out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = min_val;

        let weights_i8: Vec<i8> = self.w_out
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

impl NpuReadoutWeights {
    /// Run int8 inference on a reservoir state vector.
    /// Returns f64 outputs (dequantized from int8 accumulation).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn infer(&self, state: &[f64]) -> Vec<f64> {
        let n_res = self.reservoir_size;
        let n_out = self.output_size;
        let mut output = vec![0.0_f64; n_out];

        // Quantize state to int8
        let s_min = state.iter().cloned().fold(f64::INFINITY, f64::min);
        let s_max = state.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s_range = s_max - s_min;
        let s_scale = if s_range > 0.0 { s_range / 255.0 } else { 1.0 };

        let state_i8: Vec<i8> = state.iter().map(|&v| {
            let q = ((v - s_min) / s_scale).round() as i64 - 128;
            q.clamp(-128, 127) as i8
        }).collect();

        // Int8 matmul: W_out_i8 · state_i8 → i32 accumulator → dequantize
        for o in 0..n_out {
            let mut acc = 0_i64;
            for r in 0..n_res.min(state_i8.len()) {
                acc += (self.weights_i8[o * n_res + r] as i64) * (state_i8[r] as i64);
            }
            // Dequantize: real = (acc * w_scale * s_scale) + corrections
            output[o] = acc as f64 * self.scale * s_scale;
        }

        output
    }

    /// Classify: return argmax of int8 inference output.
    #[must_use]
    pub fn classify(&self, state: &[f64]) -> usize {
        let output = self.infer(state);
        output.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
