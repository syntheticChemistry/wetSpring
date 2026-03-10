// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reservoir generation and update logic.

use super::config::EsnConfig;

/// Simple LCG PRNG for deterministic reservoir generation.
pub struct Lcg(u64);

impl Lcg {
    /// Create a new LCG PRNG with the given seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub(super) fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        f64::from((self.0 >> 33) as u32) / f64::from(u32::MAX)
    }
    pub(super) fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(crate::tolerances::BOX_MULLER_U1_FLOOR);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Build input weight matrix `W_in`: `input_size` × `reservoir_size`, uniform [-1, 1].
pub(super) fn build_w_in(config: &EsnConfig, rng: &mut Lcg) -> Vec<f64> {
    let n_in = config.input_size;
    let n_res = config.reservoir_size;
    (0..n_in * n_res)
        .map(|_| rng.next_f64().mul_add(2.0, -1.0))
        .collect()
}

/// Build reservoir weight matrix `W_res`: sparse Gaussian, scaled to spectral radius.
pub(super) fn build_w_res(config: &EsnConfig, rng: &mut Lcg) -> Vec<f64> {
    let n_res = config.reservoir_size;
    let mut w_res = vec![0.0_f64; n_res * n_res];
    for i in 0..n_res {
        for j in 0..n_res {
            if rng.next_f64() < config.connectivity {
                w_res[i * n_res + j] = rng.next_gaussian();
            }
        }
    }

    let max_abs = w_res.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0 {
        let scale = config.spectral_radius / max_abs;
        for w in &mut w_res {
            *w *= scale;
        }
    }
    w_res
}

/// Run one reservoir update step: `state = (1-leak)*state + leak*tanh(W_in*input + W_res*state)`.
pub(super) fn update_state(
    state: &mut [f64],
    w_in: &[f64],
    w_res: &[f64],
    input: &[f64],
    config: &EsnConfig,
) {
    let n_res = config.reservoir_size;
    let n_in = config.input_size;
    let leak = config.leak_rate;

    let mut new_state = vec![0.0_f64; n_res];

    for j in 0..n_res {
        let mut val = 0.0;
        for i in 0..n_in.min(input.len()) {
            val += w_in[i * n_res + j] * input[i];
        }
        for i in 0..n_res {
            val += w_res[i * n_res + j] * state[i];
        }
        new_state[j] = (1.0 - leak).mul_add(state[j], leak * val.tanh());
    }

    state.copy_from_slice(&new_state);
}
