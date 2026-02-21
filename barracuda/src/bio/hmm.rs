// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hidden Markov Model primitives — Liu 2014 (`PhyloNet`-HMM foundation).
//!
//! Implements forward, backward, and Viterbi algorithms in log-space for
//! numerical stability. These are the building blocks for introgression
//! detection (`PhyloNet`-HMM), genomic placement, and phylogenetic HMMs.
//!
//! # References
//!
//! - Liu et al. 2014, *`PLoS` Comp Bio* 10:e1003649
//!   "`PhyloNet`-HMM for detecting introgression in phylogenomic data"
//! - Rabiner 1989, "A Tutorial on Hidden Markov Models"
//!
//! # Design
//!
//! All computation uses log-probabilities to avoid underflow. The log-sum-exp
//! trick is used for numerically stable summation:
//!   `log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))`
//!
//! This is the same primitive needed by `ToadStool`'s `LogSumExp` shader.

/// Numerically stable log-sum-exp of two values.
#[inline]
#[must_use]
pub fn log_sum_exp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max_val = a.max(b);
    max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
}

/// Numerically stable log-sum-exp of a slice.
#[must_use]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max_val
        + values
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f64>()
            .ln()
}

/// HMM specification: transition and emission probabilities in log-space.
#[derive(Debug, Clone)]
pub struct HmmModel {
    /// Number of hidden states.
    pub n_states: usize,
    /// Log initial probabilities: `ln(π[i])`, length `n_states`.
    pub log_pi: Vec<f64>,
    /// Log transition matrix: `ln(A[i][j])`, row-major `n_states × n_states`.
    /// `A[i][j]` = probability of transitioning from state i to state j.
    pub log_trans: Vec<f64>,
    /// Number of emission symbols (for discrete HMM).
    pub n_symbols: usize,
    /// Log emission matrix: `ln(B[i][k])`, row-major `n_states × n_symbols`.
    /// `B[i][k]` = probability of emitting symbol k in state i.
    pub log_emit: Vec<f64>,
}

impl HmmModel {
    #[inline]
    fn log_a(&self, from: usize, to: usize) -> f64 {
        self.log_trans[from * self.n_states + to]
    }

    #[inline]
    fn log_b(&self, state: usize, symbol: usize) -> f64 {
        self.log_emit[state * self.n_symbols + symbol]
    }
}

/// Forward algorithm result.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Log-forward variables: `alpha[t][i]` = `log P(o_1..o_t, q_t = i)`.
    /// Row-major `T × n_states`.
    pub log_alpha: Vec<f64>,
    /// Log-likelihood of the entire observation sequence.
    pub log_likelihood: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Number of states.
    pub n_states: usize,
}

/// Forward algorithm in log-space.
///
/// Returns log-forward variables and overall log-likelihood.
#[must_use]
pub fn forward(model: &HmmModel, observations: &[usize]) -> ForwardResult {
    let n = model.n_states;
    let t_len = observations.len();
    let mut log_alpha = vec![f64::NEG_INFINITY; t_len * n];

    for (i, la) in log_alpha[..n].iter_mut().enumerate() {
        *la = model.log_pi[i] + model.log_b(i, observations[0]);
    }

    // Induction
    for t in 1..t_len {
        for j in 0..n {
            let mut acc = f64::NEG_INFINITY;
            for i in 0..n {
                acc = log_sum_exp2(acc, log_alpha[(t - 1) * n + i] + model.log_a(i, j));
            }
            log_alpha[t * n + j] = acc + model.log_b(j, observations[t]);
        }
    }

    // Termination
    let log_likelihood = log_sum_exp(&log_alpha[(t_len - 1) * n..t_len * n]);

    ForwardResult {
        log_alpha,
        log_likelihood,
        n_steps: t_len,
        n_states: n,
    }
}

/// Backward algorithm result.
#[derive(Debug, Clone)]
pub struct BackwardResult {
    /// Log-backward variables: `beta[t][i]` = `log P(o_{t+1}..o_T | q_t = i)`.
    pub log_beta: Vec<f64>,
    /// Number of observation time steps.
    pub n_steps: usize,
    /// Number of hidden states.
    pub n_states: usize,
}

/// Backward algorithm in log-space.
#[must_use]
pub fn backward(model: &HmmModel, observations: &[usize]) -> BackwardResult {
    let n = model.n_states;
    let t_len = observations.len();
    let mut log_beta = vec![f64::NEG_INFINITY; t_len * n];

    // Initialization: beta[T-1][i] = 1 → log(1) = 0
    for i in 0..n {
        log_beta[(t_len - 1) * n + i] = 0.0;
    }

    // Induction (backwards)
    for t in (0..t_len - 1).rev() {
        for i in 0..n {
            let mut acc = f64::NEG_INFINITY;
            for j in 0..n {
                acc = log_sum_exp2(
                    acc,
                    model.log_a(i, j)
                        + model.log_b(j, observations[t + 1])
                        + log_beta[(t + 1) * n + j],
                );
            }
            log_beta[t * n + i] = acc;
        }
    }

    BackwardResult {
        log_beta,
        n_steps: t_len,
        n_states: n,
    }
}

/// Viterbi algorithm result.
#[derive(Debug, Clone)]
pub struct ViterbiResult {
    /// Most likely state sequence.
    pub path: Vec<usize>,
    /// Log-probability of the most likely path.
    pub log_probability: f64,
}

/// Viterbi decoding in log-space — finds the most likely state sequence.
#[must_use]
pub fn viterbi(model: &HmmModel, observations: &[usize]) -> ViterbiResult {
    let n = model.n_states;
    let t_len = observations.len();
    let mut delta = vec![f64::NEG_INFINITY; t_len * n];
    let mut psi = vec![0_usize; t_len * n];

    for (i, d) in delta[..n].iter_mut().enumerate() {
        *d = model.log_pi[i] + model.log_b(i, observations[0]);
    }

    // Recursion
    for t in 1..t_len {
        for j in 0..n {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_state = 0;
            for i in 0..n {
                let candidate = delta[(t - 1) * n + i] + model.log_a(i, j);
                if candidate > best_val {
                    best_val = candidate;
                    best_state = i;
                }
            }
            delta[t * n + j] = best_val + model.log_b(j, observations[t]);
            psi[t * n + j] = best_state;
        }
    }

    // Termination: find best final state
    let mut best_final = 0;
    let mut best_prob = f64::NEG_INFINITY;
    for i in 0..n {
        if delta[(t_len - 1) * n + i] > best_prob {
            best_prob = delta[(t_len - 1) * n + i];
            best_final = i;
        }
    }

    // Backtrack
    let mut path = vec![0_usize; t_len];
    path[t_len - 1] = best_final;
    for t in (0..t_len - 1).rev() {
        path[t] = psi[(t + 1) * n + path[t + 1]];
    }

    ViterbiResult {
        path,
        log_probability: best_prob,
    }
}

/// Posterior state probabilities using forward-backward.
///
/// Returns `gamma[t][i]` = `P(q_t = i | O)` as a row-major `T × n_states` matrix.
#[must_use]
pub fn posterior(model: &HmmModel, observations: &[usize]) -> Vec<f64> {
    let fwd = forward(model, observations);
    let bwd = backward(model, observations);
    let n = model.n_states;
    let t_len = observations.len();

    let mut gamma = vec![0.0_f64; t_len * n];
    for t in 0..t_len {
        let mut log_vals: Vec<f64> = (0..n)
            .map(|i| fwd.log_alpha[t * n + i] + bwd.log_beta[t * n + i])
            .collect();
        let normalizer = log_sum_exp(&log_vals);
        for (i, lv) in log_vals.iter_mut().enumerate() {
            gamma[t * n + i] = (*lv - normalizer).exp();
        }
    }

    gamma
}

// ─── Batch API (GPU-ready) ──────────────────────────────────────────
//
// Each sequence is independent — batch entry points enable GPU dispatch
// with one workgroup per sequence.

/// Batch forward: run forward algorithm on multiple observation sequences.
///
/// Returns one `ForwardResult` per sequence. Each sequence is independent,
/// making this embarrassingly parallel on GPU.
#[must_use]
pub fn forward_batch(model: &HmmModel, sequences: &[&[usize]]) -> Vec<ForwardResult> {
    sequences.iter().map(|obs| forward(model, obs)).collect()
}

/// Batch Viterbi: decode multiple observation sequences.
#[must_use]
pub fn viterbi_batch(model: &HmmModel, sequences: &[&[usize]]) -> Vec<ViterbiResult> {
    sequences.iter().map(|obs| viterbi(model, obs)).collect()
}

/// Batch posterior: compute posterior probabilities for multiple sequences.
///
/// Returns flat `Vec<f64>` per sequence (row-major: `[t][state]`).
#[must_use]
pub fn posterior_batch(model: &HmmModel, sequences: &[&[usize]]) -> Vec<Vec<f64>> {
    sequences.iter().map(|obs| posterior(model, obs)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_weather_model() -> HmmModel {
        // Classic: 2 states (Rainy=0, Sunny=1), 3 symbols (Walk=0, Shop=1, Clean=2)
        HmmModel {
            n_states: 2,
            log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
            log_trans: vec![
                0.7_f64.ln(),
                0.3_f64.ln(), // Rainy → Rainy/Sunny
                0.4_f64.ln(),
                0.6_f64.ln(), // Sunny → Rainy/Sunny
            ],
            n_symbols: 3,
            log_emit: vec![
                0.1_f64.ln(),
                0.4_f64.ln(),
                0.5_f64.ln(), // Rainy: Walk/Shop/Clean
                0.6_f64.ln(),
                0.3_f64.ln(),
                0.1_f64.ln(), // Sunny: Walk/Shop/Clean
            ],
        }
    }

    #[test]
    fn log_sum_exp2_basic() {
        let a = 2.0_f64.ln();
        let b = 3.0_f64.ln();
        let result = log_sum_exp2(a, b);
        assert!(
            (result - 5.0_f64.ln()).abs() < 1e-12,
            "ln(2) + ln(3) = ln(5)"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn log_sum_exp2_neg_inf() {
        assert_eq!(log_sum_exp2(f64::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_sum_exp2(1.0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn log_sum_exp_slice() {
        let vals = vec![1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()];
        let result = log_sum_exp(&vals);
        assert!((result - 6.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn forward_log_likelihood_positive() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2]; // Walk, Shop, Clean
        let result = forward(&model, &obs);
        assert!(
            result.log_likelihood < 0.0,
            "log-likelihood should be negative"
        );
        assert!(
            result.log_likelihood > f64::NEG_INFINITY,
            "log-likelihood should be finite"
        );
    }

    #[test]
    fn forward_backward_consistent() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2, 0, 1];
        let fwd = forward(&model, &obs);
        let bwd = backward(&model, &obs);

        // Forward-backward consistency: sum_i alpha[t][i]*beta[t][i] = P(O)
        for t in 0..obs.len() {
            let log_vals: Vec<f64> = (0..model.n_states)
                .map(|i| {
                    fwd.log_alpha[t * model.n_states + i] + bwd.log_beta[t * model.n_states + i]
                })
                .collect();
            let ll_t = log_sum_exp(&log_vals);
            assert!(
                (ll_t - fwd.log_likelihood).abs() < 1e-10,
                "forward-backward consistency failed at t={t}: {ll_t} vs {}",
                fwd.log_likelihood
            );
        }
    }

    #[test]
    fn viterbi_finds_valid_path() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2];
        let result = viterbi(&model, &obs);
        assert_eq!(result.path.len(), 3);
        for &s in &result.path {
            assert!(s < model.n_states, "state out of range: {s}");
        }
        assert!(result.log_probability > f64::NEG_INFINITY);
    }

    #[test]
    fn viterbi_log_prob_leq_forward() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2, 0, 1];
        let fwd = forward(&model, &obs);
        let vit = viterbi(&model, &obs);
        assert!(
            vit.log_probability <= fwd.log_likelihood + 1e-10,
            "Viterbi path can't be more likely than total: {} > {}",
            vit.log_probability,
            fwd.log_likelihood
        );
    }

    #[test]
    fn posterior_sums_to_one() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2, 0, 1];
        let gamma = posterior(&model, &obs);
        for t in 0..obs.len() {
            let row_sum: f64 = (0..model.n_states)
                .map(|i| gamma[t * model.n_states + i])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "posterior at t={t} should sum to 1, got {row_sum}"
            );
        }
    }

    #[test]
    fn posterior_values_in_unit_interval() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2];
        let gamma = posterior(&model, &obs);
        for &g in &gamma {
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&g),
                "gamma out of [0,1]: {g}"
            );
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let model = simple_weather_model();
        let obs = vec![0, 1, 2, 0, 1];
        let fwd1 = forward(&model, &obs);
        let fwd2 = forward(&model, &obs);
        assert_eq!(
            fwd1.log_likelihood.to_bits(),
            fwd2.log_likelihood.to_bits(),
            "forward should be bitwise deterministic"
        );
        let vit1 = viterbi(&model, &obs);
        let vit2 = viterbi(&model, &obs);
        assert_eq!(vit1.path, vit2.path);
    }

    #[test]
    fn three_state_model() {
        let model = HmmModel {
            n_states: 3,
            log_pi: vec![
                (1.0_f64 / 3.0).ln(),
                (1.0_f64 / 3.0).ln(),
                (1.0_f64 / 3.0).ln(),
            ],
            log_trans: vec![
                0.5_f64.ln(),
                0.3_f64.ln(),
                0.2_f64.ln(),
                0.2_f64.ln(),
                0.5_f64.ln(),
                0.3_f64.ln(),
                0.3_f64.ln(),
                0.2_f64.ln(),
                0.5_f64.ln(),
            ],
            n_symbols: 2,
            log_emit: vec![
                0.9_f64.ln(),
                0.1_f64.ln(),
                0.2_f64.ln(),
                0.8_f64.ln(),
                0.5_f64.ln(),
                0.5_f64.ln(),
            ],
        };
        let obs = vec![0, 1, 0, 0, 1, 1, 0];
        let fwd = forward(&model, &obs);
        let vit = viterbi(&model, &obs);
        let gamma = posterior(&model, &obs);

        assert!(fwd.log_likelihood.is_finite());
        assert_eq!(vit.path.len(), obs.len());

        for t in 0..obs.len() {
            let row_sum: f64 = (0..3).map(|i| gamma[t * 3 + i]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }
}
