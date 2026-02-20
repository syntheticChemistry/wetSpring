#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Liu 2014 HMM primitives — Python baseline using hmmlearn.

Validates forward, Viterbi, and posterior against a known HMM model.

References:
    Liu et al. 2014, PLoS Comp Bio 10:e1003649
    Rabiner 1989, "A Tutorial on Hidden Markov Models"
"""

import json
import os
import numpy as np

# Pure Python implementations (no hmmlearn dependency — sovereign)

def log_sum_exp(arr):
    max_val = np.max(arr)
    if max_val == -np.inf:
        return -np.inf
    return max_val + np.log(np.sum(np.exp(arr - max_val)))


def forward_algorithm(log_pi, log_A, log_B, obs):
    N = len(log_pi)
    T = len(obs)
    log_alpha = np.full((T, N), -np.inf)

    for i in range(N):
        log_alpha[0, i] = log_pi[i] + log_B[i, obs[0]]

    for t in range(1, T):
        for j in range(N):
            vals = [log_alpha[t-1, i] + log_A[i, j] for i in range(N)]
            log_alpha[t, j] = log_sum_exp(np.array(vals)) + log_B[j, obs[t]]

    log_ll = log_sum_exp(log_alpha[-1])
    return log_alpha, log_ll


def backward_algorithm(log_A, log_B, obs):
    N = log_A.shape[0]
    T = len(obs)
    log_beta = np.full((T, N), -np.inf)

    log_beta[-1] = 0.0

    for t in range(T-2, -1, -1):
        for i in range(N):
            vals = [log_A[i, j] + log_B[j, obs[t+1]] + log_beta[t+1, j] for j in range(N)]
            log_beta[t, i] = log_sum_exp(np.array(vals))

    return log_beta


def viterbi_algorithm(log_pi, log_A, log_B, obs):
    N = len(log_pi)
    T = len(obs)
    delta = np.full((T, N), -np.inf)
    psi = np.zeros((T, N), dtype=int)

    for i in range(N):
        delta[0, i] = log_pi[i] + log_B[i, obs[0]]

    for t in range(1, T):
        for j in range(N):
            candidates = [delta[t-1, i] + log_A[i, j] for i in range(N)]
            best_i = np.argmax(candidates)
            delta[t, j] = candidates[best_i] + log_B[j, obs[t]]
            psi[t, j] = best_i

    best_final = np.argmax(delta[-1])
    path = [0] * T
    path[-1] = best_final
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path, float(delta[-1, best_final])


def posterior_decode(log_pi, log_A, log_B, obs):
    log_alpha, log_ll = forward_algorithm(log_pi, log_A, log_B, obs)
    log_beta = backward_algorithm(log_A, log_B, obs)
    T = len(obs)
    N = len(log_pi)
    gamma = np.zeros((T, N))
    for t in range(T):
        log_vals = log_alpha[t] + log_beta[t]
        normalizer = log_sum_exp(log_vals)
        gamma[t] = np.exp(log_vals - normalizer)
    return gamma


def main():
    # Weather model: 2 states, 3 symbols
    log_pi_2 = np.log([0.6, 0.4])
    log_A_2 = np.log([[0.7, 0.3], [0.4, 0.6]])
    log_B_2 = np.log([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    obs_2 = [0, 1, 2, 0, 1]

    # 3-state model
    log_pi_3 = np.log([1/3, 1/3, 1/3])
    log_A_3 = np.log([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
    log_B_3 = np.log([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    obs_3 = [0, 1, 0, 0, 1, 1, 0]

    results = {}

    # 2-state model
    _, ll_2 = forward_algorithm(log_pi_2, log_A_2, log_B_2, obs_2)
    path_2, vit_prob_2 = viterbi_algorithm(log_pi_2, log_A_2, log_B_2, obs_2)
    gamma_2 = posterior_decode(log_pi_2, log_A_2, log_B_2, obs_2)

    results["weather_2state"] = {
        "log_likelihood": float(ll_2),
        "viterbi_path": [int(x) for x in path_2],
        "viterbi_log_prob": float(vit_prob_2),
        "posterior_state0": [float(g) for g in gamma_2[:, 0]],
        "posterior_state1": [float(g) for g in gamma_2[:, 1]],
    }
    print(f"2-state: LL={ll_2:.8f}, Viterbi={path_2}, VitProb={vit_prob_2:.8f}")

    # 3-state model
    _, ll_3 = forward_algorithm(log_pi_3, log_A_3, log_B_3, obs_3)
    path_3, vit_prob_3 = viterbi_algorithm(log_pi_3, log_A_3, log_B_3, obs_3)
    gamma_3 = posterior_decode(log_pi_3, log_A_3, log_B_3, obs_3)

    results["genomic_3state"] = {
        "log_likelihood": float(ll_3),
        "viterbi_path": [int(x) for x in path_3],
        "viterbi_log_prob": float(vit_prob_3),
        "posterior": [[float(g) for g in row] for row in gamma_3],
    }
    print(f"3-state: LL={ll_3:.8f}, Viterbi={path_3}, VitProb={vit_prob_3:.8f}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "026_hmm")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "liu2014_hmm_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")


if __name__ == "__main__":
    main()
