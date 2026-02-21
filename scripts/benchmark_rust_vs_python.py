#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
wetSpring — BarraCUDA CPU vs Python Timing Benchmark

Runs equivalent operations to validate_barracuda_cpu v1–v4 in pure
Python (numpy/scipy) and reports wall-clock timings for direct comparison.

Coverage: all 23 algorithmic domains that BarraCUDA validates on CPU.
"""

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def timer(func):
    """Time a function, return (result, elapsed_us)."""
    t0 = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - t0) * 1_000_000
    return result, elapsed

# ════════════════════════════════════════════════════════════════════════════
#  Domain 1: ODE Integration (RK4)
# ════════════════════════════════════════════════════════════════════════════

def rk4_step(f, y, t, dt):
    k1 = [dt * fi for fi in f(y, t)]
    k2 = [dt * fi for fi in f([yi + 0.5*k1i for yi, k1i in zip(y, k1)], t + 0.5*dt)]
    k3 = [dt * fi for fi in f([yi + 0.5*k2i for yi, k2i in zip(y, k2)], t + 0.5*dt)]
    k4 = [dt * fi for fi in f([yi + k3i for yi, k3i in zip(y, k3)], t + dt)]
    return [yi + (k1i + 2*k2i + 2*k3i + k4i)/6.0 for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)]

def rk4_integrate(f, y0, t_end, dt):
    y = list(y0)
    t = 0.0
    trajectory = [list(y)]
    while t < t_end:
        y = rk4_step(f, y, t, dt)
        t += dt
        trajectory.append(list(y))
    return trajectory

def qs_biofilm_ode(y, _t):
    S, A, H, L, B = y
    mu = 0.5; K = 1e9; d_s = 0.01; k_a = 0.1; d_a = 0.05
    k_h = 0.5; n = 2.0; K_h = 0.5; d_h = 0.1
    k_l = 0.3; d_l = 0.05; k_b = 0.2; d_b = 0.02
    dS = mu * S * (1 - S/K) - d_s * S
    dA = k_a * S - d_a * A
    dH = k_h * (A**n / (K_h**n + A**n)) - d_h * H
    dL = k_l * H - d_l * L
    dB = k_b * L * (1 - B) - d_b * B
    return [dS, dA, dH, dL, dB]


def domain_1_ode():
    y0 = [1000.0, 0.0, 0.0, 0.0, 0.0]
    traj = rk4_integrate(qs_biofilm_ode, y0, 48.0, 0.001)
    return len(traj)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 2: Stochastic Simulation (Gillespie SSA)
# ════════════════════════════════════════════════════════════════════════════

class LCG:
    def __init__(self, seed):
        self.state = seed
    def next_f64(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 11) / (1 << 53)

def gillespie_ssa(initial, propensity_fns, stoich, t_end, rng):
    state = list(initial)
    t = 0.0
    while t < t_end:
        rates = [f(state) for f in propensity_fns]
        total = sum(rates)
        if total <= 0:
            break
        u1 = rng.next_f64()
        if u1 == 0.0:
            u1 = 1e-300
        dt = -math.log(u1) / total
        t += dt
        if t > t_end:
            break
        u2 = rng.next_f64()
        cumsum = 0.0
        for i, r in enumerate(rates):
            cumsum += r
            if cumsum >= u2 * total:
                for j in range(len(state)):
                    state[j] += stoich[i][j]
                break
    return state

def domain_2_ssa():
    total = 0
    for seed in range(100):
        rng = LCG(seed)
        result = gillespie_ssa(
            [100],
            [lambda s: 0.5 * s[0], lambda s: 0.1 * s[0]],
            [[1], [-1]],
            10.0,
            rng
        )
        total += result[0]
    return total / 100


# ════════════════════════════════════════════════════════════════════════════
#  Domain 3: HMM (Forward, Viterbi)
# ════════════════════════════════════════════════════════════════════════════

def log_sum_exp(a, b):
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))

def hmm_forward(log_pi, log_trans, log_emit, obs, n_states, n_symbols):
    T = len(obs)
    alpha = [[float('-inf')] * n_states for _ in range(T)]
    for s in range(n_states):
        alpha[0][s] = log_pi[s] + log_emit[s * n_symbols + obs[0]]
    for t in range(1, T):
        for j in range(n_states):
            for i in range(n_states):
                alpha[t][j] = log_sum_exp(
                    alpha[t][j],
                    alpha[t-1][i] + log_trans[i * n_states + j]
                )
            alpha[t][j] += log_emit[j * n_symbols + obs[t]]
    ll = float('-inf')
    for s in range(n_states):
        ll = log_sum_exp(ll, alpha[T-1][s])
    return ll

def hmm_viterbi(log_pi, log_trans, log_emit, obs, n_states, n_symbols):
    T = len(obs)
    dp = [[float('-inf')] * n_states for _ in range(T)]
    bt = [[0] * n_states for _ in range(T)]
    for s in range(n_states):
        dp[0][s] = log_pi[s] + log_emit[s * n_symbols + obs[0]]
    for t in range(1, T):
        for j in range(n_states):
            for i in range(n_states):
                score = dp[t-1][i] + log_trans[i * n_states + j]
                if score > dp[t][j]:
                    dp[t][j] = score
                    bt[t][j] = i
            dp[t][j] += log_emit[j * n_symbols + obs[t]]
    best = max(range(n_states), key=lambda s: dp[T-1][s])
    path = [0] * T
    path[T-1] = best
    for t in range(T-2, -1, -1):
        path[t] = bt[t+1][path[t+1]]
    return dp[T-1][best], path

def domain_3_hmm():
    log_pi = [math.log(0.5), math.log(0.5)]
    log_trans = [math.log(0.7), math.log(0.3), math.log(0.4), math.log(0.6)]
    log_emit = [math.log(0.5), math.log(0.4), math.log(0.1),
                math.log(0.1), math.log(0.3), math.log(0.6)]
    obs = [0, 1, 2, 1, 0]
    ll = hmm_forward(log_pi, log_trans, log_emit, obs, 2, 3)
    vll, path = hmm_viterbi(log_pi, log_trans, log_emit, obs, 2, 3)
    return ll, vll, path


# ════════════════════════════════════════════════════════════════════════════
#  Domain 4: Smith-Waterman
# ════════════════════════════════════════════════════════════════════════════

def smith_waterman(q, t, match_s=2, mismatch=-1, gap_open=0, gap_ext=-2):
    m, n = len(q), len(t)
    H = [[0]*(n+1) for _ in range(m+1)]
    best = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            diag = H[i-1][j-1] + (match_s if q[i-1] == t[j-1] else mismatch)
            up = max(H[i-k][j] + gap_open + gap_ext * k for k in range(1, i+1))
            left = max(H[i][j-k] + gap_open + gap_ext * k for k in range(1, j+1))
            H[i][j] = max(0, diag, up, left)
            best = max(best, H[i][j])
    return best

def domain_4_sw():
    q = "GATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATAC"
    t = "GATCCTGGCTCAGAATGAACGCTGGCGGCATGCCTAATAC"
    return smith_waterman(q, t)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 5: Felsenstein Pruning
# ════════════════════════════════════════════════════════════════════════════

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def jc_transition_prob(branch_length, mu):
    e = math.exp(-4.0 * mu * branch_length / 3.0)
    p_same = 0.25 + 0.75 * e
    p_diff = 0.25 - 0.25 * e
    return p_same, p_diff

def felsenstein_pruning(tree, mu):
    seqs = [tree['left']['left']['seq'], tree['left']['right']['seq'], tree['right']['seq']]
    n_sites = len(seqs[0])
    ll = 0.0
    for site in range(n_sites):
        cond = compute_site(tree, site, mu)
        site_ll = sum(0.25 * cond[i] for i in range(4))
        ll += math.log(site_ll) if site_ll > 0 else float('-inf')
    return ll

def compute_site(tree, site, mu):
    if tree.get('type') == 'leaf':
        cond = [0.0]*4
        cond[BASE_MAP[tree['seq'][site]]] = 1.0
        return cond
    left_cond = compute_site(tree['left'], site, mu)
    right_cond = compute_site(tree['right'], site, mu)
    ps_l, pd_l = jc_transition_prob(tree['left_branch'], mu)
    ps_r, pd_r = jc_transition_prob(tree['right_branch'], mu)
    result = [0.0]*4
    for i in range(4):
        l_sum = sum(pd_l * left_cond[j] if j != i else ps_l * left_cond[j] for j in range(4))
        r_sum = sum(pd_r * right_cond[j] if j != i else ps_r * right_cond[j] for j in range(4))
        result[i] = l_sum * r_sum
    return result

def domain_5_felsenstein():
    tree = {
        'type': 'internal',
        'left': {
            'type': 'internal',
            'left': {'type': 'leaf', 'seq': 'ACGTACGTACGTACGTACGT'},
            'right': {'type': 'leaf', 'seq': 'ACGTACTTACGTACGTACGT'},
            'left_branch': 0.05,
            'right_branch': 0.05,
        },
        'right': {'type': 'leaf', 'seq': 'ACGTACGTACTTACGTACGT'},
        'left_branch': 0.1,
        'right_branch': 0.15,
    }
    leaves = [
        {'seq': 'ACGTACGTACGTACGTACGT'},
        {'seq': 'ACGTACTTACGTACGTACGT'},
        {'seq': 'ACGTACGTACTTACGTACGT'},
    ]
    tree['leaves'] = leaves
    return felsenstein_pruning(tree, 1.0)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 6: Diversity Metrics
# ════════════════════════════════════════════════════════════════════════════

def shannon(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h

def simpson(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c/total)**2 for c in counts)

def pielou(counts):
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    return shannon(counts) / math.log(s)

def bray_curtis(a, b):
    num = sum(abs(ai - bi) for ai, bi in zip(a, b))
    den = sum(ai + bi for ai, bi in zip(a, b))
    return num / den if den > 0 else 0.0

def domain_6_diversity():
    c = [10.0, 20.0, 30.0, 15.0, 25.0]
    return shannon(c), simpson(c), pielou(c), bray_curtis(c, [15,25,35,20,30])


# ════════════════════════════════════════════════════════════════════════════
#  Domain 7: Signal Processing
# ════════════════════════════════════════════════════════════════════════════

def find_peaks(signal, min_height=0.5):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if signal[i] >= min_height:
                peaks.append(i)
    return peaks

def domain_7_signal():
    sig = [abs(math.sin(i / 100.0 * math.tau * 3.0)) for i in range(100)]
    return find_peaks(sig)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 8: Game Theory (Cooperation ODE)
# ════════════════════════════════════════════════════════════════════════════

def cooperation_ode(y, _t):
    C, D, S = y
    mu = 0.3; K = 1e9; b = 0.1; cost = 0.05; exploit = 0.15
    dC = mu * C * (1 - (C+D)/K) + b*S*C - cost*C
    dD = mu * D * (1 - (C+D)/K) + exploit*S*D
    dS = 0.5 * C - 0.1 * S
    return [dC, dD, dS]

def domain_8_cooperation():
    traj = rk4_integrate(cooperation_ode, [500.0, 500.0, 0.0], 100.0, 0.001)
    final = traj[-1]
    return final[0] / (final[0] + final[1]) if (final[0] + final[1]) > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════════
#  Domain 9: Robinson-Foulds
# ════════════════════════════════════════════════════════════════════════════

def get_splits(newick, taxa):
    """Simplified RF for 4-taxon trees."""
    splits_a = set()
    inner = newick.strip().rstrip(';').strip('(').rstrip(')')
    parts = []
    depth = 0
    current = ''
    for ch in inner:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    parts.append(current)

    for part in parts:
        p = part.strip()
        if p.startswith('(') and p.endswith(')'):
            inner_taxa = [x.strip() for x in p[1:-1].split(',')]
            split = frozenset(inner_taxa)
            comp = frozenset(taxa) - split
            if len(split) > 0 and len(comp) > 0:
                canonical = min(split, comp, key=lambda s: sorted(s))
                splits_a.add(frozenset(canonical))
    return splits_a

def rf_distance(nw_a, nw_b, taxa):
    sa = get_splits(nw_a, taxa)
    sb = get_splits(nw_b, taxa)
    return len(sa.symmetric_difference(sb))

def domain_9_rf():
    taxa = ['A', 'B', 'C', 'D']
    return rf_distance("((A,B),(C,D));", "((A,C),(B,D));", taxa)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 10: Multi-Signal QS
# ════════════════════════════════════════════════════════════════════════════

def multi_signal_ode(y, _t):
    S, CAI1, AI2, LuxO, HapR, cdGMP, B = y
    mu = 0.5; K = 1e9; d_s = 0.01; k_c = 0.1; d_c = 0.05
    k_a = 0.1; d_a = 0.05; k_l = 0.3; n = 2; K_l = 0.5; d_l = 0.1
    k_h = 0.5; K_h = 0.5; d_h = 0.1; k_g = 0.2; d_g = 0.1
    k_b = 0.1; d_b = 0.02
    dS = mu * S * (1 - S/K) - d_s * S
    dCAI1 = k_c * S - d_c * CAI1
    dAI2 = k_a * S - d_a * AI2
    dLuxO = k_l * (K_l**n / (K_l**n + (CAI1 + AI2)**n)) - d_l * LuxO
    dHapR = k_h * (K_h**n / (K_h**n + LuxO**n)) - d_h * HapR
    dcdGMP = k_g * HapR - d_g * cdGMP
    dB = k_b * cdGMP * (1 - B) - d_b * B
    return [dS, dCAI1, dAI2, dLuxO, dHapR, dcdGMP, dB]

def domain_10_multi_signal():
    y0 = [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    traj = rk4_integrate(multi_signal_ode, y0, 48.0, 0.001)
    return len(traj)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 11: Phage Defense
# ════════════════════════════════════════════════════════════════════════════

def phage_defense_ode(y, _t):
    Bd, Bu, P, R = y
    mu = 0.5; cost = 0.1; K_r = 10.0; Y = 5e-7; ads = 1e-9
    burst = 50; eff = 0.9; decay = 0.1; inflow = 10.0; dilution = 0.1; death = 0.01
    growth_d = mu * (1 - cost) * R / (K_r + R)
    growth_u = mu * R / (K_r + R)
    inf_d = ads * (1 - eff) * P * Bd
    inf_u = ads * P * Bu
    dBd = growth_d * Bd - inf_d - death * Bd
    dBu = growth_u * Bu - inf_u - death * Bu
    dP = burst * (inf_d + inf_u) - decay * P - ads * P * (Bd + Bu)
    dR = inflow - dilution * R - (growth_d * Bd + growth_u * Bu) / Y
    return [dBd, dBu, dP, dR]

def domain_11_phage():
    y0 = [1e6, 1e6, 1e4, 100.0]
    traj = rk4_integrate(phage_defense_ode, y0, 48.0, 0.001)
    return len(traj)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 12: Bootstrap Resampling
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_resample(alignment, n_sites, rng):
    resampled = []
    for _ in range(n_sites):
        col = int(rng.next_f64() * n_sites) % n_sites
        resampled.append(col)
    return resampled

def domain_12_bootstrap():
    seq_a = "ACGTACGT"
    seq_b = "ACTTACTT"
    rng = LCG(42)
    lls = []
    for _ in range(100):
        cols = bootstrap_resample(None, 8, rng)
        ll = 0.0
        for col in cols:
            if seq_a[col] == seq_b[col]:
                ll += math.log(0.25 + 0.75 * math.exp(-4.0/3.0 * 0.1))
            else:
                ll += math.log(0.25 - 0.25 * math.exp(-4.0/3.0 * 0.1))
        lls.append(ll)
    return lls


# ════════════════════════════════════════════════════════════════════════════
#  Domain 13: Phylogenetic Placement
# ════════════════════════════════════════════════════════════════════════════

def domain_13_placement():
    query = "ACGTACGTACGT"
    refs = ["ACGTACGTACGT", "ACGTACTTACGT", "ACTTACTTACTT"]
    scores = []
    for ref in refs:
        ll = sum(
            math.log(0.25 + 0.75 * math.exp(-4.0/3.0 * 0.05))
            if query[i] == ref[i]
            else math.log(0.25 - 0.25 * math.exp(-4.0/3.0 * 0.05))
            for i in range(len(query))
        )
        scores.append(ll)
    return scores


# ════════════════════════════════════════════════════════════════════════════
#  Domain 14: Decision Tree
# ════════════════════════════════════════════════════════════════════════════

def decision_stump_predict(feature_idx, threshold, sample, class_left, class_right):
    return class_left if sample[feature_idx] < threshold else class_right

def domain_14_dt():
    samples = [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    return [decision_stump_predict(0, 5.0, s, 0, 1) for s in samples]


# ════════════════════════════════════════════════════════════════════════════
#  Domain 15: Spectral Matching
# ════════════════════════════════════════════════════════════════════════════

def cosine_similarity(mz_q, int_q, mz_r, int_r, tol):
    matched = []
    used_r = set()
    for i, mq in enumerate(mz_q):
        best_j = -1
        best_diff = tol + 1
        for j, mr in enumerate(mz_r):
            if j not in used_r and abs(mq - mr) <= tol:
                if abs(mq - mr) < best_diff:
                    best_diff = abs(mq - mr)
                    best_j = j
        if best_j >= 0:
            matched.append((i, best_j))
            used_r.add(best_j)
    if not matched:
        return 0.0
    dot = sum(int_q[i] * int_r[j] for i, j in matched)
    norm_q = math.sqrt(sum(int_q[i]**2 for i, _ in matched))
    norm_r = math.sqrt(sum(int_r[j]**2 for _, j in matched))
    return dot / (norm_q * norm_r) if norm_q > 0 and norm_r > 0 else 0.0

def domain_15_spectral():
    mz = [100.0, 200.0, 300.0, 400.0, 500.0]
    int_a = [1000.0, 500.0, 800.0, 300.0, 600.0]
    int_b = [900.0, 550.0, 750.0, 350.0, 550.0]
    s1 = cosine_similarity(mz, int_a, mz, int_a, 0.5)
    s2 = cosine_similarity(mz, int_a, mz, int_b, 0.5)
    return s1, s2


# ════════════════════════════════════════════════════════════════════════════
#  Domain 16: Extended Diversity
# ════════════════════════════════════════════════════════════════════════════

def chao1(counts):
    observed = sum(1 for c in counts if c > 0)
    f1 = sum(1 for c in counts if c == 1)
    f2 = sum(1 for c in counts if c == 2)
    if f2 == 0:
        return observed + f1 * (f1 - 1) / 2.0 if f1 > 0 else float(observed)
    return observed + f1**2 / (2.0 * f2)

def domain_16_ext_diversity():
    even = [25.0]*4
    uneven = [97.0, 1.0, 1.0, 1.0]
    return (
        pielou(even), pielou(uneven),
        bray_curtis([10,20,30,40], [10,20,30,40]),
        bray_curtis([10,20,30,40], [40,30,20,10]),
        chao1([10, 5, 1, 1, 0, 0, 0, 0]),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Domain 17: K-mer Counting
# ════════════════════════════════════════════════════════════════════════════

ENCODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def count_kmers(seq, k):
    counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = 0
        valid = True
        for j in range(k):
            b = ENCODE.get(seq[i+j])
            if b is None:
                valid = False
                break
            kmer = (kmer << 2) | b
        if valid:
            counts[kmer] += 1
    return counts

def domain_17_kmer():
    return count_kmers("ACGTACGTACGTACGT", 4)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 18: Integrated Pipeline
# ════════════════════════════════════════════════════════════════════════════

def domain_18_pipeline():
    c = [10.0, 20.0, 30.0, 15.0, 25.0]
    h = shannon(c)
    bc = bray_curtis([10, 20, 30], [15, 25, 35])
    cs = cosine_similarity(
        [100, 200, 300], [1000, 500, 800],
        [100, 200, 300], [950, 520, 780], 0.5
    )
    return h, bc, cs


# ════════════════════════════════════════════════════════════════════════════
#  Domain 19: ANI (Average Nucleotide Identity)
# ════════════════════════════════════════════════════════════════════════════

def pairwise_ani(seq1, seq2):
    aligned = identical = 0
    for a, b in zip(seq1, seq2):
        a, b = a.upper(), b.upper()
        if a in ('-', '.', 'N') or b in ('-', '.', 'N'):
            continue
        aligned += 1
        if a == b:
            identical += 1
    return (identical / aligned if aligned > 0 else 0.0), aligned, identical

def domain_19_ani():
    seqs = [
        "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        "ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG",
        "CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
    ]
    return [pairwise_ani(seqs[i], seqs[j])
            for i in range(len(seqs)) for j in range(i)]


# ════════════════════════════════════════════════════════════════════════════
#  Domain 20: SNP Calling
# ════════════════════════════════════════════════════════════════════════════

VALID_BASES_SNP = ['A', 'C', 'G', 'T']

def call_snps(sequences):
    if not sequences:
        return [], 0, 0
    aln_len = len(sequences[0])
    variants = []
    for pos in range(aln_len):
        counts = {b: 0 for b in VALID_BASES_SNP}
        depth = 0
        for seq in sequences:
            base = seq[pos].upper() if pos < len(seq) else '-'
            if base in ('-', '.', 'N'):
                continue
            depth += 1
            if base in counts:
                counts[base] += 1
        n_alleles = sum(1 for c in counts.values() if c > 0)
        if n_alleles < 2 or depth < 2:
            continue
        variants.append(pos)
    return variants

def domain_20_snp():
    seqs = [
        "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        "ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG",
        "CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
        "ATGATCATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
    ]
    return call_snps(seqs)


# ════════════════════════════════════════════════════════════════════════════
#  Domain 21: dN/dS (Nei-Gojobori 1986)
# ════════════════════════════════════════════════════════════════════════════

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def translate_codon_d21(codon):
    return GENETIC_CODE.get(codon.upper())

def codon_sites_d21(codon):
    orig_aa = translate_codon_d21(codon)
    if orig_aa is None or orig_aa == '*':
        return 0.0, 0.0
    syn = 0.0
    for pos in range(3):
        syn_changes = total = 0
        for base in 'ACGT':
            if base == codon[pos].upper():
                continue
            mutant = list(codon.upper())
            mutant[pos] = base
            total += 1
            new_aa = translate_codon_d21(''.join(mutant))
            if new_aa and new_aa != '*' and new_aa == orig_aa:
                syn_changes += 1
        if total > 0:
            syn += syn_changes / total
    return syn, 3.0 - syn

def pairwise_dnds(seq1, seq2):
    from itertools import permutations
    ts, tn, sd, nd = 0.0, 0.0, 0.0, 0.0
    for i in range(0, len(seq1), 3):
        c1, c2 = seq1[i:i+3], seq2[i:i+3]
        if '-' in c1 or '.' in c1 or '-' in c2 or '.' in c2:
            continue
        s1s, s1n = codon_sites_d21(c1)
        s2s, s2n = codon_sites_d21(c2)
        ts += (s1s + s2s) / 2.0
        tn += (s1n + s2n) / 2.0
        diff_pos = [j for j in range(3) if c1[j].upper() != c2[j].upper()]
        if not diff_pos:
            continue
        if len(diff_pos) == 1:
            aa1 = translate_codon_d21(c1)
            aa2 = translate_codon_d21(c2)
            if aa1 and aa2 and aa1 == aa2:
                sd += 1.0
            else:
                nd += 1.0
        else:
            ps, pn = 0.0, 0.0
            for perm in permutations(diff_pos):
                cur = list(c1.upper())
                trg = list(c2.upper())
                for p in perm:
                    a_before = translate_codon_d21(''.join(cur))
                    cur[p] = trg[p]
                    a_after = translate_codon_d21(''.join(cur))
                    if a_before and a_after and a_before == a_after:
                        ps += 1.0
                    else:
                        pn += 1.0
            np_ = math.factorial(len(diff_pos))
            sd += ps / np_
            nd += pn / np_
    p_s = sd / ts if ts > 0 else 0.0
    p_n = nd / tn if tn > 0 else 0.0
    jc = lambda p: 0.0 if p <= 0 else (-0.75 * math.log(max(1 - 4*p/3, 1e-300)))
    return jc(p_n), jc(p_s)

def domain_21_dnds():
    return pairwise_dnds("ATGGCTAAATTTGCTGCTGCTGCTGCTGCT",
                         "ATGGCCGAATTTGCTGCTGCTGCTGCCGCT")


# ════════════════════════════════════════════════════════════════════════════
#  Domain 22: Molecular Clock
# ════════════════════════════════════════════════════════════════════════════

def domain_22_clock():
    bl = [0.0, 0.1, 0.2, 0.05, 0.05, 0.15, 0.15]
    par = [None, 0, 0, 1, 1, 2, 2]
    dist = [0.0] * 7
    for i in range(7):
        if par[i] is not None:
            dist[i] = dist[par[i]] + bl[i]
    tree_h = max(dist)
    rate = tree_h / 3500.0
    ages = [3500.0 - d / rate for d in dist]
    rates = [0.0] * 7
    for i in range(7):
        if par[i] is not None:
            span = ages[par[i]] - ages[i]
            if span > 0:
                rates[i] = bl[i] / span
    pos = [r for r in rates if r > 0]
    mean = sum(pos) / len(pos)
    var = sum((r - mean)**2 for r in pos) / (len(pos) - 1)
    cv = math.sqrt(var) / mean
    return rate, ages, cv


# ════════════════════════════════════════════════════════════════════════════
#  Domain 23: Pangenome Analysis
# ════════════════════════════════════════════════════════════════════════════

def domain_23_pangenome():
    presence = [
        [True]*5, [True]*5, [True]*5,
        [True, True, False, False, False],
        [False, True, True, False, False],
        [True, False, False, False, False],
        [False, False, False, False, True],
    ]
    core = acc = uniq = 0
    for row in presence:
        c = sum(row)
        if c == 5: core += 1
        elif c == 1: uniq += 1
        elif c > 1: acc += 1
    # Hypergeometric + BH
    def hyper_p(k, n, big_k, big_n):
        if big_n == 0: return 1.0
        exp = n * big_k / big_n
        if k <= exp: return 1.0
        var = n*big_k*(big_n-big_k)*(big_n-n)/(big_n*big_n*max(big_n-1,1))
        if var <= 0: return 0.0 if k > exp else 1.0
        z = (k - exp) / math.sqrt(var)
        return 1 - 0.5*(1 + math.erf(z / math.sqrt(2)))
    pvals = [hyper_p(8,10,20,100), hyper_p(2,10,20,100), hyper_p(5,10,20,100)]
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0]*3
    cm = float('inf')
    for i in range(2, -1, -1):
        a = indexed[i][1] * 3 / (i+1)
        cm = min(cm, a, 1.0)
        adj[indexed[i][0]] = cm
    return core, acc, uniq, adj


# ════════════════════════════════════════════════════════════════════════════
#  Main: Run all domains, collect timings
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  wetSpring — Python Timing Baseline (23 Domains)")
    print("=" * 60)
    print()

    results = {}
    domains = [
        ("D01: ODE Integration (RK4)", domain_1_ode),
        ("D02: Gillespie SSA (100 reps)", domain_2_ssa),
        ("D03: HMM (Forward + Viterbi)", domain_3_hmm),
        ("D04: Smith-Waterman (40bp)", domain_4_sw),
        ("D05: Felsenstein (20bp, 3 taxa)", domain_5_felsenstein),
        ("D06: Diversity (Shannon+Simpson)", domain_6_diversity),
        ("D07: Signal Processing", domain_7_signal),
        ("D08: Cooperation ODE (100h)", domain_8_cooperation),
        ("D09: Robinson-Foulds (4 taxa)", domain_9_rf),
        ("D10: Multi-Signal QS (48h)", domain_10_multi_signal),
        ("D11: Phage Defense (48h)", domain_11_phage),
        ("D12: Bootstrap (100 reps, 8bp)", domain_12_bootstrap),
        ("D13: Placement (3 taxa, 12bp)", domain_13_placement),
        ("D14: Decision Tree (4 samples)", domain_14_dt),
        ("D15: Spectral Match (5 peaks)", domain_15_spectral),
        ("D16: Extended Diversity", domain_16_ext_diversity),
        ("D17: K-mer Counting (16bp, k=4)", domain_17_kmer),
        ("D18: Integrated Pipeline", domain_18_pipeline),
        ("D19: ANI (3 seqs, 50bp)", domain_19_ani),
        ("D20: SNP Calling (4 seqs, 50bp)", domain_20_snp),
        ("D21: dN/dS (10 codons)", domain_21_dnds),
        ("D22: Molecular Clock (7 nodes)", domain_22_clock),
        ("D23: Pangenome (7 genes)", domain_23_pangenome),
    ]

    total_us = 0.0
    for name, func in domains:
        _, us = timer(func)
        results[name] = us
        total_us += us
        print(f"  {name:<40} {us:>12.0f} µs")

    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':<40} {total_us:>12.0f} µs")
    print()

    out_dir = Path(__file__).parent.parent / "experiments" / "results" / "059_23_domain_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "python_timing.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_us": total_us,
            "domains": {k: v for k, v in results.items()},
        }, f, indent=2)
    print(f"  Timings written to {out_file}")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
