#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-21
"""
wetSpring — BarraCUDA CPU v4: Track 1c Python Timing Baseline

Pure-Python implementations of the 5 Track 1c domains for timing
comparison against Rust. Covers: ANI, SNP, dN/dS, molecular clock,
and pangenome analysis.

Usage:
    python3 scripts/barracuda_cpu_v4_baseline.py
"""

import math
import time

def timer(func):
    t0 = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - t0) * 1_000_000
    return result, elapsed


# ════════════════════════════════════════════════════════════════════════
#  Domain 19: ANI (Average Nucleotide Identity)
# ════════════════════════════════════════════════════════════════════════

def pairwise_ani(seq1, seq2):
    aligned = 0
    identical = 0
    for a, b in zip(seq1, seq2):
        a, b = a.upper(), b.upper()
        if a in ('-', '.', 'N') or b in ('-', '.', 'N'):
            continue
        aligned += 1
        if a == b:
            identical += 1
    ani = identical / aligned if aligned > 0 else 0.0
    return ani, aligned, identical

def ani_matrix(sequences):
    n = len(sequences)
    matrix = []
    for i in range(1, n):
        for j in range(i):
            ani, _, _ = pairwise_ani(sequences[i], sequences[j])
            matrix.append(ani)
    return matrix


# ════════════════════════════════════════════════════════════════════════
#  Domain 20: SNP Calling
# ════════════════════════════════════════════════════════════════════════

VALID_BASES = ['A', 'C', 'G', 'T']

def call_snps(sequences):
    if not sequences:
        return [], 0, 0
    aln_len = len(sequences[0])
    variants = []
    for pos in range(aln_len):
        counts = {b: 0 for b in VALID_BASES}
        depth = 0
        for seq in sequences:
            if pos >= len(seq):
                continue
            base = seq[pos].upper()
            if base in ('-', '.', 'N'):
                continue
            depth += 1
            if base in counts:
                counts[base] += 1
        n_alleles = sum(1 for c in counts.values() if c > 0)
        if n_alleles < 2 or depth < 2:
            continue
        ref_base = max(counts, key=counts.get)
        alt_alleles = [(b, c) for b, c in counts.items() if b != ref_base and c > 0]
        variants.append({
            'position': pos,
            'ref_allele': ref_base,
            'alt_alleles': alt_alleles,
            'depth': depth,
        })
    return variants, aln_len, len(sequences)


# ════════════════════════════════════════════════════════════════════════
#  Domain 21: dN/dS (Nei-Gojobori 1986)
# ════════════════════════════════════════════════════════════════════════

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

def translate_codon(codon):
    return GENETIC_CODE.get(codon.upper())

def codon_sites(codon):
    orig_aa = translate_codon(codon)
    if orig_aa is None or orig_aa == '*':
        return 0.0, 0.0
    syn = 0.0
    for pos in range(3):
        syn_changes = 0
        total = 0
        for base in 'ACGT':
            if base == codon[pos].upper():
                continue
            mutant = list(codon.upper())
            mutant[pos] = base
            total += 1
            new_aa = translate_codon(''.join(mutant))
            if new_aa and new_aa != '*' and new_aa == orig_aa:
                syn_changes += 1
        if total > 0:
            syn += syn_changes / total
    return syn, 3.0 - syn

def jukes_cantor(p):
    if p <= 0.0:
        return 0.0
    arg = 1.0 - 4.0 * p / 3.0
    if arg <= 0.0:
        return float('inf')
    return -0.75 * math.log(arg)

def pairwise_dnds(seq1, seq2):
    from itertools import permutations
    total_syn_sites = 0.0
    total_nonsyn_sites = 0.0
    total_syn_diffs = 0.0
    total_nonsyn_diffs = 0.0

    for i in range(0, len(seq1), 3):
        c1, c2 = seq1[i:i+3], seq2[i:i+3]
        if '-' in c1 or '.' in c1 or '-' in c2 or '.' in c2:
            continue
        s1s, s1n = codon_sites(c1)
        s2s, s2n = codon_sites(c2)
        total_syn_sites += (s1s + s2s) / 2.0
        total_nonsyn_sites += (s1n + s2n) / 2.0

        diff_pos = [j for j in range(3) if c1[j].upper() != c2[j].upper()]
        if not diff_pos:
            continue
        if len(diff_pos) == 1:
            aa1 = translate_codon(c1)
            aa2 = translate_codon(c2)
            if aa1 and aa2 and aa1 == aa2:
                total_syn_diffs += 1.0
            else:
                total_nonsyn_diffs += 1.0
        else:
            total_s = 0.0
            total_n = 0.0
            for perm in permutations(diff_pos):
                current = list(c1.upper())
                target = list(c2.upper())
                for pos in perm:
                    aa_before = translate_codon(''.join(current))
                    current[pos] = target[pos]
                    aa_after = translate_codon(''.join(current))
                    if aa_before and aa_after and aa_before == aa_after:
                        total_s += 1.0
                    else:
                        total_n += 1.0
            n_perms = math.factorial(len(diff_pos))
            total_syn_diffs += total_s / n_perms
            total_nonsyn_diffs += total_n / n_perms

    p_s = total_syn_diffs / total_syn_sites if total_syn_sites > 0 else 0.0
    p_n = total_nonsyn_diffs / total_nonsyn_sites if total_nonsyn_sites > 0 else 0.0
    ds = jukes_cantor(p_s)
    dn = jukes_cantor(p_n)
    omega = dn / ds if ds > 1e-10 else None

    return {'dn': dn, 'ds': ds, 'omega': omega,
            'syn_sites': total_syn_sites, 'nonsyn_sites': total_nonsyn_sites}


# ════════════════════════════════════════════════════════════════════════
#  Domain 22: Molecular Clock
# ════════════════════════════════════════════════════════════════════════

def strict_clock(branch_lengths, parent_indices, root_age_ma):
    n = len(branch_lengths)
    dist_from_root = [0.0] * n
    for i in range(n):
        if parent_indices[i] is not None:
            dist_from_root[i] = dist_from_root[parent_indices[i]] + branch_lengths[i]
    tree_height = max(dist_from_root)
    if tree_height <= 0.0 or root_age_ma <= 0.0:
        return None
    rate = tree_height / root_age_ma
    node_ages = [root_age_ma - d / rate for d in dist_from_root]
    return {'rate': rate, 'node_ages': node_ages}

def relaxed_clock_rates(branch_lengths, node_ages, parent_indices):
    n = len(branch_lengths)
    rates = [0.0] * n
    for i in range(n):
        if parent_indices[i] is not None:
            span = node_ages[parent_indices[i]] - node_ages[i]
            if span > 0.0:
                rates[i] = branch_lengths[i] / span
    return rates

def rate_variation_cv(rates):
    positive = [r for r in rates if r > 0.0]
    if len(positive) < 2:
        return 0.0
    mean = sum(positive) / len(positive)
    if mean <= 0.0:
        return 0.0
    var = sum((r - mean)**2 for r in positive) / (len(positive) - 1)
    return math.sqrt(var) / mean


# ════════════════════════════════════════════════════════════════════════
#  Domain 23: Pangenome Analysis
# ════════════════════════════════════════════════════════════════════════

def pangenome_analyze(presence_matrix, n_genomes):
    core = accessory = unique = 0
    for row in presence_matrix:
        count = sum(1 for p in row if p)
        if count == n_genomes:
            core += 1
        elif count == 1:
            unique += 1
        elif count > 1:
            accessory += 1
    return core, accessory, unique

def hypergeometric_pvalue(k, n, big_k, big_n):
    if big_n == 0 or n == 0 or big_k == 0:
        return 1.0
    expected = n * big_k / big_n
    if k <= expected:
        return 1.0
    var = n * big_k * (big_n - big_k) * (big_n - n) / (big_n * big_n * max(big_n - 1, 1))
    if var <= 0.0:
        return 0.0 if k > expected else 1.0
    z = (k - expected) / math.sqrt(var)
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def benjamini_hochberg(pvalues):
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummin = float('inf')
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = indexed[i][1] * n / rank
        cummin = min(cummin, adj, 1.0)
        adjusted[indexed[i][0]] = cummin
    return adjusted


# ════════════════════════════════════════════════════════════════════════
#  Main — Timing Benchmark
# ════════════════════════════════════════════════════════════════════════

def main():
    timings = []

    # Domain 19: ANI
    seqs = [
        "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        "ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG",
        "CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
    ]
    _, us = timer(lambda: [pairwise_ani(seqs[i], seqs[j])
                           for i in range(len(seqs)) for j in range(i)])
    timings.append(("ANI (3 seqs, 50bp)", us))

    # Domain 20: SNP
    snp_seqs = [
        "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        "ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG",
        "CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
        "ATGATCATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
    ]
    _, us = timer(lambda: call_snps(snp_seqs))
    timings.append(("SNP calling (4 seqs, 50bp)", us))

    # Domain 21: dN/dS
    dnds_seq1 = "ATGGCTAAATTTGCTGCTGCTGCTGCTGCT"
    dnds_seq2 = "ATGGCCGAATTTGCTGCTGCTGCTGCCGCT"
    _, us = timer(lambda: pairwise_dnds(dnds_seq1, dnds_seq2))
    timings.append(("dN/dS (10 codons)", us))

    # Domain 22: Molecular Clock
    branch_lengths = [0.0, 0.1, 0.2, 0.05, 0.05, 0.15, 0.15]
    parents = [None, 0, 0, 1, 1, 2, 2]
    def clock_work():
        sc = strict_clock(branch_lengths, parents, 3500.0)
        rates = relaxed_clock_rates(branch_lengths, sc['node_ages'], parents)
        cv = rate_variation_cv(rates)
        return sc, rates, cv
    _, us = timer(clock_work)
    timings.append(("Molecular clock (7-node tree)", us))

    # Domain 23: Pangenome
    presence = [
        [True, True, True, True, True],   # core
        [True, True, True, True, True],   # core
        [True, True, True, True, True],   # core
        [True, True, False, False, False], # accessory
        [False, True, True, False, False], # accessory
        [True, False, False, False, False], # unique
        [False, False, False, False, True], # unique
    ]
    def pan_work():
        c, a, u = pangenome_analyze(presence, 5)
        pvals = [hypergeometric_pvalue(8, 10, 20, 100),
                 hypergeometric_pvalue(2, 10, 20, 100),
                 hypergeometric_pvalue(5, 10, 20, 100)]
        adj = benjamini_hochberg(pvals)
        return c, a, u, adj
    _, us = timer(pan_work)
    timings.append(("Pangenome (7 genes, 5 genomes)", us))

    # Summary
    print("═" * 60)
    print(" BarraCUDA CPU v4 — Python Timing Baseline (Track 1c)")
    print("═" * 60)
    print(f"\n  {'Domain':<40} {'Time (µs)':>12}")
    print(f"  {'-'*55}")
    total = 0.0
    for name, us in timings:
        print(f"  {name:<40} {us:>12.0f}")
        total += us
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<40} {total:>12.0f}")
    print()
    print("ALL 5 TRACK 1c DOMAINS TIMED")

if __name__ == "__main__":
    main()
