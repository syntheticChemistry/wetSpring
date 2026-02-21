#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anderson 2014 — Viral metagenomics at hydrothermal vents.

Generates Python baselines for Exp052 validation. Includes dN/dS
estimation via the Nei-Gojobori (1986) method for comparison with
the new Rust `bio::dnds` module.

Paper: Anderson et al. (2014) PLoS ONE 9:e109696
DOI: 10.1371/journal.pone.0109696

Usage:
    python scripts/anderson2014_viral_metagenomics.py

Requires: no external dependencies (pure Python + math)
Python: 3.10+
Date: 2026-02-20
"""

import json
import math
import os
from pathlib import Path


# ── Standard genetic code ─────────────────────────────────────────

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

BASES = ["A", "C", "G", "T"]


def translate(codon):
    return CODON_TABLE.get(codon.upper(), None)


def codon_sites(codon):
    """Count synonymous and nonsynonymous sites for a codon."""
    codon = codon.upper()
    orig_aa = translate(codon)
    if orig_aa is None or orig_aa == "*":
        return (0.0, 0.0)

    syn_sites = 0.0
    for pos in range(3):
        syn_changes = 0
        total_changes = 0
        for base in BASES:
            if base == codon[pos]:
                continue
            mutant = list(codon)
            mutant[pos] = base
            mutant = "".join(mutant)
            total_changes += 1
            new_aa = translate(mutant)
            if new_aa is not None and new_aa != "*" and new_aa == orig_aa:
                syn_changes += 1
        if total_changes > 0:
            syn_sites += syn_changes / total_changes

    return (syn_sites, 3.0 - syn_sites)


def pathway_diffs(c1, c2, order):
    """Walk one mutation pathway and classify changes."""
    current = list(c1)
    syn = 0.0
    non = 0.0
    for pos in order:
        aa_before = translate("".join(current))
        current[pos] = c2[pos]
        aa_after = translate("".join(current))
        if aa_before is not None and aa_after is not None and aa_before == aa_after:
            syn += 1.0
        else:
            non += 1.0
    return syn, non


def permutations(items):
    if len(items) <= 1:
        return [list(items)]
    result = []
    for i, item in enumerate(items):
        rest = items[:i] + items[i + 1:]
        for perm in permutations(rest):
            result.append([item] + perm)
    return result


def count_codon_diffs(c1, c2):
    """Count syn/nonsyn differences between two codons."""
    c1 = c1.upper()
    c2 = c2.upper()
    diff_pos = [i for i in range(3) if c1[i] != c2[i]]
    n_diffs = len(diff_pos)

    if n_diffs == 0:
        return (0.0, 0.0)

    if n_diffs == 1:
        aa1 = translate(c1)
        aa2 = translate(c2)
        if aa1 is not None and aa2 is not None and aa1 == aa2:
            return (1.0, 0.0)
        return (0.0, 1.0)

    perms = permutations(diff_pos)
    total_syn = 0.0
    total_non = 0.0
    for perm in perms:
        s, n = pathway_diffs(list(c1), list(c2), perm)
        total_syn += s
        total_non += n

    count = len(perms)
    return (total_syn / count, total_non / count)


def jukes_cantor(p):
    if p <= 0.0:
        return 0.0
    arg = 1.0 - 4.0 * p / 3.0
    if arg <= 0.0:
        return float("inf")
    return -0.75 * math.log(arg)


def pairwise_dnds(seq1, seq2):
    """Nei-Gojobori dN/dS estimation."""
    assert len(seq1) == len(seq2)
    assert len(seq1) % 3 == 0

    total_syn_sites = 0.0
    total_nonsyn_sites = 0.0
    total_syn_diffs = 0.0
    total_nonsyn_diffs = 0.0

    for i in range(0, len(seq1), 3):
        c1 = seq1[i:i + 3]
        c2 = seq2[i:i + 3]
        if "-" in c1 or "-" in c2 or "." in c1 or "." in c2:
            continue

        s1_syn, s1_non = codon_sites(c1)
        s2_syn, s2_non = codon_sites(c2)
        total_syn_sites += (s1_syn + s2_syn) / 2.0
        total_nonsyn_sites += (s1_non + s2_non) / 2.0

        sd, nd = count_codon_diffs(c1, c2)
        total_syn_diffs += sd
        total_nonsyn_diffs += nd

    p_s = total_syn_diffs / total_syn_sites if total_syn_sites > 0 else 0.0
    p_n = total_nonsyn_diffs / total_nonsyn_sites if total_nonsyn_sites > 0 else 0.0

    ds = jukes_cantor(p_s)
    dn = jukes_cantor(p_n)
    omega = dn / ds if ds > 1e-10 else None

    return {
        "dn": dn,
        "ds": ds,
        "omega": omega,
        "syn_sites": total_syn_sites,
        "nonsyn_sites": total_nonsyn_sites,
        "syn_diffs": total_syn_diffs,
        "nonsyn_diffs": total_nonsyn_diffs,
    }


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


def main():
    results = {}

    # ── Diversity: viral vs cellular functional profiles ──────────
    # Synthetic KEGG-category abundances modeled after Anderson 2014 Table S3
    viral_kegg = [120, 80, 60, 45, 30, 25, 20, 15, 10, 8, 5, 3, 2, 1]
    cellular_kegg = [200, 180, 150, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25,
                     20, 15, 12, 10, 8, 5]

    results["viral_shannon"] = shannon(viral_kegg)
    results["cellular_shannon"] = shannon(cellular_kegg)

    # ── dN/dS test cases ─────────────────────────────────────────
    # Case 1: Identical sequences
    seq_id = "ATGATGATG"
    r = pairwise_dnds(seq_id, seq_id)
    results["dnds_identical"] = r

    # Case 2: Synonymous only (TTT→TTC = Phe→Phe)
    r = pairwise_dnds("TTTGCTAAA", "TTCGCTAAA")
    results["dnds_synonymous"] = r

    # Case 3: Nonsynonymous only (AAA→GAA = Lys→Glu)
    r = pairwise_dnds("AAAGCTGCT", "GAAGCTGCT")
    results["dnds_nonsynonymous"] = r

    # Case 4: Mixed changes (longer sequence)
    seq1 = "ATGGCTAAATTTGCTGCTGCTGCTGCTGCT"
    seq2 = "ATGGCCAAATTTGCTGCTGCTGCTGCCGCT"
    r = pairwise_dnds(seq1, seq2)
    results["dnds_mixed"] = r

    # Case 5: Purifying selection example (conserved gene)
    gene1 = "ATGGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCT"
    gene2 = "ATGGCCGCTGCTGCTGCTGCTGCTGCCGCTGCTGCTGCTGCTGCTGCT"
    r = pairwise_dnds(gene1, gene2)
    results["dnds_purifying"] = r

    out_dir = Path("experiments/results/052_viral_metagenomics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anderson2014_python_baseline.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nViral Shannon:    {results['viral_shannon']:.12f}")
    print(f"Cellular Shannon: {results['cellular_shannon']:.12f}")
    print(f"\ndN/dS identical:     dN={results['dnds_identical']['dn']:.12f}, dS={results['dnds_identical']['ds']:.12f}")
    print(f"dN/dS synonymous:    dN={results['dnds_synonymous']['dn']:.12f}, dS={results['dnds_synonymous']['ds']:.12f}, omega={results['dnds_synonymous']['omega']}")
    print(f"dN/dS nonsynonymous: dN={results['dnds_nonsynonymous']['dn']:.12f}")
    print(f"dN/dS mixed:         dN={results['dnds_mixed']['dn']:.12f}, dS={results['dnds_mixed']['ds']:.12f}")
    print(f"dN/dS purifying:     dN={results['dnds_purifying']['dn']:.12f}, dS={results['dnds_purifying']['ds']:.12f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
