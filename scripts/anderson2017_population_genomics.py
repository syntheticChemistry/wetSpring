#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anderson 2017 — Population genomics at hydrothermal vents.

Generates Python baselines for Exp055 validation. Validates ANI and
SNP calling primitives against synthetic metagenomic data.

Paper: Anderson et al. (2017) Nature Communications 8:1114
DOI: 10.1038/s41467-017-01228-6
Data: PRJNA283159

Usage:
    python scripts/anderson2017_population_genomics.py

Python: 3.10+
Date: 2026-02-20
"""

import json
import os
from pathlib import Path


def pairwise_ani(seq1, seq2):
    """Compute ANI between two aligned sequences."""
    length = min(len(seq1), len(seq2))
    identical = 0
    aligned = 0
    for i in range(length):
        a = seq1[i].upper()
        b = seq2[i].upper()
        if a in ('-', '.', 'N') or b in ('-', '.', 'N'):
            continue
        aligned += 1
        if a == b:
            identical += 1
    ani = identical / aligned if aligned > 0 else 0.0
    return {"ani": ani, "aligned": aligned, "identical": identical}


def call_snps(sequences):
    """Call SNPs from aligned sequences."""
    if not sequences:
        return {"variants": [], "aln_len": 0}
    aln_len = len(sequences[0])
    variants = []
    for pos in range(aln_len):
        counts = {}
        for seq in sequences:
            if pos >= len(seq):
                continue
            base = seq[pos].upper()
            if base in ('-', '.', 'N'):
                continue
            counts[base] = counts.get(base, 0) + 1
        if len(counts) >= 2 and sum(counts.values()) >= 2:
            ref_allele = max(counts, key=counts.get)
            depth = sum(counts.values())
            alt = [(b, c) for b, c in counts.items() if b != ref_allele]
            variants.append({
                "position": pos,
                "ref": ref_allele,
                "alt": alt,
                "depth": depth,
            })
    return {"variants": variants, "aln_len": aln_len, "n_variants": len(variants)}


def main():
    results = {}

    # ── ANI test cases ───────────────────────────────────────────
    results["ani_identical"] = pairwise_ani("ATGATGATG", "ATGATGATG")
    results["ani_half"] = pairwise_ani("AATT", "AAGC")
    results["ani_with_gaps"] = pairwise_ani("A-TG", "ACTG")

    # Same species pair (>95% identical)
    seq1 = "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG"
    seq2 = list(seq1)
    seq2[0] = 'C'
    seq2[10] = 'C'
    seq2 = ''.join(seq2)
    results["ani_same_species"] = pairwise_ani(seq1, seq2)

    # Different species pair (<95%)
    seq3 = list(seq1)
    for i in range(0, len(seq3), 10):
        seq3[i] = 'C' if seq3[i] != 'C' else 'G'
    seq3 = ''.join(seq3)
    results["ani_diff_species"] = pairwise_ani(seq1, seq3)

    # ── SNP test cases ───────────────────────────────────────────
    results["snp_identical"] = call_snps(["ATGATG", "ATGATG", "ATGATG"])
    results["snp_single"] = call_snps(["ATGATG", "ATGATG", "ATGTTG"])
    results["snp_multiple"] = call_snps(["ATGATG", "CTGATG", "ATGTTG"])

    # Allele frequency test
    results["snp_frequency"] = call_snps(["A", "A", "A", "T"])

    # Write output
    out_dir = Path("experiments/results/055_population_genomics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anderson2017_python_baseline.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nANI (identical): {results['ani_identical']['ani']:.6f}")
    print(f"ANI (half):      {results['ani_half']['ani']:.6f}")
    print(f"ANI (same sp):   {results['ani_same_species']['ani']:.6f}")
    print(f"ANI (diff sp):   {results['ani_diff_species']['ani']:.6f}")
    print(f"SNPs (identical): {results['snp_identical']['n_variants']}")
    print(f"SNPs (single):    {results['snp_single']['n_variants']}")
    print(f"SNPs (multiple):  {results['snp_multiple']['n_variants']}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
