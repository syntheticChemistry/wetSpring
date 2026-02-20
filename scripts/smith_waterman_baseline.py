#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Smith-Waterman local alignment — Python baseline.

Pure-Python SW with affine gaps for validation against Rust.
No external dependencies (sovereign).
"""

import json, os

def smith_waterman(query, target, match=2, mismatch=-1, gap_open=-3, gap_extend=-1):
    m, n = len(query), len(target)
    if m == 0 or n == 0:
        return {"score": 0, "aligned_query": "", "aligned_target": ""}

    H = [[0]*(n+1) for _ in range(m+1)]
    E = [[-(10**9)]*(n+1) for _ in range(m+1)]
    F = [[-(10**9)]*(n+1) for _ in range(m+1)]

    best_score, best_i, best_j = 0, 0, 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if query[i-1].upper() == target[j-1].upper() else mismatch
            E[i][j] = max(H[i][j-1] + gap_open + gap_extend, E[i][j-1] + gap_extend)
            F[i][j] = max(H[i-1][j] + gap_open + gap_extend, F[i-1][j] + gap_extend)
            H[i][j] = max(0, H[i-1][j-1] + s, E[i][j], F[i][j])
            if H[i][j] > best_score:
                best_score, best_i, best_j = H[i][j], i, j

    # Traceback
    aq, at = [], []
    i, j = best_i, best_j
    while i > 0 and j > 0 and H[i][j] > 0:
        s = match if query[i-1].upper() == target[j-1].upper() else mismatch
        if H[i][j] == H[i-1][j-1] + s:
            aq.append(query[i-1]); at.append(target[j-1]); i -= 1; j -= 1
        elif H[i][j] == F[i][j]:
            aq.append(query[i-1]); at.append("-"); i -= 1
        else:
            aq.append("-"); at.append(target[j-1]); j -= 1

    return {
        "score": best_score,
        "aligned_query": "".join(reversed(aq)),
        "aligned_target": "".join(reversed(at)),
        "query_start": i,
        "target_start": j,
    }

def main():
    cases = {
        "identical": ("ACGTACGT", "ACGTACGT"),
        "mismatch": ("ACGT", "ACTT"),
        "gap": ("ACGTACGT", "ACGACGT"),
        "local": ("XXXACGTACGTXXX", "ACGTACGT"),
        "no_match": ("AAAA", "CCCC"),
        "16s_fragment": (
            "GATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATAC",
            "GATCCTGGCTCAGAATGAACGCTGGCGGCATGCCTAATAC",
        ),
    }

    results = {}
    for name, (q, t) in cases.items():
        r = smith_waterman(q, t)
        results[name] = r
        print(f"{name:15s}: score={r['score']:3d}  "
              f"q='{r['aligned_query'][:30]}' t='{r['aligned_target'][:30]}'")

    # Special case: harsh penalties
    r = smith_waterman("AAAA", "CCCC", match=1, mismatch=-3, gap_open=-5, gap_extend=-2)
    results["no_match_harsh"] = r
    print(f"{'no_match_harsh':15s}: score={r['score']:3d}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "028_alignment")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smith_waterman_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")

if __name__ == "__main__":
    main()
