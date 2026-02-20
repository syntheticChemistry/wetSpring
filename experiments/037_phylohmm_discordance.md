# Exp037 — HMM Gene Tree Discordance Detection

**Date:** February 20, 2026
**Status:** COMPLETE — 10/10 checks PASS
**Track:** 1b (Comparative Genomics & Phylogenetics)
**Phase:** Exp019 Phase 3

---

## Objective

Validate the HMM module on a PhyloNet-HMM style analysis using real
PhyNetPy gene trees. Models consecutive gene tree RF distances as a
2-state HMM: concordant blocks (low RF) vs discordant blocks (high RF).

Mirrors Liu 2014's PhyloNet-HMM approach for introgression detection
(DOI: 10.1371/journal.pcbi.1003649).

## Data Source

- **Trees:** 200 gene trees from PhyNetPy DEFJ (2 scenarios × 10 replicates × 10 trees)
- **Observation sequence:** 199 consecutive pairwise RF distances
- **Discretization:** RF ≤ median → 0 (concordant), RF > median → 1 (discordant)
- **Median RF:** 18

## HMM Model

- 2 hidden states: concordant (state 0) and discordant (state 1)
- Transition: switch probability = 0.1 (persistence of genomic blocks)
- Emission: state 0 emits low RF (obs=0) with p=0.8; state 1 emits high RF with p=0.8
- Initial: π = [0.6, 0.4]

## Python Baseline

Script: `scripts/phylohmm_introgression_baseline.py`
Output: `experiments/results/037_phylohmm/python_baseline.json`

## Validation Checks (10/10)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | Forward log-likelihood finite/negative | ✓ | PASS |
| 2 | Viterbi path length | 20 | PASS |
| 3 | Mostly discordant (≥15/20 high-RF obs) | ✓ | PASS |
| 4 | All states valid (0 or 1) | ✓ | PASS |
| 5 | Viterbi LL ≤ Forward LL | ✓ | PASS |
| 6 | Forward deterministic | exact | PASS |
| 7 | Viterbi length deterministic | = | PASS |
| 8 | Viterbi path deterministic | = | PASS |
| 9 | All-concordant: mostly state 0 | ✓ | PASS |
| 10 | Single observation finite | ✓ | PASS |

## Key Findings

- HMM log-likelihood on 20-obs high-RF subsequence: -8.53
- Viterbi correctly classifies predominantly discordant observations as state 1
- All-concordant control sequence correctly classified as state 0
- Forward-Viterbi consistency: Viterbi path probability never exceeds sum-over-paths

## GPU Promotion Path

Forward and Viterbi algorithms are matrix-chain multiplications along the
time axis. Batch processing across multiple observation sequences is already
implemented (`forward_batch`, `viterbi_batch`). GPU promotion: one workgroup
per sequence, shared memory for transition matrix.

## Run

```bash
cargo run --bin validate_phylohmm
python3 scripts/phylohmm_introgression_baseline.py
```
