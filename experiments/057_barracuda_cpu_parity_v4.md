# Experiment 057: BarraCUDA CPU Parity v4 — Track 1c Domains

**Date:** February 21, 2026
**Status:** COMPLETE
**Track:** cross (CPU parity)

---

## Purpose

Extend the BarraCUDA CPU parity validation to the 5 new Track 1c modules
(deep-sea metagenomics / microbial evolution). Proves that the Rust
implementations are:
1. Pure math — no external tool calls, no Python dependencies
2. Correct — matches analytical expectations and Python baselines
3. Fast — benchmarked against Python timing baselines
4. GPU-ready — batch APIs validate the GPU-friendly patterns

This is the fourth CPU parity experiment in the progression:

```
v1 (Exp035, 9 domains) → v2 (Exp035, +5 batch) → v3 (Exp043, +9) → [THIS] v4 (+5 Track 1c)
```

Total after v4: **23 algorithmic domains** validated in pure Rust.

---

## Domains Validated

| Domain | Module | Algorithm | Python Baseline |
|--------|--------|-----------|-----------------|
| 19 | `bio::ani` | Average Nucleotide Identity (pairwise + matrix + batch) | `anderson2017_population_genomics.py` |
| 20 | `bio::snp` | SNP calling (position-parallel, allele frequency, flat SoA) | `anderson2017_population_genomics.py` |
| 21 | `bio::dnds` | Nei-Gojobori 1986 dN/dS (codon sites, Jukes-Cantor, batch) | `anderson2014_viral_metagenomics.py` |
| 22 | `bio::molecular_clock` | Strict/relaxed clock, calibration, CV | `mateos2023_sulfur_phylogenomics.py` |
| 23 | `bio::pangenome` | Core/accessory/unique, Heap's law, enrichment, BH FDR, flat matrix | `moulana2020_pangenomics.py` |

---

## Validation Design

### Analytical checks (known-value)
- ANI: identical sequences → 1.0; completely different → 0.0; half-match → 0.5
- SNP: invariant sites produce no variants; known allele frequencies
- dN/dS: synonymous-only → dN=0; Met codon has 3.0 nonsynonymous sites
- Clock: strict clock on known tree → exact rate; CV=0 for uniform rates
- Pangenome: known core/accessory/unique partition from synthetic matrix
- Enrichment: hypergeometric p-value for known (k, n, K, N)
- BH FDR: corrected p-values monotonically bounded in [0,1]

### Python parity checks
- Each domain compared against Python baseline expected values
- Tolerances: 1e-6 for floating-point (generous for f64 math)

### Batch API checks (GPU-ready patterns)
- `pairwise_ani_batch()` produces same results as sequential calls
- `call_snps_flat()` SoA layout matches `call_snps()` AoS layout
- `pairwise_dnds_batch()` produces same results as sequential calls
- `presence_matrix_flat()` correctly linearizes boolean presence matrix

### Timing
- Wall-clock microseconds per domain, release build
- Python baseline timing from `scripts/barracuda_cpu_v4_baseline.py`

---

## Actual Checks: 44/44 PASS

- Domain 19 (ANI): 9 checks (pairwise + matrix + batch API)
- Domain 20 (SNP): 8 checks (calling + freq + density + flat SoA)
- Domain 21 (dN/dS): 9 checks (identical + syn + mixed + batch API)
- Domain 22 (Clock): 7 checks (strict + relaxed + CV + calibration)
- Domain 23 (Pangenome): 11 checks (classify + flat matrix + enrichment + BH FDR)

---

## Data

All synthetic — no external data dependencies.
