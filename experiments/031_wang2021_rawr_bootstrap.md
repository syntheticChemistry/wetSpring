# Experiment 031: Wang 2021 — RAWR Bootstrap Resampling

**Paper**: Wang et al. "Build a better bootstrap and the RAWR shall beat a random path to your door" — *Bioinformatics* (ISMB) 37:i111-i119, 2021

**Track**: 1b — Comparative Genomics & Phylogenetics (Liu)

---

## Objective

Implement the core RAWR (Resampling Aligned Weighted Reads) primitive for phylogenetic bootstrap confidence estimation. Column resampling with replacement generates replicate alignments; per-replicate Felsenstein likelihood comparison provides statistical support for tree topologies.

## Model

- **Column resampling**: Sample N columns with replacement from the original alignment
- **Per-replicate likelihood**: Felsenstein pruning under JC69 model
- **Bootstrap support**: Fraction of replicates favoring tree A over tree B
- **Seeded PRNG**: LCG64 for deterministic replication

## Validation Checks

| # | Check | Status |
|---|-------|--------|
| 1 | Original LL finite and negative | PASS |
| 2 | Resampling preserves n_taxa, n_sites | PASS |
| 3 | 100 bootstrap replicates generated | PASS |
| 4 | All replicate LLs finite and negative | PASS |
| 5 | Mean LL near original (within ±5.0) | PASS |
| 6 | Bootstrap support ∈ [0,1] | PASS |
| 7 | Deterministic (bit-exact reruns with same seed) | PASS |

## Baseline Provenance

- **Python script**: `scripts/wang2021_rawr_bootstrap.py`
- **Rust module**: `barracuda::bio::bootstrap`
- **Dependencies**: `barracuda::bio::felsenstein`, `barracuda::bio::gillespie::Lcg64`
- **Validation binary**: `validate_bootstrap` — 11/11 checks

## GPU Promotion Path

**Tier A — Rewire**: Embarrassingly parallel across replicates. Each bootstrap replicate is independent — column resampling + Felsenstein likelihood in one workgroup. ToadStool `batch_likelihood` primitive. The column resampling step is a simple index permutation (GPU-native).

---

*Date*: 2026-02-20 | *Checks*: 11/11 PASS
