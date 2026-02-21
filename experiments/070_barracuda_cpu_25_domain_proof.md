# Experiment 070: BarraCUDA CPU — 25-Domain Pure Rust Math Proof

**Date:** 2026-02-21
**Status:** COMPLETE — 25-domain pure Rust math proof (50/50 checks PASS)
**Track:** cross-cutting / barracuda

## Objective

Consolidate all 25 algorithmic domains into a single definitive validation
binary that proves BarraCUDA pure Rust math:
1. **Matches Python/paper baselines** — parity checks for every domain
2. **Is faster than interpreted language** — timing comparison with Python

This is the bridge between "paper validation done" and "GPU promotion":
every domain proven correct in pure Rust CPU math can then be shown
portable to GPU via barracuda GPU.

## Domains (25)

| # | Domain | Paper/Baseline | Check |
|---|--------|---------------|-------|
| 1 | ODE Integration (RK4) | Waters 2008 / scipy.odeint | Biomass steady-state |
| 2 | Gillespie SSA | Massie 2012 | Mean final population |
| 3 | HMM Forward | Liu 2014 | Log-likelihood |
| 4 | Smith-Waterman | Needleman 1970 | Alignment score |
| 5 | Felsenstein Pruning | Felsenstein 1981 | Log-likelihood |
| 6 | Shannon/Simpson | Shannon 1948 / Simpson 1949 | Entropy value |
| 7 | Signal Processing | Peak detection | Peak count |
| 8 | Game Theory (QS) | Bruger 2018 | Cooperation steady-state |
| 9 | Robinson-Foulds | Robinson & Foulds 1981 | Tree distance |
| 10 | Multi-Signal QS | Srivastava 2011 | Signal steady-state |
| 11 | Phage Defense | Hsueh 2022 | Equilibrium phage |
| 12 | Bootstrap | Wang 2021 | Replicate LLs |
| 13 | Phylo Placement | Alamin 2024 | Edge likelihoods |
| 14 | Decision Tree | sklearn spec | Classification |
| 15 | Spectral Matching | Cosine similarity | Cosine score |
| 16 | Extended Diversity | Bray-Curtis, Chao1 | BC value |
| 17 | K-mer Counting | Jaccard distance | Jaccard |
| 18 | Integrated Pipeline | NJ + Diversity | Pipeline check |
| 19 | ANI | Goris 2007 | Pairwise identity |
| 20 | SNP Calling | Anderson 2017 | Allele frequency |
| 21 | dN/dS | Nei & Gojobori 1986 | Ratio |
| 22 | Molecular Clock | Zuckerkandl 1965 | Branch rate |
| 23 | Pangenome | Moulana 2020 | Core gene fraction |
| 24 | Random Forest | sklearn spec | Majority vote |
| 25 | GBM | sklearn spec | Predicted value |

## Protocol

1. Run `python3 scripts/benchmark_python_baseline.py` → JSON baseline
2. Run `cargo run --release --bin validate_barracuda_cpu_full`
3. Binary validates each domain (one canonical parity check per domain)
4. Binary times each domain
5. Binary loads Python baseline JSON for timing comparison
6. Summary table: Domain | Checks | Rust µs | Python µs | Speedup

## Provenance

| Field | Value |
|-------|-------|
| Baseline commit | current HEAD |
| Baseline tool | scipy, numpy, dendropy, sklearn (per-domain Python scripts) |
| Baseline date | 2026-02-21 |
| Exact command | `cargo run --release --bin validate_barracuda_cpu_full` |
| Data | Synthetic test vectors (hardcoded, reproducible) |
| Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!_OS 22.04) |
