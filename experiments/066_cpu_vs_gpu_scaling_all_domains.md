# Exp066: CPU vs GPU Scaling Benchmark — All GPU Domains

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross/GPU
**Binary:** `benchmark_all_domains_cpu_gpu`
**Command:** `cargo run --features gpu --release --bin benchmark_all_domains_cpu_gpu`

---

## Objective

Extend the existing CPU vs GPU benchmark (Exp015/benchmark_cpu_gpu) to cover
ALL GPU-eligible domains: ANI, SNP, dN/dS, pangenome, Random Forest, and HMM.
Characterizes WHERE GPU wins and WHERE CPU wins at various data sizes — the
scaling evidence that drives metalForge routing decisions.

---

## Domains Benchmarked

| Domain | GPU Primitive | Scaling Dimension |
|--------|-------------|------------------|
| ANI pairwise | `ani_batch_f64.wgsl` | Number of sequence pairs |
| SNP calling | `snp_calling_f64.wgsl` | Alignment length × sequence count |
| dN/dS | `dnds_batch_f64.wgsl` | Number of codon pairs |
| Pangenome | `pangenome_classify.wgsl` | Genes × genomes |
| Random Forest | `rf_batch_inference.wgsl` | Samples × trees |
| HMM forward | `hmm_forward_f64.wgsl` | Sequences × observation length |

---

## Protocol

For each domain at multiple data sizes:
1. Run CPU reference implementation with timing
2. Run GPU implementation with timing (including dispatch overhead)
3. Report: CPU µs, GPU µs, speedup ratio (GPU/CPU)
4. Identify crossover point where GPU dispatch overhead < compute savings

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
| Exact command | `cargo run --features gpu --release --bin benchmark_all_domains_cpu_gpu` |
| Data | Synthetic sequences/vectors at increasing sizes |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04 |
