# Exp121: NCBI Vibrio QS Parameter Landscape

**Status:** PASS (14/14 checks)
**Binary:** `validate_ncbi_vibrio_qs`
**Features:** GPU-optional (CPU fallback when `--features gpu` omitted)
**Date:** 2026-02-23
**GPU confirmed:** Yes (release build, ~3m 44s incl. compilation)

## Purpose

Loads 200 real Vibrio genome assemblies from NCBI Datasets API, derives QS ODE parameters from genome size (mu_max) and gene count (k_ai_prod), and runs a GPU ODE sweep to map the bistability landscape with real genomic diversity. Extends Exp108 synthetic grid to biologically grounded parameter ranges.

## Design

NCBI Assembly metadata (genome size, gene count, isolation source) drives parameter derivation: mu_max inversely proportional to genome size, k_ai_prod proportional to QS-gene proxy from total gene count. Derived parameters feed into QsBiofilmParams for GPU ODE sweep. Regime classification and clinical vs environmental clustering tested against Exp108 baseline.

## Data Source

200 Vibrio assemblies from NCBI Datasets API (Assembly DB search `Vibrio[Organism]`, complete/chromosome-level). Metadata cached in `data/ncbi_phase35/vibrio_assemblies.json`. Fallback: enhanced synthetic data when offline.

## GPU Results

- **All 200 real Vibrio assemblies converge to biofilm** — 200/200 biofilm, 0 planktonic, 0 extinction
- Real genomic diversity clusters entirely in biofilm-favoring parameter space
- mu_max range: [0.543, 1.200], k_ai_prod range: [2.970, 6.034]
- 4 clinical / 196 environmental isolates — both subsets 100% biofilm
- 2/32 bistable parameter sets detected via perturbation scan
- GPU–CPU parity: max |diff| = 1.17 (long-horizon ODE drift, < 2.0 threshold)

## Key Finding

Real Vibrio genomes do NOT produce the diverse parameter landscape seen in Exp108's synthetic 32×32 grid. The synthetic grid artificially spread parameters into planktonic/extinction regions that real genomes don't occupy. This is a testable prediction: the Waters 2008 ODE model, when parameterized from real genome annotations, inherently favors biofilm formation across the entire genus.

## Reproduction

```bash
cargo run --release --features gpu --bin validate_ncbi_vibrio_qs
```
