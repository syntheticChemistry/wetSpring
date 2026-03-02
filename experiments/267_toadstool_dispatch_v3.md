# Exp267: ToadStool Dispatch v3 — Pure Rust Math Validation

| Field | Value |
|-------|-------|
| Binary | `validate_toadstool_dispatch_v3` |
| Date | 2026-03-01 |
| ToadStool Pin | S71+++ (`1dd7e338`) |
| Command | `cargo run --release --bin validate_toadstool_dispatch_v3` |
| Checks | 41/41 PASS |

## Purpose

Validates pure Rust math across all 6 `barracuda` domains consumed by wetSpring.
Proves that every ToadStool abstraction layer preserves mathematical correctness
from analytical formulae through CPU compute.

## Sections

| Section | Domain | Checks |
|---------|--------|--------|
| S1 | `barracuda::stats` — bootstrap, jackknife, correlation, regression | 12 |
| S2 | `barracuda::linalg` — Laplacian, effective rank, ridge, NMF | 11 |
| S3 | `barracuda::special` — erf, erfc, ln_gamma | 6 |
| S4 | `barracuda::numerical` — trapezoidal integration, numerical Hessian | 4 |
| S5 | wetSpring `bio::diversity` → `barracuda::stats` identity proof | 4 |
| S6 | `barracuda::spectral` — Anderson 3D, Lanczos, level spacing ratio | 4 |

## Validation Chain

Paper Math → CPU v20 → GPU v11 → Parity v7 → metalForge v12 → **ToadStool v3 (this)**

## Key Results

- S1: Perfect linear fit (slope=2.0, r²=1.0), Pearson=1.0, Spearman=1.0
- S2: Laplacian row sums exactly zero, NMF fully non-negative
- S3: erf(x) + erfc(x) = 1 to 1e-14 precision
- S4: Hessian of x²+y² = diag(2,2) to 1e-4 (numerical differentiation)
- S5: wetSpring diversity delegates are bit-identical to barracuda::stats
- S6: Anderson localization LSR=0.6687 (W=4, L=4, 30 eigenvalues)
