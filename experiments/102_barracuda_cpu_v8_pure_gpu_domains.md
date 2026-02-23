# Exp102: BarraCUDA CPU v8 — Pure Rust Math for 13 GPU-Promoted Domains

| Field    | Value                                       |
|----------|---------------------------------------------|
| Script   | `validate_barracuda_cpu_v8`                 |
| Command  | `cargo run --release --bin validate_barracuda_cpu_v8` |
| Status   | **PASS** (84/84)                            |
| Phase    | 28                                          |
| Depends  | Exp079 (v6), Exp085 (v7), Exp101            |

## Purpose

Validates that the 13 modules promoted to GPU in the pure GPU completion pass
produce correct CPU results from analytical known-values. This proves **pure
Rust math** matches paper equations before any GPU dispatch — the foundation
for claiming GPU portability.

Extends BarraCUDA CPU v1–v7 (205 + 43 checks) to cover all 13 newly promoted
domains with known-value validation.

## Modules Validated

| # | Module | Paper/Domain | Key Checks |
|---|--------|-------------|------------|
| 1 | `cooperation` | Bruger & Waters 2018 | Flat RT, coexistence, cooperator persistence, biofilm |
| 2 | `capacitor` | Mhatre et al. 2020 | Flat RT, cell growth, VpsR accumulation, mixed phenotype |
| 3 | `kmd` | Kendrick 1963 | Result count, Kendrick mass analytical, CF2 spread |
| 4 | `gbm` | GBM ensemble | Prediction count, probability bounds, monotonicity |
| 5 | `merge_pairs` | DADA2 pipeline | Input count, merge accounting, merged length |
| 6 | `signal` | Peak detection | Peak count (≥2), Gaussian/Lorentzian positions, height threshold |
| 7 | `feature_table` | LC-MS features | Empty-input identity (0 features, 0 mass tracks) |
| 8 | `robinson_foulds` | Robinson & Foulds 1981 | Different topology > 0, self-distance = 0, 5-taxon |
| 9 | `derep` | Dereplication | Input/unique counts, max abundance, sort invariance |
| 10 | `chimera` | UCHIME-inspired | Input count, results = input, retained + found = input |
| 11 | `neighbor_joining` | Saitou & Nei 1987 | Matrix size, diagonal zeros, monotonicity, symmetry, NJ tree |
| 12 | `reconciliation` | Zheng et al. 2023 | Cost ≥ 0, cost finite, higher costs → higher total |
| 13 | `molecular_clock` | Molecular clock | Strict rate > 0, relaxed count, root rate = 0, non-negative |

## Results

| Module | Checks | Status | Notes |
|--------|--------|--------|-------|
| M01: Cooperation | 6/6 | PASS | Flat RT + ODE steady-state + biofilm |
| M02: Capacitor | 6/6 | PASS | Flat RT + 3 phenotypes + VpsR |
| M03: KMD | 7/7 | PASS | CF2 series analytical |
| M04: GBM | 5/5 | PASS | Ensemble predictions |
| M05: Merge Pairs | 3/3 | PASS | Overlap accounting |
| M06: Signal | 5/5 | PASS | Gaussian + Lorentzian peaks |
| M07: Feature Table | 2/2 | PASS | Empty identity |
| M08: Robinson-Foulds | 3/3 | PASS | Known tree distances |
| M09: Dereplication | 5/5 | PASS | Exact duplicate counting |
| M10: Chimera | 3/3 | PASS | Chimeric detection |
| M11: Neighbor Joining | 24/24 | PASS | JC69 diagonal, monotonicity, symmetry, NJ tree |
| M12: Reconciliation | 2/2 | PASS | DTL cost bounds |
| M13: Molecular Clock | 13/13 | PASS | Strict + relaxed rates |
| **Total** | **84/84** | **PASS** | |

## BarraCUDA CPU Parity Running Total

| Version | Exp | Domains | Checks |
|---------|-----|---------|--------|
| v1–v5 | 035–062 | 25 core | 205 |
| v6 | 079 | +6 ODE flat | 48 |
| v7 | 085 | +3 Tier A layouts | 43 |
| **v8** | **102** | **+13 GPU-promoted** | **84** |
| **Total** | | **34 domain groups** | **380** |
