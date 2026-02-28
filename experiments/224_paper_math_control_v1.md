# Exp224: Paper Math Control — 18 Papers Published Equations via BarraCuda CPU

**Track:** cross
**Phase:** 71
**Status:** PASS — 58/58 checks
**Binary:** `validate_paper_math_control_v1`
**Features:** none (CPU only)

## Purpose

Validates that published equations from 18 foundational papers are correctly
implemented in BarraCuda CPU. Serves as the foundation of the three-tier
chain (CPU → GPU → metalForge).

## Paper/Reference

| Paper | Domain |
|-------|--------|
| Waters 2008 | QS ODE bistable |
| Massie 2012 | Gillespie stochastic |
| Fernandez 2020 | Bistable ODE |
| Srivastava 2011 | Multi-signal ODE |
| Bruger 2018 | Cooperation model |
| Hsueh 2022 | Phage defense ODE |
| Mhatre 2020 | Capacitor model |
| Liu 2014 | Phylogenetic placement |
| Felsenstein 1981 | Felsenstein pruning |
| Jones PFAS | PFAS screening |
| Fajgenbaum 2019 | Pathway analysis |
| Martínez-García 2023 | Pore geometry |
| Islam 2014 | Brandt Farm no-till |
| Zuber 2016 | Meta-analysis |
| Feng 2024 | Pore diversity |
| Liang 2015 | Long-term tillage |
| Tecon 2017 | Biofilm aggregate |
| Bourgain & Kachkovskiy 2018 | Anderson spectral |

## Model / Equations

Each paper contributes one or more published equations implemented as pure
Rust CPU kernels. No GPU, no Python — analytical correctness only.

## Validation

- 58 checks across all 18 paper domains
- Reference values from published figures/tables or analytical solutions
- Tolerance: `ANALYTICAL_F64` where applicable

## Status

PASS — 58/58 checks
