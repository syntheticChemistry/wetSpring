# Exp192: metalForge V59 Cross-Substrate — CPU↔GPU Parity

**Date:** February 26, 2026
**Phase:** V59 — Three-tier controls
**Binary:** `validate_metalforge_v59_science`
**Command:** `cargo run --features gpu --release --bin validate_metalforge_v59_science`
**Status:** DONE (metalForge, 36 checks PASS)

## Purpose

Proves that V59 science computations produce identical results on CPU
and GPU substrates. Exercises diversity, Bray-Curtis, and Anderson
spectral analysis on both substrates and compares outputs for parity.

## Domains Validated

| Domain | Checks | What it proves |
|--------|:------:|---------------|
| MF01: Diversity CPU↔GPU | 16 | Shannon, Simpson, S_obs, Pielou same on both substrates |
| MF02: Bray-Curtis CPU↔GPU | 8 | Full matrix + condensed distance match exactly |
| MF03: Anderson spectral CPU↔GPU | 10 | Anderson 3D → Lanczos → r parity across 7 (W, seed) pairs |
| MF04: Pipeline summary | 2 | End-to-end: FASTA → diversity → Anderson → classification |

## Acceptance Criteria

- All 36 checks PASS with --features gpu
- CPU↔GPU parity within tolerances::EXACT (bitwise for deterministic math)
- Full pipeline produces valid classification on both substrates
