# Exp190: BarraCuda CPU v10 — V59 Science Extensions

**Date:** February 26, 2026
**Phase:** V59 — Three-tier controls
**Binary:** `validate_barracuda_cpu_v10`
**Command:** `cargo run --release --bin validate_barracuda_cpu_v10`
**Status:** DONE (CPU, 75 checks PASS)

## Purpose

Consolidates the CPU math proof for all V59 science domains (Exp184-188).
Validates diversity pipeline, Bray-Curtis distance mathematics, dynamic
Anderson W(t) functions, NPU int8 quantization, and FASTA→diversity
end-to-end processing using pure Rust math via barracuda always-on.

## Domains Validated

| Domain | Source Exp | Checks | barracuda Primitives |
|--------|-----------|:------:|---------------------|
| D01: Sovereign diversity | Exp184/185 | 19 | `bio::diversity::{shannon, simpson, observed_features, pielou_evenness}` |
| D02: Bray-Curtis distance | Exp184/185 | 24 | `bio::diversity::{bray_curtis_matrix, bray_curtis_condensed}` |
| D03: Dynamic Anderson W(t) | Exp186 | 12 | Analytical functions (tillage, antibiotic, seasonal) |
| D04: NPU int8 quantization | Exp188 | 13 | Affine quantization + round-trip fidelity |
| D05: FASTA → diversity | Exp184 | 7 | FASTA parsing + diversity computation |

## Acceptance Criteria

- All 75 checks PASS
- Pure CPU execution (no --features gpu)
- Validates barracuda always-on math for 5 V59 science domains
