# Exp089: ToadStool Streaming Dispatch Proof

**Status**: PASS — 25/25 checks  
**Binary**: `validate_streaming_dispatch`  
**Date**: 2026-02-22

## Purpose

Prove that ToadStool's unidirectional streaming pattern (data flows
source → GPU → GPU → sink without CPU round-trips) produces results
identical to the naive round-trip pattern. This validates the core
architectural claim that streaming reduces dispatch overhead while
preserving mathematical correctness.

## Validated Patterns

| Pattern | Stages | Test | Checks |
|---------|--------|------|--------|
| Round-trip | 3 (CPU → GPU → CPU each) | kmer → Shannon → BC | 3 |
| Streaming | 3 (CPU → GPU → GPU → CPU) | same ops, no CPU intermediate | 3 |
| 3-stage chain | kmer → diversity → taxonomy | RT ↔ stream parity per sample | 8 |
| 5-stage chain | kmer → diversity → taxonomy → UniFrac → classify | full pipeline + f64/int8 | 8 |
| Determinism | 3 runs of streaming | bitwise identical results | 3 |

## Key Results

- **Round-trip = Streaming**: Identical Shannon, Simpson, Bray-Curtis values
- **3-stage parity**: All 4 samples produce identical Shannon and taxonomy across patterns
- **5-stage parity**: Full pipeline through flat CSR matches original UniFrac
- **Determinism**: Streaming produces bitwise-identical results across 3 runs
- **f64 ↔ int8**: Classification agreement on all samples in 5-stage chain

## Reproduction

```bash
cargo run --release --bin validate_streaming_dispatch
# Expected: 25/25 PASS, exit 0
```
