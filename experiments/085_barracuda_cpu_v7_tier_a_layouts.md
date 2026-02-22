# Exp085: BarraCUDA CPU v7 — Tier A Data Layout Fidelity

**Status**: PASS — 43/43 checks  
**Binary**: `validate_barracuda_cpu_v7`  
**Date**: 2026-02-22

## Purpose

Prove that the 3 newly Tier A modules preserve mathematical correctness
through their GPU/NPU-ready flat data layouts. This is the bridge from
"CPU math is correct" to "CPU math is correct *through the serialization
path that GPU/NPU will consume*."

## Evolution Chain Position

```
Python baseline → Rust CPU (Exp001–077) → Flat Layout CPU (THIS)
                                            ↓
                                     GPU shader consumption (next)
```

## Validated Domains

| Module | Layout | Round-Trip Test | Checks |
|--------|--------|-----------------|--------|
| kmer | `to_histogram` / `from_histogram` (dense 4^k) | count → flatten → restore → verify | 9 |
| kmer | `to_sorted_pairs` / `from_sorted_pairs` (sparse) | count → sort → restore → verify | 4 |
| kmer | Multi-sequence histogram | combine → flatten → restore → verify | 3 |
| unifrac | `to_flat_tree` / `to_phylo_tree` (CSR) | parse Newick → flatten → reconstruct → compute | 5 |
| unifrac | Weighted/unweighted properties | self-distance = 0, max-diff > 0, finite | 4 |
| unifrac | `to_sample_matrix` (GPU buffer layout) | abundance table → dense matrix → verify dims | 5 |
| taxonomy | `to_int8_weights` / `classify_quantized` | train → quantize → classify → match f64 | 6 |
| taxonomy | Multi-taxon int8 parity | 3 taxa × (f64 confidence + int8 match) + priors | 7 |

## Key Results

- **kmer**: Histogram and sorted-pairs round-trips are bitwise identical (zero tolerance)
- **unifrac**: Flat tree CSR → PhyloTree reconstruction preserves UniFrac distances to 1e-12
- **taxonomy**: Int8 quantized argmax matches f64 argmax on all unambiguous inputs
- **Total new checks**: 43 (extends CPU parity from 205 to 248)

## Reproduction

```bash
cargo run --release --bin validate_barracuda_cpu_v7
# Expected: 43/43 PASS, exit 0
```
