# Exp115: ESN Phylogenetic Placement → NPU

**Status**: PASS (9/9)  
**Phase**: 33 — NPU Reservoir Deployment  
**Binary**: `validate_npu_phylo_placement`  
**Depends on**: Exp109 (Large-Scale Phylogenetic Placement)

## Purpose

Train an ESN on JC69 distance-feature vectors to classify metagenomic
reads into 8 clades without full NJ tree construction. Quantize for NPU.

## Data

- 512 training / 256 test distance-feature vectors (8 features per sample)
- 64 taxa, 8 clades, JC69 pairwise distances
- Features: per-clade representative distances with divergence noise

## Architecture

- ESN: 8 input → 300 reservoir (ρ=0.95, c=0.15, α=0.2) → 8 output

## Results

| Metric | Value |
|--------|-------|
| F64 accuracy | 27.0% (8-class) |
| NPU int8 accuracy | 27.0% |
| F64 ↔ NPU agreement | 97.7% |
| NPU speedup vs full placement | ~154× |
| Energy ratio | ~9,000× |

## Key Findings

1. **97.7% quantization fidelity** on 8-class problem — NPU faithfully
   reproduces f64 decisions.
2. **Accuracy limited by diagonal regression** — ToadStool's full ESN
   with matrix readout will improve significantly.
3. **154× throughput gain** over full distance-matrix placement: the ESN
   eliminates O(N²) distance computation at inference time.

## Reproduction

```bash
cargo run --release --bin validate_npu_phylo_placement
```
