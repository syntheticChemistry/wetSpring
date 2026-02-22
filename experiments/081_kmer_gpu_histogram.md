# Exp081: K-mer GPU Histogram Preparation

**Status**: COMPLETE
**Date**: 2026-02-22
**Module**: `barracuda/src/bio/kmer.rs`
**New tests**: 4

## Purpose

Adds GPU-compatible flat representations to the k-mer counting engine,
replacing `HashMap<u64, u32>` with contiguous buffers suitable for GPU
histogram reduction and radix sort dispatch.

## APIs Added

| Method | Layout | Use Case |
|--------|--------|----------|
| `to_histogram()` | `Vec<u32>` of size 4^k | Dense GPU buffer (k ≤ 12) |
| `from_histogram(hist, k)` | Round-trip reconstruction | Validation |
| `to_sorted_pairs()` | `Vec<(u64, u32)>` sorted by kmer | Compact GPU buffer (large k) |
| `from_sorted_pairs(pairs, k)` | Round-trip reconstruction | Validation |

## GPU Dispatch Pattern

```
CPU: count_kmers(seq, k) → KmerCounts (HashMap)
     ↓ to_histogram()
GPU: flat u32[4^k] buffer → parallel reduction / radix sort
     ↓ readback
CPU: from_histogram() → KmerCounts (verified identical)
```

For k=8: 65,536 u32 entries = 256 KB — fits in GPU workgroup shared memory.
For k=4: 256 u32 entries = 1 KB — trivial GPU buffer.

## Tests

| Test | What it validates |
|------|-------------------|
| `histogram_round_trip` | HashMap → histogram → HashMap preserves all counts |
| `sorted_pairs_round_trip` | HashMap → sorted pairs → HashMap preserves all counts; pairs sorted |
| `histogram_preserves_top_n` | Top-1 k-mer identical through round-trip |
| `histogram_gpu_buffer_size` | k=8 produces 65,536-entry buffer |

## Tier Promotion

kmer: **B → A** (GPU-ready, pending `hash_table_u64.wgsl` or histogram-based shader)
