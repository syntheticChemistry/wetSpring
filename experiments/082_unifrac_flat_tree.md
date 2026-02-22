# Exp082: UniFrac Flat Tree (CSR) for GPU

**Status**: COMPLETE
**Date**: 2026-02-22
**Module**: `barracuda/src/bio/unifrac.rs`
**New tests**: 4

## Purpose

Adds GPU-compatible flat tree representation using Compressed Sparse Row
(CSR) layout, enabling the pairwise UniFrac distance matrix to be computed
on GPU where each (i, j) pair runs independently.

## APIs Added

| Type / Method | Description |
|--------------|-------------|
| `FlatTree` struct | GPU buffer: `parent[]`, `branch_length[]`, CSR children |
| `PhyloTree::to_flat_tree()` | Convert tree to CSR layout |
| `FlatTree::to_phylo_tree()` | Round-trip reconstruction |
| `to_sample_matrix(flat, samples)` | Dense n_samples × n_leaves matrix |

## FlatTree Layout

```
parent:           [u32; n_nodes]      — parent index per node
branch_length:    [f64; n_nodes]      — branch length per node
n_children:       [u32; n_nodes]      — child count per node
children_offset:  [u32; n_nodes]      — CSR offset into children_flat
children_flat:    [u32; total_edges]  — all children contiguously
leaf_indices:     [u32; n_leaves]     — which nodes are leaves
leaf_labels:      [String; n_leaves]  — for sample mapping
```

## GPU Dispatch Pattern

```
CPU: PhyloTree::from_newick() → to_flat_tree() → upload buffers
     to_sample_matrix() → upload sample matrix
GPU: parallel over (i, j) pairs:
       propagate(sample_i) + propagate(sample_j) → unifrac distance
CPU: readback distance matrix → from_phylo_tree() for verification
```

Pairwise loop is O(n²) — trivially parallel on GPU workgroups.

## Tests

| Test | What it validates |
|------|-------------------|
| `flat_tree_round_trip` | CSR ↔ PhyloTree preserves topology, labels, branch lengths |
| `flat_tree_unifrac_parity` | Unweighted + weighted UniFrac identical through flat path |
| `sample_matrix_layout` | Dense matrix has correct dimensions and total abundance |
| `flat_tree_csr_consistency` | Every child's parent pointer matches CSR structure |

## Tier Promotion

unifrac: **B → A** (GPU-ready, pending tree-propagation WGSL shader)
