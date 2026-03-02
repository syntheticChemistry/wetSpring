# Exp265: metalForge v12 — V87 Extended Cross-System Dispatch

**Status:** PASS (63/63 checks)
**Date:** 2026-03-01
**Binary:** `validate_metalforge_v12_extended`
**Command:** `cargo run --release --features gpu,ipc --bin validate_metalforge_v12_extended`
**Feature gate:** `gpu`, `ipc`

## Purpose

Extends metalForge v11 (23-workload, 43 checks) to cover the 5 new GPU
domains (PCoA, K-mer, Bootstrap, KMD, Kriging) and adds vault-aware
dispatch with provenance chain tracking.

## Workload Catalog (28 total)

| Hardware | Count | New | Examples |
|----------|-------|-----|----------|
| GPU | 21 | +5 | pcoa_gpu, kmer_gpu, bootstrap_gpu, kmd_gpu, vault_provenance |
| NPU | 3 | — | esn_classify, taxonomy_classify, signal_peak_detect |
| CPU | 4 | — | quality_filter, sw_align, merge_pairs, relaxed_clock |

## New Sections (N16–N21)

| Section | Checks | Description |
|---------|--------|-------------|
| N16 | 2 | PCoA CPU dispatch |
| N17 | 2 | K-mer CPU dispatch |
| N18 | 2 | Bootstrap statistics |
| N19 | 1 | KMD PFAS screening |
| N20 | 1 | Kriging variogram |
| N21 | 7 | Vault-aware dispatch (provenance + consent + store/retrieve) |

## Chain

Parity v7 (Exp264) → **metalForge v12 (this)** → NUCLEUS v3 (Exp266)
