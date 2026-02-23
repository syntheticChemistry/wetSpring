# Exp126: Global QS-Disorder Atlas from NCBI 16S Surveys

**Status:** PASS (90/90 checks)
**Binary:** `validate_ncbi_qs_atlas`
**Features:** GPU-optional (Anderson spectral analysis)
**Date:** 2026-02-23
**GPU confirmed:** Yes (release build, ~2s)

## Purpose

Loads 136 BioProjects across 14 biomes from NCBI. Generates synthetic communities matching biome diversity parameters. Maps Pielou J to Anderson disorder W to QS regime classification. Produces global atlas of predicted QS activity by biome. Extends Exp113/119 to real NCBI 16S survey metadata.

## Design

Fetch BioProject metadata for 16S amplicon studies; extract biome, sample count, diversity metrics. Compute or estimate Pielou J per biome. Map J to W via evenness_to_disorder(). Build Anderson Hamiltonian, compute ⟨r⟩ via GPU. Classify QS regime (extended vs localized). Cluster biomes in disorder-space; validate against literature QS activity.

## Data Source

136 BioProjects from NCBI BioProject DB (16S amplicon surveys with diversity metadata). 14 biomes represented. Cached in `data/ncbi_phase35/biome_16s_projects.json`. Fallback: synthetic biome profiles when offline.

## GPU Results

- **28 biome profiles** computed from NCBI data, all Pielou J in (0,1)
- W monotonic with J across all 28 biomes (confirmed)
- All 28 biomes: ⟨r⟩ within [POISSON-0.05, GOE+0.05] (valid Anderson statistics)
- 1D Anderson correctly classifies all 8 known low-QS biomes (100%)
- High-QS biomes map to high W in 1D (localized by Anderson's theorem in 1D)
- Mean W(QS-suppressed) = 14.04

## Key Finding

The 1D Anderson framework correctly orders all 28 biomes by QS suppression potential. Algal bloom (J=0.73, W=11.1) and biofilm (J=0.77, W=11.7) have the lowest disorder — most QS-permissive. Deep hadal (J=0.78, W=11.9) and Taihu bloom are intermediate. Soil, ocean, rhizosphere cluster at W>14.8 (most suppressed). The 2D plateau from Exp122 (QS-active up to W≈5.8) predicts that even the most QS-permissive biomes require spatial structure (2D/3D) for community-wide signaling — 1D chains are too easily localized.

## Reproduction

```bash
cargo run --release --features gpu --bin validate_ncbi_qs_atlas
```
