# Exp178: Wang 2025 Tillage × Compartment

**Date:** February 2026
**Track:** 4 (No-Till Soil QS)
**Paper:** Wang et al. "Effects of tillage practices in stover-return on endosphere and rhizosphere microbiomes" npj Sustainable Agriculture 3:12 (2025)
**Binary:** `validate_tillage_microbiome_2025`
**Status:** PASS (15 checks)

---

## Hypothesis

This experiment validates that endosphere and rhizosphere compartment effects, and stover return interactions with tillage, are correctly captured in microbiome diversity and composition metrics.

## Method

Validation partitions microbial communities by compartment (endosphere vs. rhizosphere) and applies tillage × stover-return treatments. Diversity and composition patterns are checked against the Wang et al. findings.

## Results

All 15 checks PASS. See `cargo run --bin validate_tillage_microbiome_2025` for full output.

## Key Finding

Endosphere/rhizosphere compartment effects, stover return.

## Modules Validated

- `bio::diversity`
