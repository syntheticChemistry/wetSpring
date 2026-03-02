# Exp272: Bio Brain Cross-Spring Validation — BioNautilusBrain + Attention State Machine

**Status:** PASS (64/64 checks)
**Date:** 2026-03-02
**Binary:** `validate_bio_brain_s79`
**Command:** `cargo run --release --bin validate_bio_brain_s79`
**Feature gate:** none

## Purpose

Validates hotSpring 4-layer brain architecture adapted to bio sentinel context. BioNautilusBrain from bingocube-nautilus, BioBrain adapter with attention state machine.

## Sections/Domains

| Section | Checks | Description |
|---------|--------|-------------|
| D01 | 8 | ESN reservoir |
| D02 | 10 | Attention state machine |
| D03 | 8 | Observation features |
| D04 | 12 | Brain IPC dispatch |
| D05 | 8 | Cross-spring provenance |
| D06 | 10 | Urgency pipeline |
| D07 | 8 | Head group alignment |

## Chain

Cross-Spring S79 (Exp271) → **This** → Immuno-Anderson (Exp273)
