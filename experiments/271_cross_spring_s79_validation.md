# Exp271: Cross-Spring S79 Validation — Five Springs via ToadStool

**Status:** PASS (73/73 checks)
**Date:** 2026-03-02
**Binary:** `validate_cross_spring_evolution_v71`
**Command:** `cargo run --release --features gpu --bin validate_cross_spring_evolution_v71`
**Feature gate:** gpu

## Purpose

Validates all five Springs' primitives consumed by wetSpring via ToadStool S79. 13 domain sweep across hotSpring precision, wetSpring bio, neuralSpring ML, airSpring hydro, groundSpring bootstrap.

## Sections/Domains

| Section | Checks | Description |
|---------|--------|-------------|
| D01–D13 | 73 | Cross-spring evolution across all 5 springs (hotSpring, wetSpring, neuralSpring, airSpring, groundSpring) |

## Chain

NUCLEUS v3 (Exp266) → **This** → Bio Brain (Exp272)
