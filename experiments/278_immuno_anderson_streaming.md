# Exp278: Immuno-Anderson Streaming — Batched GPU Pipeline

**Status:** PASS (31/31 checks)
**Date:** 2026-03-02
**Binary:** `validate_immuno_anderson_streaming`
**Command:** `cargo run --release --features gpu --bin validate_immuno_anderson_streaming`
**Feature gate:** gpu

## Purpose

Batched GPU pipeline for immuno-Anderson. Streaming dispatch reduces round-trips vs individual calls.

## Chain

GPU Validation (Exp277) → **This** → metalForge (Exp279)
