# Exp277: Immuno-Anderson GPU Validation — Shannon, Simpson, Bray-Curtis on GPU

**Status:** PASS (21/21 checks)
**Date:** 2026-03-02
**Binary:** `validate_immuno_anderson_gpu`
**Command:** `cargo run --release --features gpu --bin validate_immuno_anderson_gpu`
**Feature gate:** gpu

## Purpose

GPU portability of immuno-Anderson math. Shannon, Simpson, Bray-Curtis on GPU via ToadStool dispatch.

## Chain

CPU Parity (Exp276) → **This** → Streaming (Exp278)
