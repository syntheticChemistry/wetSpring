# Exp284: Gonzales GPU Validation — Pharmacological Math on GPU

**Status:** PASS (17/17 checks)
**Date:** 2026-03-02
**Binary:** `validate_gonzales_gpu`
**Command:** `cargo run --release --features gpu --bin validate_gonzales_gpu`
**Feature gate:** gpu

## Purpose

GPU portability of pharmacological math. Shannon, Simpson, Pielou, Bray-Curtis on GPU for Gonzales datasets.

## Chain

CPU Parity (Exp283) → **This** → Streaming (Exp285)
