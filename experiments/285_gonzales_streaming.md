# Exp285: Gonzales Streaming — Batched GPU Pipeline for Pharmacology

**Status:** PASS (37/37 checks)
**Date:** 2026-03-02
**Binary:** `validate_gonzales_streaming`
**Command:** `cargo run --release --features gpu --bin validate_gonzales_streaming`
**Feature gate:** gpu

## Purpose

Streaming dispatch for pharmacological workloads. Shannon, Simpson, Bray-Curtis matrix via batched GPU pipeline.

## Chain

GPU Validation (Exp284) → **This** → metalForge (Exp286)
