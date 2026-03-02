# Exp286: Gonzales metalForge — Cross-Hardware Portability for Drug Reproductions

**Status:** PASS (36/36 checks)
**Date:** 2026-03-02
**Binary:** `validate_gonzales_metalforge`
**Command:** `cargo run --release --features gpu,ipc --bin validate_gonzales_metalforge`
**Feature gate:** gpu,ipc

## Purpose

Cross-hardware portability for Gonzales drug reproductions. CPU↔GPU + Hill + Anderson + NUCLEUS atomics.

## Chain

Streaming (Exp285) → **This** → next wave
