# Exp349: ToadStool Dispatch v4

**Date:** March 2026
**Track:** V109 — Upstream Rewire + NUCLEUS Atomics
**Binary:** `validate_toadstool_dispatch_v4`
**Status:** PASS (32 checks)

---

## Hypothesis

ToadStool compute dispatch layer preserves mathematical correctness from analytical formulae through CPU. Extends v3 with Track 6 biogas kinetics and Anderson W mapping.

## Method

6 sections: stats regression (S7), linalg (S8), special functions (S9), numerical integration (S10), bio diversity round-trip (S11), Track 6 kinetics dispatch (S12). All barracuda primitives consumed by wetSpring validated through the dispatch abstraction.

## Results

All 32 checks PASS. See `cargo run --release --features gpu --bin validate_toadstool_dispatch_v4`.

## Key Finding

Every ToadStool abstraction layer preserves mathematical correctness. Track 6 kinetics (Gompertz, Monod, Haldane) dispatch correctly through the compute layer. Numerical integration (trapz) matches analytical result.
