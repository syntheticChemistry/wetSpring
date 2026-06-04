# wetSpring V195 — Wave 76 Deep Debt

**Date:** June 3, 2026
**From:** wetSpring (southGate)
**Phase:** Wave 76 deep debt — architecture evolution

---

## Changes

- **TCP transport:** `Transport::Tcp(addr)` variant, `tcp_jsonrpc_line()`, unified `jsonrpc_line()` dispatch, `{ENV_VAR}_TCP` env-var priority. 4 new tests.
- **Primal name centralization:** Replaced `"nestgate"`, `"petaltongue"`, `"rhizocrypt"`, `"loamspine"`, `"sweetgrass"` literals with `primal_names::*` constants across 4 files.
- **Songbird registration fix:** IPC-only builds now register `niche::CAPABILITIES` instead of empty list.
- **macOS RSS:** `bench::peak_rss_mb()` safe subprocess for macOS (respects `#![forbid(unsafe_code)]`).
- **CLI port wired:** `wetspring serve --port` now accepted and logged.
- **Ionic bonding modernized:** `try_from` safe casts, `ok_or_else` lazy error, dead match arm removed, redundant clone eliminated.
- **Dependencies bumped:** axum 0.8.9, blake3 1.8.5, proptest 1.11, tempfile 3.27, tower-http 0.6.11.
- **Registries const-ified:** `ScenarioRegistry` and `BenchmarkRegistry` constructors/accessors.
- **Stale lints removed:** 2 unfulfilled `#[expect]` attributes cleaned.

## Build Gate

- **2,089 tests**, 0 failures, 3 ignored
- **0 clippy warnings** (pedantic + nursery, production code)
- **0 unsafe code** (`#![forbid(unsafe_code)]`)
- All docs updated to V195

## Gaps for Upstream

- Songbird: TCP mesh relay for cross-gate `capability.call`
- bearDog: `crypto.ionic_bond.seal/propose/verify_proposal` (Ed25519)
- toadStool: `compute.fan_out` scheduler (Tenaillon 264-clone batch)
- biomeOS: `nest.sync` E2E (WS-2)
- Forgejo: SSH key registration still pending

---

**ACK:** FRAGO wave76-parity-sprint-springs complete.
