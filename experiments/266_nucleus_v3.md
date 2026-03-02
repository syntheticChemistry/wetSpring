# Exp266: NUCLEUS v3 — Tower→Node→Nest + Vault + biomeOS

**Status:** PASS (106/106 checks)
**Date:** 2026-03-01
**Binary:** `validate_nucleus_v3_extended`
**Command:** `cargo run --release -p wetspring-forge --bin validate_nucleus_v3_extended`
**Feature gate:** none

## Purpose

Extends NUCLEUS v2 (62 checks) with V87 validation covering mixed
hardware routing, vault integration, and expanded workload dispatch.
Validates the Tower→Node→Nest atomics with PCIe bypass, bandwidth-aware
routing, and biomeOS coordination.

## Sections (S1–S8)

| Section | Focus | Key Checks |
|---------|-------|------------|
| S1 | Tower Discovery | Substrates, CPU present, capabilities |
| S2 | Nest Protocol | NestGate store/retrieve or sovereign fallback |
| S3 | Node Dispatch | 13 workloads (8 inherited + 5 new: PCoA, K-mer, KMD, diversity_fusion, PFAS) |
| S4 | Workload Catalog | 49+ workloads, absorption tracking, origin_summary |
| S5 | Cross-System Pipeline | 6-stage GPU→NPU→CPU with PCoA+K-mer+Diversity chain |
| S6 | biomeOS Coordination | Songbird/NestGate socket discovery, sovereign mode |
| S7 | Mixed Hardware | PCIe bypass, bandwidth tiers, GPU chaining, BW-aware routing |
| S8 | Vault Integration | Provenance chains, origin_summary, compute substrate |

## NUCLEUS Atomics

| Atomic | Role | Implementation |
|--------|------|----------------|
| **Tower** | Substrate discovery + capability matching | `inventory::discover()` |
| **Node** | Compute dispatch + bandwidth-aware routing | `dispatch::route()`, `route_bandwidth_aware()` |
| **Nest** | Storage protocol (NestGate or sovereign) | `nest::NestClient` |

## Hardware Detected

- CPU: always present
- GPU: NVIDIA TITAN V + RTX 4070 (PCIe bandwidth tiers detected)
- NPU: via pipeline stages (quantized inference)

## Chain

metalForge v12 (Exp265) → **NUCLEUS v3 (this)** → ToadStool absorption
