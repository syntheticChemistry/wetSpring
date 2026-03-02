# Exp263: BarraCuda CPU v20 — V87 Vault + DF64 + Cross-Domain

**Status:** PASS (37/37 checks)
**Date:** 2026-03-01
**Binary:** `validate_barracuda_cpu_v20`
**Command:** `cargo run --release --bin validate_barracuda_cpu_v20`
**Feature gate:** none

## Purpose

Extends CPU v19 (D01–D26, 99 checks) with V87 domains covering vault
protocol validation, DF64 double-float arithmetic, encoding sovereignty,
and tolerance registry verification.

## New Domains (D27–D32)

| Domain | Checks | Module | Description |
|--------|--------|--------|-------------|
| D27 | 8 | `vault::provenance` | BLAKE3 Merkle chain, tamper detection, actor filtering |
| D28 | 8 | `vault::consent` | Scope hierarchy, revocation, expiry, authorization |
| D29 | 7 | `vault::storage` | ChaCha20-Poly1305 encrypt/decrypt, key rejection, cross-owner |
| D30 | 6 | `df64_host` | Pack/unpack roundtrip, try_unpack_slice error handling |
| D31 | 4 | `encoding` | Base64 RFC 4648, roundtrip identity, empty edge cases |
| D32 | 4 | `tolerances` | GPU_VS_CPU_F64, DF64_ROUNDTRIP, EXACT bounds |

## Provenance

Expected values are **analytical** — derived from cryptographic and
mathematical invariants (BLAKE3 chain integrity, ChaCha20-Poly1305
AEAD rejection, DF64 precision bounds, base64 round-trip identity).

## Chain

Paper (Exp251) → CPU v19 (Exp252) → **CPU v20 (this)** → GPU v11 (Exp254)
