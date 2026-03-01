# Exp259: Genomic Vault — Consent + Encrypted Storage + Provenance

**Status:** PASS (30/30 checks)
**Date:** 2026-03-01
**Binary:** `validate_genomic_vault`
**Command:** `cargo run --release --bin validate_genomic_vault`
**Feature gate:** none (sovereign implementation, zero external deps)

## Purpose

Validates the Genomic Vault protocol — sovereign encrypted storage for
personal biological data with consent-gated access and tamper-evident
provenance.

## Organ Model

Genomic data is treated like an organ: it belongs to the individual.
No pipeline can touch it without a signed, time-bounded, revocable
consent ticket.

## Modules Created

| Module | Tests | Purpose |
|--------|-------|---------|
| `vault::consent` | 7 | Consent ticket protocol |
| `vault::provenance` | 5 | Append-only audit chain |
| `vault::storage` | 8 | Encrypted vault storage |

## Checks

### Phase 1: Consent Ticket Protocol (10 checks)
- Create, validate, revoke consent tickets
- Scope hierarchy (FullPipeline implies diversity/Anderson, NOT raw sequences)
- Time-bounded expiration
- Revocation

### Phase 2: Encrypted Vault Storage (6 checks)
- Store 16S sequence data encrypted
- Retrieve and verify plaintext matches original
- Wrong key → integrity check fails
- Cross-owner access rejected
- Unauthorized access rejected

### Phase 3: Multi-Sample Vault (3 checks)
- Store 5 samples, list by owner
- Unknown owner returns empty

### Phase 4: Provenance Chain Integrity (6 checks)
- Chain integrity verified
- Store and retrieve operations recorded
- All chain links verified
- Filter by actor works

### Phase 5: Standalone Provenance Chain (4 checks)
- Multi-actor chain (nestgate → wetspring → toadstool → wetspring)
- Filter by actor

### Phase 6: Organ Model Summary (1 check — display)

## Absorb Targets

- BearDog: BLAKE3 hashing, ChaCha20-Poly1305 encryption, Ed25519 signing
- NestGate: content-addressed blob storage, append-only provenance CAS
- Songbird: consent request/approve/deny workflow
- biomeOS: Neural API `vault.*` capability routing

## Chain

Exp256-258 (NUCLEUS) → **Exp259 (Genomic Vault)** → primal absorption
