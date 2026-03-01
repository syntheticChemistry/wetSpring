// SPDX-License-Identifier: AGPL-3.0-or-later
//! Genomic Vault — sovereign encrypted storage for personal biological data.
//!
//! Treats genomic data like an organ: it belongs to the individual.
//! No pipeline can touch it without a signed, time-bounded, revocable
//! consent ticket.
//!
//! # Architecture
//!
//! ```text
//! Owner (lineage seed)
//!   → ConsentTicket (scope, duration, BearDog-signed)
//!     → VaultBlob (ChaCha20-Poly1305 encrypted, BLAKE3 content-addressed)
//!       → ProvenanceEntry (append-only audit chain)
//! ```
//!
//! # Primal Integration
//!
//! | Capability | Primal | Method | Status |
//! |------------|--------|--------|--------|
//! | Encrypt/decrypt | `BearDog` | `encryption.encrypt` / `encryption.decrypt` | absorb target |
//! | Key derivation | `BearDog` | `genetic.derive_lineage_key` | absorb target |
//! | Blob storage | `NestGate` | `storage.store_blob` / `storage.retrieve_blob` | absorb target |
//! | Consent | `Songbird` | `POST /consent/request` | absorb target |
//! | Signing | `BearDog` | Ed25519 lineage certs | absorb target |
//!
//! Local implementations here are sovereign (zero external deps). They
//! validate the vault protocol so specific primals can absorb.

pub mod consent;
pub mod provenance;
pub mod storage;

pub use consent::{ConsentScope, ConsentTicket};
pub use provenance::{ProvenanceChain, ProvenanceEntry};
pub use storage::{VaultBlob, VaultStore};
