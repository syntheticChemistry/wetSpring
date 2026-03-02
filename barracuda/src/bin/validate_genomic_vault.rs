// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp259: Genomic Vault — Consent + Encrypted Storage + Provenance
//!
//! Validates the sovereign genomic vault protocol:
//!
//! 1. Consent tickets: create, validate, revoke, scope hierarchy
//! 2. Encrypted storage: store, retrieve, wrong-key rejection
//! 3. Provenance chain: append-only, tamper-evident, filterable
//! 4. End-to-end: consent → encrypt → store → audit → retrieve → verify
//! 5. Unauthorized access: every path without valid consent fails
//!
//! ## Organ Model
//!
//! Genomic data is treated like an organ: it belongs to the individual.
//! No pipeline can touch it without a signed, time-bounded, revocable
//! consent ticket. The vault enforces this at every operation.
//!
//! ## Chain
//!
//! Exp256-258 (NUCLEUS) → **Exp259 (Genomic Vault)** → `BearDog`/`NestGate` absorb
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | Command | `cargo run --release --bin validate_genomic_vault` |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use std::time::Duration;

use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::vault::consent::{ConsentScope, ConsentTicket};
use wetspring_barracuda::vault::provenance::ProvenanceChain;
use wetspring_barracuda::vault::storage::VaultStore;

fn main() {
    let mut v = Validator::new("Exp259: Genomic Vault — Consent + Encrypted Storage + Provenance");

    v.section("Phase 1: Consent Ticket Protocol");

    let ticket = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::FullPipeline,
        "wetspring-pipeline",
        Duration::from_secs(3600),
    );
    v.check_pass("Consent: ticket created and valid", ticket.is_valid());
    v.check_pass(
        "Consent: remaining > 0",
        ticket.remaining() > Duration::ZERO,
    );
    v.check_pass("Consent: not revoked", !ticket.revoked);
    println!(
        "  Ticket ID: {:02x}{:02x}{:02x}{:02x}...",
        ticket.id[0], ticket.id[1], ticket.id[2], ticket.id[3]
    );
    println!("  Owner: {}", ticket.owner_id);
    println!(
        "  Scope: {} (sensitivity {})",
        ticket.scope.label(),
        ticket.scope.sensitivity()
    );
    println!("  Grantee: {}", ticket.grantee);
    println!("  Duration: {}s", ticket.duration.as_secs());

    let full_ticket = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::FullPipeline,
        "wetspring-pipeline",
        Duration::from_secs(3600),
    );
    v.check_pass(
        "Consent: FullPipeline authorizes DiversityAnalysis",
        full_ticket.authorizes(&ConsentScope::DiversityAnalysis),
    );
    v.check_pass(
        "Consent: FullPipeline authorizes AndersonClassification",
        full_ticket.authorizes(&ConsentScope::AndersonClassification),
    );
    v.check_pass(
        "Consent: FullPipeline does NOT authorize ReadRawSequences",
        !full_ticket.authorizes(&ConsentScope::ReadRawSequences),
    );

    let mut revocable = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::ReadRawSequences,
        "external-lab",
        Duration::from_secs(86400),
    );
    v.check_pass(
        "Consent: raw sequences ticket valid before revoke",
        revocable.is_valid(),
    );
    revocable.revoke();
    v.check_pass("Consent: revoked ticket is invalid", !revocable.is_valid());
    v.check_pass(
        "Consent: revoked ticket remaining = 0",
        revocable.remaining() == Duration::ZERO,
    );

    let mut expired = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::DiversityAnalysis,
        "wetspring-pipeline",
        Duration::from_secs(1),
    );
    expired.issued_at = 0;
    expired.duration = Duration::from_secs(1);
    v.check_pass("Consent: expired ticket is invalid", !expired.is_valid());

    v.section("Phase 2: Encrypted Vault Storage");

    let mut vault = VaultStore::new("eastgate");
    let key = [42u8; 32];
    let store_ticket = ConsentTicket::new(
        "patient-001",
        ConsentScope::ReadRawSequences,
        "wetspring-vault",
        Duration::from_secs(3600),
    );

    let sample_16s = b">sample_001 16S rRNA gene partial sequence\nATCGATCGATCGATCGATCGATCGATCGATCG\nGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT\n";
    let hash = vault
        .store(
            sample_16s,
            "sample_001.fasta",
            "patient-001",
            &key,
            &store_ticket,
        )
        .unwrap();
    v.check_pass("Vault: 16S sequence stored", vault.blob_count() == 1);
    println!(
        "  Stored: sample_001.fasta ({} bytes) → hash {:02x}{:02x}{:02x}{:02x}...",
        sample_16s.len(),
        hash[0],
        hash[1],
        hash[2],
        hash[3]
    );

    let result = vault.retrieve(&hash, &key, &store_ticket).unwrap();
    v.check_pass(
        "Vault: retrieved plaintext matches original",
        result.plaintext == sample_16s,
    );
    v.check_pass("Vault: content hash matches", result.content_hash == hash);
    v.check_pass("Vault: label preserved", result.label == "sample_001.fasta");

    let wrong_key = [99u8; 32];
    let wrong_result = vault.retrieve(&hash, &wrong_key, &store_ticket);
    v.check_pass("Vault: wrong key rejected", wrong_result.is_err());

    let other_owner_ticket = ConsentTicket::new(
        "patient-002",
        ConsentScope::ReadRawSequences,
        "wetspring-vault",
        Duration::from_secs(3600),
    );
    let cross_owner = vault.retrieve(&hash, &key, &other_owner_ticket);
    v.check_pass("Vault: cross-owner access rejected", cross_owner.is_err());

    let unauth = vault.retrieve_unauthorized(&hash);
    v.check_pass("Vault: unauthorized access rejected", unauth.is_err());

    v.section("Phase 3: Multi-Sample Vault");

    let samples = [
        (
            "sample_002.fasta",
            b"ATCGATCG metagenome fragment 002" as &[u8],
        ),
        ("sample_003.fasta", b"GCTAGCTA metagenome fragment 003"),
        ("sample_004.fasta", b"TTAACCGG metagenome fragment 004"),
        ("sample_005.fasta", b"CCGGTTAA metagenome fragment 005"),
    ];

    for (label, data) in &samples {
        vault
            .store(data, label, "patient-001", &key, &store_ticket)
            .unwrap();
    }
    v.check_pass("Vault: 5 samples stored", vault.blob_count() == 5);

    let listing = vault.list("patient-001");
    v.check_pass("Vault: list returns 5 entries", listing.len() == 5);

    let empty_list = vault.list("patient-999");
    v.check_pass(
        "Vault: list for unknown owner is empty",
        empty_list.is_empty(),
    );

    v.section("Phase 4: Provenance Chain Integrity");

    v.check_pass(
        "Provenance: chain integrity verified",
        vault.verify_provenance(),
    );
    let chain = vault.provenance();
    v.check_pass("Provenance: chain has entries", !chain.is_empty());
    println!("  Chain length: {} entries", chain.len());

    let store_ops: Vec<_> = chain
        .iter()
        .filter(|e| e.operation == "vault.store")
        .collect();
    let retrieve_ops: Vec<_> = chain
        .iter()
        .filter(|e| e.operation == "vault.retrieve")
        .collect();
    v.check_pass(
        "Provenance: 5 store operations recorded",
        store_ops.len() == 5,
    );
    v.check_pass(
        "Provenance: 1 retrieve operation recorded",
        retrieve_ops.len() == 1,
    );
    println!(
        "  Operations: {} store + {} retrieve",
        store_ops.len(),
        retrieve_ops.len()
    );

    for (i, entry) in chain.iter().enumerate() {
        if i > 0 {
            let prev = &chain.iter().nth(i - 1).unwrap();
            assert_eq!(entry.parent, prev.hash, "chain link broken at entry {i}");
        }
    }
    v.check_pass("Provenance: all chain links verified", true);

    v.check_pass(
        "Provenance: filter by actor works",
        chain.by_actor("wetspring").count() == chain.len(),
    );

    v.section("Phase 5: Standalone Provenance Chain");

    let mut standalone = ProvenanceChain::new();
    v.check_pass(
        "Provenance: empty chain valid",
        standalone.verify_integrity(),
    );

    standalone.append("ingest", "nestgate", [1u8; 32], [2u8; 32], "eastgate");
    standalone.append("diversity", "wetspring", [1u8; 32], [3u8; 32], "eastgate");
    standalone.append("anderson", "toadstool", [1u8; 32], [4u8; 32], "eastgate");
    standalone.append("export", "wetspring", [1u8; 32], [5u8; 32], "eastgate");

    v.check_pass(
        "Provenance: 4-entry chain valid",
        standalone.verify_integrity(),
    );
    v.check_pass(
        "Provenance: head is export",
        standalone.head().unwrap().operation == "export",
    );

    v.check_pass(
        "Provenance: toadstool did 1 operation",
        standalone.by_actor("toadstool").count() == 1,
    );

    v.section("Phase 6: Organ Model Summary");

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Genomic Vault — Organ Model                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  Principle: Genomic data belongs to the individual.          ║");
    println!("║  Like an organ — explicit, informed, revocable consent.      ║");
    println!("║                                                              ║");
    println!("║  Enforcement:                                                ║");
    println!("║    ✓ No access without valid consent ticket                  ║");
    println!("║    ✓ Tickets are time-bounded and revocable                  ║");
    println!("║    ✓ Scope hierarchy (diversity < pipeline < raw sequences)  ║");
    println!("║    ✓ Cross-owner access rejected                             ║");
    println!("║    ✓ Wrong key → integrity check fails                       ║");
    println!("║    ✓ Every operation audited in provenance chain             ║");
    println!("║    ✓ Chain is tamper-evident (Merkle-linked)                  ║");
    println!("║                                                              ║");
    println!("║  Absorb Targets:                                             ║");
    println!("║    BearDog  → Ed25519 signing, ChaCha20-Poly1305, BLAKE3    ║");
    println!("║    NestGate → content-addressed blob storage on ZFS          ║");
    println!("║    Songbird → consent request/approve/deny workflow          ║");
    println!("║    biomeOS  → Neural API vault.* capability routing          ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    v.finish();
}
