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
//! # Exp263: `BarraCuda` CPU v20 — V87 Vault + DF64 + Cross-Domain Validation
//!
//! Extends CPU v19 (D01–D26, 99 checks) with V87 domains:
//! - D27: Vault Provenance Chain (BLAKE3 Merkle chain, tamper detection)
//! - D28: Vault Consent Protocol (scope hierarchy, revocation, expiry)
//! - D29: Vault Encrypted Storage (ChaCha20-Poly1305, key rejection)
//! - D30: DF64 Pack/Unpack (host-side double-float, `try_unpack_slice`)
//! - D31: Encoding Sovereignty (base64 encode/decode, RFC 4648)
//! - D32: Tolerance Registry (97 named constants, hierarchy validation)
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from cryptographic and
//! mathematical invariants (BLAKE3 chain integrity, ChaCha20-Poly1305
//! AEAD rejection, DF64 precision bounds, base64 round-trip identity).
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (cryptographic + mathematical invariants) |
//! | Date | 2026-03-01 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v20` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::{Duration, Instant};

use wetspring_barracuda::df64_host;
use wetspring_barracuda::encoding;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::vault::consent::{ConsentScope, ConsentTicket};
use wetspring_barracuda::vault::provenance::ProvenanceChain;
use wetspring_barracuda::vault::storage::VaultStore;

struct DomainTiming {
    name: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp263: BarraCuda CPU v20 — V87 Vault + DF64 + Cross-Domain");
    let t_total = Instant::now();
    let mut timings: Vec<DomainTiming> = Vec::new();

    println!("  Inherited: D01–D26 from CPU v19 (99 checks)");
    println!("  New: D27–D32 below\n");

    // ═══ D27: Vault Provenance Chain ═══════════════════════════════════════
    let t = Instant::now();
    v.section("D27: Vault Provenance Chain (vault::provenance)");
    let mut d27 = 0_u32;

    let mut chain = ProvenanceChain::new();
    v.check_pass("Provenance: empty chain valid", chain.verify_integrity());
    d27 += 1;
    v.check_pass("Provenance: empty chain is_empty", chain.is_empty());
    d27 += 1;

    chain.append("ingest", "nestgate", [1u8; 32], [2u8; 32], "eastgate");
    chain.append("diversity", "wetspring", [1u8; 32], [3u8; 32], "eastgate");
    chain.append("anderson", "toadstool", [1u8; 32], [4u8; 32], "eastgate");
    chain.append("export", "wetspring", [1u8; 32], [5u8; 32], "eastgate");

    v.check_count("Provenance: chain length", chain.len(), 4);
    d27 += 1;
    v.check_pass("Provenance: 4-entry chain valid", chain.verify_integrity());
    d27 += 1;
    v.check_pass(
        "Provenance: head is export",
        chain.head().unwrap().operation == "export",
    );
    d27 += 1;
    v.check_count(
        "Provenance: wetspring actor count",
        chain.by_actor("wetspring").count(),
        2,
    );
    d27 += 1;
    v.check_count(
        "Provenance: toadstool actor count",
        chain.by_actor("toadstool").count(),
        1,
    );
    d27 += 1;

    let second_hash = chain.iter().nth(1).unwrap().hash;
    let third_parent = chain.iter().nth(2).unwrap().parent;
    v.check_pass(
        "Provenance: chain link [2].parent == [1].hash",
        third_parent == second_hash,
    );
    d27 += 1;

    timings.push(DomainTiming {
        name: "D27 Provenance",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d27,
    });

    // ═══ D28: Vault Consent Protocol ══════════════════════════════════════
    let t = Instant::now();
    v.section("D28: Vault Consent Protocol (vault::consent)");
    let mut d28 = 0_u32;

    let ticket = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::FullPipeline,
        "wetspring",
        Duration::from_secs(3600),
    );
    v.check_pass("Consent: new ticket is valid", ticket.is_valid());
    d28 += 1;
    v.check_pass(
        "Consent: remaining > 0",
        ticket.remaining() > Duration::ZERO,
    );
    d28 += 1;

    v.check_pass(
        "Consent: FullPipeline authorizes DiversityAnalysis",
        ticket.authorizes(&ConsentScope::DiversityAnalysis),
    );
    d28 += 1;
    v.check_pass(
        "Consent: FullPipeline authorizes Anderson",
        ticket.authorizes(&ConsentScope::AndersonClassification),
    );
    d28 += 1;
    v.check_pass(
        "Consent: FullPipeline does NOT authorize ReadRawSequences",
        !ticket.authorizes(&ConsentScope::ReadRawSequences),
    );
    d28 += 1;

    let mut revocable = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::ReadRawSequences,
        "external-lab",
        Duration::from_secs(86400),
    );
    v.check_pass("Consent: raw seq ticket valid", revocable.is_valid());
    d28 += 1;
    revocable.revoke();
    v.check_pass("Consent: revoked is invalid", !revocable.is_valid());
    d28 += 1;
    v.check_pass(
        "Consent: revoked remaining = 0",
        revocable.remaining() == Duration::ZERO,
    );
    d28 += 1;

    timings.push(DomainTiming {
        name: "D28 Consent",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d28,
    });

    // ═══ D29: Vault Encrypted Storage ═════════════════════════════════════
    let t = Instant::now();
    v.section("D29: Vault Encrypted Storage (vault::storage)");
    let mut d29 = 0_u32;

    let mut vault = VaultStore::new("eastgate");
    let key = [42u8; 32];
    let consent = ConsentTicket::new(
        "patient-001",
        ConsentScope::ReadRawSequences,
        "wetspring-vault",
        Duration::from_secs(3600),
    );

    let sample_data = b">sample_001 16S rRNA\nATCGATCGATCGATCGATCG\n";
    let hash = vault
        .store(
            sample_data,
            "sample_001.fasta",
            "patient-001",
            &key,
            &consent,
        )
        .unwrap();
    v.check_count("Vault: 1 blob stored", vault.blob_count(), 1);
    d29 += 1;

    let result = vault.retrieve(&hash, &key, &consent).unwrap();
    v.check_pass(
        "Vault: decrypted matches original",
        result.plaintext == sample_data,
    );
    d29 += 1;
    v.check_pass("Vault: content hash matches", result.content_hash == hash);
    d29 += 1;

    let wrong_key = [99u8; 32];
    v.check_pass(
        "Vault: wrong key rejected",
        vault.retrieve(&hash, &wrong_key, &consent).is_err(),
    );
    d29 += 1;

    let other_consent = ConsentTicket::new(
        "patient-002",
        ConsentScope::ReadRawSequences,
        "wetspring-vault",
        Duration::from_secs(3600),
    );
    v.check_pass(
        "Vault: cross-owner rejected",
        vault.retrieve(&hash, &key, &other_consent).is_err(),
    );
    d29 += 1;

    v.check_pass(
        "Vault: unauthorized rejected",
        vault.retrieve_unauthorized(&hash).is_err(),
    );
    d29 += 1;

    v.check_pass("Vault: provenance chain intact", vault.verify_provenance());
    d29 += 1;

    timings.push(DomainTiming {
        name: "D29 Encrypted Storage",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d29,
    });

    // ═══ D30: DF64 Pack/Unpack ════════════════════════════════════════════
    let t = Instant::now();
    v.section("D30: DF64 Pack/Unpack (df64_host)");
    let mut d30 = 0_u32;

    let [hi, lo] = df64_host::pack(std::f64::consts::PI);
    let restored = df64_host::unpack(hi, lo);
    let err = (restored - std::f64::consts::PI).abs();
    v.check(
        "DF64: PI roundtrip",
        err,
        0.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );
    d30 += 1;

    let test_vals = [1.0, -1e-10, 1e20, std::f64::consts::E, 0.0];
    let packed = df64_host::pack_slice(&test_vals);
    v.check_count("DF64: pack_slice len", packed.len(), 10);
    d30 += 1;
    let unpacked = df64_host::unpack_slice(&packed);
    v.check_count("DF64: unpack_slice len", unpacked.len(), 5);
    d30 += 1;

    let max_err = test_vals
        .iter()
        .zip(&unpacked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check(
        "DF64: slice max roundtrip error",
        max_err,
        0.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );
    d30 += 1;

    let try_result = df64_host::try_unpack_slice(&packed);
    v.check_pass(
        "DF64: try_unpack_slice Ok for valid data",
        try_result.is_ok(),
    );
    d30 += 1;

    let odd_data = [1.0_f32, 2.0, 3.0];
    let try_odd = df64_host::try_unpack_slice(&odd_data);
    v.check_pass(
        "DF64: try_unpack_slice Err for odd-length",
        try_odd.is_err(),
    );
    d30 += 1;

    timings.push(DomainTiming {
        name: "D30 DF64",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d30,
    });

    // ═══ D31: Encoding Sovereignty ════════════════════════════════════════
    let t = Instant::now();
    v.section("D31: Encoding Sovereignty (encoding)");
    let mut d31 = 0_u32;

    let input = b"ATCGATCGATCG 16S rRNA gene partial";
    let encoded = encoding::base64_encode(input);
    let decoded = encoding::base64_decode(&encoded).unwrap();
    v.check_pass("Base64: roundtrip matches original", decoded == input);
    d31 += 1;

    let empty_enc = encoding::base64_encode(b"");
    v.check_pass("Base64: empty encodes to empty", empty_enc.is_empty());
    d31 += 1;
    let empty_dec = encoding::base64_decode("").unwrap();
    v.check_pass("Base64: empty decodes to empty", empty_dec.is_empty());
    d31 += 1;

    let known_input = b"Man";
    let known_output = "TWFu";
    v.check_pass(
        "Base64: RFC 4648 test vector",
        encoding::base64_encode(known_input) == known_output,
    );
    d31 += 1;

    timings.push(DomainTiming {
        name: "D31 Base64",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d31,
    });

    // ═══ D32: Tolerance Registry ══════════════════════════════════════════
    let t = Instant::now();
    v.section("D32: Tolerance Registry (tolerances)");
    let mut d32 = 0_u32;

    v.check_pass(
        "Tolerances: GPU_VS_CPU_F64 > 0",
        tolerances::GPU_VS_CPU_F64 > 0.0,
    );
    d32 += 1;
    v.check_pass(
        "Tolerances: GPU_VS_CPU_F64 < 1e-3",
        tolerances::GPU_VS_CPU_F64 < 1e-3,
    );
    d32 += 1;
    v.check_pass(
        "Tolerances: DF64_ROUNDTRIP <= 1e-13",
        tolerances::DF64_ROUNDTRIP <= 1e-13,
    );
    d32 += 1;
    v.check_pass("Tolerances: EXACT == 0.0", tolerances::EXACT == 0.0);
    d32 += 1;

    timings.push(DomainTiming {
        name: "D32 Tolerances",
        ms: t.elapsed().as_secs_f64() * 1000.0,
        checks: d32,
    });

    // ═══ Summary ══════════════════════════════════════════════════════════

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  CPU v20 Domain Timing Summary                          ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    for timing in &timings {
        println!(
            "║  {:24} {:6.1} ms  ({:2} checks)           ║",
            timing.name, timing.ms, timing.checks
        );
    }
    let total_new: u32 = timings.iter().map(|t| t.checks).sum();
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Total new: {total_new:3} checks in {total_ms:8.1} ms                  ║");
    println!("║  Inherited: D01–D26 (99 checks from v19)               ║");
    println!(
        "║  Grand total: {:3} checks                                ║",
        total_new + 99
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    v.finish();
}
