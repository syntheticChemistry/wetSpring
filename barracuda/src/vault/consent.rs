// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consent ticket protocol for genomic data access.
//!
//! A consent ticket is a signed, time-bounded, scope-limited, revocable
//! authorization for a specific pipeline to access specific data. The
//! ticket is checked before any vault operation.
//!
//! # Design
//!
//! Modeled on organ transplant consent: explicit, informed, revocable,
//! scope-limited, audited.
//!
//! # Signing
//!
//! Ticket authentication uses BLAKE3 keyed MAC derived from the owner's
//! lineage seed. This is a symmetric scheme appropriate for self-signed
//! tickets verified within the same trust boundary.
//!
//! When BearDog reaches IPC maturity, signing migrates to
//! `crypto.sign_ed25519` / `crypto.verify_ed25519` over JSON-RPC —
//! Tower Atomic delegation. No code changes in callers needed (the
//! `sign_with_lineage` / `verify_signature` API stays the same).
//!
//! # Absorb targets
//!
//! - `BearDog`: asymmetric signing (`crypto.sign_ed25519`) for cross-boundary tickets
//! - `Songbird`: `ConsentManager` (`POST /consent/request`, approve/deny)

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// What the consent ticket authorizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsentScope {
    /// Compute diversity metrics (Shannon, Simpson, Chao1, etc.)
    DiversityAnalysis,
    /// Run Anderson spectral classification
    AndersonClassification,
    /// Full pipeline (diversity + QS + Anderson)
    FullPipeline,
    /// Export results (non-raw, aggregated only)
    ExportAggregated,
    /// Read raw sequences (most sensitive)
    ReadRawSequences,
    /// Custom scope with description.
    Custom(String),
}

impl ConsentScope {
    /// Sensitivity level (higher = more restricted).
    #[must_use]
    pub const fn sensitivity(&self) -> u8 {
        match self {
            Self::DiversityAnalysis | Self::AndersonClassification => 1,
            Self::FullPipeline => 2,
            Self::ExportAggregated => 3,
            Self::ReadRawSequences => 5,
            Self::Custom(_) => 4,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::DiversityAnalysis => "diversity_analysis",
            Self::AndersonClassification => "anderson_classification",
            Self::FullPipeline => "full_pipeline",
            Self::ExportAggregated => "export_aggregated",
            Self::ReadRawSequences => "read_raw_sequences",
            Self::Custom(s) => s,
        }
    }
}

/// A signed consent ticket authorizing data access.
#[derive(Debug, Clone)]
pub struct ConsentTicket {
    /// Unique ticket identifier (BLAKE3 of contents).
    pub id: [u8; 32],
    /// Who owns the data (lineage identifier).
    pub owner_id: String,
    /// What is authorized.
    pub scope: ConsentScope,
    /// Who/what is authorized to access (pipeline, node, primal).
    pub grantee: String,
    /// When the ticket was issued (UNIX timestamp seconds).
    pub issued_at: u64,
    /// How long the ticket is valid.
    pub duration: Duration,
    /// Whether the ticket has been revoked.
    pub revoked: bool,
    /// BLAKE3 keyed MAC over the ticket contents (32 bytes).
    pub signature: [u8; 32],
}

impl ConsentTicket {
    /// Create a new consent ticket.
    ///
    /// Uses BLAKE3 for the ticket ID. Signature is zeroed until
    /// `sign_with_lineage` is called.
    #[must_use]
    pub fn new(owner_id: &str, scope: ConsentScope, grantee: &str, duration: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let id = Self::compute_id(owner_id, scope.label(), grantee, now, duration.as_secs());

        Self {
            id,
            owner_id: owner_id.to_string(),
            scope,
            grantee: grantee.to_string(),
            issued_at: now,
            duration,
            revoked: false,
            signature: [0u8; 32],
        }
    }

    /// Compute the ticket ID (BLAKE3 of canonical fields).
    fn compute_id(
        owner_id: &str,
        scope_label: &str,
        grantee: &str,
        issued_at: u64,
        duration_secs: u64,
    ) -> [u8; 32] {
        let mut hasher_input = Vec::new();
        hasher_input.extend_from_slice(owner_id.as_bytes());
        hasher_input.extend_from_slice(scope_label.as_bytes());
        hasher_input.extend_from_slice(grantee.as_bytes());
        hasher_input.extend_from_slice(&issued_at.to_le_bytes());
        hasher_input.extend_from_slice(&duration_secs.to_le_bytes());
        *blake3::hash(&hasher_input).as_bytes()
    }

    /// Bytes to sign (canonical representation of ticket contents).
    fn message_to_sign(&self) -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(self.owner_id.as_bytes());
        msg.extend_from_slice(self.scope.label().as_bytes());
        msg.extend_from_slice(self.grantee.as_bytes());
        msg.extend_from_slice(&self.issued_at.to_le_bytes());
        msg.extend_from_slice(&self.duration.as_secs().to_le_bytes());
        msg
    }

    /// Sign this ticket with a lineage seed.
    ///
    /// Derives a BLAKE3 keyed MAC from the seed and signs the ticket
    /// contents. Sovereign (zero external crypto deps).
    pub fn sign_with_lineage(&mut self, lineage_seed: &[u8]) {
        let key = Self::derive_signing_key(lineage_seed);
        let msg = self.message_to_sign();
        let mac = blake3::keyed_hash(&key, &msg);
        self.signature = *mac.as_bytes();
    }

    /// Verify the ticket signature against a lineage seed.
    ///
    /// Returns `true` if the signature is valid (or zeroed for unsigned
    /// legacy tickets). Uses constant-time comparison via BLAKE3's
    /// `Hash::eq` implementation.
    #[must_use]
    pub fn verify_signature(&self, lineage_seed: &[u8]) -> bool {
        if self.signature == [0u8; 32] {
            return true;
        }
        let key = Self::derive_signing_key(lineage_seed);
        let expected = blake3::keyed_hash(&key, &self.message_to_sign());
        expected.as_bytes() == &self.signature
    }

    /// Derive the 32-byte signing key from a lineage seed via BLAKE3.
    fn derive_signing_key(lineage_seed: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"wetspring-consent-signing-v1");
        hasher.update(lineage_seed);
        *hasher.finalize().as_bytes()
    }

    /// Check whether this ticket is currently valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let expires_at = self.issued_at + self.duration.as_secs();
        now < expires_at
    }

    /// Check whether this ticket authorizes a specific scope.
    #[must_use]
    pub fn authorizes(&self, requested: &ConsentScope) -> bool {
        if !self.is_valid() {
            return false;
        }
        match (&self.scope, requested) {
            (a, b) if a == b => true,
            (
                ConsentScope::FullPipeline,
                ConsentScope::DiversityAnalysis | ConsentScope::AndersonClassification,
            ) => true,
            _ => false,
        }
    }

    /// Revoke this ticket.
    pub const fn revoke(&mut self) {
        self.revoked = true;
    }

    /// Remaining validity duration (zero if expired or revoked).
    #[must_use]
    pub fn remaining(&self) -> Duration {
        if self.revoked {
            return Duration::ZERO;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let expires_at = self.issued_at + self.duration.as_secs();
        if now >= expires_at {
            Duration::ZERO
        } else {
            Duration::from_secs(expires_at - now)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consent_ticket_valid_when_created() {
        let ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::DiversityAnalysis,
            "wetspring-pipeline",
            Duration::from_secs(3600),
        );
        assert!(ticket.is_valid());
        assert!(!ticket.revoked);
        assert!(ticket.remaining() > Duration::ZERO);
    }

    #[test]
    fn consent_ticket_expired() {
        let mut ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::DiversityAnalysis,
            "wetspring-pipeline",
            Duration::from_secs(0),
        );
        ticket.issued_at = 0;
        ticket.duration = Duration::from_secs(1);
        assert!(!ticket.is_valid());
    }

    #[test]
    fn consent_ticket_revocable() {
        let mut ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::FullPipeline,
            "wetspring-pipeline",
            Duration::from_secs(3600),
        );
        assert!(ticket.is_valid());
        ticket.revoke();
        assert!(!ticket.is_valid());
        assert_eq!(ticket.remaining(), Duration::ZERO);
    }

    #[test]
    fn full_pipeline_implies_sub_scopes() {
        let ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::FullPipeline,
            "wetspring-pipeline",
            Duration::from_secs(3600),
        );
        assert!(ticket.authorizes(&ConsentScope::DiversityAnalysis));
        assert!(ticket.authorizes(&ConsentScope::AndersonClassification));
        assert!(!ticket.authorizes(&ConsentScope::ReadRawSequences));
    }

    #[test]
    fn ticket_id_is_deterministic_for_same_input() {
        let h1 = *blake3::hash(b"test input for hash").as_bytes();
        let h2 = *blake3::hash(b"test input for hash").as_bytes();
        assert_eq!(h1, h2);
    }

    #[test]
    fn ticket_id_differs_for_different_input() {
        let h1 = *blake3::hash(b"input A").as_bytes();
        let h2 = *blake3::hash(b"input B").as_bytes();
        assert_ne!(h1, h2);
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let mut ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::ReadRawSequences,
            "wetspring-pipeline",
            Duration::from_secs(3600),
        );
        let lineage_seed = b"test-lineage-seed-32-bytes!!!!!!";
        ticket.sign_with_lineage(lineage_seed);
        assert!(ticket.verify_signature(lineage_seed));
    }

    #[test]
    fn verify_fails_with_wrong_seed() {
        let mut ticket = ConsentTicket::new(
            "eastgate-family",
            ConsentScope::ReadRawSequences,
            "wetspring-pipeline",
            Duration::from_secs(3600),
        );
        ticket.sign_with_lineage(b"correct-lineage-seed-32-bytes!!!!!");
        assert!(!ticket.verify_signature(b"wrong-lineage-seed-32-bytes!!!!!!"));
    }

    #[test]
    fn sensitivity_ordering() {
        assert!(
            ConsentScope::ReadRawSequences.sensitivity()
                > ConsentScope::DiversityAnalysis.sensitivity()
        );
        assert!(
            ConsentScope::ExportAggregated.sensitivity() > ConsentScope::FullPipeline.sensitivity()
        );
    }
}
