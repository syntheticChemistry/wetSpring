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
//! # Absorb targets
//!
//! - `BearDog`: Ed25519 signing (`genetic.sign_lineage_certificate`)
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
    /// Signature over the ticket contents (Ed25519 placeholder — `BearDog` absorb target).
    pub signature: [u8; 64],
}

impl ConsentTicket {
    /// Create a new consent ticket.
    ///
    /// Uses local BLAKE3 for the ticket ID. Signature is zeroed until
    /// `BearDog` absorbs the signing protocol.
    #[must_use]
    pub fn new(owner_id: &str, scope: ConsentScope, grantee: &str, duration: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut hasher_input = Vec::new();
        hasher_input.extend_from_slice(owner_id.as_bytes());
        hasher_input.extend_from_slice(scope.label().as_bytes());
        hasher_input.extend_from_slice(grantee.as_bytes());
        hasher_input.extend_from_slice(&now.to_le_bytes());
        hasher_input.extend_from_slice(&duration.as_secs().to_le_bytes());

        let id = sovereign_blake3(&hasher_input);

        Self {
            id,
            owner_id: owner_id.to_string(),
            scope,
            grantee: grantee.to_string(),
            issued_at: now,
            duration,
            revoked: false,
            signature: [0u8; 64],
        }
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

/// Sovereign BLAKE3 hash (minimal, no external deps).
///
/// Uses a simplified Merkle-Damgard construction with the BLAKE3 constants.
/// `BearDog` absorb target: replace with `beardog-security` BLAKE3.
fn sovereign_blake3(input: &[u8]) -> [u8; 32] {
    let mut h: [u64; 4] = [
        0x6A09_E667_F3BC_C908,
        0xBB67_AE85_84CA_A73B,
        0x3C6E_F372_FE94_F82B,
        0xA54F_F53A_5F1D_36F1,
    ];

    for chunk in input.chunks(32) {
        for (i, byte) in chunk.iter().enumerate() {
            h[i % 4] ^= u64::from(*byte) << ((i % 8) * 8);
            h[i % 4] = h[i % 4].wrapping_mul(0x517C_C1B7_2722_0A95).rotate_left(17);
        }
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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
        let h1 = sovereign_blake3(b"test input for hash");
        let h2 = sovereign_blake3(b"test input for hash");
        assert_eq!(h1, h2);
    }

    #[test]
    fn ticket_id_differs_for_different_input() {
        let h1 = sovereign_blake3(b"input A");
        let h2 = sovereign_blake3(b"input B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn sensitivity_ordering() {
        assert!(ConsentScope::ReadRawSequences.sensitivity() > ConsentScope::DiversityAnalysis.sensitivity());
        assert!(ConsentScope::ExportAggregated.sensitivity() > ConsentScope::FullPipeline.sensitivity());
    }
}
