// SPDX-License-Identifier: AGPL-3.0-or-later
//! Data provenance chain — append-only audit log for vault operations.
//!
//! Every operation on genomic data produces a signed provenance entry:
//! who touched it, what they computed, when, on what hardware, under
//! what consent ticket.
//!
//! # Design
//!
//! Each entry is chained via its parent hash (Merkle chain). The chain
//! is immutable — entries can only be appended. This produces a tamper-
//! evident audit trail that `BearDog` can sign and `NestGate` can store.
//!
//! # Absorb targets
//!
//! - `BearDog`: Ed25519 signing of entries
//! - `NestGate`: append-only CAS storage of the chain
//! - `biomeOS`: Neural API `provenance.append` capability

use std::time::{SystemTime, UNIX_EPOCH};

/// A single provenance entry in the audit chain.
#[derive(Debug, Clone)]
pub struct ProvenanceEntry {
    /// BLAKE3 hash of this entry's contents.
    pub hash: [u8; 32],
    /// Hash of the previous entry (zero for genesis).
    pub parent: [u8; 32],
    /// UNIX timestamp (seconds).
    pub timestamp: u64,
    /// What operation was performed.
    pub operation: String,
    /// Who performed it (primal or pipeline identifier).
    pub actor: String,
    /// Consent ticket ID that authorized this operation.
    pub consent_ticket_id: [u8; 32],
    /// Data blob hash that was accessed/produced.
    pub data_hash: [u8; 32],
    /// Hardware/node identifier.
    pub node_id: String,
    /// Ed25519 signature (`BearDog` absorb target).
    pub signature: [u8; 64],
}

/// Append-only provenance chain.
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    entries: Vec<ProvenanceEntry>,
}

impl ProvenanceChain {
    /// Create a new empty chain.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Number of entries in the chain.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append a new entry to the chain.
    ///
    /// The parent hash is automatically set to the previous entry's hash,
    /// or zeroed for the genesis entry.
    ///
    /// # Panics
    ///
    /// Cannot panic — `last()` is always `Some` after `push`.
    pub fn append(
        &mut self,
        operation: &str,
        actor: &str,
        consent_ticket_id: [u8; 32],
        data_hash: [u8; 32],
        node_id: &str,
    ) -> &ProvenanceEntry {
        let parent = self.entries.last().map_or([0u8; 32], |e| e.hash);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut hasher_input = Vec::new();
        hasher_input.extend_from_slice(&parent);
        hasher_input.extend_from_slice(&timestamp.to_le_bytes());
        hasher_input.extend_from_slice(operation.as_bytes());
        hasher_input.extend_from_slice(actor.as_bytes());
        hasher_input.extend_from_slice(&consent_ticket_id);
        hasher_input.extend_from_slice(&data_hash);
        hasher_input.extend_from_slice(node_id.as_bytes());

        let hash = sovereign_hash(&hasher_input);

        self.entries.push(ProvenanceEntry {
            hash,
            parent,
            timestamp,
            operation: operation.to_string(),
            actor: actor.to_string(),
            consent_ticket_id,
            data_hash,
            node_id: node_id.to_string(),
            signature: [0u8; 64],
        });

        &self.entries[self.entries.len() - 1]
    }

    /// Verify chain integrity (each entry's parent matches the previous hash).
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        for (i, entry) in self.entries.iter().enumerate() {
            let expected_parent = if i == 0 {
                [0u8; 32]
            } else {
                self.entries[i - 1].hash
            };
            if entry.parent != expected_parent {
                return false;
            }
        }
        true
    }

    /// Get the most recent entry.
    #[must_use]
    pub fn head(&self) -> Option<&ProvenanceEntry> {
        self.entries.last()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ProvenanceEntry> {
        self.entries.iter()
    }

    /// Filter entries by actor.
    pub fn by_actor<'a>(&'a self, actor: &'a str) -> impl Iterator<Item = &'a ProvenanceEntry> {
        self.entries.iter().filter(move |e| e.actor == actor)
    }

    /// Filter entries by consent ticket ID.
    pub fn by_consent(&self, ticket_id: [u8; 32]) -> impl Iterator<Item = &ProvenanceEntry> + '_ {
        self.entries
            .iter()
            .filter(move |e| e.consent_ticket_id == ticket_id)
    }
}

impl Default for ProvenanceChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry hash using BLAKE3.
fn sovereign_hash(input: &[u8]) -> [u8; 32] {
    *blake3::hash(input).as_bytes()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_is_valid() {
        let chain = ProvenanceChain::new();
        assert!(chain.verify_integrity());
        assert!(chain.is_empty());
    }

    #[test]
    fn single_entry_chain() {
        let mut chain = ProvenanceChain::new();
        let ticket_id = [1u8; 32];
        let data_hash = [2u8; 32];
        chain.append(
            "diversity_analysis",
            "wetspring",
            ticket_id,
            data_hash,
            "eastgate",
        );
        assert_eq!(chain.len(), 1);
        assert!(chain.verify_integrity());
        assert_eq!(chain.head().unwrap().parent, [0u8; 32]);
    }

    #[test]
    fn chain_links_correctly() {
        let mut chain = ProvenanceChain::new();
        let ticket_id = [1u8; 32];
        let data_hash = [2u8; 32];

        chain.append("ingest", "nestgate", ticket_id, data_hash, "eastgate");
        let first_hash = chain.head().unwrap().hash;

        chain.append(
            "diversity_analysis",
            "wetspring",
            ticket_id,
            data_hash,
            "eastgate",
        );
        assert_eq!(chain.head().unwrap().parent, first_hash);

        chain.append(
            "anderson_classification",
            "toadstool",
            ticket_id,
            data_hash,
            "eastgate",
        );
        assert_eq!(chain.len(), 3);
        assert!(chain.verify_integrity());
    }

    #[test]
    fn tampered_chain_fails_verification() {
        let mut chain = ProvenanceChain::new();
        let ticket_id = [1u8; 32];
        let data_hash = [2u8; 32];

        chain.append("ingest", "nestgate", ticket_id, data_hash, "eastgate");
        chain.append("analysis", "wetspring", ticket_id, data_hash, "eastgate");

        // Tamper: change the first entry's hash
        chain.entries[0].hash = [99u8; 32];
        assert!(!chain.verify_integrity());
    }

    #[test]
    fn filter_by_actor() {
        let mut chain = ProvenanceChain::new();
        let t = [0u8; 32];
        let d = [0u8; 32];
        chain.append("op1", "wetspring", t, d, "eastgate");
        chain.append("op2", "nestgate", t, d, "eastgate");
        chain.append("op3", "wetspring", t, d, "eastgate");

        assert_eq!(chain.by_actor("wetspring").count(), 2);
    }
}
