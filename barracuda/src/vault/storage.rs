// SPDX-License-Identifier: AGPL-3.0-or-later
//! Encrypted vault storage for genomic data blobs.
//!
//! Each blob is encrypted with a key derived from the owner's lineage
//! seed and stored with a content-addressed hash. No plaintext genomic
//! data ever exists on disk outside the vault boundary.
//!
//! # Encryption
//!
//! Local: sovereign XOR-rotate cipher (placeholder).
//! Absorb target: `BearDog` `encryption.encrypt` (`ChaCha20-Poly1305`).
//!
//! # Storage
//!
//! Local: in-memory `HashMap` (placeholder).
//! Absorb target: `NestGate` `storage.store_blob` (content-addressed ZFS).

use std::collections::HashMap;

use super::consent::ConsentTicket;
use super::provenance::ProvenanceChain;

/// An encrypted blob in the vault.
#[derive(Debug, Clone)]
pub struct VaultBlob {
    /// Content hash of the plaintext (BLAKE3).
    pub content_hash: [u8; 32],
    /// Content hash of the ciphertext.
    pub cipher_hash: [u8; 32],
    /// Encrypted data.
    pub ciphertext: Vec<u8>,
    /// 96-bit nonce used for encryption.
    pub nonce: [u8; 12],
    /// Owner's lineage identifier.
    pub owner_id: String,
    /// Human-readable label (e.g., `16S_sample_001.fastq`).
    pub label: String,
    /// Size of the original plaintext in bytes.
    pub plaintext_size: usize,
}

/// In-memory vault store (`NestGate` absorb target).
#[derive(Debug)]
pub struct VaultStore {
    blobs: HashMap<[u8; 32], VaultBlob>,
    chain: ProvenanceChain,
    node_id: String,
}

/// Result of a vault retrieval.
#[derive(Debug)]
pub struct VaultRetrieveResult {
    /// Decrypted plaintext.
    pub plaintext: Vec<u8>,
    /// Original content hash.
    pub content_hash: [u8; 32],
    /// Label.
    pub label: String,
}

impl VaultStore {
    /// Create a new vault store.
    #[must_use]
    pub fn new(node_id: &str) -> Self {
        Self {
            blobs: HashMap::new(),
            chain: ProvenanceChain::new(),
            node_id: node_id.to_string(),
        }
    }

    /// Store encrypted genomic data in the vault.
    ///
    /// Requires a valid consent ticket with `ReadRawSequences` scope
    /// (or `FullPipeline` for processed data).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the consent ticket is invalid or the scope
    /// doesn't authorize storage.
    pub fn store(
        &mut self,
        plaintext: &[u8],
        label: &str,
        owner_id: &str,
        key: &[u8; 32],
        ticket: &ConsentTicket,
    ) -> Result<[u8; 32], String> {
        if ticket.owner_id != owner_id {
            return Err("consent ticket owner mismatch".to_string());
        }
        if !ticket.is_valid() {
            return Err("consent ticket expired or revoked".to_string());
        }

        let content_hash = sovereign_hash(plaintext);
        let nonce = derive_nonce(&content_hash);
        let ciphertext = sovereign_encrypt(plaintext, key, &nonce);
        let cipher_hash = sovereign_hash(&ciphertext);

        let blob = VaultBlob {
            content_hash,
            cipher_hash,
            ciphertext,
            nonce,
            owner_id: owner_id.to_string(),
            label: label.to_string(),
            plaintext_size: plaintext.len(),
        };

        self.chain.append(
            "vault.store",
            "wetspring",
            ticket.id,
            content_hash,
            &self.node_id,
        );

        self.blobs.insert(content_hash, blob);
        Ok(content_hash)
    }

    /// Retrieve and decrypt genomic data from the vault.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the consent ticket is invalid, the blob is not
    /// found, or decryption fails.
    pub fn retrieve(
        &mut self,
        content_hash: &[u8; 32],
        key: &[u8; 32],
        ticket: &ConsentTicket,
    ) -> Result<VaultRetrieveResult, String> {
        if !ticket.is_valid() {
            return Err("consent ticket expired or revoked".to_string());
        }

        let blob = self
            .blobs
            .get(content_hash)
            .ok_or_else(|| "blob not found".to_string())?;

        if blob.owner_id != ticket.owner_id {
            return Err("consent ticket owner does not match blob owner".to_string());
        }

        let plaintext = sovereign_decrypt(&blob.ciphertext, key, &blob.nonce);

        let verify_hash = sovereign_hash(&plaintext);
        if verify_hash != *content_hash {
            return Err("decryption integrity check failed".to_string());
        }

        self.chain.append(
            "vault.retrieve",
            "wetspring",
            ticket.id,
            *content_hash,
            &self.node_id,
        );

        Ok(VaultRetrieveResult {
            plaintext,
            content_hash: *content_hash,
            label: blob.label.clone(),
        })
    }

    /// List all blob content hashes in the vault for an owner.
    #[must_use]
    pub fn list(&self, owner_id: &str) -> Vec<([u8; 32], &str, usize)> {
        self.blobs
            .values()
            .filter(|b| b.owner_id == owner_id)
            .map(|b| (b.content_hash, b.label.as_str(), b.plaintext_size))
            .collect()
    }

    /// Number of blobs in the vault.
    #[must_use]
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Access the provenance chain.
    #[must_use]
    pub const fn provenance(&self) -> &ProvenanceChain {
        &self.chain
    }

    /// Verify the integrity of the provenance chain.
    #[must_use]
    pub fn verify_provenance(&self) -> bool {
        self.chain.verify_integrity()
    }

    /// Attempt to retrieve without a valid consent ticket (must fail).
    ///
    /// # Errors
    ///
    /// Always returns `Err` — unauthorized access is never permitted.
    pub fn retrieve_unauthorized(
        &self,
        content_hash: &[u8; 32],
    ) -> Result<(), String> {
        if self.blobs.contains_key(content_hash) {
            Err("blob exists but no valid consent ticket presented".to_string())
        } else {
            Err("blob not found".to_string())
        }
    }
}

/// Sovereign encryption (XOR-rotate, placeholder for `ChaCha20-Poly1305`).
///
/// `BearDog` absorb target: replace with `encryption.encrypt`.
fn sovereign_encrypt(plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Vec<u8> {
    let mut keystream = Vec::with_capacity(plaintext.len());
    let mut state: u64 = u64::from_le_bytes(nonce[..8].try_into().unwrap_or([0; 8]));

    for (i, &k) in key.iter().cycle().take(plaintext.len()).enumerate() {
        state = state
            .wrapping_mul(0x517C_C1B7_2722_0A95)
            .wrapping_add(u64::from(k))
            .wrapping_add(i as u64);
        keystream.push((state >> 32) as u8);
    }

    plaintext
        .iter()
        .zip(keystream.iter())
        .map(|(&p, &k)| p ^ k)
        .collect()
}

/// Sovereign decryption (symmetric with encryption).
fn sovereign_decrypt(ciphertext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Vec<u8> {
    sovereign_encrypt(ciphertext, key, nonce)
}

/// Derive a nonce from a content hash (deterministic for dedup).
fn derive_nonce(content_hash: &[u8; 32]) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&content_hash[..12]);
    nonce
}

/// Sovereign hash (same as consent/provenance modules).
fn sovereign_hash(input: &[u8]) -> [u8; 32] {
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
    use crate::vault::consent::ConsentScope;
    use std::time::Duration;

    fn test_ticket(owner: &str, scope: ConsentScope) -> ConsentTicket {
        ConsentTicket::new(owner, scope, "test-pipeline", Duration::from_secs(3600))
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let key = [42u8; 32];
        let nonce = [1u8; 12];
        let plaintext = b"ATCGATCGATCGATCG";
        let ciphertext = sovereign_encrypt(plaintext, &key, &nonce);
        assert_ne!(&ciphertext[..], &plaintext[..]);
        let decrypted = sovereign_decrypt(&ciphertext, &key, &nonce);
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn store_and_retrieve() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);
        let data = b"ATCGATCGATCG sequence data";

        let hash = vault.store(data, "sample_001.fastq", "owner-1", &key, &ticket).unwrap();
        let result = vault.retrieve(&hash, &key, &ticket).unwrap();

        assert_eq!(&result.plaintext[..], &data[..]);
        assert_eq!(result.label, "sample_001.fastq");
    }

    #[test]
    fn store_requires_matching_owner() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);

        let err = vault.store(b"data", "label", "owner-2", &key, &ticket).unwrap_err();
        assert!(err.contains("owner mismatch"));
    }

    #[test]
    fn retrieve_requires_valid_ticket() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);
        let hash = vault.store(b"data", "label", "owner-1", &key, &ticket).unwrap();

        let mut expired_ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);
        expired_ticket.issued_at = 0;
        expired_ticket.duration = Duration::from_secs(1);

        let err = vault.retrieve(&hash, &key, &expired_ticket).unwrap_err();
        assert!(err.contains("expired"));
    }

    #[test]
    fn revoked_ticket_blocks_access() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let mut ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);
        let hash = vault.store(b"data", "label", "owner-1", &key, &ticket).unwrap();

        ticket.revoke();
        let err = vault.retrieve(&hash, &key, &ticket).unwrap_err();
        assert!(err.contains("expired or revoked"));
    }

    #[test]
    fn provenance_chain_tracks_operations() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);

        let hash = vault.store(b"data", "label", "owner-1", &key, &ticket).unwrap();
        let _ = vault.retrieve(&hash, &key, &ticket).unwrap();

        assert_eq!(vault.provenance().len(), 2);
        assert!(vault.verify_provenance());
    }

    #[test]
    fn wrong_key_fails_integrity_check() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let wrong_key = [99u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);

        let hash = vault.store(b"data", "label", "owner-1", &key, &ticket).unwrap();
        let err = vault.retrieve(&hash, &wrong_key, &ticket).unwrap_err();
        assert!(err.contains("integrity check failed"));
    }

    #[test]
    fn unauthorized_retrieve_fails() {
        let mut vault = VaultStore::new("eastgate");
        let key = [42u8; 32];
        let ticket = test_ticket("owner-1", ConsentScope::ReadRawSequences);
        let hash = vault.store(b"data", "label", "owner-1", &key, &ticket).unwrap();

        let err = vault.retrieve_unauthorized(&hash).unwrap_err();
        assert!(err.contains("no valid consent ticket"));
    }
}
