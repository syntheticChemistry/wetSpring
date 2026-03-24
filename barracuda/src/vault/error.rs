// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed errors for the Genomic Vault module.

/// Errors produced by vault storage operations.
#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    /// Consent ticket's owner does not match the requested owner.
    #[error("consent ticket owner mismatch")]
    ConsentOwnerMismatch,
    /// Consent ticket has expired or been revoked.
    #[error("consent ticket expired or revoked")]
    ConsentExpiredOrRevoked,
    /// Requested blob was not found in the vault.
    #[error("blob not found")]
    BlobNotFound,
    /// Decryption failed (wrong key or tampered ciphertext).
    #[error("decryption failed (wrong key or tampered ciphertext)")]
    DecryptionFailed,
    /// Post-decryption integrity check failed (content hash mismatch).
    #[error("decryption integrity check failed")]
    IntegrityCheckFailed,
    /// Encryption failed.
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),
    /// Blob exists but no valid consent ticket was presented.
    #[error("blob exists but no valid consent ticket presented")]
    Unauthorized,
    /// Invalid key length for cipher construction.
    #[error("invalid key length")]
    InvalidKeyLength,
}
