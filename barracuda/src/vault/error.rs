// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed errors for the Genomic Vault module.

use std::fmt;

/// Errors produced by vault storage operations.
#[derive(Debug)]
pub enum VaultError {
    /// Consent ticket's owner does not match the requested owner.
    ConsentOwnerMismatch,
    /// Consent ticket has expired or been revoked.
    ConsentExpiredOrRevoked,
    /// Requested blob was not found in the vault.
    BlobNotFound,
    /// Decryption failed (wrong key or tampered ciphertext).
    DecryptionFailed,
    /// Post-decryption integrity check failed (content hash mismatch).
    IntegrityCheckFailed,
    /// Encryption failed.
    EncryptionFailed(String),
    /// Blob exists but no valid consent ticket was presented.
    Unauthorized,
    /// Invalid key length for cipher construction.
    InvalidKeyLength,
}

impl fmt::Display for VaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConsentOwnerMismatch => write!(f, "consent ticket owner mismatch"),
            Self::ConsentExpiredOrRevoked => {
                write!(f, "consent ticket expired or revoked")
            }
            Self::BlobNotFound => write!(f, "blob not found"),
            Self::DecryptionFailed => {
                write!(f, "decryption failed (wrong key or tampered ciphertext)")
            }
            Self::IntegrityCheckFailed => write!(f, "decryption integrity check failed"),
            Self::EncryptionFailed(msg) => write!(f, "encryption failed: {msg}"),
            Self::Unauthorized => {
                write!(f, "blob exists but no valid consent ticket presented")
            }
            Self::InvalidKeyLength => write!(f, "invalid key length"),
        }
    }
}

impl std::error::Error for VaultError {}
