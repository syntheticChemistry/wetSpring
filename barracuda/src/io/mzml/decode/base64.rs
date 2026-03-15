// SPDX-License-Identifier: AGPL-3.0-or-later
//! Base64 decoding for mzML binary arrays.
//!
//! Wraps the sovereign base64 decoder with mzML-specific trimming of
//! whitespace around the encoded payload.

use crate::encoding::base64_decode;
use crate::error::Result;

/// Decode a base64-encoded string to raw bytes.
///
/// Trims whitespace from the input before decoding, as mzML binary
/// elements may contain newlines or indentation.
///
/// # Errors
///
/// Returns [`Error::Base64`] for invalid base64 input.
pub(super) fn decode_base64(encoded: &str) -> Result<Vec<u8>> {
    base64_decode(encoded.trim())
}
