// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zlib decompression and float array parsing for mzML binary arrays.
//!
//! Handles optional zlib decompression and conversion of raw bytes
//! to f32/f64 arrays (little-endian).

use crate::error::{Error, Result};
use flate2::read::ZlibDecoder;
use std::io::Read;

/// Decompress zlib-compressed bytes into the given buffer.
///
/// The buffer is cleared before writing. Caller should reuse the buffer
/// across multiple decompressions to amortize allocations.
///
/// # Errors
///
/// Returns [`Error::Zlib`] for decompression failures.
pub(super) fn decompress_zlib(bytes: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let mut decoder = ZlibDecoder::new(bytes);
    decoder
        .read_to_end(out)
        .map_err(|e| Error::Zlib(format!("{e}")))?;
    Ok(())
}

/// Parse raw bytes as a little-endian float array.
///
/// When `is_64bit` is true, interprets 8-byte chunks as f64.
/// Otherwise interprets 4-byte chunks as f32 (promoted to f64).
///
/// # Errors
///
/// Returns [`Error::BinaryFormat`] if the byte length is not divisible
/// by the element size (8 for f64, 4 for f32).
pub(super) fn decode_float_array(bytes: &[u8], is_64bit: bool) -> Result<Vec<f64>> {
    if is_64bit {
        if !bytes.len().is_multiple_of(8) {
            return Err(Error::BinaryFormat(format!(
                "f64 array length {} not divisible by 8",
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| {
                let mut arr = [0_u8; 8];
                arr.copy_from_slice(chunk);
                f64::from_le_bytes(arr)
            })
            .collect())
    } else {
        if !bytes.len().is_multiple_of(4) {
            return Err(Error::BinaryFormat(format!(
                "f32 array length {} not divisible by 4",
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(chunk);
                f64::from(f32::from_le_bytes(arr))
            })
            .collect())
    }
}
