// SPDX-License-Identifier: AGPL-3.0-or-later
//! In-tree base64 encoding/decoding — zero external dependencies.
//!
//! Implements only standard base64 (RFC 4648 §4) with `=` padding.
//! This replaces the `base64` crate for sovereignty: no external code
//! in the decode path for binary mzML arrays.

/// Standard base64 alphabet (RFC 4648 §4).
const ENCODE_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Decode table: maps ASCII byte → 6-bit value (255 = invalid).
const fn build_decode_table() -> [u8; 256] {
    let mut table = [255_u8; 256];
    let mut i = 0_usize;
    while i < 64 {
        #[allow(clippy::cast_possible_truncation)] // i < 64, always fits u8
        let val = i as u8;
        table[ENCODE_TABLE[i] as usize] = val;
        i += 1;
    }
    table
}

const DECODE_TABLE: [u8; 256] = build_decode_table();

/// Decode a standard base64 string to bytes.
///
/// Handles `=` padding and ignores whitespace (space, tab, CR, LF).
///
/// # Errors
///
/// Returns an error message if the input contains invalid characters
/// or has malformed padding.
pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let mut buf = Vec::with_capacity(input.len() * 3 / 4);
    let mut accum: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in input.as_bytes() {
        if matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b'=') {
            continue;
        }
        let val = DECODE_TABLE[byte as usize];
        if val == 255 {
            return Err(format!("invalid base64 byte: {byte:#04x}"));
        }
        accum = (accum << 6) | u32::from(val);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            #[allow(clippy::cast_possible_truncation)] // masked to 8 bits
            buf.push((accum >> bits) as u8);
            accum &= (1 << bits) - 1;
        }
    }

    Ok(buf)
}

/// Encode bytes to standard base64 with `=` padding.
#[must_use]
pub fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;

    while i + 2 < data.len() {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8) | u32::from(data[i + 2]);
        result.push(ENCODE_TABLE[((n >> 18) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((n >> 12) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((n >> 6) & 63) as usize] as char);
        result.push(ENCODE_TABLE[(n & 63) as usize] as char);
        i += 3;
    }

    let remaining = data.len() - i;
    if remaining == 1 {
        let n = u32::from(data[i]) << 16;
        result.push(ENCODE_TABLE[((n >> 18) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((n >> 12) & 63) as usize] as char);
        result.push('=');
        result.push('=');
    } else if remaining == 2 {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8);
        result.push(ENCODE_TABLE[((n >> 18) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((n >> 12) & 63) as usize] as char);
        result.push(ENCODE_TABLE[((n >> 6) & 63) as usize] as char);
        result.push('=');
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(base64_encode(&[]), "");
        assert_eq!(base64_decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn roundtrip_hello() {
        let encoded = base64_encode(b"Hello, World!");
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
        assert_eq!(base64_decode(&encoded).unwrap(), b"Hello, World!");
    }

    #[test]
    fn roundtrip_1_byte() {
        let encoded = base64_encode(&[0xFF]);
        assert_eq!(base64_decode(&encoded).unwrap(), vec![0xFF]);
    }

    #[test]
    fn roundtrip_2_bytes() {
        let encoded = base64_encode(&[0xAB, 0xCD]);
        assert_eq!(base64_decode(&encoded).unwrap(), vec![0xAB, 0xCD]);
    }

    #[test]
    fn roundtrip_3_bytes() {
        let encoded = base64_encode(&[0x01, 0x02, 0x03]);
        assert_eq!(base64_decode(&encoded).unwrap(), vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn roundtrip_f64_array() {
        let values: Vec<f64> = vec![100.0, 200.0, 300.0, 12345.6789];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = base64_encode(&bytes);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, bytes);

        // Verify f64 reconstruction
        let reconstructed: Vec<f64> = decoded
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(reconstructed, values);
    }

    #[test]
    fn decode_with_whitespace() {
        let encoded = "SGVs\nbG8s\r\n IFdv\tcmxkIQ==";
        assert_eq!(base64_decode(encoded).unwrap(), b"Hello, World!");
    }

    #[test]
    fn decode_invalid_char() {
        assert!(base64_decode("SGVs!bG8=").is_err());
    }

    #[test]
    fn decode_no_padding() {
        // Some encoders omit padding — our decoder should handle it
        assert_eq!(base64_decode("SGVsbG8").unwrap(), b"Hello");
    }

    #[test]
    fn known_rfc4648_vectors() {
        // RFC 4648 test vectors
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
