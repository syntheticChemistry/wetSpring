// SPDX-License-Identifier: AGPL-3.0-or-later
//! Host-side DF64 (double-float f32-pair) pack/unpack utilities.
//!
//! barraCuda S68+ provides universal precision: shaders authored in f64
//! canonical form are compiled to any target via `compile_shader_universal`.
//! When targeting `Precision::Df64`, GPU storage uses `vec2<f32>` pairs
//! (hi, lo) instead of native f64. These utilities handle the host-side
//! conversion between f64 and the DF64 wire format.
//!
//! # Wire format
//!
//! ```text
//! f64 value → [hi: f32, lo: f32]
//!   hi = value as f32          (truncated to f32)
//!   lo = (value - hi) as f32   (residual)
//!   value ≈ hi + lo            (~48-bit mantissa)
//! ```
//!
//! On GPU: `struct Df64 { hi: f32, lo: f32 }`, stored as `vec2<f32>` in
//! storage buffers. `df64_core.wgsl` provides arithmetic (add, mul, fma,
//! sqrt, div) that preserves the double-float invariant.

/// Split a single f64 into a DF64 (hi, lo) pair.
///
/// Matches `df64_from_f64()` in `df64_core.wgsl`.
#[must_use]
pub const fn pack(v: f64) -> [f32; 2] {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    [hi, lo]
}

/// Reconstruct f64 from a DF64 (hi, lo) pair.
///
/// Matches `df64_to_f64()` in `df64_core.wgsl`.
#[must_use]
pub const fn unpack(hi: f32, lo: f32) -> f64 {
    hi as f64 + lo as f64
}

/// Pack a slice of f64 values into DF64 wire format (interleaved f32 pairs).
///
/// Output length = 2 × input length. Each f64 becomes two consecutive f32s
/// `[hi, lo]`, matching `array<vec2<f32>>` in WGSL.
#[must_use]
pub fn pack_slice(data: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let [hi, lo] = pack(v);
        out.push(hi);
        out.push(lo);
    }
    out
}

/// Unpack DF64 wire format (interleaved f32 pairs) back to f64 values.
///
/// Input length must be even. Each pair of f32s `[hi, lo]` is reconstructed
/// to f64. Output length = input length / 2.
///
/// # Errors
///
/// Returns [`InvalidInput`](crate::error::Error::InvalidInput) if `data.len()` is odd.
pub fn try_unpack_slice(data: &[f32]) -> crate::error::Result<Vec<f64>> {
    if !data.len().is_multiple_of(2) {
        return Err(crate::error::Error::InvalidInput(format!(
            "DF64 data must have even length, got {}",
            data.len()
        )));
    }
    Ok(data.chunks_exact(2).map(|c| unpack(c[0], c[1])).collect())
}

/// Unpack DF64 wire format (interleaved f32 pairs) back to f64 values.
///
/// Equivalent to [`try_unpack_slice`] but panics on odd-length input.
/// Prefer [`try_unpack_slice`] in library code; this is provided for
/// validation binaries where input is always from [`pack_slice`].
///
/// # Panics
///
/// Panics if `data.len()` is odd.
#[must_use]
pub fn unpack_slice(data: &[f32]) -> Vec<f64> {
    assert!(
        data.len().is_multiple_of(2),
        "DF64 data must have even length"
    );
    data.chunks_exact(2).map(|c| unpack(c[0], c[1])).collect()
}

/// Precision of a DF64 round-trip: pack then unpack.
///
/// Returns the maximum absolute error for the given value. DF64 preserves
/// ~48 bits of mantissa (vs f32's 24 bits), so round-trip error is typically
/// in the range of f32 ULP² ≈ 1e-15 for values near 1.0.
#[must_use]
pub fn roundtrip_error(v: f64) -> f64 {
    let [hi, lo] = pack(v);
    (unpack(hi, lo) - v).abs()
}

#[cfg(test)]
#[expect(clippy::float_cmp, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn pack_unpack_roundtrip_exact_for_f32_representable() {
        let v = 1.0_f64;
        let [hi, lo] = pack(v);
        assert_eq!(hi, 1.0_f32);
        assert_eq!(lo, 0.0_f32);
        assert_eq!(unpack(hi, lo), 1.0);
    }

    #[test]
    fn pack_unpack_roundtrip_high_precision() {
        let v = std::f64::consts::PI;
        let err = roundtrip_error(v);
        assert!(
            err < tolerances::PYTHON_PARITY_TIGHT,
            "roundtrip error {err} exceeds DF64 precision"
        );
    }

    #[test]
    fn pack_unpack_roundtrip_small_value() {
        let v = 1e-10_f64;
        let err = roundtrip_error(v);
        assert!(err < 1e-25, "roundtrip error {err} for small value");
    }

    #[test]
    fn pack_unpack_roundtrip_large_value() {
        let v = 1e20_f64;
        let err = roundtrip_error(v);
        let rel = err / v;
        assert!(
            rel < tolerances::MATRIX_EPS,
            "relative roundtrip error {rel}"
        );
    }

    #[test]
    fn pack_unpack_zero() {
        let [hi, lo] = pack(0.0);
        assert_eq!(hi, 0.0_f32);
        assert_eq!(lo, 0.0_f32);
        assert_eq!(unpack(hi, lo), 0.0);
    }

    #[test]
    fn pack_unpack_negative() {
        let v = -std::f64::consts::E;
        let err = roundtrip_error(v);
        assert!(
            err < tolerances::PYTHON_PARITY_TIGHT,
            "negative roundtrip error {err}"
        );
    }

    #[test]
    fn slice_roundtrip() {
        let data = [1.0, 2.0, std::f64::consts::PI, -1e-8, 1e15];
        let packed = pack_slice(&data);
        assert_eq!(packed.len(), 10);
        let unpacked = unpack_slice(&packed);
        assert_eq!(unpacked.len(), 5);
        for (orig, restored) in data.iter().zip(&unpacked) {
            let err = (orig - restored).abs();
            assert!(
                err < tolerances::PYTHON_PARITY_TIGHT,
                "slice roundtrip error {err} for {orig}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "even length")]
    fn unpack_odd_panics() {
        let _ = unpack_slice(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn try_unpack_odd_returns_error() {
        let result = try_unpack_slice(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("even length"),
            "error should mention even length: {msg}"
        );
    }

    #[test]
    fn try_unpack_roundtrip() {
        let data = [1.0, 2.0, std::f64::consts::PI];
        let packed = pack_slice(&data);
        let unpacked = try_unpack_slice(&packed).expect("valid packed data");
        assert_eq!(unpacked.len(), data.len());
    }

    #[test]
    fn df64_precision_vs_f32() {
        let v = 1.000_000_1_f64;
        let f32_err = (v - f64::from(v as f32)).abs();
        let df64_err = roundtrip_error(v);
        assert!(
            df64_err < f32_err,
            "DF64 error {df64_err} should be less than f32 error {f32_err}"
        );
    }
}
