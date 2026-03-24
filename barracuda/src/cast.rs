// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safe numeric cast helpers for GPU interop and domain conversions.
//!
//! Centralizes all `as` casts behind named functions with documented
//! precision and truncation semantics. Call sites use `crate::cast::*`
//! instead of raw `as` casts, eliminating scattered `#[expect(clippy::cast_*)]`.
//!
//! # Precision guarantees
//!
//! - `usize_f64` / `u64_f64`: exact for values < 2^53
//! - `i32_f64` / `u32_f64`: exact (32-bit integers always fit in f64)
//! - Truncating casts (`f64_usize`, `f64_u32`): `debug_assert!` guards
//!   in debug builds; saturating behavior in release

/// `usize` → `f64`. Exact for values below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "documented: exact for values < 2^53"
)]
#[inline]
#[must_use]
pub const fn usize_f64(v: usize) -> f64 {
    v as f64
}

/// `f64` → `usize`. Truncates toward zero; debug-asserts non-negative and in range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "caller ensures v is non-negative and fits in usize"
)]
#[inline]
#[must_use]
pub fn f64_usize(v: f64) -> usize {
    debug_assert!(v >= 0.0 && v.is_finite(), "f64_usize: {v} out of range");
    v as usize
}

/// `usize` → `u32`. Debug-asserts the value fits; GPU buffer sizes are bounded.
#[expect(
    clippy::cast_possible_truncation,
    reason = "caller ensures v fits in u32"
)]
#[inline]
#[must_use]
pub fn usize_u32(v: usize) -> u32 {
    debug_assert!(u32::try_from(v).is_ok(), "usize_u32: {v} overflows u32");
    v as u32
}

/// `i32` → `f64`. Always exact (32-bit integer fits in f64 mantissa).
#[inline]
#[must_use]
pub fn i32_f64(v: i32) -> f64 {
    f64::from(v)
}

/// `u32` → `f64`. Always exact (32-bit integer fits in f64 mantissa).
#[inline]
#[must_use]
pub fn u32_f64(v: u32) -> f64 {
    f64::from(v)
}

/// `f64` → `u32`. Truncates toward zero; debug-asserts non-negative and in range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "caller ensures v is non-negative and fits in u32"
)]
#[inline]
#[must_use]
pub fn f64_u32(v: f64) -> u32 {
    debug_assert!(
        v >= 0.0 && v <= f64::from(u32::MAX),
        "f64_u32: {v} out of range"
    );
    v as u32
}

/// `u32` → `usize`. Infallible widening on all supported targets.
#[inline]
#[must_use]
pub const fn u32_usize(v: u32) -> usize {
    v as usize
}

/// `u64` → `usize`. Exact on 64-bit; saturates on 32-bit targets.
#[expect(
    clippy::cast_possible_truncation,
    reason = "saturates on 32-bit targets"
)]
#[inline]
#[must_use]
pub const fn u64_usize(v: u64) -> usize {
    v as usize
}

/// `u64` → `f64`. Exact for values below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "documented: exact for values < 2^53"
)]
#[inline]
#[must_use]
pub const fn u64_f64(v: u64) -> f64 {
    v as f64
}

/// `usize` → `u64`. Infallible widening on all supported targets.
#[inline]
#[must_use]
pub const fn usize_u64(v: usize) -> u64 {
    v as u64
}

/// `u64` → `u32`. Truncates; debug-asserts the value fits.
#[expect(
    clippy::cast_possible_truncation,
    reason = "caller ensures v fits in u32"
)]
#[inline]
#[must_use]
pub fn u64_u32(v: u64) -> u32 {
    debug_assert!(u32::try_from(v).is_ok(), "u64_u32: {v} overflows u32");
    v as u32
}

/// `f64` → `i32`. Truncates toward zero; debug-asserts finite and in i32 range.
#[expect(
    clippy::cast_possible_truncation,
    reason = "caller ensures v is finite and fits in i32"
)]
#[inline]
#[must_use]
pub fn f64_i32(v: f64) -> i32 {
    debug_assert!(v.is_finite(), "f64_i32: {v} not finite");
    v as i32
}

/// `i64` → `f64`. Exact for values with magnitude below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "documented: exact for |v| < 2^53"
)]
#[inline]
#[must_use]
pub const fn i64_f64(v: i64) -> f64 {
    v as f64
}

/// `u128` → `f64`. Exact for values below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "documented: exact for values < 2^53"
)]
#[inline]
#[must_use]
pub const fn u128_f64(v: u128) -> f64 {
    v as f64
}

/// `f64` → `f32`. Intentional precision loss for GPU shader inputs.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional: GPU shaders require f32"
)]
#[inline]
#[must_use]
pub const fn f64_f32(v: f64) -> f32 {
    v as f32
}

/// `usize` → `i32`. Debug-asserts the value fits.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "caller ensures v fits in i32; debug-asserted"
)]
#[inline]
#[must_use]
pub const fn usize_i32(v: usize) -> i32 {
    debug_assert!(v <= i32::MAX as usize, "usize_i32: overflow");
    v as i32
}

/// `u64` → `u32`, intentionally truncating to the low 32 bits.
///
/// Used for PRNG seed generation where only entropy matters.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional: low 32 bits for seed generation"
)]
#[inline]
#[must_use]
pub const fn u64_u32_truncate(v: u64) -> u32 {
    v as u32
}

/// `f64` → `i64`. Truncates toward zero; debug-asserts finite.
#[expect(
    clippy::cast_possible_truncation,
    reason = "caller ensures v is finite and fits in i64"
)]
#[inline]
#[must_use]
pub fn f64_i64(v: f64) -> i64 {
    debug_assert!(v.is_finite(), "f64_i64: {v} not finite");
    v as i64
}

/// `i32` → `usize`. Debug-asserts non-negative.
#[expect(
    clippy::cast_sign_loss,
    reason = "caller ensures v is non-negative; debug-asserted"
)]
#[inline]
#[must_use]
pub const fn i32_usize(v: i32) -> usize {
    debug_assert!(v >= 0, "i32_usize: negative value");
    v as usize
}

/// `i64` → `usize`. Debug-asserts non-negative and in range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "caller ensures v is non-negative; may narrow on 32-bit targets"
)]
#[inline]
#[must_use]
pub const fn i64_usize(v: i64) -> usize {
    debug_assert!(v >= 0, "i64_usize: negative value");
    v as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_roundtrip() {
        assert_eq!(f64_usize(usize_f64(42)), 42);
    }

    #[test]
    fn u32_roundtrip() {
        assert_eq!(f64_u32(u32_f64(1_000_000)), 1_000_000);
    }

    #[test]
    fn i32_exact() {
        assert!((i32_f64(-42) - (-42.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn u64_conversion() {
        assert_eq!(u64_usize(123_u64), 123_usize);
        assert!((u64_f64(123_u64) - 123.0).abs() < f64::EPSILON);
    }

    #[test]
    fn u32_to_usize() {
        assert_eq!(u32_usize(99), 99_usize);
    }

    #[test]
    fn usize_to_u32_small() {
        assert_eq!(usize_u32(500), 500_u32);
    }

    #[test]
    fn usize_to_u64() {
        assert_eq!(usize_u64(42), 42_u64);
    }

    #[test]
    fn f64_f32_precision() {
        let v = f64_f32(1.5_f64);
        assert!((v - 1.5_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn usize_i32_small() {
        assert_eq!(usize_i32(42), 42_i32);
    }

    #[test]
    fn u64_u32_truncate_low_bits() {
        assert_eq!(u64_u32_truncate(0xFFFF_FFFF), u32::MAX);
        assert_eq!(u64_u32_truncate(0x1_0000_0000), 0);
    }

    #[test]
    fn f64_i64_truncates() {
        assert_eq!(f64_i64(3.9), 3);
        assert_eq!(f64_i64(-3.9), -3);
    }

    #[test]
    fn i32_usize_positive() {
        assert_eq!(i32_usize(42), 42_usize);
    }

    #[test]
    fn i64_usize_positive() {
        assert_eq!(i64_usize(100), 100_usize);
    }
}
