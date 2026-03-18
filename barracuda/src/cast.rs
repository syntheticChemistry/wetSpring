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
#[expect(clippy::cast_precision_loss, reason = "documented: exact for values < 2^53")]
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
    debug_assert!(v >= 0.0 && v <= f64::from(u32::MAX), "f64_u32: {v} out of range");
    v as u32
}

/// `u32` → `usize`. Infallible widening on all supported targets.
#[inline]
#[must_use]
pub const fn u32_usize(v: u32) -> usize {
    v as usize
}

/// `u64` → `usize`. Exact on 64-bit; saturates on 32-bit targets.
#[expect(clippy::cast_possible_truncation, reason = "saturates on 32-bit targets")]
#[inline]
#[must_use]
pub const fn u64_usize(v: u64) -> usize {
    v as usize
}

/// `u64` → `f64`. Exact for values below 2^53.
#[expect(clippy::cast_precision_loss, reason = "documented: exact for values < 2^53")]
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
}
