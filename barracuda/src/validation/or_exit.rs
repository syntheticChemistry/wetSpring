// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero-panic error handling for validation binaries.

/// [`Result::unwrap`]/[`Option::expect`] replacement via stderr + exit 1.
pub trait OrExit<T> {
    /// Unwrap or print to stderr and `process::exit(1)`.
    fn or_exit(self, context: &str) -> T;
}

impl<T, E: std::fmt::Display> OrExit<T> for Result<T, E> {
    fn or_exit(self, context: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) => {
                eprintln!("FATAL: {context}: {e}");
                std::process::exit(1)
            }
        }
    }
}

impl<T> OrExit<T> for Option<T> {
    #[expect(
        clippy::option_if_let_else,
        reason = "explicit if-let is clearer for a fatal-exit path"
    )]
    fn or_exit(self, context: &str) -> T {
        if let Some(v) = self {
            v
        } else {
            eprintln!("FATAL: {context}");
            std::process::exit(1)
        }
    }
}
