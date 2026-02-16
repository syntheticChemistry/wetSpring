// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for wetSpring I/O and computation.
//!
//! All parser and algorithm errors use [`Error`], with variants for each
//! failure mode. No external error crates — zero-dependency error type.

use std::fmt;
use std::path::PathBuf;

/// Errors produced by wetSpring parsers and algorithms.
#[derive(Debug)]
pub enum Error {
    /// File I/O error with path context.
    Io {
        /// Path that caused the error.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// FASTQ parsing error (sovereign parser).
    Fastq(String),
    /// mzML XML parsing error.
    Xml(String),
    /// Base64 decoding error in binary arrays.
    Base64(String),
    /// Zlib decompression error in binary arrays.
    Zlib(String),
    /// Binary data format error (wrong length, alignment).
    BinaryFormat(String),
    /// MS2 text format parsing error.
    Ms2(String),
    /// GPU compute error (device creation, shader compilation, dispatch).
    Gpu(String),
    /// Invalid input parameters (dimensions, ranges, constraints).
    InvalidInput(String),
}

/// Result type alias for wetSpring operations.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Fastq(msg) => write!(f, "FASTQ parse error: {msg}"),
            Self::Xml(msg) => write!(f, "mzML parse error: {msg}"),
            Self::Base64(msg) => write!(f, "base64 decode error: {msg}"),
            Self::Zlib(msg) => write!(f, "zlib decompress error: {msg}"),
            Self::BinaryFormat(msg) => write!(f, "binary format error: {msg}"),
            Self::Ms2(msg) => write!(f, "MS2 parse error: {msg}"),
            Self::Gpu(msg) => write!(f, "GPU compute error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Fastq(_)
            | Self::Xml(_)
            | Self::Base64(_)
            | Self::Zlib(_)
            | Self::BinaryFormat(_)
            | Self::Ms2(_)
            | Self::Gpu(_)
            | Self::InvalidInput(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_io_error() {
        let err = Error::Io {
            path: PathBuf::from("test_data/sample.fastq"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        assert!(err.to_string().contains("sample.fastq"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn display_parse_errors() {
        let err = Error::Fastq("bad record".into());
        assert!(err.to_string().contains("FASTQ"));
        assert!(err.to_string().contains("bad record"));
    }

    #[test]
    fn display_all_variants() {
        let cases: Vec<(Error, &str)> = vec![
            (Error::Xml("tag mismatch".into()), "mzML parse error"),
            (Error::Base64("invalid char".into()), "base64 decode error"),
            (
                Error::Zlib("inflate failed".into()),
                "zlib decompress error",
            ),
            (
                Error::BinaryFormat("wrong length".into()),
                "binary format error",
            ),
            (Error::Ms2("missing scan".into()), "MS2 parse error"),
            (Error::Gpu("no device".into()), "GPU compute error"),
            (
                Error::InvalidInput("bad dimensions".into()),
                "invalid input",
            ),
        ];
        for (err, expected_prefix) in cases {
            let msg = err.to_string();
            assert!(
                msg.starts_with(expected_prefix),
                "'{msg}' should start with '{expected_prefix}'"
            );
        }
    }

    #[test]
    fn error_source_chain() {
        let io_err = Error::Io {
            path: PathBuf::from("x"),
            source: std::io::Error::other("inner"),
        };
        assert!(std::error::Error::source(&io_err).is_some());

        let parse_err = Error::Xml("bad xml".into());
        assert!(std::error::Error::source(&parse_err).is_none());
    }

    #[test]
    fn source_none_for_all_string_variants() {
        let variants: Vec<Error> = vec![
            Error::Fastq("x".into()),
            Error::Base64("x".into()),
            Error::Zlib("x".into()),
            Error::BinaryFormat("x".into()),
            Error::Ms2("x".into()),
            Error::Gpu("x".into()),
            Error::InvalidInput("x".into()),
        ];
        for err in &variants {
            assert!(std::error::Error::source(err).is_none());
        }
    }
}
