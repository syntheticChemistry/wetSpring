// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for wetSpring I/O and computation.
//!
//! All parser and algorithm errors use [`Error`], with variants for each
//! failure mode. No external error crates — zero-dependency error type.

use std::fmt;
use std::path::PathBuf;

/// Typed IPC error variants for structured error recovery.
///
/// Evolved from opaque `Ipc(String)` following the healthSpring/biomeOS pattern
/// so callers can match on failure category (retry on [`Connect`](Self::Connect),
/// degrade on [`RpcReject`](Self::RpcReject), abort on [`SocketPath`](Self::SocketPath)).
#[derive(Debug)]
pub enum IpcError {
    /// Socket path is invalid or inaccessible (create/remove/bind).
    SocketPath(String),
    /// Cannot connect to a primal socket.
    Connect(String),
    /// I/O error during an established RPC exchange (write, read, flush, timeout, shutdown).
    Transport(String),
    /// JSON serialization or deserialization failure.
    Codec(String),
    /// Remote primal returned a JSON-RPC error response.
    RpcReject {
        /// JSON-RPC error code.
        code: i64,
        /// Human-readable error message from the remote primal.
        message: String,
    },
    /// Remote primal returned an empty or missing response.
    EmptyResponse,
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SocketPath(msg) => write!(f, "socket: {msg}"),
            Self::Connect(msg) => write!(f, "connect: {msg}"),
            Self::Transport(msg) => write!(f, "transport: {msg}"),
            Self::Codec(msg) => write!(f, "codec: {msg}"),
            Self::RpcReject { code, message } => write!(f, "RPC reject [{code}]: {message}"),
            Self::EmptyResponse => write!(f, "empty response"),
        }
    }
}

impl std::error::Error for IpcError {}

impl IpcError {
    /// Whether this error is likely transient and the operation can be retried.
    ///
    /// This is the coarse transport-layer view: [`Connect`](Self::Connect),
    /// [`Transport`](Self::Transport), and [`EmptyResponse`](Self::EmptyResponse) only.
    /// For JSON-RPC–aware classification (including when to retry
    /// [`RpcReject`](Self::RpcReject)), see [`is_recoverable`](Self::is_recoverable).
    #[must_use]
    pub const fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::Connect(_) | Self::Transport(_) | Self::EmptyResponse
        )
    }

    /// Whether the failure may succeed on retry (transient) vs will not (permanent).
    ///
    /// Returns `true` for connection/transport/empty-response failures and for
    /// JSON-RPC responses that are typically transient: internal error (`-32603`),
    /// plus non-standard codes whose message suggests timeout, connection loss,
    /// or temporary unavailability.
    ///
    /// Returns `false` for local configuration/codec errors and for JSON-RPC
    /// client/request errors that will not change on retry: invalid request
    /// (`-32600`), method not found (`-32601`), invalid params (`-32602`),
    /// parse error (`-32700`).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Connect(_) | Self::Transport(_) | Self::EmptyResponse => true,
            Self::SocketPath(_) | Self::Codec(_) => false,
            Self::RpcReject { code, message } => rpc_reject_is_recoverable(*code, message),
        }
    }

    /// Whether this error is likely caused by a timeout or slow peer.
    ///
    /// Heuristic: checks for common timeout-related substrings in transport
    /// and connect messages. Useful for circuit-breaker / backoff logic.
    #[must_use]
    pub fn is_timeout_likely(&self) -> bool {
        match self {
            Self::Transport(msg) | Self::Connect(msg) => {
                let lower = msg.to_ascii_lowercase();
                lower.contains("timeout")
                    || lower.contains("timed out")
                    || lower.contains("deadline")
                    || lower.contains("wouldblock")
            }
            Self::EmptyResponse => true,
            Self::SocketPath(_) | Self::Codec(_) | Self::RpcReject { .. } => false,
        }
    }

    /// Whether the remote primal rejected the call as "method not found" (`-32601`).
    #[must_use]
    pub const fn is_method_not_found(&self) -> bool {
        matches!(self, Self::RpcReject { code: -32601, .. })
    }

    /// Whether this is a connection-level failure (socket unreachable).
    #[must_use]
    pub const fn is_connection_error(&self) -> bool {
        matches!(self, Self::Connect(_) | Self::SocketPath(_))
    }
}

/// JSON-RPC 2.0 recoverability for [`IpcError::RpcReject`].
fn rpc_reject_is_recoverable(code: i64, message: &str) -> bool {
    match code {
        -32600 | -32601 | -32602 | -32700 => false,
        -32603 => true,
        _ => rpc_message_suggests_transient(message),
    }
}

fn rpc_message_suggests_transient(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("connection reset")
        || lower.contains("reset by peer")
        || lower.contains("econnreset")
        || lower.contains("broken pipe")
        || lower.contains("unavailable")
        || lower.contains("temporarily unavailable")
        || lower.contains("try again")
        || lower.contains("overloaded")
        || lower.contains("deadline exceeded")
}

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
    /// NCBI Entrez / SRA / `NestGate` error (network, parsing, protocol).
    Ncbi(String),
    /// NPU compute error (discovery, model load, inference, DMA).
    Npu(String),
    /// Nanopore raw signal parsing error (POD5/FAST5).
    Nanopore(String),
    /// Structured IPC server/client protocol error.
    Ipc(IpcError),
    /// JCAMP-DX spectroscopy format parsing error.
    Jcamp(String),
}

impl From<IpcError> for Error {
    fn from(e: IpcError) -> Self {
        Self::Ipc(e)
    }
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
            Self::Ncbi(msg) => write!(f, "NCBI error: {msg}"),
            Self::Npu(msg) => write!(f, "NPU error: {msg}"),
            Self::Nanopore(msg) => write!(f, "nanopore parse error: {msg}"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::Jcamp(msg) => write!(f, "JCAMP-DX parse error: {msg}"),
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
            | Self::InvalidInput(_)
            | Self::Ncbi(_)
            | Self::Npu(_)
            | Self::Nanopore(_)
            | Self::Jcamp(_) => None,
            Self::Ipc(e) => Some(e),
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
            (Error::Ncbi("connection refused".into()), "NCBI error"),
            (Error::Npu("no device".into()), "NPU error"),
            (
                Error::Nanopore("truncated signal".into()),
                "nanopore parse error",
            ),
            (
                Error::Ipc(IpcError::Transport("socket refused".into())),
                "IPC error",
            ),
            (Error::Jcamp("missing TITLE".into()), "JCAMP-DX parse error"),
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

        let ipc_err = Error::Ipc(IpcError::Connect("refused".into()));
        assert!(std::error::Error::source(&ipc_err).is_some());
    }

    #[test]
    fn ipc_error_display_all_variants() {
        assert!(
            IpcError::SocketPath("bad".into())
                .to_string()
                .contains("socket")
        );
        assert!(
            IpcError::Connect("refused".into())
                .to_string()
                .contains("connect")
        );
        assert!(
            IpcError::Transport("broken pipe".into())
                .to_string()
                .contains("transport")
        );
        assert!(
            IpcError::Codec("invalid json".into())
                .to_string()
                .contains("codec")
        );
        assert!(
            IpcError::RpcReject {
                code: -32601,
                message: "not found".into()
            }
            .to_string()
            .contains("-32601")
        );
        assert!(IpcError::EmptyResponse.to_string().contains("empty"));
    }

    #[test]
    fn ipc_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(IpcError::Connect("refused".into()));
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn ipc_error_from_conversion() {
        let ipc: IpcError = IpcError::Transport("write failed".into());
        let err: Error = ipc.into();
        assert!(err.to_string().contains("IPC error"));
        assert!(err.to_string().contains("transport"));
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
            Error::Ncbi("x".into()),
            Error::Npu("x".into()),
            Error::Nanopore("x".into()),
            Error::Jcamp("x".into()),
        ];
        for err in &variants {
            assert!(std::error::Error::source(err).is_none());
        }
    }

    #[test]
    fn ipc_variant_has_source() {
        let err = Error::Ipc(IpcError::EmptyResponse);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn ipc_retriable_variants() {
        assert!(IpcError::Connect("refused".into()).is_retriable());
        assert!(IpcError::Transport("broken pipe".into()).is_retriable());
        assert!(IpcError::EmptyResponse.is_retriable());

        assert!(!IpcError::SocketPath("bad".into()).is_retriable());
        assert!(!IpcError::Codec("invalid".into()).is_retriable());
        assert!(
            !IpcError::RpcReject {
                code: -32601,
                message: "not found".into()
            }
            .is_retriable()
        );
    }

    #[test]
    fn ipc_timeout_detection() {
        assert!(IpcError::Transport("read timeout exceeded".into()).is_timeout_likely());
        assert!(IpcError::Connect("connection timed out".into()).is_timeout_likely());
        assert!(IpcError::Transport("WouldBlock on read".into()).is_timeout_likely());
        assert!(IpcError::EmptyResponse.is_timeout_likely());

        assert!(!IpcError::Transport("broken pipe".into()).is_timeout_likely());
        assert!(!IpcError::Connect("refused".into()).is_timeout_likely());
        assert!(!IpcError::Codec("invalid json".into()).is_timeout_likely());
    }

    #[test]
    fn ipc_method_not_found() {
        assert!(
            IpcError::RpcReject {
                code: -32601,
                message: "method not found".into()
            }
            .is_method_not_found()
        );

        assert!(
            !IpcError::RpcReject {
                code: -32602,
                message: "invalid params".into()
            }
            .is_method_not_found()
        );

        assert!(!IpcError::Connect("refused".into()).is_method_not_found());
    }

    #[test]
    fn ipc_connection_error() {
        assert!(IpcError::Connect("refused".into()).is_connection_error());
        assert!(IpcError::SocketPath("bad path".into()).is_connection_error());
        assert!(!IpcError::Transport("broken pipe".into()).is_connection_error());
        assert!(!IpcError::EmptyResponse.is_connection_error());
    }

    #[test]
    fn ipc_recoverable_transport_and_empty() {
        assert!(IpcError::Connect("refused".into()).is_recoverable());
        assert!(IpcError::Transport("reset".into()).is_recoverable());
        assert!(IpcError::EmptyResponse.is_recoverable());
    }

    #[test]
    fn ipc_not_recoverable_socket_codec() {
        assert!(!IpcError::SocketPath("/bad".into()).is_recoverable());
        assert!(!IpcError::Codec("oops".into()).is_recoverable());
    }

    #[test]
    fn ipc_recoverable_rpc_standard_codes() {
        assert!(!IpcError::RpcReject {
            code: -32600,
            message: "bad".into()
        }
        .is_recoverable());
        assert!(!IpcError::RpcReject {
            code: -32601,
            message: "nope".into()
        }
        .is_recoverable());
        assert!(!IpcError::RpcReject {
            code: -32602,
            message: "bad params".into()
        }
        .is_recoverable());
        assert!(IpcError::RpcReject {
            code: -32603,
            message: "internal".into()
        }
        .is_recoverable());
        assert!(!IpcError::RpcReject {
            code: -32700,
            message: "parse".into()
        }
        .is_recoverable());
    }

    #[test]
    fn ipc_recoverable_rpc_custom_code_uses_message_heuristic() {
        assert!(IpcError::RpcReject {
            code: -32000,
            message: "upstream timeout".into()
        }
        .is_recoverable());
        assert!(IpcError::RpcReject {
            code: -32001,
            message: "Service Unavailable: overloaded".into()
        }
        .is_recoverable());
        assert!(!IpcError::RpcReject {
            code: -32099,
            message: "business rule violation".into()
        }
        .is_recoverable());
    }
}
