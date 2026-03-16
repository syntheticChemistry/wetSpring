// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed errors for the wetSpring Forge crate.

use std::fmt;
use std::path::PathBuf;

/// Errors produced by `NestGate` client operations.
#[derive(Debug, PartialEq, Eq)]
pub enum NestError {
    /// Unix socket path is invalid.
    InvalidSocket(String),
    /// Connection to `NestGate` failed.
    Connect(String),
    /// Timeout configuration failed.
    Timeout(String),
    /// Write to socket failed.
    Write(String),
    /// Flush to socket failed.
    Flush(String),
    /// Read from socket failed.
    Read(String),
    /// `NestGate` returned an empty response.
    EmptyResponse,
    /// File I/O error.
    FileRead {
        /// The path that could not be read.
        path: PathBuf,
        /// Underlying error.
        source: String,
    },
}

impl fmt::Display for NestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSocket(e) => write!(f, "invalid NestGate socket: {e}"),
            Self::Connect(e) => write!(f, "NestGate connect: {e}"),
            Self::Timeout(e) => write!(f, "timeout: {e}"),
            Self::Write(e) => write!(f, "write: {e}"),
            Self::Flush(e) => write!(f, "flush: {e}"),
            Self::Read(e) => write!(f, "read: {e}"),
            Self::EmptyResponse => write!(f, "NestGate returned empty response"),
            Self::FileRead { path, source } => {
                write!(f, "read {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for NestError {}

/// Errors produced by `Songbird` mesh discovery.
#[derive(Debug)]
pub enum SongbirdError {
    /// Unix socket path is invalid.
    InvalidSocket(String),
    /// Connection to `Songbird` failed.
    Connect(String),
    /// Timeout configuration failed.
    Timeout(String),
    /// Write to socket failed.
    Write(String),
    /// Flush to socket failed.
    Flush(String),
    /// Read from socket failed.
    Read(String),
    /// `Songbird` returned an empty response.
    EmptyResponse,
    /// `Songbird` returned an error response.
    ErrorResponse,
}

impl fmt::Display for SongbirdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSocket(e) => write!(f, "invalid Songbird socket: {e}"),
            Self::Connect(e) => write!(f, "Songbird connect: {e}"),
            Self::Timeout(e) => write!(f, "set timeout: {e}"),
            Self::Write(e) => write!(f, "write: {e}"),
            Self::Flush(e) => write!(f, "flush: {e}"),
            Self::Read(e) => write!(f, "read: {e}"),
            Self::EmptyResponse => write!(f, "Songbird returned empty response"),
            Self::ErrorResponse => write!(f, "Songbird returned error"),
        }
    }
}

impl std::error::Error for SongbirdError {}

/// Errors produced by assembly statistics computation.
#[derive(Debug)]
pub enum AssemblyError {
    /// FASTA file contains no sequences.
    NoSequences,
    /// External tool (gzip) failed.
    ToolFailed(String),
    /// Line read error.
    ReadLine(String),
    /// Directory listing failed.
    ReadDir(String),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSequences => write!(f, "no sequences in FASTA"),
            Self::ToolFailed(e) => write!(f, "gzip: {e}"),
            Self::ReadLine(e) => write!(f, "read line: {e}"),
            Self::ReadDir(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

/// Errors produced by NCBI E-utilities operations.
#[derive(Debug)]
pub enum NcbiError {
    /// HTTP request (curl) failed or is unavailable.
    HttpRequest(String),
    /// Response body is not valid UTF-8.
    InvalidUtf8(String),
    /// NCBI assembly not found.
    AssemblyNotFound(String),
    /// `NestGate` cache operation failed.
    CacheFailed(String),
    /// Filesystem error during local assembly lookup.
    FileSystem(String),
}

impl fmt::Display for NcbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HttpRequest(e) => write!(f, "HTTP request failed: {e}"),
            Self::InvalidUtf8(e) => write!(f, "invalid UTF-8: {e}"),
            Self::AssemblyNotFound(acc) => write!(f, "assembly {acc} not found in NCBI"),
            Self::CacheFailed(e) => write!(f, "NestGate cache: {e}"),
            Self::FileSystem(e) => write!(f, "filesystem: {e}"),
        }
    }
}

impl std::error::Error for NcbiError {}

/// Errors produced by `NestGate` data pipeline RPC.
#[derive(Debug)]
pub enum DataError {
    /// Socket path is invalid.
    InvalidSocket(String),
    /// Connection to `NestGate` failed.
    Connect(String),
    /// Timeout configuration failed.
    Timeout(String),
    /// Write to socket failed.
    Write(String),
    /// Flush to socket failed.
    Flush(String),
    /// Read from socket failed.
    Read(String),
    /// `NestGate` returned empty response.
    EmptyResponse,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSocket(e) => write!(f, "invalid NestGate socket: {e}"),
            Self::Connect(e) => write!(f, "NestGate connect: {e}"),
            Self::Timeout(e) => write!(f, "timeout: {e}"),
            Self::Write(e) => write!(f, "write: {e}"),
            Self::Flush(e) => write!(f, "flush: {e}"),
            Self::Read(e) => write!(f, "read: {e}"),
            Self::EmptyResponse => write!(f, "NestGate returned empty response"),
        }
    }
}

impl std::error::Error for DataError {}
