// SPDX-License-Identifier: AGPL-3.0-or-later
//! I/O parsers for bioinformatics, mass spectrometry, and nanopore data formats.
//!
//! All parsers stream from disk — no full-file buffering.

#[cfg(feature = "json")]
pub mod biom;
pub mod fastq;
pub mod jcamp;
pub mod ms2;
pub mod mzml;
pub mod mzxml;
pub mod nanopore;
pub mod xml;
