// SPDX-License-Identifier: AGPL-3.0-or-later
//! I/O parsers for bioinformatics and mass spectrometry data formats.
//!
//! All parsers stream from disk — no full-file buffering.

pub mod fastq;
pub mod ms2;
pub mod mzml;
pub(crate) mod xml;
