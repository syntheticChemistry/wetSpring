// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nanopore Read Stream (NRS) wire format — binary streaming I/O.
//!
//! Layout:
//!   `magic`:    [u8; 4] = b"NRS1"
//!   `n_reads`:  u64 (little-endian)
//!   Per-read:
//!     `read_id`:            [u8; 16]
//!     `channel`:            u32 (LE)
//!     `sample_rate`:        f64 (LE)
//!     `calibration_offset`: f64 (LE)
//!     `calibration_scale`:  f64 (LE)
//!     `signal_length`:      u64 (LE)
//!     `signal`:             [i16; `signal_length`] (LE)

use super::types::NanoporeRead;
use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::Path;

/// Magic bytes identifying the NRS wire format.
const NRS_MAGIC: &[u8; 4] = b"NRS1";

/// Streaming iterator over nanopore reads from an NRS file.
///
/// Opens a file and yields [`NanoporeRead`] values one at a time,
/// following the same pattern as [`FastqIter`](crate::io::fastq::FastqIter).
pub struct NanoporeIter {
    reader: BufReader<std::fs::File>,
    remaining: u64,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for NanoporeIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NanoporeIter")
            .field("remaining", &self.remaining)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl NanoporeIter {
    /// Open an NRS file and prepare to stream reads.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if the file cannot be opened, or
    /// `Error::Nanopore` if the file header is invalid.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        read_exact(&mut reader, &mut magic, path)?;
        if &magic != NRS_MAGIC {
            return Err(Error::Nanopore(format!(
                "invalid NRS magic: expected NRS1, got {}",
                String::from_utf8_lossy(&magic)
            )));
        }

        let n_reads = read_u64_le(&mut reader, path)?;

        Ok(Self {
            reader,
            remaining: n_reads,
            path: path.to_path_buf(),
        })
    }
}

impl Iterator for NanoporeIter {
    type Item = Result<NanoporeRead>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(read_one_nrs_record(&mut self.reader, &self.path))
    }
}

/// Write a collection of nanopore reads to NRS wire format.
///
/// Used by test generators to create synthetic data files.
///
/// # Errors
///
/// Returns `Error::Io` if writing fails.
pub fn write_nrs(path: &Path, reads: &[NanoporeRead]) -> Result<()> {
    use std::io::Write;

    let file = std::fs::File::create(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    writer.write_all(NRS_MAGIC).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let n = reads.len() as u64;
    writer.write_all(&n.to_le_bytes()).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    for read in reads {
        write_one_nrs_record(&mut writer, read, path)?;
    }

    Ok(())
}

fn read_exact(reader: &mut impl Read, buf: &mut [u8], path: &Path) -> Result<()> {
    reader.read_exact(buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

fn read_u32_le(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_exact(reader, &mut buf, path)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(reader: &mut impl Read, path: &Path) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact(reader, &mut buf, path)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64_le(reader: &mut impl Read, path: &Path) -> Result<f64> {
    let mut buf = [0u8; 8];
    read_exact(reader, &mut buf, path)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_one_nrs_record(reader: &mut impl BufRead, path: &Path) -> Result<NanoporeRead> {
    let mut read_id = [0u8; 16];
    read_exact(reader, &mut read_id, path)?;

    let channel = read_u32_le(reader, path)?;
    let sample_rate = read_f64_le(reader, path)?;
    let calibration_offset = read_f64_le(reader, path)?;
    let calibration_scale = read_f64_le(reader, path)?;
    let signal_length = read_u64_le(reader, path)?;

    if signal_length > 100_000_000 {
        return Err(Error::Nanopore(format!(
            "signal length {signal_length} exceeds 100M samples — likely corrupt"
        )));
    }

    let n = signal_length as usize;
    let mut signal = vec![0i16; n];

    let byte_buf: &mut [u8] = bytemuck::cast_slice_mut(&mut signal);
    read_exact(reader, byte_buf, path)?;

    #[cfg(target_endian = "big")]
    for sample in &mut signal {
        *sample = i16::from_le_bytes(sample.to_ne_bytes());
    }

    Ok(NanoporeRead {
        read_id,
        signal,
        channel,
        sample_rate,
        calibration_offset,
        calibration_scale,
    })
}

fn write_one_nrs_record(
    writer: &mut impl std::io::Write,
    read: &NanoporeRead,
    path: &Path,
) -> Result<()> {
    let w = |writer: &mut dyn std::io::Write, data: &[u8]| -> Result<()> {
        writer.write_all(data).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })
    };

    w(writer, &read.read_id)?;
    w(writer, &read.channel.to_le_bytes())?;
    w(writer, &read.sample_rate.to_le_bytes())?;
    w(writer, &read.calibration_offset.to_le_bytes())?;
    w(writer, &read.calibration_scale.to_le_bytes())?;
    w(writer, &(read.signal.len() as u64).to_le_bytes())?;

    #[cfg(target_endian = "little")]
    {
        let byte_slice: &[u8] = bytemuck::cast_slice(&read.signal);
        w(writer, byte_slice)?;
    }
    #[cfg(target_endian = "big")]
    {
        for &sample in &read.signal {
            w(writer, &sample.to_le_bytes())?;
        }
    }

    Ok(())
}
