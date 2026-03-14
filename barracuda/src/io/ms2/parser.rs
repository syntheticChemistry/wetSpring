// SPDX-License-Identifier: AGPL-3.0-or-later
//! MS2 streaming parser.

use super::types::Ms2Spectrum;
use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Streaming iterator that yields one [`Ms2Spectrum`] at a time without
/// buffering the entire file.
///
/// Uses `read_line` into a reusable buffer to avoid per-line allocation
/// (unlike `Lines` which allocates a fresh `String` per line).
pub struct Ms2Iter {
    reader: Box<dyn BufRead>,
    line_buf: String,
    path: std::path::PathBuf,
    pending: Option<Ms2Spectrum>,
    done: bool,
}

impl Ms2Iter {
    /// Open an MS2 file for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(file));
        Ok(Self {
            reader,
            line_buf: String::new(),
            path: path.to_path_buf(),
            pending: None,
            done: false,
        })
    }

    fn parse_i_line(spec: &mut Ms2Spectrum, line: &str) -> std::result::Result<(), Error> {
        let mut parts = line.split_whitespace();
        if let (Some(_), Some(key), Some(val)) = (parts.next(), parts.next(), parts.next()) {
            match key {
                "RTime" => {
                    spec.rt_minutes = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid RTime: {val}")))?;
                }
                "TIC" => {
                    spec.tic = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid TIC: {val}")))?;
                }
                "BPI" => {
                    spec.bpi = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid BPI: {val}")))?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl Iterator for Ms2Iter {
    type Item = Result<Ms2Spectrum>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => {
                    self.done = true;
                    return self.pending.take().map(Ok);
                }
                Ok(_) => {
                    let line = self.line_buf.trim_end_matches(['\n', '\r']);
                    if line.is_empty() {
                        continue;
                    }
                    let first_byte = line.as_bytes()[0];
                    match first_byte {
                        b'H' => {}
                        b'S' => {
                            let emit = self.pending.take();
                            let mut parts = line.split_whitespace();
                            let scan: u32 = match parts.nth(1).and_then(|s| s.parse().ok()) {
                                Some(v) => v,
                                None => {
                                    return Some(Err(Error::Ms2(format!(
                                        "S-line missing scan number: {line}"
                                    ))));
                                }
                            };
                            let pmz: f64 = match parts.nth(1).and_then(|s| s.parse().ok()) {
                                Some(v) => v,
                                None => {
                                    return Some(Err(Error::Ms2(format!(
                                        "S-line missing precursor m/z: {line}"
                                    ))));
                                }
                            };
                            self.pending = Some(Ms2Spectrum {
                                scan,
                                precursor_mz: pmz,
                                rt_minutes: 0.0,
                                tic: 0.0,
                                bpi: 0.0,
                                charge: 0,
                                mz_array: Vec::new(),
                                intensity_array: Vec::new(),
                            });
                            if emit.is_some() {
                                return emit.map(Ok);
                            }
                        }
                        b'I' => {
                            if let Some(ref mut spec) = self.pending {
                                if let Err(e) = Self::parse_i_line(spec, line) {
                                    return Some(Err(e));
                                }
                            }
                        }
                        b'Z' => {
                            if let Some(ref mut spec) = self.pending {
                                if let Some(cs) = line.split_whitespace().nth(1) {
                                    match cs.parse() {
                                        Ok(v) => spec.charge = v,
                                        Err(_) => {
                                            return Some(Err(Error::Ms2(format!(
                                                "Z-line invalid charge: {cs}"
                                            ))));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            if let Some(ref mut spec) = self.pending {
                                let mut parts = line.split_whitespace();
                                if let (Some(mz_str), Some(int_str)) = (parts.next(), parts.next())
                                {
                                    if let (Ok(mz), Ok(intensity)) =
                                        (mz_str.parse::<f64>(), int_str.parse::<f64>())
                                    {
                                        spec.mz_array.push(mz);
                                        spec.intensity_array.push(intensity);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(Error::Io {
                        path: self.path.clone(),
                        source: e,
                    }));
                }
            }
        }
    }
}

/// Process each spectrum without collecting to a `Vec`.
///
/// The callback receives each [`Ms2Spectrum`] as it is parsed. For
/// single-pass aggregation (statistics, filtering), this avoids
/// buffering all spectra in memory.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened,
/// [`Error::Ms2`] if a record is malformed, or propagates the
/// callback's [`Result`].
pub fn for_each_spectrum<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(Ms2Spectrum) -> Result<()>,
{
    for result in Ms2Iter::open(path)? {
        f(result?)?;
    }
    Ok(())
}
