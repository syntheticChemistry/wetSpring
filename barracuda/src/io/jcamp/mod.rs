// SPDX-License-Identifier: AGPL-3.0-or-later
//! JCAMP-DX streaming parser for spectroscopy data.
//!
//! Streams from disk via `BufReader` — the file is never loaded into memory.
//!
//! Supports JCAMP-DX versions 4.24 and 5.x with `XYDATA`, `PEAK TABLE`,
//! and simple XY pair formats common in IR, UV-Vis, Raman, and NMR
//! spectroscopy.
//!
//! Reference: IUPAC Recommendations 1988 (Pure Appl. Chem. 60, 1365–1376)
//! and the 1993 update for NMR (Applied Spectroscopy 47, 1093–1099).

use crate::error::{Error, Result};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single JCAMP-DX data block (spectrum or compound data).
#[derive(Debug, Clone)]
pub struct JcampBlock {
    /// Title of the data block (from `##TITLE=`).
    pub title: String,
    /// Data type string (e.g. "INFRARED SPECTRUM").
    pub data_type: String,
    /// X-axis units (e.g. "1/CM", "MICROMETERS").
    pub x_units: String,
    /// Y-axis units (e.g. "ABSORBANCE", "TRANSMITTANCE").
    pub y_units: String,
    /// X data values.
    pub x: Vec<f64>,
    /// Y data values.
    pub y: Vec<f64>,
    /// All labeled data records (LDR) as key-value pairs.
    pub metadata: BTreeMap<String, String>,
}

/// Streaming iterator over JCAMP-DX blocks in a file.
///
/// Handles compound files (multiple `##TITLE=`...`##END=` blocks) by
/// yielding one [`JcampBlock`] per logical data block.
///
/// Uses a single reusable `String` buffer for line I/O — no per-line
/// allocation (unlike `BufRead::lines()`).
pub struct JcampIter {
    reader: BufReader<std::fs::File>,
    line_buf: String,
    pending_line: Option<String>,
    done: bool,
}

impl JcampIter {
    /// Open a JCAMP-DX file for streaming.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(Self {
            reader: BufReader::new(file),
            line_buf: String::new(),
            pending_line: None,
            done: false,
        })
    }

    fn read_line(&mut self) -> std::result::Result<Option<&str>, std::io::Error> {
        self.line_buf.clear();
        match self.reader.read_line(&mut self.line_buf)? {
            0 => Ok(None),
            _ => Ok(Some(&self.line_buf)),
        }
    }
}

impl Iterator for JcampIter {
    type Item = Result<JcampBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut metadata = BTreeMap::new();
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut in_data = false;
        let mut first_x: Option<f64> = None;
        let mut delta_x: Option<f64> = None;
        let mut found_title = false;

        loop {
            let line: String = if let Some(pending) = self.pending_line.take() {
                pending
            } else {
                match self.read_line() {
                    Ok(Some(l)) => l.to_owned(),
                    Ok(None) => {
                        self.done = true;
                        return if found_title {
                            Some(Ok(build_block(&metadata, x_vals, y_vals)))
                        } else {
                            None
                        };
                    }
                    Err(e) => {
                        self.done = true;
                        return Some(Err(Error::Jcamp(format!("I/O: {e}"))));
                    }
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(ldr) = trimmed.strip_prefix("##") {
                if in_data {
                    in_data = false;
                }

                if let Some((key, value)) = split_ldr(ldr) {
                    let key_upper = key.to_ascii_uppercase();

                    if key_upper == "END" {
                        return Some(Ok(build_block(&metadata, x_vals, y_vals)));
                    }

                    if key_upper == "TITLE" {
                        if found_title && !x_vals.is_empty() {
                            self.pending_line = Some(line);
                            return Some(Ok(build_block(&metadata, x_vals, y_vals)));
                        }
                        found_title = true;
                    }

                    if key_upper == "FIRSTX" {
                        first_x = value.trim().parse().ok();
                    }
                    if key_upper == "DELTAX" {
                        delta_x = value.trim().parse().ok();
                    }

                    if key_upper == "XYDATA" || key_upper == "PEAK TABLE" {
                        in_data = true;
                    }

                    metadata.insert(key_upper, value.to_string());
                }
            } else if in_data {
                parse_data_line(trimmed, &mut x_vals, &mut y_vals, first_x, delta_x);
            }
        }
    }
}

/// Parse all blocks from a JCAMP-DX file into memory.
///
/// # Errors
///
/// Returns [`Error::Io`] on file open failure or [`Error::Jcamp`] on parse errors.
pub fn parse_jcamp(path: &Path) -> Result<Vec<JcampBlock>> {
    JcampIter::open(path)?.collect()
}

/// Stream through a JCAMP-DX file, calling `f` for each data block.
///
/// # Errors
///
/// Returns the first error encountered during parsing.
pub fn for_each_block<F: FnMut(&JcampBlock)>(path: &Path, mut f: F) -> Result<()> {
    for block in JcampIter::open(path)? {
        f(&block?);
    }
    Ok(())
}

fn split_ldr(ldr: &str) -> Option<(&str, &str)> {
    let eq = ldr.find('=')?;
    let key = ldr[..eq].trim();
    let value = ldr[eq + 1..].trim();
    Some((key, value))
}

fn build_block(metadata: &BTreeMap<String, String>, x: Vec<f64>, y: Vec<f64>) -> JcampBlock {
    JcampBlock {
        title: metadata.get("TITLE").cloned().unwrap_or_default(),
        data_type: metadata.get("DATA TYPE").cloned().unwrap_or_default(),
        x_units: metadata.get("XUNITS").cloned().unwrap_or_default(),
        y_units: metadata.get("YUNITS").cloned().unwrap_or_default(),
        x,
        y,
        metadata: metadata.clone(),
    }
}

fn parse_data_line(
    line: &str,
    x_vals: &mut Vec<f64>,
    y_vals: &mut Vec<f64>,
    first_x: Option<f64>,
    delta_x: Option<f64>,
) {
    let tokens: Vec<&str> = line
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return;
    }

    // XYDATA (X++(Y..Y)) format: first token is X, rest are Y values
    // PEAK TABLE (XY..XY) format: alternating X, Y pairs
    // Detect: if we have FIRSTX and DELTAX metadata, treat as packed Y
    if let (Some(fx), Some(dx)) = (first_x, delta_x) {
        if tokens[0].parse::<f64>().is_ok() {
            for tok in &tokens[1..] {
                if let Ok(yv) = parse_jcamp_value(tok) {
                    #[expect(clippy::cast_precision_loss)]
                    let xi = dx.mul_add(x_vals.len() as f64, fx);
                    x_vals.push(xi);
                    y_vals.push(yv);
                }
            }
        }
    } else if tokens.len() >= 2 && tokens.len().is_multiple_of(2) {
        // XY pairs
        let mut i = 0;
        while i + 1 < tokens.len() {
            if let (Ok(xv), Ok(yv)) = (tokens[i].parse::<f64>(), tokens[i + 1].parse::<f64>()) {
                x_vals.push(xv);
                y_vals.push(yv);
            }
            i += 2;
        }
    } else {
        // X++(Y..Y) without DELTAX: first token is X, rest are Y at implicit X
        if let Ok(start_x) = tokens[0].parse::<f64>() {
            if tokens.len() == 2 {
                if let Ok(yv) = tokens[1].parse::<f64>() {
                    x_vals.push(start_x);
                    y_vals.push(yv);
                }
            } else {
                for tok in &tokens[1..] {
                    if let Ok(yv) = parse_jcamp_value(tok) {
                        x_vals.push(start_x);
                        y_vals.push(yv);
                    }
                }
            }
        }
    }
}

/// Parse a JCAMP value token, handling SQZ (squeezed) digit encoding.
///
/// SQZ uses letters to encode the first digit with sign:
/// `@`=0, `A`-`I`=+1..+9, `a`-`i`=-1..-9, `J`-`R`=+0..+8 (DIF)
fn parse_jcamp_value(token: &str) -> std::result::Result<f64, ()> {
    if token.is_empty() {
        return Err(());
    }
    let first = token.as_bytes()[0];

    // Plain numeric
    if first.is_ascii_digit() || first == b'+' || first == b'-' || first == b'.' {
        return token.parse::<f64>().map_err(|_| ());
    }

    // SQZ encoding
    let (sign, digit) = match first {
        b'@' => (1.0, 0),
        b'A'..=b'I' => (1.0, u32::from(first - b'A' + 1)),
        b'a'..=b'i' => (-1.0, u32::from(first - b'a' + 1)),
        _ => return Err(()),
    };

    let rest: String = token[1..]
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let mantissa = if rest.is_empty() {
        f64::from(digit)
    } else {
        format!("{digit}{rest}").parse::<f64>().map_err(|_| ())?
    };

    Ok(sign * mantissa)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn write_ir_jcamp(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("test_ir.jcamp");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "##TITLE= Test IR Spectrum").unwrap();
        writeln!(f, "##JCAMP-DX= 5.00").unwrap();
        writeln!(f, "##DATA TYPE= INFRARED SPECTRUM").unwrap();
        writeln!(f, "##XUNITS= 1/CM").unwrap();
        writeln!(f, "##YUNITS= ABSORBANCE").unwrap();
        writeln!(f, "##FIRSTX= 400").unwrap();
        writeln!(f, "##LASTX= 402").unwrap();
        writeln!(f, "##DELTAX= 1").unwrap();
        writeln!(f, "##NPOINTS= 3").unwrap();
        writeln!(f, "##XYDATA= (X++(Y..Y))").unwrap();
        writeln!(f, "400.0  0.123  0.456  0.789").unwrap();
        writeln!(f, "##END=").unwrap();
        path
    }

    #[test]
    fn parse_ir_spectrum() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_ir_jcamp(dir.path());
        let blocks = parse_jcamp(&path).unwrap();

        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert_eq!(b.title, "Test IR Spectrum");
        assert_eq!(b.data_type, "INFRARED SPECTRUM");
        assert_eq!(b.x_units, "1/CM");
        assert_eq!(b.y_units, "ABSORBANCE");
        assert_eq!(b.x.len(), 3);
        assert_eq!(b.y.len(), 3);
        assert!((b.x[0] - 400.0).abs() < f64::EPSILON);
        assert!((b.x[1] - 401.0).abs() < f64::EPSILON);
        assert!((b.x[2] - 402.0).abs() < f64::EPSILON);
        assert!((b.y[0] - 0.123).abs() < 1e-9);
        assert!((b.y[1] - 0.456).abs() < 1e-9);
        assert!((b.y[2] - 0.789).abs() < 1e-9);
    }

    #[test]
    fn parse_peak_table_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("peaks.jcamp");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "##TITLE= Peak List").unwrap();
        writeln!(f, "##DATA TYPE= MASS SPECTRUM").unwrap();
        writeln!(f, "##XUNITS= M/Z").unwrap();
        writeln!(f, "##YUNITS= RELATIVE INTENSITY").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "100.0, 45.0").unwrap();
        writeln!(f, "200.0, 100.0").unwrap();
        writeln!(f, "300.0, 23.0").unwrap();
        writeln!(f, "##END=").unwrap();

        let blocks = parse_jcamp(&path).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].x.len(), 3);
        assert!((blocks[0].x[0] - 100.0).abs() < f64::EPSILON);
        assert!((blocks[0].y[1] - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_compound_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compound.jcamp");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "##TITLE= Block 1").unwrap();
        writeln!(f, "##DATA TYPE= IR").unwrap();
        writeln!(f, "##XUNITS= 1/CM").unwrap();
        writeln!(f, "##YUNITS= T").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "1000.0, 0.5").unwrap();
        writeln!(f, "##END=").unwrap();
        writeln!(f, "##TITLE= Block 2").unwrap();
        writeln!(f, "##DATA TYPE= RAMAN").unwrap();
        writeln!(f, "##XUNITS= 1/CM").unwrap();
        writeln!(f, "##YUNITS= COUNTS").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "500.0, 1000.0").unwrap();
        writeln!(f, "600.0, 2000.0").unwrap();
        writeln!(f, "##END=").unwrap();

        let blocks = parse_jcamp(&path).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].title, "Block 1");
        assert_eq!(blocks[0].x.len(), 1);
        assert_eq!(blocks[1].title, "Block 2");
        assert_eq!(blocks[1].x.len(), 2);
    }

    #[test]
    fn iter_matches_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_ir_jcamp(dir.path());

        let all = parse_jcamp(&path).unwrap();
        let streamed: Vec<JcampBlock> = JcampIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(all.len(), streamed.len());
        assert_eq!(all[0].x.len(), streamed[0].x.len());
        assert_eq!(all[0].title, streamed[0].title);
    }

    #[test]
    fn empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jcamp");
        File::create(&path).unwrap();

        let blocks = parse_jcamp(&path).unwrap();
        assert!(blocks.is_empty());
    }

    #[test]
    fn nonexistent_file() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_jcamp_8f2a.jcamp");
        let result = parse_jcamp(&path);
        assert!(result.is_err());
    }

    #[test]
    fn sqz_encoding() {
        assert!((parse_jcamp_value("@").unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((parse_jcamp_value("A").unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((parse_jcamp_value("I").unwrap() - 9.0).abs() < f64::EPSILON);
        assert!((parse_jcamp_value("a").unwrap() - (-1.0)).abs() < f64::EPSILON);
        assert!((parse_jcamp_value("B23").unwrap() - 223.0).abs() < f64::EPSILON);
        assert!((parse_jcamp_value("b45").unwrap() - (-245.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.jcamp");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "##TITLE= Caffeine NMR").unwrap();
        writeln!(f, "##JCAMP-DX= 5.01").unwrap();
        writeln!(f, "##DATA TYPE= NMR SPECTRUM").unwrap();
        writeln!(f, "##XUNITS= HZ").unwrap();
        writeln!(f, "##YUNITS= ARBITRARY UNITS").unwrap();
        writeln!(f, "##.OBSERVE FREQUENCY= 400.13").unwrap();
        writeln!(f, "##.OBSERVE NUCLEUS= ^1H").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "1600.5, 1.0").unwrap();
        writeln!(f, "##END=").unwrap();

        let blocks = parse_jcamp(&path).unwrap();
        assert_eq!(blocks[0].metadata.get("JCAMP-DX").unwrap(), "5.01");
        assert_eq!(
            blocks[0].metadata.get(".OBSERVE FREQUENCY").unwrap(),
            "400.13"
        );
        assert_eq!(blocks[0].metadata.get(".OBSERVE NUCLEUS").unwrap(), "^1H");
    }

    #[test]
    fn for_each_block_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_ir_jcamp(dir.path());
        let mut count = 0;
        for_each_block(&path, |_| count += 1).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn xy_without_deltax() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("simple_xy.jcamp");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "##TITLE= Simple").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "10.0 50.0").unwrap();
        writeln!(f, "20.0 60.0").unwrap();
        writeln!(f, "30.0 70.0").unwrap();
        writeln!(f, "##END=").unwrap();

        let blocks = parse_jcamp(&path).unwrap();
        assert_eq!(blocks[0].x.len(), 3);
        assert!((blocks[0].x[0] - 10.0).abs() < f64::EPSILON);
        assert!((blocks[0].x[2] - 30.0).abs() < f64::EPSILON);
        assert!((blocks[0].y[2] - 70.0).abs() < f64::EPSILON);
    }
}
