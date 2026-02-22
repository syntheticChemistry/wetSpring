// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared helpers for validation binaries (SILVA loading, etc.).
//!
//! All file I/O uses streaming — no `read_to_string` on potentially
//! large reference databases.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::bio::taxonomy::{Lineage, NaiveBayesClassifier, ReferenceSeq};

/// Load and subsample SILVA 138.1 NR99 reference, train `NaiveBayesClassifier`.
///
/// Subsamples to ~5000 sequences (every 87th) for tractable training.
/// Streams both the taxonomy TSV and FASTA files — never loads either
/// fully into memory.
///
/// Returns `None` if reference files are missing or empty.
#[must_use]
pub fn load_silva_classifier(ref_dir: &Path) -> Option<NaiveBayesClassifier> {
    let fasta_path = ref_dir.join("silva_138_99_seqs.fasta");
    let tax_path = ref_dir.join("silva_138_99_taxonomy.tsv");

    if !fasta_path.exists() || !tax_path.exists() {
        println!(
            "  [INFO] SILVA reference not found at {} — skipping taxonomy",
            ref_dir.display()
        );
        return None;
    }

    println!("  Loading SILVA 138.1 NR99 reference database...");

    let tax_map = stream_taxonomy_tsv(&tax_path)?;
    println!("  Loaded {} taxonomy entries", tax_map.len());

    let refs = stream_fasta_subsampled(&fasta_path, &tax_map, 87)?;
    let n_total = refs.len().saturating_mul(87);

    println!(
        "  Subsampled {} reference sequences from ~{} total",
        refs.len(),
        n_total
    );

    if refs.is_empty() {
        println!("  [WARN] No reference sequences loaded — skipping taxonomy");
        return None;
    }

    println!("  Training NaiveBayes classifier (k=8)...");
    let classifier = NaiveBayesClassifier::train(&refs, 8);
    println!("  Classifier ready: {} taxa", classifier.n_taxa());
    Some(classifier)
}

/// Stream-parse a SILVA taxonomy TSV (id → lineage string) without
/// loading the entire file into memory.
fn stream_taxonomy_tsv(path: &Path) -> Option<HashMap<String, String>> {
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.ok()?;
        if i == 0 {
            continue; // header
        }
        if let Some((id, tax)) = line.split_once('\t') {
            map.insert(id.to_string(), tax.trim().to_string());
        }
    }
    Some(map)
}

/// Stream-parse a FASTA file, keeping every `nth` sequence that has a
/// taxonomy entry. Each sequence is built incrementally from continuation
/// lines without buffering the whole file.
fn stream_fasta_subsampled(
    path: &Path,
    tax_map: &HashMap<String, String>,
    nth: usize,
) -> Option<Vec<ReferenceSeq>> {
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::new(file);

    let mut refs = Vec::new();
    let mut current_id = String::new();
    let mut current_seq: Vec<u8> = Vec::new();
    let mut n_parsed = 0_usize;

    for line_result in reader.lines() {
        let line = line_result.ok()?;
        if let Some(header) = line.strip_prefix('>') {
            if !current_id.is_empty() && !current_seq.is_empty() {
                n_parsed += 1;
                if n_parsed % nth == 0 {
                    if let Some(tax) = tax_map.get(&current_id) {
                        refs.push(ReferenceSeq {
                            id: std::mem::take(&mut current_id),
                            sequence: std::mem::take(&mut current_seq),
                            lineage: Lineage::from_taxonomy_string(tax),
                        });
                    }
                }
            }
            current_id = header.split_whitespace().next().unwrap_or("").to_string();
            current_seq.clear();
        } else {
            current_seq.extend(
                line.trim()
                    .bytes()
                    .filter(u8::is_ascii_alphabetic)
                    .map(|b| b.to_ascii_uppercase()),
            );
        }
    }

    // Handle last sequence
    if !current_id.is_empty() && !current_seq.is_empty() {
        n_parsed += 1;
        if n_parsed % nth == 0 {
            if let Some(tax) = tax_map.get(&current_id) {
                refs.push(ReferenceSeq {
                    id: current_id,
                    sequence: current_seq,
                    lineage: Lineage::from_taxonomy_string(tax),
                });
            }
        }
    }

    Some(refs)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_test_silva(dir: &Path) {
        let tax = dir.join("silva_138_99_taxonomy.tsv");
        let fasta = dir.join("silva_138_99_seqs.fasta");

        let mut tf = std::fs::File::create(&tax).unwrap();
        writeln!(tf, "id\ttaxonomy").unwrap();
        for i in 0..200 {
            writeln!(tf, "seq{i}\tBac;Firm;Bac;Lac;Lac;Sp{i}").unwrap();
        }

        let mut ff = std::fs::File::create(&fasta).unwrap();
        for i in 0..200 {
            writeln!(ff, ">seq{i} description").unwrap();
            writeln!(ff, "ACGTACGTACGTACGTACGT").unwrap();
        }
    }

    #[test]
    fn load_silva_classifier_subsamples() {
        let dir = TempDir::new().unwrap();
        write_test_silva(dir.path());
        let classifier = load_silva_classifier(dir.path());
        assert!(classifier.is_some());
    }

    #[test]
    fn load_silva_classifier_missing_files_returns_none() {
        let dir = TempDir::new().unwrap();
        assert!(load_silva_classifier(dir.path()).is_none());
    }

    #[test]
    fn stream_taxonomy_tsv_parses_entries() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tax.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id\ttaxonomy").unwrap();
        writeln!(f, "A\tBac;Prot").unwrap();
        writeln!(f, "B\tArch;Eury").unwrap();

        let map = stream_taxonomy_tsv(&path).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["A"], "Bac;Prot");
        assert_eq!(map["B"], "Arch;Eury");
    }

    #[test]
    fn stream_fasta_subsampled_respects_nth() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("seqs.fasta");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 0..10 {
            writeln!(f, ">s{i}").unwrap();
            writeln!(f, "ACGT").unwrap();
        }

        let mut tax = HashMap::new();
        for i in 0..10 {
            tax.insert(format!("s{i}"), format!("Tax{i}"));
        }

        let refs = stream_fasta_subsampled(&path, &tax, 3).unwrap();
        // Sequences 3, 6, 9 match (n_parsed % 3 == 0)
        assert_eq!(refs.len(), 3);
    }
}
