// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared helpers for validation binaries (SILVA loading, etc.).

use std::collections::HashMap;
use std::path::Path;

use crate::bio::taxonomy::{Lineage, NaiveBayesClassifier, ReferenceSeq};

/// Load and subsample SILVA 138.1 NR99 reference, train `NaiveBayesClassifier`.
///
/// Subsamples to ~5000 sequences (every 87th) for tractable training.
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

    let tax_content = std::fs::read_to_string(&tax_path).ok()?;
    let mut tax_map: HashMap<String, String> = HashMap::new();
    for line in tax_content.lines().skip(1) {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() == 2 {
            tax_map.insert(parts[0].to_string(), parts[1].trim().to_string());
        }
    }
    println!("  Loaded {} taxonomy entries", tax_map.len());

    let fasta_content = std::fs::read_to_string(&fasta_path).ok()?;
    let mut refs = Vec::new();
    let mut current_id = String::new();
    let mut current_seq: Vec<u8> = Vec::new();
    let mut n_parsed = 0_usize;

    for line in fasta_content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !current_id.is_empty() && !current_seq.is_empty() {
                n_parsed += 1;
                if n_parsed % 87 == 0 {
                    if let Some(tax) = tax_map.get(&current_id) {
                        refs.push(ReferenceSeq {
                            id: current_id.clone(),
                            sequence: current_seq.clone(),
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
    if !current_id.is_empty() && !current_seq.is_empty() {
        n_parsed += 1;
        if n_parsed % 87 == 0 {
            if let Some(tax) = tax_map.get(&current_id) {
                refs.push(ReferenceSeq {
                    id: current_id,
                    sequence: current_seq,
                    lineage: Lineage::from_taxonomy_string(tax),
                });
            }
        }
    }

    println!(
        "  Subsampled {} reference sequences from {} total",
        refs.len(),
        n_parsed
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
