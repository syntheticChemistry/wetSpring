// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use std::io::Write;

fn write_temp_fasta(contents: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::with_suffix(".fasta").unwrap();
    f.write_all(contents).unwrap();
    f.flush().unwrap();
    f
}

#[test]
fn single_record() {
    let f = write_temp_fasta(b">seq1 test sequence\nACGTACGT\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "seq1");
    assert_eq!(records[0].description, "seq1 test sequence");
    assert_eq!(records[0].sequence, b"ACGTACGT");
    assert_eq!(records[0].len(), 8);
    assert!(!records[0].is_empty());
}

#[test]
fn multi_line_sequence() {
    let f = write_temp_fasta(b">genome\nACGT\nTGCA\nAAAA\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].sequence, b"ACGTTGCAAAAA");
    assert_eq!(records[0].len(), 12);
}

#[test]
fn multiple_records() {
    let f = write_temp_fasta(b">seq1\nACGT\n>seq2\nTGCA\n>seq3\nGGCC\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].id, "seq1");
    assert_eq!(records[1].id, "seq2");
    assert_eq!(records[2].id, "seq3");
    assert_eq!(records[0].sequence, b"ACGT");
    assert_eq!(records[1].sequence, b"TGCA");
    assert_eq!(records[2].sequence, b"GGCC");
}

#[test]
fn gc_content() {
    let f = write_temp_fasta(b">test\nGCGC\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert!((records[0].gc_content() - 1.0).abs() < 1e-10);

    let f2 = write_temp_fasta(b">test\nATAT\n");
    let records2 = FastaRecord::load_all(f2.path()).unwrap();
    assert!(records2[0].gc_content().abs() < 1e-10);
}

#[test]
fn empty_sequence_gc() {
    let f = write_temp_fasta(b">empty\n\n>next\nA\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    let empty_rec = records.iter().find(|r| r.id == "empty").unwrap();
    assert!(empty_rec.is_empty());
    assert!(empty_rec.gc_content().abs() < 1e-10);
}

#[test]
fn case_normalization() {
    let f = write_temp_fasta(b">test\nacgtACGT\n");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert_eq!(records[0].sequence, b"ACGTACGT");
}

#[test]
fn streaming_iterator() {
    let f = write_temp_fasta(b">a\nACGT\n>b\nTGCA\n");
    let iter = FastaIter::open(f.path()).unwrap();
    let records: Vec<_> = iter.map(|r| r.unwrap()).collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "a");
    assert_eq!(records[1].id, "b");
}

#[test]
fn no_trailing_newline() {
    let f = write_temp_fasta(b">seq\nACGT");
    let records = FastaRecord::load_all(f.path()).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].sequence, b"ACGT");
}

#[test]
fn bad_header_errors() {
    let f = write_temp_fasta(b"not_a_header\nACGT\n");
    let result = FastaRecord::load_all(f.path());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("FASTA"));
}

#[test]
fn genbank_minimal() {
    let gbk = r#"LOCUS       REL606    100 bp
FEATURES             Location/Qualifiers
     CDS             10..50
                     /gene="testGene"
                     /product="test protein"
                     /locus_tag="b0001"
     CDS             complement(60..90)
                     /gene="revGene"
ORIGIN
        1 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac
       51 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac
//
"#;
    let mut f = tempfile::NamedTempFile::with_suffix(".gbk").unwrap();
    f.write_all(gbk.as_bytes()).unwrap();
    f.flush().unwrap();

    let rec = GenBankRecord::load(f.path()).unwrap();
    assert_eq!(rec.locus, "REL606");
    assert_eq!(rec.sequence.len(), 100);
    assert_eq!(rec.cds_count(), 2);

    let cds1 = &rec.features[0];
    assert_eq!(cds1.feature_type, "CDS");
    assert_eq!(cds1.start, 10);
    assert_eq!(cds1.end, 50);
    assert!(cds1.forward);
    assert_eq!(cds1.gene.as_deref(), Some("testGene"));
    assert_eq!(cds1.product.as_deref(), Some("test protein"));
    assert_eq!(cds1.locus_tag.as_deref(), Some("b0001"));

    let cds2 = &rec.features[1];
    assert!(!cds2.forward);
    assert_eq!(cds2.gene.as_deref(), Some("revGene"));
}

#[test]
fn feature_at_lookup() {
    let gbk = "LOCUS       T    50 bp\nFEATURES             Location/Qualifiers\n     CDS             10..30\n                     /gene=\"abc\"\nORIGIN\n        1 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt acgtacgtac\n//\n";
    let mut f = tempfile::NamedTempFile::with_suffix(".gbk").unwrap();
    f.write_all(gbk.as_bytes()).unwrap();
    f.flush().unwrap();

    let rec = GenBankRecord::load(f.path()).unwrap();
    assert!(rec.feature_at(15).is_some());
    assert_eq!(rec.feature_at(15).unwrap().gene.as_deref(), Some("abc"));
    assert!(rec.feature_at(5).is_none());
    assert!(rec.feature_at(35).is_none());
}
