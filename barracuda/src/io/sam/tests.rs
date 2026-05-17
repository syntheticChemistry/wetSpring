// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

fn make_record() -> SamRecord {
    SamRecord {
        qname: "read1".into(),
        flag: 0,
        rname: "chr1".into(),
        pos: 100,
        mapq: 60,
        cigar: vec![
            CigarOp {
                len: 50,
                op: CigarType::Match,
            },
            CigarOp {
                len: 2,
                op: CigarType::Insertion,
            },
            CigarOp {
                len: 48,
                op: CigarType::Match,
            },
        ],
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"ACGT".to_vec(),
        qual: b"IIII".to_vec(),
    }
}

#[test]
fn cigar_parse_roundtrip() {
    let cigar_str = "50M2I3D10S5H";
    let ops = parse_cigar(cigar_str).unwrap();
    assert_eq!(ops.len(), 5);
    assert_eq!(ops[0].len, 50);
    assert_eq!(ops[0].op, CigarType::Match);
    assert_eq!(ops[1].len, 2);
    assert_eq!(ops[1].op, CigarType::Insertion);
    assert_eq!(ops[2].len, 3);
    assert_eq!(ops[2].op, CigarType::Deletion);
    assert_eq!(ops[3].len, 10);
    assert_eq!(ops[3].op, CigarType::SoftClip);
    assert_eq!(ops[4].len, 5);
    assert_eq!(ops[4].op, CigarType::HardClip);

    let formatted = format_cigar(&ops);
    assert_eq!(formatted, cigar_str);
}

#[test]
fn cigar_star() {
    let ops = parse_cigar("*").unwrap();
    assert!(ops.is_empty());
    assert_eq!(format_cigar(&[]), "*");
}

#[test]
fn cigar_with_eq_and_x() {
    let ops = parse_cigar("10=5X3=").unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].op, CigarType::SeqMatch);
    assert_eq!(ops[1].op, CigarType::SeqMismatch);
    assert_eq!(ops[2].op, CigarType::SeqMatch);
}

#[test]
fn reference_and_query_lengths() {
    let ops = parse_cigar("50M2I3D10S").unwrap();
    assert_eq!(reference_length(&ops), 53); // 50M + 3D
    assert_eq!(query_length(&ops), 62); // 50M + 2I + 10S
}

#[test]
fn record_end_pos() {
    let rec = make_record();
    assert_eq!(rec.end_pos(), 100 + 50 + 48); // pos + 50M + 48M (I doesn't consume ref)
}

#[test]
fn record_flags() {
    let mut rec = make_record();
    assert!(rec.is_mapped());
    assert!(rec.is_primary());
    assert!(!rec.is_reverse());

    rec.flag = FLAG_REVERSE | FLAG_SECONDARY;
    assert!(rec.is_reverse());
    assert!(!rec.is_primary());
    assert!(rec.is_mapped());

    rec.flag = FLAG_UNMAPPED;
    assert!(!rec.is_mapped());
}

#[test]
fn parse_sam_line() {
    let line = "read1\t0\tchr1\t100\t60\t50M\t*\t0\t0\tACGT\tIIII";
    let rec = parse_record(line).unwrap();
    assert_eq!(rec.qname, "read1");
    assert_eq!(rec.flag, 0);
    assert_eq!(rec.rname, "chr1");
    assert_eq!(rec.pos, 100);
    assert_eq!(rec.mapq, 60);
    assert_eq!(rec.cigar.len(), 1);
    assert_eq!(rec.cigar[0].len, 50);
    assert_eq!(rec.seq, b"ACGT");
    assert_eq!(rec.qual, b"IIII");
}

#[test]
fn too_few_fields_error() {
    let result = parse_record("read1\t0\tchr1");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("11+"));
}

#[test]
fn write_roundtrip() {
    let rec = make_record();
    let mut buf = Vec::new();
    rec.write_to(&mut buf).unwrap();
    let line = String::from_utf8(buf).unwrap();
    let parsed = parse_record(&line).unwrap();
    assert_eq!(parsed.qname, rec.qname);
    assert_eq!(parsed.flag, rec.flag);
    assert_eq!(parsed.pos, rec.pos);
    assert_eq!(parsed.mapq, rec.mapq);
    assert_eq!(parsed.cigar.len(), rec.cigar.len());
    assert_eq!(parsed.seq, rec.seq);
}

#[test]
fn sam_file_roundtrip() {
    let mut f = tempfile::NamedTempFile::with_suffix(".sam").unwrap();

    let header = SamHeader {
        header_lines: vec!["@HD\tVN:1.6\tSO:coordinate".into()],
        references: vec![SamRefSeq {
            name: "chr1".into(),
            length: 1000,
        }],
    };

    let rec1 = SamRecord {
        qname: "r1".into(),
        flag: 0,
        rname: "chr1".into(),
        pos: 100,
        mapq: 60,
        cigar: parse_cigar("50M").unwrap(),
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"ACGT".to_vec(),
        qual: b"IIII".to_vec(),
    };
    let rec2 = SamRecord {
        qname: "r2".into(),
        flag: FLAG_REVERSE,
        rname: "chr1".into(),
        pos: 200,
        mapq: 42,
        cigar: parse_cigar("30M2D20M").unwrap(),
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"TGCA".to_vec(),
        qual: b"HHHH".to_vec(),
    };

    {
        let mut writer = SamWriter::new(&mut f);
        writer.write_header(&header).unwrap();
        writer.write_record(&rec1).unwrap();
        writer.write_record(&rec2).unwrap();
        writer.flush().unwrap();
    }

    let reader = SamReader::open(f.path()).unwrap();
    assert_eq!(reader.header.references.len(), 1);
    assert_eq!(reader.header.references[0].name, "chr1");
    assert_eq!(reader.header.references[0].length, 1000);

    let records: Vec<_> = reader.map(|r| r.unwrap()).collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].qname, "r1");
    assert_eq!(records[0].pos, 100);
    assert_eq!(records[1].qname, "r2");
    assert!(records[1].is_reverse());
}

#[test]
fn sort_by_position_works() {
    let mut records = vec![
        SamRecord {
            qname: "b".into(),
            flag: 0,
            rname: "chr1".into(),
            pos: 300,
            mapq: 60,
            cigar: vec![],
            rnext: "*".into(),
            pnext: 0,
            tlen: 0,
            seq: vec![],
            qual: vec![],
        },
        SamRecord {
            qname: "a".into(),
            flag: 0,
            rname: "chr1".into(),
            pos: 100,
            mapq: 60,
            cigar: vec![],
            rnext: "*".into(),
            pnext: 0,
            tlen: 0,
            seq: vec![],
            qual: vec![],
        },
    ];

    sort_by_position(&mut records);
    assert_eq!(records[0].pos, 100);
    assert_eq!(records[1].pos, 300);
}

#[test]
fn cigar_consumes_correctly() {
    assert!(CigarType::Match.consumes_query());
    assert!(CigarType::Match.consumes_reference());
    assert!(CigarType::Insertion.consumes_query());
    assert!(!CigarType::Insertion.consumes_reference());
    assert!(!CigarType::Deletion.consumes_query());
    assert!(CigarType::Deletion.consumes_reference());
    assert!(CigarType::SoftClip.consumes_query());
    assert!(!CigarType::SoftClip.consumes_reference());
    assert!(!CigarType::HardClip.consumes_query());
    assert!(!CigarType::HardClip.consumes_reference());
}
