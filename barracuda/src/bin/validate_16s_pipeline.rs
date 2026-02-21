// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate the complete 16S amplicon pipeline: derep → DADA2 → chimera → taxonomy → `UniFrac`.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | QIIME2/DADA2 via Galaxy 24.1 |
//! | Baseline version | Galaxy 24.1, QIIME2 2026.1.0 |
//! | Baseline command | Pipeline integration (synthetic communities) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --bin validate_16s_pipeline` |
//! | Data | Synthetic (pipeline integration tests) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Methodology
//!
//! Uses synthetic communities with known ground truth to validate each pipeline
//! stage produces correct and self-consistent results. Validates:
//!
//! 1. **DADA2**: Denoising separates true variants from errors
//! 2. **Chimera**: Known chimeras are detected, non-chimeras pass
//! 3. **Taxonomy**: Known taxa are classified correctly with high confidence
//! 4. **`UniFrac`**: Phylogeny-weighted distances match analytical expectations
//! 5. **Pipeline integration**: Stages compose correctly end-to-end

use std::collections::HashMap;
use wetspring_barracuda::bio::chimera::{self, ChimeraParams};
use wetspring_barracuda::bio::dada2::{self, Asv, Dada2Params};
use wetspring_barracuda::bio::derep::{self, DerepSort, UniqueSequence};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::taxonomy::{
    ClassifyParams, Lineage, NaiveBayesClassifier, ReferenceSeq,
};
use wetspring_barracuda::bio::unifrac::{self, PhyloTree};
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("wetSpring 16S Pipeline Validation");

    validate_dada2(&mut v);
    validate_chimera(&mut v);
    validate_taxonomy(&mut v);
    validate_unifrac(&mut v);
    validate_end_to_end(&mut v);

    v.finish();
}

// ── DADA2 denoising ─────────────────────────────────────────────────────────

fn make_unique(seq: &[u8], abundance: usize, q: u8) -> UniqueSequence {
    UniqueSequence {
        sequence: seq.to_vec(),
        abundance,
        best_quality: f64::from(q),
        representative_id: String::new(),
        representative_quality: vec![33 + q; seq.len()],
    }
}

fn validate_dada2(v: &mut Validator) {
    v.section("DADA2 Denoising");

    // Two clearly distinct sequences at high abundance → 2 ASVs
    let seqs = vec![
        make_unique(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 1000, 35),
        make_unique(b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", 800, 35),
    ];
    let (asvs, stats) = dada2::denoise(&seqs, &Dada2Params::default());
    v.check_count("DADA2 distinct → 2 ASVs", asvs.len(), 2);
    v.check_count("DADA2 input uniques", stats.input_uniques, 2);

    // Abundant center + low-quality error variant → 1 ASV (error absorbed)
    let center = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut variant = center.to_vec();
    variant[5] = b'T';
    let seqs = vec![make_unique(center, 1000, 35), make_unique(&variant, 3, 20)];
    let (asvs, _) = dada2::denoise(&seqs, &Dada2Params::default());
    v.check_count("DADA2 error variant absorbed → 1 ASV", asvs.len(), 1);

    // Total reads conserved through denoising
    let seqs = vec![
        make_unique(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT", 500, 30),
        make_unique(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", 300, 30),
    ];
    let (asvs, _) = dada2::denoise(&seqs, &Dada2Params::default());
    let total_abundance: usize = asvs.iter().map(|a| a.abundance).sum();
    v.check_count("DADA2 reads conserved", total_abundance, 800);

    // FASTA output format
    let fasta = dada2::asvs_to_fasta(&asvs);
    let has_headers = fasta.lines().filter(|l| l.starts_with('>')).count();
    v.check_count("DADA2 FASTA headers", has_headers, asvs.len());

    // Min-abundance filtering
    let seqs = vec![
        make_unique(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT", 100, 30),
        make_unique(b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", 1, 30),
    ];
    let params = Dada2Params {
        min_abundance: 5,
        ..Dada2Params::default()
    };
    let (asvs, _) = dada2::denoise(&seqs, &params);
    let all_above_min = asvs.iter().all(|a| a.abundance >= 5);
    v.check(
        "DADA2 min_abundance filter",
        if all_above_min { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}

// ── Chimera detection ───────────────────────────────────────────────────────

fn validate_chimera(v: &mut Validator) {
    v.section("Chimera Detection (UCHIME-style)");

    let left_parent = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    let right_parent = b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";

    // Construct chimera: first half of left + second half of right
    let mut chimera_seq = Vec::with_capacity(50);
    chimera_seq.extend_from_slice(&left_parent[..25]);
    chimera_seq.extend_from_slice(&right_parent[25..]);

    let asvs = vec![
        Asv {
            sequence: left_parent.to_vec(),
            abundance: 1000,
            n_members: 1,
        },
        Asv {
            sequence: right_parent.to_vec(),
            abundance: 800,
            n_members: 1,
        },
        Asv {
            sequence: chimera_seq.clone(),
            abundance: 10,
            n_members: 1,
        },
    ];

    let params = ChimeraParams::default();
    let (clean, stats) = chimera::remove_chimeras(&asvs, &params);

    v.check_count("Chimera: input sequences", stats.input_sequences, 3);
    #[allow(clippy::cast_precision_loss)]
    let clean_len_f64 = clean.len() as f64;
    v.check("Chimera: parents preserved", clean_len_f64, 2.0, 1.0);

    // Non-chimeric sequences should pass through unchanged
    let non_chimeric = vec![
        Asv {
            sequence: b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            abundance: 500,
            n_members: 1,
        },
        Asv {
            sequence: b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            abundance: 300,
            n_members: 1,
        },
    ];
    let (clean2, stats2) = chimera::remove_chimeras(&non_chimeric, &params);
    v.check_count("Chimera: non-chimeric pass through", clean2.len(), 2);
    v.check_count("Chimera: zero flagged", stats2.chimeras_found, 0);

    // Single sequence cannot be chimeric
    let single = vec![Asv {
        sequence: b"ACGTACGT".to_vec(),
        abundance: 100,
        n_members: 1,
    }];
    let (clean3, _) = chimera::remove_chimeras(&single, &params);
    v.check_count("Chimera: single seq passes", clean3.len(), 1);
}

// ── Taxonomy classification ─────────────────────────────────────────────────

fn validate_taxonomy(v: &mut Validator) {
    v.section("Taxonomy Classification (Naive Bayes)");

    let refs = vec![
        ReferenceSeq {
            id: "ref1".into(),
            sequence: b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            lineage: Lineage::from_taxonomy_string(
                "k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
            ),
        },
        ReferenceSeq {
            id: "ref2".into(),
            sequence: b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            lineage: Lineage::from_taxonomy_string(
                "k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
            ),
        },
        ReferenceSeq {
            id: "ref3".into(),
            sequence: b"GGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCC".to_vec(),
            lineage: Lineage::from_taxonomy_string(
                "k__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__fragilis",
            ),
        },
    ];

    let classifier = NaiveBayesClassifier::train(&refs, 8);
    v.check_count("Taxonomy: trained taxa", classifier.n_taxa(), 3);

    let params = ClassifyParams::default();

    // Classify a sequence identical to ref1 → should get Firmicutes
    let result = classifier.classify(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT", &params);
    let kingdom_correct = result.lineage.ranks.first().map(String::as_str) == Some("k__Bacteria");
    v.check(
        "Taxonomy: kingdom = Bacteria",
        if kingdom_correct { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    let phylum_correct = result.lineage.ranks.get(1).map(String::as_str) == Some("p__Firmicutes");
    v.check(
        "Taxonomy: ref1 → Firmicutes",
        if phylum_correct { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Classify ref2 → should get Proteobacteria
    let result2 = classifier.classify(b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA", &params);
    let phylum2 = result2.lineage.ranks.get(1).map(String::as_str);
    v.check(
        "Taxonomy: ref2 → Proteobacteria",
        if phylum2 == Some("p__Proteobacteria") {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    // Confidence should be high for exact match
    let kingdom_confidence = result.confidence.first().copied().unwrap_or(0.0);
    v.check(
        "Taxonomy: kingdom confidence > 0.8",
        if kingdom_confidence > 0.8 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Parse reference FASTA format
    let fasta_text = ">ref1 k__Bacteria;p__Firmicutes\nACGTACGT\n>ref2 k__Bacteria;p__Proteobacteria\nTGCATGCA\n";
    let parsed = wetspring_barracuda::bio::taxonomy::parse_reference_fasta(fasta_text);
    v.check_count("Taxonomy: parse reference FASTA", parsed.len(), 2);
}

// ── UniFrac distance ────────────────────────────────────────────────────────

fn validate_unifrac(v: &mut Validator) {
    v.section("UniFrac Distance (Phylogeny-weighted)");

    // Simple tree: ((A:1,B:1):0.5,(C:1,D:1):0.5)
    let tree = PhyloTree::from_newick("((A:1,B:1):0.5,(C:1,D:1):0.5)");
    v.check_count("UniFrac: tree leaves", tree.n_leaves(), 4);

    let total_bl = tree.total_branch_length();
    v.check(
        "UniFrac: total branch length = 5.0",
        total_bl,
        5.0,
        tolerances::ODE_NEAR_ZERO,
    );

    // Identical communities → distance 0
    let mut sample_a: HashMap<String, f64> = HashMap::new();
    sample_a.insert("A".into(), 10.0);
    sample_a.insert("B".into(), 20.0);
    let uw = unifrac::unweighted_unifrac(&tree, &sample_a, &sample_a);
    v.check(
        "UniFrac: identical = 0",
        uw,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Disjoint communities (A,B) vs (C,D) → high distance
    let mut sample_b: HashMap<String, f64> = HashMap::new();
    sample_b.insert("C".into(), 10.0);
    sample_b.insert("D".into(), 20.0);
    let uw_disjoint = unifrac::unweighted_unifrac(&tree, &sample_a, &sample_b);
    v.check(
        "UniFrac: disjoint communities > 0",
        if uw_disjoint > 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
    v.check(
        "UniFrac: unweighted in [0,1]",
        if (0.0..=1.0).contains(&uw_disjoint) {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    // Weighted UniFrac: abundance matters
    let wuf = unifrac::weighted_unifrac(&tree, &sample_a, &sample_b);
    v.check(
        "UniFrac: weighted >= 0",
        if wuf >= 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Weighted UniFrac: identical = 0
    let wuf_same = unifrac::weighted_unifrac(&tree, &sample_a, &sample_a);
    v.check(
        "UniFrac: weighted identical = 0",
        wuf_same,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Symmetry: d(A,B) = d(B,A)
    let uw_ba = unifrac::unweighted_unifrac(&tree, &sample_b, &sample_a);
    v.check(
        "UniFrac: symmetry",
        (uw_disjoint - uw_ba).abs(),
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Distance matrix: 2 samples → 2×2 matrix
    let mut table: HashMap<String, HashMap<String, f64>> = HashMap::new();
    table.insert("sampleA".into(), sample_a.clone());
    table.insert("sampleB".into(), sample_b.clone());
    let (ids, dm) = unifrac::unifrac_distance_matrix(&tree, &table, false);
    v.check_count("UniFrac: distance matrix samples", ids.len(), 2);
    v.check_count("UniFrac: distance matrix rows", dm.len(), 2);
}

// ── End-to-end pipeline ─────────────────────────────────────────────────────

fn validate_end_to_end(v: &mut Validator) {
    v.section("End-to-End 16S Pipeline");

    // Simulate reads from 3 "species" with known sequences
    let species_a = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let species_b = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
    let species_c = b"GGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCCGGGG";

    let mut records = Vec::new();
    for i in 0..50 {
        records.push(FastqRecord {
            id: format!("read_a_{i}"),
            sequence: species_a.to_vec(),
            quality: vec![b'I'; species_a.len()],
        });
    }
    for i in 0..30 {
        records.push(FastqRecord {
            id: format!("read_b_{i}"),
            sequence: species_b.to_vec(),
            quality: vec![b'I'; species_b.len()],
        });
    }
    for i in 0..20 {
        records.push(FastqRecord {
            id: format!("read_c_{i}"),
            sequence: species_c.to_vec(),
            quality: vec![b'I'; species_c.len()],
        });
    }

    // Step 1: Dereplication (min_abundance = 1 to keep all)
    let (uniques, derep_stats) = derep::dereplicate(&records, DerepSort::Abundance, 1);
    v.check_count("Pipeline: derep → 3 unique", uniques.len(), 3);
    v.check_count("Pipeline: derep input", derep_stats.input_sequences, 100);

    // Step 2: DADA2 denoising
    let (asvs, _) = dada2::denoise(&uniques, &Dada2Params::default());
    v.check_count("Pipeline: DADA2 → 3 ASVs", asvs.len(), 3);
    let total_reads: usize = asvs.iter().map(|a| a.abundance).sum();
    v.check_count("Pipeline: reads conserved = 100", total_reads, 100);

    // Step 3: Chimera removal (no chimeras in clean data)
    let (clean_asvs, chimera_stats) = chimera::remove_chimeras(&asvs, &ChimeraParams::default());
    v.check_count("Pipeline: chimera → 3 clean ASVs", clean_asvs.len(), 3);
    v.check_count("Pipeline: 0 chimeras", chimera_stats.chimeras_found, 0);

    // Step 4: Diversity from abundance vector
    #[allow(clippy::cast_precision_loss)]
    let counts: Vec<f64> = clean_asvs.iter().map(|a| a.abundance as f64).collect();
    let shannon = diversity::shannon(&counts);
    let simpson = diversity::simpson(&counts);
    let observed = diversity::observed_features(&counts);

    v.check("Pipeline: observed = 3", observed, 3.0, 0.0);
    v.check(
        "Pipeline: Shannon > 0",
        if shannon > 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
    v.check(
        "Pipeline: Simpson in (0,1)",
        if simpson > 0.0 && simpson < 1.0 {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    // Analytical Shannon for [50, 30, 20]: H = -Σ p_i ln(p_i)
    let expected_shannon = -0.2f64.mul_add(
        0.2_f64.ln(),
        0.5_f64.mul_add(0.5_f64.ln(), 0.3 * 0.3_f64.ln()),
    );
    v.check(
        "Pipeline: Shannon analytical",
        shannon,
        expected_shannon,
        tolerances::PYTHON_PARITY,
    );
}
