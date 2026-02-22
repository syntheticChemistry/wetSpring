// SPDX-License-Identifier: AGPL-3.0-or-later
//! Determinism tests: rerun identical inputs, expect bitwise-identical output
//! via `to_bits()` equality.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;
use wetspring_barracuda::bio::{
    alignment, ani, chimera, dada2, derep, diversity, dnds, felsenstein, hmm, kmd, merge_pairs,
    quality, signal, taxonomy,
};
use wetspring_barracuda::io::fastq;

// ═══════════════════════════════════════════════════════════════════
// Determinism tests — rerun identical, expect bitwise-identical output
// ═══════════════════════════════════════════════════════════════════

#[test]
fn diversity_deterministic_across_runs() {
    let counts: Vec<f64> = (1..=200).map(|i| f64::from(i * 7 % 50 + 1)).collect();
    let run1 = diversity::alpha_diversity(&counts);
    let run2 = diversity::alpha_diversity(&counts);
    assert_eq!(run1.shannon.to_bits(), run2.shannon.to_bits());
    assert_eq!(run1.simpson.to_bits(), run2.simpson.to_bits());
    assert_eq!(run1.observed.to_bits(), run2.observed.to_bits());
    assert_eq!(run1.chao1.to_bits(), run2.chao1.to_bits());
    assert_eq!(run1.evenness.to_bits(), run2.evenness.to_bits());
}

#[test]
fn bray_curtis_deterministic_across_runs() {
    let samples: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            (0..100)
                .map(|j| f64::from((i * 37 + j * 13) % 50 + 1))
                .collect()
        })
        .collect();
    let run1 = diversity::bray_curtis_condensed(&samples);
    let run2 = diversity::bray_curtis_condensed(&samples);
    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn dada2_deterministic_across_runs() {
    let seqs: Vec<derep::UniqueSequence> = (0..10)
        .map(|i| {
            let base = b"ACGTACGTACGTACGTACGTACGTACGT";
            let mut seq = base.to_vec();
            if i > 0 {
                let len = seq.len();
                seq[i % len] = b"TGCA"[i % 4];
            }
            let len = seq.len();
            derep::UniqueSequence {
                representative_id: format!("seq{i}"),
                sequence: seq,
                representative_quality: vec![40; len],
                abundance: if i == 0 { 100 } else { 5 },
                best_quality: 40.0,
            }
        })
        .collect();

    let params = dada2::Dada2Params {
        min_abundance: 2,
        ..dada2::Dada2Params::default()
    };

    let (asvs1, stats1) = dada2::denoise(&seqs, &params);
    let (asvs2, stats2) = dada2::denoise(&seqs, &params);
    assert_eq!(stats1.output_asvs, stats2.output_asvs);
    assert_eq!(stats1.output_reads, stats2.output_reads);
    assert_eq!(asvs1.len(), asvs2.len());
    for (a, b) in asvs1.iter().zip(asvs2.iter()) {
        assert_eq!(a.sequence, b.sequence);
        assert_eq!(a.abundance, b.abundance);
    }
}

#[test]
fn chimera_detection_deterministic_across_runs() {
    let seqs: Vec<dada2::Asv> = (0..6)
        .map(|i| {
            let seq = match i {
                0 => b"AAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_vec(),
                1 => b"CCCCCCCCCCCCCCCCCCCCCCCCCCCC".to_vec(),
                2 => {
                    let mut v = b"AAAAAAAAAAAAAA".to_vec();
                    v.extend_from_slice(b"CCCCCCCCCCCCCC");
                    v
                }
                _ => vec![b"ACGT"[i % 4]; 28],
            };
            dada2::Asv {
                sequence: seq,
                abundance: if i < 2 { 100 } else { 5 },
                n_members: 1,
            }
        })
        .collect();

    let params = chimera::ChimeraParams::default();
    let (results1, _) = chimera::detect_chimeras(&seqs, &params);
    let (results2, _) = chimera::detect_chimeras(&seqs, &params);
    assert_eq!(results1.len(), results2.len());
    for (a, b) in results1.iter().zip(results2.iter()) {
        assert_eq!(a.is_chimera, b.is_chimera);
        assert_eq!(a.query_idx, b.query_idx);
    }
}

#[test]
fn taxonomy_classification_deterministic_across_runs() {
    let refs = vec![
        taxonomy::ReferenceSeq {
            id: "ref1".into(),
            sequence: b"ACGTACGTACGTACGTACGT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Bac;Firm;Bac;Lac;Lac;Lacto"),
        },
        taxonomy::ReferenceSeq {
            id: "ref2".into(),
            sequence: b"GGGTTTTGGGTTTTGGGTTTT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Bac;Prot;Gamma;Enter;Enter;Ecoli"),
        },
    ];
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, 8);
    let params = taxonomy::ClassifyParams::default();
    let query = b"ACGTACGTACGTACGT";

    let r1 = classifier.classify(query, &params);
    let r2 = classifier.classify(query, &params);
    assert_eq!(r1.taxon_idx, r2.taxon_idx);
    for (c1, c2) in r1.confidence.iter().zip(r2.confidence.iter()) {
        assert_eq!(c1.to_bits(), c2.to_bits());
    }
}

#[test]
fn full_16s_pipeline_deterministic_across_runs() {
    let dir = TempDir::new().unwrap();
    let fwd_content = generate_16s_reads(100, 250, 42);
    let rev_content = generate_16s_reads(100, 250, 137);
    let fwd_path = dir.path().join("R1.fastq");
    let rev_path = dir.path().join("R2.fastq");
    File::create(&fwd_path)
        .unwrap()
        .write_all(fwd_content.as_bytes())
        .unwrap();
    File::create(&rev_path)
        .unwrap()
        .write_all(rev_content.as_bytes())
        .unwrap();

    let run = || {
        let fwd_reads = fastq::parse_fastq(&fwd_path).unwrap();
        let rev_reads = fastq::parse_fastq(&rev_path).unwrap();

        let qparams = quality::QualityParams {
            min_length: 20,
            ..quality::QualityParams::default()
        };
        let (fwd_filt, _) = quality::filter_reads(&fwd_reads, &qparams);
        let (rev_filt, _) = quality::filter_reads(&rev_reads, &qparams);

        let n_pairs = fwd_filt.len().min(rev_filt.len());
        let (merged, _) = merge_pairs::merge_pairs(
            &fwd_filt[..n_pairs],
            &rev_filt[..n_pairs],
            &merge_pairs::MergeParams::default(),
        );

        let (uniques, _) = derep::dereplicate(&merged, derep::DerepSort::Abundance, 0);
        let counts = derep::abundance_vector(&uniques);
        diversity::alpha_diversity(&counts)
    };

    let alpha1 = run();
    let alpha2 = run();
    assert_eq!(alpha1.shannon.to_bits(), alpha2.shannon.to_bits());
    assert_eq!(alpha1.simpson.to_bits(), alpha2.simpson.to_bits());
    assert_eq!(alpha1.observed.to_bits(), alpha2.observed.to_bits());
}

// ── Non-stochastic algorithm determinism (bitwise identical) ────────

#[test]
fn determinism_shannon() {
    let counts = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let r1 = diversity::shannon(&counts);
    let r2 = diversity::shannon(&counts);
    assert_eq!(r1.to_bits(), r2.to_bits(), "Shannon must be deterministic");
}

#[test]
fn determinism_simpson() {
    let counts = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let r1 = diversity::simpson(&counts);
    let r2 = diversity::simpson(&counts);
    assert_eq!(r1.to_bits(), r2.to_bits(), "Simpson must be deterministic");
}

#[test]
fn determinism_bray_curtis() {
    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![15.0, 25.0, 5.0, 45.0];
    let r1 = diversity::bray_curtis(&a, &b);
    let r2 = diversity::bray_curtis(&a, &b);
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "Bray-Curtis must be deterministic"
    );
}

#[test]
fn determinism_ani() {
    let seq1 = b"ACGTACGTACGTACGTACGT";
    let seq2 = b"ACGTACGTACGTACGTACGA";
    let r1 = ani::pairwise_ani(seq1, seq2);
    let r2 = ani::pairwise_ani(seq1, seq2);
    assert_eq!(
        r1.ani.to_bits(),
        r2.ani.to_bits(),
        "ANI must be deterministic"
    );
    assert_eq!(r1.aligned_length, r2.aligned_length);
    assert_eq!(r1.identical_positions, r2.identical_positions);
}

#[test]
fn determinism_dnds() {
    let seq1 = b"ATGGCCAAACCCGGGTTTAAATGG";
    let seq2 = b"ATGGCTAAACCCGGGTTTAAATGG";
    let r1 = dnds::pairwise_dnds(seq1, seq2).unwrap();
    let r2 = dnds::pairwise_dnds(seq1, seq2).unwrap();
    assert_eq!(
        r1.dn.to_bits(),
        r2.dn.to_bits(),
        "dN/dS dn must be deterministic"
    );
    assert_eq!(
        r1.ds.to_bits(),
        r2.ds.to_bits(),
        "dN/dS ds must be deterministic"
    );
    match (&r1.omega, &r2.omega) {
        (Some(o1), Some(o2)) => assert_eq!(o1.to_bits(), o2.to_bits()),
        (None, None) => {}
        _ => panic!("dN/dS omega must be deterministic"),
    }
}

#[test]
fn determinism_hmm_forward() {
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![(0.5_f64).ln(), (0.5_f64).ln()],
        log_trans: vec![
            (0.9_f64).ln(),
            (0.1_f64).ln(),
            (0.1_f64).ln(),
            (0.9_f64).ln(),
        ],
        n_symbols: 2,
        log_emit: vec![
            (0.8_f64).ln(),
            (0.2_f64).ln(),
            (0.2_f64).ln(),
            (0.8_f64).ln(),
        ],
    };
    let observations = vec![0, 1, 0, 1, 0];
    let r1 = hmm::forward(&model, &observations);
    let r2 = hmm::forward(&model, &observations);
    assert_eq!(
        r1.log_likelihood.to_bits(),
        r2.log_likelihood.to_bits(),
        "HMM forward log_likelihood must be deterministic"
    );
    assert_eq!(r1.log_alpha.len(), r2.log_alpha.len());
    for (a, b) in r1.log_alpha.iter().zip(r2.log_alpha.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "HMM log_alpha must be deterministic"
        );
    }
}

#[test]
fn determinism_smith_waterman() {
    let query = b"ACGTACGTACGT";
    let target = b"ACGTACGTACGA";
    let params = alignment::ScoringParams::default();
    let r1 = alignment::smith_waterman(query, target, &params);
    let r2 = alignment::smith_waterman(query, target, &params);
    assert_eq!(r1.score, r2.score);
    assert_eq!(r1.aligned_query, r2.aligned_query);
    assert_eq!(r1.aligned_target, r2.aligned_target);
    assert_eq!(r1.query_start, r2.query_start);
    assert_eq!(r1.target_start, r2.target_start);
}

#[test]
fn determinism_felsenstein() {
    use wetspring_barracuda::bio::felsenstein::TreeNode;
    let tree = TreeNode::Internal {
        left: Box::new(TreeNode::Leaf {
            name: "A".into(),
            states: vec![0, 1, 2, 3, 0],
        }),
        right: Box::new(TreeNode::Leaf {
            name: "B".into(),
            states: vec![0, 1, 2, 3, 1],
        }),
        left_branch: 0.1,
        right_branch: 0.1,
    };
    let r1 = felsenstein::log_likelihood(&tree, 1.0);
    let r2 = felsenstein::log_likelihood(&tree, 1.0);
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "Felsenstein log_likelihood must be deterministic"
    );
}

#[test]
fn determinism_signal_peak() {
    let n = 100;
    #[allow(clippy::cast_precision_loss)]
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = f64::from(i);
            10000.0 * (-0.5 * ((x - 30.0) / 5.0).powi(2)).exp()
                + 5000.0 * (-0.5 * ((x - 70.0) / 4.0).powi(2)).exp()
        })
        .collect();
    let params = signal::PeakParams::default();
    let r1 = signal::find_peaks(&data, &params);
    let r2 = signal::find_peaks(&data, &params);
    assert_eq!(r1.len(), r2.len(), "Peak count must be deterministic");
    for (p1, p2) in r1.iter().zip(r2.iter()) {
        assert_eq!(p1.index, p2.index);
        assert_eq!(p1.height.to_bits(), p2.height.to_bits());
        assert_eq!(p1.prominence.to_bits(), p2.prominence.to_bits());
        assert_eq!(p1.width.to_bits(), p2.width.to_bits());
        assert_eq!(p1.left_ips.to_bits(), p2.left_ips.to_bits());
        assert_eq!(p1.right_ips.to_bits(), p2.right_ips.to_bits());
    }
}

#[test]
fn determinism_kmd() {
    let mz = vec![100.0, 150.0, 200.0, 250.0, 300.0];
    let r1 = kmd::kendrick_mass_defect(&mz, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
    let r2 = kmd::kendrick_mass_defect(&mz, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
    assert_eq!(r1.len(), r2.len());
    for (k1, k2) in r1.iter().zip(r2.iter()) {
        assert_eq!(k1.exact_mass.to_bits(), k2.exact_mass.to_bits());
        assert_eq!(k1.kendrick_mass.to_bits(), k2.kendrick_mass.to_bits());
        assert_eq!(k1.kmd.to_bits(), k2.kmd.to_bits());
        assert_eq!(k1.nominal_km.to_bits(), k2.nominal_km.to_bits());
    }
}

fn generate_16s_reads(n: usize, len: usize, seed: u64) -> String {
    let mut rng = seed;
    let bases = [b'A', b'C', b'G', b'T'];
    let mut out = String::new();
    let template = |rng: &mut u64| -> (Vec<u8>, Vec<u8>) {
        let mut seq = Vec::with_capacity(len);
        let mut qual = Vec::with_capacity(len);
        for _ in 0..len {
            *rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seq.push(bases[(*rng >> 33) as usize % 4]);
            qual.push(b'I');
        }
        (seq, qual)
    };

    let species: Vec<Vec<u8>> = (0..3)
        .map(|_| {
            let (s, _) = template(&mut rng);
            s
        })
        .collect();

    for i in 0..n {
        let sp = &species[i % species.len()];
        let mut seq = sp.clone();
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        if (rng >> 33) % 20 == 0 {
            let pos = (rng >> 17) as usize % len;
            seq[pos] = bases[(rng >> 10) as usize % 4];
        }
        let qual = vec![b'I'; len];
        let _ = write!(
            out,
            "@read{i}\n{}\n+\n{}\n",
            String::from_utf8_lossy(&seq),
            String::from_utf8_lossy(&qual)
        );
    }
    out
}
