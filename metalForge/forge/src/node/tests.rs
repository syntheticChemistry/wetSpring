// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::module_inception)]
mod tests {
    use crate::node::assembly;
    use crate::node::{
        AssemblyStats, compute_collection_from_dir, list_assembly_files, shannon_entropy_binned,
    };
    use std::path::Path;

    use wetspring_barracuda::tolerances;

    #[test]
    fn n50_basic() {
        let contigs = vec![100, 80, 50, 30, 20, 10];
        let total: u64 = contigs.iter().sum();
        assert_eq!(assembly::compute_n50(&contigs, total), 80);
    }

    #[test]
    fn n50_single_contig() {
        assert_eq!(assembly::compute_n50(&[1000], 1000), 1000);
    }

    #[test]
    fn n50_empty() {
        assert_eq!(assembly::compute_n50(&[], 0), 0);
    }

    #[test]
    fn gc_count_basic() {
        let seqs = vec![b"ATGCGC".to_vec()];
        let (gc, total) = assembly::count_gc(&seqs);
        assert_eq!(gc, 4);
        assert_eq!(total, 6);
    }

    #[test]
    fn gc_handles_n_bases() {
        let seqs = vec![b"ATGCN".to_vec()];
        let (gc, total) = assembly::count_gc(&seqs);
        assert_eq!(gc, 2);
        assert_eq!(total, 5);
    }

    #[test]
    fn gc_case_insensitive() {
        let seqs = vec![b"atgcGC".to_vec()];
        let (gc, total) = assembly::count_gc(&seqs);
        assert_eq!(gc, 4);
        assert_eq!(total, 6);
    }

    #[test]
    fn mean_basic() {
        assert!((barracuda::stats::mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_empty() {
        assert!((barracuda::stats::mean(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn std_dev_basic() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let _m = barracuda::stats::mean(&values);
        let sd = barracuda::stats::correlation::std_dev(&values).unwrap_or(0.0);
        assert!(
            (sd - 2.138).abs() < tolerances::ODE_STEADY_STATE,
            "sample std dev: {sd}"
        );
    }

    #[test]
    fn std_dev_single() {
        let sd = barracuda::stats::correlation::std_dev(&[42.0]).unwrap_or(0.0);
        assert!((sd - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn shannon_entropy_uniform() {
        let values: Vec<f64> = (0..100).map(|i| f64::from(i) / 100.0).collect();
        let h = shannon_entropy_binned(&values, 10);
        assert!(
            h > 2.0,
            "uniform distribution should have high entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_constant() {
        let values = vec![0.5; 100];
        let h = shannon_entropy_binned(&values, 10);
        assert!(
            h.abs() < f64::EPSILON,
            "constant values should have zero entropy"
        );
    }

    #[test]
    fn shannon_entropy_empty() {
        assert!((shannon_entropy_binned(&[], 10) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_fasta_basic() {
        let fasta = b">seq1\nATGC\nATGC\n>seq2\nGGGG\n";
        let reader = std::io::BufReader::new(&fasta[..]);
        let seqs = assembly::parse_fasta_sequences(reader).unwrap();
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0], b"ATGCATGC");
        assert_eq!(seqs[1], b"GGGG");
    }

    #[test]
    fn parse_fasta_empty() {
        let reader = std::io::BufReader::new(&b""[..]);
        let seqs = assembly::parse_fasta_sequences(reader).unwrap();
        assert!(seqs.is_empty());
    }

    #[test]
    fn aggregate_collection_basic() {
        let assemblies = vec![
            AssemblyStats {
                accession: "A".to_string(),
                num_contigs: 10,
                total_length: 5_000_000,
                n50: 100_000,
                gc_content: 0.45,
                largest_contig: 500_000,
            },
            AssemblyStats {
                accession: "B".to_string(),
                num_contigs: 20,
                total_length: 4_000_000,
                n50: 80_000,
                gc_content: 0.50,
                largest_contig: 300_000,
            },
        ];

        let coll = assembly::aggregate_collection("test", assemblies);
        assert_eq!(coll.assembly_count, 2);
        assert!((coll.mean_gc - 0.475).abs() < tolerances::ODE_STEADY_STATE);
        assert!((coll.mean_genome_size - 4_500_000.0).abs() < 1.0);
    }

    #[test]
    fn list_assembly_files_handles_missing_dir() {
        let result = list_assembly_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn list_assembly_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = list_assembly_files(dir.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn list_assembly_files_filters_fna_gz() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("GCF_001.fna.gz"), "data").unwrap();
        std::fs::write(dir.path().join("GCF_002.fna.gz"), "data").unwrap();
        std::fs::write(dir.path().join("README.md"), "docs").unwrap();
        std::fs::write(dir.path().join("data.csv"), "csv").unwrap();

        let result = list_assembly_files(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
        assert!(
            result[0]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(".fna.gz")
        );
    }

    #[test]
    fn parse_fasta_multiline() {
        let fasta = b">seq1\nATGC\nGCTA\nAAAA\n>seq2\nTTTT\n";
        let reader = std::io::BufReader::new(&fasta[..]);
        let seqs = assembly::parse_fasta_sequences(reader).unwrap();
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0], b"ATGCGCTAAAAA");
        assert_eq!(seqs[1], b"TTTT");
    }

    #[test]
    fn parse_fasta_no_header() {
        let fasta = b"ATGCATGC\n";
        let reader = std::io::BufReader::new(&fasta[..]);
        let seqs = assembly::parse_fasta_sequences(reader).unwrap();
        assert!(seqs.is_empty());
    }

    #[test]
    fn parse_fasta_trailing_whitespace() {
        let fasta = b">seq\nATGC  \nGCTA\n";
        let reader = std::io::BufReader::new(&fasta[..]);
        let seqs = assembly::parse_fasta_sequences(reader).unwrap();
        assert_eq!(seqs[0], b"ATGCGCTA");
    }

    #[test]
    fn gc_empty_sequences() {
        let (gc, total) = assembly::count_gc(&[]);
        assert_eq!(gc, 0);
        assert_eq!(total, 0);
    }

    #[test]
    fn gc_only_n_bases() {
        let seqs = vec![b"NNNNN".to_vec()];
        let (gc, total) = assembly::count_gc(&seqs);
        assert_eq!(gc, 0);
        assert_eq!(total, 5);
    }

    #[test]
    fn n50_equal_contigs() {
        let contigs = vec![100, 100, 100];
        assert_eq!(assembly::compute_n50(&contigs, 300), 100);
    }

    #[test]
    fn n50_dominated_by_largest() {
        let contigs = vec![900, 50, 50];
        assert_eq!(assembly::compute_n50(&contigs, 1000), 900);
    }

    #[test]
    fn shannon_entropy_two_bins() {
        let values = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let h = shannon_entropy_binned(&values, 2);
        let expected = 2.0_f64.ln();
        assert!(
            (h - expected).abs() < tolerances::ODE_STEADY_STATE,
            "expected ~{expected}, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_zero_bins() {
        assert!((shannon_entropy_binned(&[0.5, 0.6], 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_collection_single() {
        let assemblies = vec![AssemblyStats {
            accession: "X".to_string(),
            num_contigs: 5,
            total_length: 1_000_000,
            n50: 200_000,
            gc_content: 0.40,
            largest_contig: 400_000,
        }];
        let coll = assembly::aggregate_collection("single", assemblies);
        assert_eq!(coll.assembly_count, 1);
        assert!((coll.mean_gc - 0.40).abs() < f64::EPSILON);
        assert!((coll.mean_genome_size - 1_000_000.0).abs() < 1.0);
        assert!((coll.gc_std).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_collection_from_dir_no_fna_gz() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.txt"), "not fasta").unwrap();
        let result = compute_collection_from_dir("test", dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no .fna.gz"),);
    }

    #[test]
    fn compute_assembly_stats_pipeline() {
        let dir = tempfile::tempdir().unwrap();

        let fasta = b">contig1\nATGCGCGCATGCATGCATATGCGCATGC\n>contig2\nAAAATTTTCCCCGGGG\n";
        let compressed = {
            use std::io::Write;
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder.write_all(fasta).unwrap();
            encoder.finish().unwrap()
        };
        std::fs::write(dir.path().join("GCF_test.fna.gz"), &compressed).unwrap();

        let stats = assembly::compute_assembly_stats_from_file(
            "GCF_test",
            &dir.path().join("GCF_test.fna.gz"),
        )
        .unwrap();
        assert_eq!(stats.num_contigs, 2);
        assert_eq!(stats.accession, "GCF_test");
        assert!(stats.total_length > 0);
        assert!(stats.gc_content > 0.0 && stats.gc_content < 1.0);
        assert!(stats.n50 > 0);
        assert!(stats.largest_contig >= stats.n50);

        let coll = compute_collection_from_dir("test", dir.path()).unwrap();
        assert_eq!(coll.assembly_count, 1);
        assert!(coll.mean_gc > 0.0);
    }
}
