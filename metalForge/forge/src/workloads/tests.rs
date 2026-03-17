// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::expect_used,
    reason = "test module: assertions use expect for clarity"
)]

use crate::substrate::SubstrateKind;
use crate::workloads::*;

#[test]
fn all_workloads_has_entries() {
    let all = all_workloads();
    assert!(all.len() >= 54, "expected at least 54 workloads");
}

#[test]
fn origin_counts_match() {
    let (absorbed, local, cpu_only) = origin_summary();
    assert_eq!(
        absorbed, 52,
        "52 absorbed domains (28 base + 11 extension + 6 NUCLEUS data-driven + 7 S86 science)"
    );
    assert_eq!(local, 0, "0 local WGSL extensions (all absorbed)");
    assert_eq!(
        cpu_only, 2,
        "2 CPU-only domains (fastq_parsing + ncbi_assembly_ingest)"
    );
}

#[test]
fn absorbed_ode_workloads_have_dims() {
    for w in [
        phage_defense_ode(),
        bistable_ode(),
        multi_signal_ode(),
        cooperation_ode(),
        capacitor_ode(),
    ] {
        assert!(w.is_absorbed());
        assert!(
            w.ode_dims.is_some(),
            "{} should have ODE dims",
            w.workload.name
        );
    }
}

#[test]
fn absorbed_workloads_have_primitive() {
    for w in all_workloads() {
        if !w.is_absorbed() {
            continue;
        }
        assert!(
            w.primitive.is_some(),
            "{} should have primitive name",
            w.workload.name
        );
    }
}

#[test]
fn qs_biofilm_is_absorbed_ode() {
    let w = qs_biofilm_ode();
    assert!(w.is_absorbed());
    let dims = w.ode_dims.expect("should have dims");
    assert_eq!(dims.n_vars, 4);
    assert_eq!(dims.n_params, 17);
}

#[test]
fn taxonomy_prefers_npu() {
    let w = taxonomy();
    assert_eq!(w.workload.preferred_substrate, Some(SubstrateKind::Npu));
}

#[test]
fn fastq_parsing_is_cpu_only() {
    let w = fastq_parsing();
    assert!(w.is_cpu_only());
    assert!(!w.is_local());
    assert!(!w.is_absorbed());
}

#[test]
fn diversity_is_absorbed() {
    let w = diversity();
    assert!(w.is_absorbed());
    assert!(!w.is_local());
    assert!(!w.is_cpu_only());
}

#[test]
fn all_workloads_no_duplicate_names() {
    let all = all_workloads();
    let mut names: Vec<&str> = all.iter().map(|w| w.workload.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate workload names found");
}
