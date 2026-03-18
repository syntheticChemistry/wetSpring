// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring shader provenance for wetSpring.
//!
//! Wires `barracuda::shaders::provenance` to expose wetSpring-specific views:
//! which shaders wetSpring originated, which it consumes from other springs,
//! and the full cross-spring dependency matrix.
//!
//! # Write → Absorb → Lean
//!
//! wetSpring bio shaders (Smith-Waterman, Felsenstein, Gillespie SSA, HMM,
//! `fused_map_reduce`) were written here, absorbed by barraCuda, and are now
//! consumed upstream by neuralSpring (batched inference, evolutionary dynamics)
//! and other springs. Conversely, wetSpring leans on hotSpring precision
//! math (DF64), neuralSpring statistics (KL divergence, chi-squared),
//! airSpring hydrology, and groundSpring universal primitives (Welford).

use barracuda::shaders::provenance::{
    self,
    types::{ShaderRecord, SpringDomain},
};

/// Shaders that wetSpring originated and barraCuda absorbed.
#[must_use]
pub fn shaders_authored() -> Vec<&'static ShaderRecord> {
    provenance::shaders_from(SpringDomain::WET_SPRING)
}

/// All shaders wetSpring consumes (from any spring, including self).
#[must_use]
pub fn shaders_consumed() -> Vec<&'static ShaderRecord> {
    provenance::shaders_consumed_by(SpringDomain::WET_SPRING)
}

/// Shaders from other springs that wetSpring benefits from.
#[must_use]
pub fn shaders_from_other_springs() -> Vec<&'static ShaderRecord> {
    provenance::shaders_consumed_by(SpringDomain::WET_SPRING)
        .into_iter()
        .filter(|r| r.origin != SpringDomain::WET_SPRING)
        .collect()
}

/// Full cross-spring evolution report from barraCuda provenance registry.
#[must_use]
pub fn cross_spring_report() -> String {
    provenance::report::evolution_report()
}

// ═══════════════════════════════════════════════════════════════════
// Python baseline provenance registry
// ═══════════════════════════════════════════════════════════════════

/// A Python baseline provenance record.
pub struct PythonBaseline {
    /// Rust validation binary name (e.g. `"validate_fastq"`).
    pub binary: &'static str,
    /// Python script path relative to workspace (e.g. `"scripts/validate_exp001.py"`).
    pub script: Option<&'static str>,
    /// Git commit of the baseline run.
    pub commit: &'static str,
    /// Date of the baseline run (ISO 8601).
    pub date: &'static str,
    /// Baseline category.
    pub category: BaselineCategory,
}

/// Categories of validation baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineCategory {
    /// Python parity — Rust reproduces Python/scipy results.
    PythonParity,
    /// GPU parity — GPU reproduces CPU Rust results.
    GpuParity,
    /// Analytical — Rust matches closed-form known values.
    Analytical,
    /// Published — Rust reproduces published paper values.
    Published,
    /// Visualization — scenario structure and curve-shape checks.
    Visualization,
}

/// Canonical commit hashes for major baseline epochs.
pub mod commits {
    /// Initial Python parity baselines (Exp001–097, Track 1–3).
    pub const PYTHON_PARITY_V1: &str = "e4358c5";
    /// Python script generation commit (all 58 scripts).
    pub const SCRIPTS_GENERATED: &str = "756df26";
    /// GPU parity baselines (Exp074+, Track 4 QS).
    pub const GPU_PARITY_V1: &str = "1f9f80e";
    /// petalTongue visualization baselines (Exp355+).
    pub const VIZ_V1: &str = "5e6a00b";
}

/// All registered Python baseline provenance records.
///
/// This is the single source of truth for which validation binaries
/// correspond to which Python baselines. When a Python script is rerun,
/// update the commit and date here.
#[must_use]
pub const fn python_baselines() -> &'static [PythonBaseline] {
    &[
        PythonBaseline {
            binary: "validate_fastq",
            script: Some("scripts/validate_exp001.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_diversity",
            script: Some("scripts/validate_exp001.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_peaks",
            script: Some("scripts/generate_peak_baselines.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_qs_ode",
            script: Some("scripts/waters2008_qs_ode.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_rare_biosphere",
            script: Some("scripts/anderson2015_rare_biosphere.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-20",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_epa_pfas_ml",
            script: Some("scripts/epa_pfas_ml_baseline.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-20",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_sulfur_phylogenomics",
            script: Some("scripts/mateos2023_sulfur_phylogenomics.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-20",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_soil_qs_cpu_parity",
            script: None,
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-25",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_substrate_router",
            script: None,
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-21",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_streaming_ode_phylo",
            script: None,
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-23",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_dynamic_anderson",
            script: None,
            commit: "wetSpring Phase 59",
            date: "2026-02-26",
            category: BaselineCategory::Analytical,
        },
        PythonBaseline {
            binary: "validate_cold_seep_pipeline",
            script: None,
            commit: commits::SCRIPTS_GENERATED,
            date: "2026-02-26",
            category: BaselineCategory::Analytical,
        },
        PythonBaseline {
            binary: "validate_gonzales_ic50_s79",
            script: None,
            commit: "Gonzales 2014 Table 1",
            date: "2026-03-02",
            category: BaselineCategory::Published,
        },
        PythonBaseline {
            binary: "validate_petaltongue_biogas_v1",
            script: None,
            commit: commits::VIZ_V1,
            date: "2026-03-14",
            category: BaselineCategory::Visualization,
        },
    ]
}

/// wetSpring-focused provenance summary for validation binaries and handoffs.
#[must_use]
pub fn wetspring_provenance_summary() -> String {
    use std::fmt::Write;

    let authored = shaders_authored();
    let consumed = shaders_consumed();
    let from_others = shaders_from_other_springs();
    let matrix = provenance::cross_spring_matrix();
    let total = provenance::report::shader_count();

    let mut s = String::new();
    let _ = writeln!(s, "# wetSpring Shader Provenance Summary\n");
    let _ = writeln!(s, "Registry: {total} shaders tracked across all springs\n");

    let _ = writeln!(s, "## Authored by wetSpring ({} shaders)\n", authored.len());
    for r in &authored {
        let consumers: Vec<_> = r
            .consumers
            .iter()
            .filter(|c| **c != SpringDomain::WET_SPRING)
            .map(|c| format!("{c}"))
            .collect();
        if consumers.is_empty() {
            let _ = writeln!(s, "- `{}` [{}]", r.path, r.category);
        } else {
            let _ = writeln!(
                s,
                "- `{}` [{}] → consumed by {}",
                r.path,
                r.category,
                consumers.join(", ")
            );
        }
    }

    let _ = writeln!(
        s,
        "\n## Consumed from other springs ({} shaders)\n",
        from_others.len()
    );
    for r in &from_others {
        let _ = writeln!(s, "- `{}` [{}] from {}", r.path, r.category, r.origin);
    }

    let _ = writeln!(s, "\n## Total consumed: {} shaders\n", consumed.len());

    let _ = writeln!(s, "## Cross-Spring Flows involving wetSpring\n");
    let inbound: usize = matrix
        .iter()
        .filter(|((_, to), _)| *to == SpringDomain::WET_SPRING)
        .map(|(_, v)| v)
        .sum();
    let outbound: usize = matrix
        .iter()
        .filter(|((from, _), _)| *from == SpringDomain::WET_SPRING)
        .map(|(_, v)| v)
        .sum();
    let _ = writeln!(s, "- Inbound (other→wetSpring): {inbound} shader-flows");
    let _ = writeln!(s, "- Outbound (wetSpring→other): {outbound} shader-flows");

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_includes_bio_shaders() {
        let authored = shaders_authored();
        assert!(
            authored.len() >= 4,
            "wetSpring should author 4+ bio shaders"
        );
        let paths: Vec<_> = authored.iter().map(|r| r.path).collect();
        assert!(
            paths.iter().any(|p| p.contains("smith_waterman")),
            "Smith-Waterman missing from wetSpring authored"
        );
        assert!(
            paths.iter().any(|p| p.contains("hmm_forward")),
            "HMM forward missing from wetSpring authored"
        );
    }

    #[test]
    fn consumed_includes_cross_spring() {
        let consumed = shaders_consumed();
        assert!(consumed.len() >= 10, "wetSpring should consume 10+ shaders");
        let origins: Vec<_> = consumed.iter().map(|r| r.origin).collect();
        assert!(origins.contains(&SpringDomain::HOT_SPRING));
        assert!(origins.contains(&SpringDomain::NEURAL_SPRING));
    }

    #[test]
    fn from_other_springs_excludes_self() {
        let others = shaders_from_other_springs();
        for r in &others {
            assert_ne!(r.origin, SpringDomain::WET_SPRING);
        }
    }

    #[test]
    fn summary_contains_key_sections() {
        let summary = wetspring_provenance_summary();
        assert!(summary.contains("Authored by wetSpring"));
        assert!(summary.contains("Consumed from other springs"));
        assert!(summary.contains("Inbound"));
        assert!(summary.contains("Outbound"));
    }

    #[test]
    fn cross_spring_report_not_empty() {
        let report = cross_spring_report();
        assert!(report.contains("Cross-Spring Shader Evolution Report"));
        assert!(report.contains("wetSpring"));
    }

    #[test]
    fn python_baselines_not_empty() {
        let baselines = python_baselines();
        assert!(
            baselines.len() >= 10,
            "should have 10+ registered baselines"
        );
    }

    #[test]
    fn python_baselines_have_valid_dates() {
        for b in python_baselines() {
            assert!(
                b.date.starts_with("2026-"),
                "{}: date should be 2026-*",
                b.binary
            );
        }
    }

    #[test]
    fn python_baselines_have_nonempty_commits() {
        for b in python_baselines() {
            assert!(
                !b.commit.is_empty(),
                "{}: commit should not be empty",
                b.binary
            );
        }
    }

    #[test]
    fn canonical_commits_are_short_hashes() {
        assert_eq!(commits::PYTHON_PARITY_V1.len(), 7);
        assert_eq!(commits::SCRIPTS_GENERATED.len(), 7);
        assert_eq!(commits::GPU_PARITY_V1.len(), 7);
        assert_eq!(commits::VIZ_V1.len(), 7);
    }
}
