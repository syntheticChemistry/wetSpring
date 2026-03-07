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
    provenance::shaders_from(SpringDomain::WetSpring)
}

/// All shaders wetSpring consumes (from any spring, including self).
#[must_use]
pub fn shaders_consumed() -> Vec<&'static ShaderRecord> {
    provenance::shaders_consumed_by(SpringDomain::WetSpring)
}

/// Shaders from other springs that wetSpring benefits from.
#[must_use]
pub fn shaders_from_other_springs() -> Vec<&'static ShaderRecord> {
    provenance::shaders_consumed_by(SpringDomain::WetSpring)
        .into_iter()
        .filter(|r| r.origin != SpringDomain::WetSpring)
        .collect()
}

/// Full cross-spring evolution report from barraCuda provenance registry.
#[must_use]
pub fn cross_spring_report() -> String {
    provenance::report::evolution_report()
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
            .filter(|c| **c != SpringDomain::WetSpring)
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
        .filter(|((_, to), _)| *to == SpringDomain::WetSpring)
        .map(|(_, v)| v)
        .sum();
    let outbound: usize = matrix
        .iter()
        .filter(|((from, _), _)| *from == SpringDomain::WetSpring)
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
        assert!(origins.contains(&SpringDomain::HotSpring));
        assert!(origins.contains(&SpringDomain::NeuralSpring));
    }

    #[test]
    fn from_other_springs_excludes_self() {
        let others = shaders_from_other_springs();
        for r in &others {
            assert_ne!(r.origin, SpringDomain::WetSpring);
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
}
