// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-domain timing summaries for cross-spring validators.

/// Per-domain timing row for [`print_domain_summary`].
#[derive(Debug)]
pub struct DomainResult {
    /// Domain or section name.
    pub name: &'static str,
    /// Optional originating spring (cross-spring validators).
    pub spring: Option<&'static str>,
    /// Elapsed time in milliseconds.
    pub ms: f64,
    /// Number of validation checks in this domain.
    pub checks: u32,
}

/// Box-drawing domain summary table; optional Spring column when any row sets [`DomainResult::spring`].
pub fn print_domain_summary(title: &str, domains: &[DomainResult]) {
    let has_spring = domains.iter().any(|d| d.spring.is_some());
    let mut total_checks: u32 = 0;
    let mut total_ms: f64 = 0.0;
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║  {title:<64} ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    if has_spring {
        println!(
            "║ {:<22} │ {:<18} │ {:>7} │ {:>3} ║",
            "Domain", "Spring", "Time", "✓"
        );
    } else {
        println!("║ {:<22} │ {:>7} │ {:>3} ║", "Domain", "Time", "✓");
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    for d in domains {
        total_checks += d.checks;
        total_ms += d.ms;
        if has_spring {
            let spring = d.spring.unwrap_or("—");
            println!(
                "║ {:<22} │ {:<18} │ {:>5.1}ms │ {:>3} ║",
                d.name, spring, d.ms, d.checks
            );
        } else {
            println!("║ {:<22} │ {:>5.1}ms │ {:>3} ║", d.name, d.ms, d.checks);
        }
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    if has_spring {
        println!(
            "║ {:<22} │ {:<18} │ {:>5.1}ms │ {:>3} ║",
            "TOTAL", "", total_ms, total_checks
        );
    } else {
        println!(
            "║ {:<22} │ {:>5.1}ms │ {:>3} ║",
            "TOTAL", total_ms, total_checks
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════╝");
}
