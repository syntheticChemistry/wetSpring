// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "UniBin: stdout is the output medium for certification and status"
)]
//! # wetspring_unibin — Eukaryotic UniBin for wetSpring
//!
//! Single binary absorbing guidestone certification, validation scenarios,
//! IPC serve, status, and version. Follows the primalSpring v0.9.25
//! eukaryotic evolution pattern.
//!
//! ## Subcommands
//!
//! - `certify` — Layered certification (L0–L6)
//! - `validate` — Two-tier scenario validation
//! - `serve` — JSON-RPC IPC server
//! - `status` — Composition health summary
//! - `version` — Version info

mod cli;

use clap::Parser;
use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use wetspring_barracuda::certification;
use wetspring_barracuda::validation::scenarios::{self, Tier, Track};

fn main() {
    let parsed = cli::Cli::parse();

    match parsed.command {
        cli::Commands::Certify { layer, bare } => cmd_certify(layer, bare),
        cli::Commands::Validate {
            ref track,
            ref scenario,
            ref tier,
            list,
        } => cmd_validate(track.as_deref(), scenario.as_deref(), tier.as_deref(), list),
        cli::Commands::Serve => cmd_serve(),
        cli::Commands::Status => cmd_status(),
        cli::Commands::Version => cmd_version(),
    }
}

fn cmd_certify(layer: Option<u8>, bare: bool) {
    let max_layer = if bare {
        0
    } else {
        layer.unwrap_or(certification::MAX_LAYER)
    };
    let result = certification::certify(max_layer);
    let code = if result.exit_code() == 0 && max_layer < 3 {
        2
    } else {
        result.exit_code()
    };
    std::process::exit(code);
}

fn cmd_validate(
    track_filter: Option<&str>,
    scenario_id: Option<&str>,
    tier_filter: Option<&str>,
    list: bool,
) {
    let registry = scenarios::build_registry();

    if list {
        println!(
            "  {:30} {:15} {:6} {}",
            "ID", "TRACK", "TIER", "DESCRIPTION"
        );
        println!("  {}", "─".repeat(80));
        for s in registry.all() {
            println!(
                "  {:30} {:15} {:6} {}",
                s.meta.id, s.meta.track, s.meta.tier, s.meta.description
            );
        }
        return;
    }

    let mut v = ValidationResult::new("wetSpring Validation — Scenario Suite");
    ValidationResult::print_banner("wetSpring Validation — Scenario Suite");
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();
    let mut ran = 0_u32;

    for s in registry.all() {
        if let Some(id) = scenario_id {
            if s.meta.id != id {
                continue;
            }
        }
        if let Some(track_str) = track_filter {
            if let Some(track) = Track::from_str_loose(track_str) {
                if s.meta.track != track {
                    continue;
                }
            }
        }
        if let Some(tier_str) = tier_filter {
            if let Some(tier) = Tier::from_str_loose(tier_str) {
                if s.meta.tier != tier && s.meta.tier != Tier::Both && tier != Tier::Both {
                    continue;
                }
            }
        }

        v.section(&format!("Scenario: {} [{}]", s.meta.id, s.meta.tier));
        (s.run)(&mut v, &mut ctx);
        ran += 1;
    }

    if ran == 0 {
        eprintln!("No scenarios matched the filter.");
        std::process::exit(1);
    }

    v.finish();
    std::process::exit(v.exit_code());
}

fn cmd_serve() {
    eprintln!("wetspring_unibin serve: delegating to wetspring IPC server");
    eprintln!("(Use `wetspring serve` for full IPC server — UniBin serve is scaffolded)");
    std::process::exit(0);
}

fn cmd_status() {
    println!("wetSpring UniBin Status");
    println!("  version:     {}", env!("CARGO_PKG_VERSION"));
    println!("  domain:      {}", wetspring_barracuda::PRIMAL_DOMAIN);
    println!(
        "  niche:       {}",
        wetspring_barracuda::niche::NICHE_DESCRIPTION
    );
    println!(
        "  guidestone:  L{}",
        wetspring_barracuda::niche::GUIDESTONE_READINESS
    );

    let registry = scenarios::build_registry();
    println!("  scenarios:   {}", registry.len());

    let tier1 = registry.filter_by_tier(Tier::Rust).count();
    let tier2 = registry.filter_by_tier(Tier::Live).count();
    println!("  tier 1 (rust): {tier1}");
    println!("  tier 2 (live): {tier2}");

    let _ctx = CompositionContext::from_live_discovery_with_fallback();
    println!("  primals:     (live discovery attempted)");
}

fn cmd_version() {
    println!("wetspring_unibin {}", env!("CARGO_PKG_VERSION"));
}
