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
        cli::Commands::Certify {
            layer,
            bare,
            format,
        } => cmd_certify(layer, bare, format),
        cli::Commands::Validate {
            ref track,
            ref scenario,
            ref tier,
            list,
            format,
        } => cmd_validate(
            track.as_deref(),
            scenario.as_deref(),
            tier.as_deref(),
            list,
            format,
        ),
        cli::Commands::Serve => cmd_serve(),
        cli::Commands::Status { format } => cmd_status(format),
        cli::Commands::Version => cmd_version(),
    }
}

fn cmd_certify(layer: Option<u8>, bare: bool, format: cli::OutputFormat) {
    let max_layer = if bare {
        0
    } else {
        layer.unwrap_or(certification::MAX_LAYER)
    };
    let result = certification::certify(max_layer);
    if matches!(format, cli::OutputFormat::Json) {
        if let Ok(json) = result.to_json() {
            println!("{json}");
        }
    }
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
    format: cli::OutputFormat,
) {
    let registry = scenarios::build_registry();
    let json_mode = matches!(format, cli::OutputFormat::Json);

    if list {
        if json_mode {
            let items: Vec<serde_json::Value> = registry
                .all()
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "id": s.meta.id,
                        "track": format!("{}", s.meta.track),
                        "tier": format!("{}", s.meta.tier),
                        "description": s.meta.description,
                    })
                })
                .collect();
            if let Ok(json) = serde_json::to_string_pretty(&items) {
                println!("{json}");
            }
        } else {
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
        }
        return;
    }

    let mut v = ValidationResult::new("wetSpring Validation — Scenario Suite");
    if !json_mode {
        ValidationResult::print_banner("wetSpring Validation — Scenario Suite");
    }
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
        if json_mode {
            println!(r#"{{"error": "No scenarios matched the filter."}}"#);
        } else {
            eprintln!("No scenarios matched the filter.");
        }
        std::process::exit(1);
    }

    if json_mode {
        if let Ok(json) = v.to_json() {
            println!("{json}");
        }
    } else {
        v.finish();
    }
    std::process::exit(v.exit_code());
}

fn cmd_serve() {
    tracing_subscriber::fmt::init();

    tracing::info!(
        primal = wetspring_barracuda::PRIMAL_NAME,
        version = env!("CARGO_PKG_VERSION"),
        "starting science primal (UniBin)"
    );

    let server = match wetspring_barracuda::ipc::Server::bind_default() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "cannot bind socket");
            std::process::exit(1);
        }
    };

    tracing::info!(socket = %server.socket_path().display(), "bound");

    let _heartbeat = wetspring_barracuda::ipc::songbird::discover_socket().map_or_else(
        || {
            tracing::info!("Songbird not found, standalone mode");
            None
        },
        |songbird_socket| {
            tracing::info!(songbird = %songbird_socket.display(), "Songbird discovered");
            Some(wetspring_barracuda::ipc::songbird::start_heartbeat_loop(
                songbird_socket,
                server.socket_path().to_path_buf(),
            ))
        },
    );

    tracing::info!(
        capabilities = wetspring_barracuda::ipc::handlers::CAPABILITIES.len(),
        "ready"
    );

    server.run();
}

fn cmd_status(format: cli::OutputFormat) {
    let registry = scenarios::build_registry();
    let tier1 = registry.filter_by_tier(Tier::Rust).count();
    let tier2 = registry.filter_by_tier(Tier::Live).count();
    let _ctx = CompositionContext::from_live_discovery_with_fallback();

    match format {
        cli::OutputFormat::Json => {
            let status = serde_json::json!({
                "binary": "wetspring_unibin",
                "version": env!("CARGO_PKG_VERSION"),
                "domain": wetspring_barracuda::PRIMAL_DOMAIN,
                "niche": wetspring_barracuda::niche::NICHE_DESCRIPTION,
                "guidestone_level": wetspring_barracuda::niche::GUIDESTONE_READINESS,
                "scenarios": registry.len(),
                "tier1_rust": tier1,
                "tier2_live": tier2,
            });
            if let Ok(json) = serde_json::to_string_pretty(&status) {
                println!("{json}");
            }
        }
        cli::OutputFormat::Text => {
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
            println!("  scenarios:   {}", registry.len());
            println!("  tier 1 (rust): {tier1}");
            println!("  tier 2 (live): {tier2}");
            println!("  primals:     (live discovery attempted)");
        }
    }
}

fn cmd_version() {
    println!("wetspring_unibin {}", env!("CARGO_PKG_VERSION"));
}
