// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI definition for the wetSpring UniBin binary.

/// wetSpring UniBin — single binary for certification, validation, and serve.
#[derive(clap::Parser)]
#[command(name = "wetspring_unibin", about = "wetSpring eukaryotic UniBin")]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// UniBin subcommands.
#[derive(clap::Subcommand)]
pub enum Commands {
    /// Run layered certification (L0–L6).
    Certify {
        /// Maximum layer to certify (0–6, default: all).
        #[arg(long, value_name = "N")]
        layer: Option<u8>,
        /// Bare mode — layer 0 only, no primals needed.
        #[arg(long, default_value_t = false)]
        bare: bool,
    },
    /// Run validation scenarios.
    Validate {
        /// Filter by track (science, pharmacology, composition, pipeline).
        #[arg(long)]
        track: Option<String>,
        /// Run a single scenario by id.
        #[arg(long)]
        scenario: Option<String>,
        /// Filter by tier (rust, live, both).
        #[arg(long)]
        tier: Option<String>,
        /// List all scenarios without running them.
        #[arg(long, default_value_t = false)]
        list: bool,
    },
    /// Start JSON-RPC IPC server (biomeOS science primal).
    Serve,
    /// Print composition health status.
    Status,
    /// Print version information.
    Version,
}
