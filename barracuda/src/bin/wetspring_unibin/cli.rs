// SPDX-License-Identifier: AGPL-3.0-or-later
//! CLI definition for the wetSpring UniBin binary.

/// wetSpring UniBin — single binary for certification, validation, and serve.
#[derive(clap::Parser)]
#[command(name = "wetspring", about = "wetSpring eukaryotic UniBin")]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Output format for structured results.
#[derive(Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text (default).
    #[default]
    Text,
    /// Machine-readable JSON (for projectNUCLEUS Tier 2 ingestion).
    Json,
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
        /// Output format (text or json).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
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
        /// Output format (text or json).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Start JSON-RPC IPC server (biomeOS science primal).
    #[command(alias = "server")]
    Serve {
        /// Bind UDS at this path (overrides WETSPRING_SOCKET / XDG discovery).
        #[arg(long)]
        socket: Option<std::path::PathBuf>,
        /// Bind TCP on this port (opt-in; UDS is default transport).
        #[arg(long)]
        port: Option<u16>,
        /// Family identifier for BTSP socket naming.
        #[arg(long)]
        family_id: Option<String>,
    },
    /// Print composition health status.
    Status {
        /// Output format (text or json).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Run benchmark scenarios.
    Benchmark {
        /// Run a single benchmark by id.
        #[arg(long)]
        scenario: Option<String>,
        /// List all benchmarks without running them.
        #[arg(long, default_value_t = false)]
        list: bool,
        /// Output format (text or json).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Print version information.
    Version,
}
