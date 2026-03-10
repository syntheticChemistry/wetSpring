// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! wetSpring IPC server — `biomeOS` science primal.
//!
//! Listens on a Unix socket (or TCP via `WETSPRING_TCP_ADDR`) and handles
//! JSON-RPC 2.0 requests for science capabilities (diversity, QS model,
//! NCBI fetch, Anderson). Registers with Songbird for capability-based
//! discovery when available.
//!
//! # Usage
//!
//! ```text
//! wetspring-server [--help | --version | serve]
//! ```
//!
//! # Environment
//!
//! - `WETSPRING_SOCKET` — Override the default socket path
//! - `WETSPRING_TCP_ADDR` — Bind TCP instead of Unix socket (e.g. `127.0.0.1:9800`)
//! - `SONGBIRD_SOCKET` — Override Songbird discovery socket

use wetspring_barracuda::ipc::{Server, songbird};

const PRIMAL: &str = wetspring_barracuda::PRIMAL_NAME;
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_version() {
    println!("{PRIMAL} {VERSION}");
}

fn print_help() {
    println!("{PRIMAL} {VERSION} — biomeOS science primal (BarraCuda-powered)");
    println!();
    println!("USAGE:");
    println!("  wetspring-server [COMMAND]");
    println!();
    println!("COMMANDS:");
    println!("  serve       Start the IPC server (default)");
    println!("  version     Print version and exit");
    println!("  help        Print this help and exit");
    println!();
    println!("OPTIONS:");
    println!("  --help, -h       Print help");
    println!("  --version, -V    Print version");
    println!();
    println!("ENVIRONMENT:");
    println!("  WETSPRING_SOCKET     Override Unix socket path");
    println!("  WETSPRING_TCP_ADDR   Bind TCP (e.g. 127.0.0.1:9800)");
    println!("  SONGBIRD_SOCKET      Override Songbird socket");
    println!();
    println!("CAPABILITIES:");
    println!("  health.check, science.diversity, science.anderson,");
    println!("  science.qs_model, science.ncbi_fetch, science.full_pipeline,");
    println!("  metrics.snapshot");
}

fn run_server() {
    eprintln!("{PRIMAL} v{VERSION}");
    eprintln!("  Science primal for biomeOS — BarraCuda-powered");

    let server = match Server::bind_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FATAL: cannot bind socket: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("  Socket: {}", server.socket_path().display());

    let _heartbeat = songbird::discover_socket().map_or_else(
        || {
            eprintln!("  Songbird: not found (standalone mode)");
            None
        },
        |songbird_socket| {
            eprintln!("  Songbird: {}", songbird_socket.display());
            Some(songbird::start_heartbeat_loop(
                songbird_socket,
                server.socket_path().to_path_buf(),
            ))
        },
    );

    eprintln!(
        "  Capabilities: health.check, science.{{diversity,anderson,qs_model,ncbi_fetch,full_pipeline}}"
    );
    eprintln!("  Ready.");

    server.run();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => print_help(),
        Some("--version" | "-V" | "version") => print_version(),
        Some("serve") | None => run_server(),
        Some(unknown) => {
            eprintln!("error: unknown command '{unknown}'");
            eprintln!("Run '{PRIMAL} --help' for usage.");
            std::process::exit(2);
        }
    }
}
