// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! wetSpring — biomeOS science primal (UniBin).
//!
//! Listens on a Unix socket (or TCP via `WETSPRING_TCP_ADDR`) and handles
//! JSON-RPC 2.0 requests for science capabilities (diversity, QS model,
//! NCBI fetch, Anderson). Registers with Songbird for capability-based
//! discovery when available.
//!
//! # Usage
//!
//! ```text
//! wetspring <server|status|version|help>
//! ```
//!
//! # Environment
//!
//! - `WETSPRING_SOCKET` — Override the default socket path
//! - `WETSPRING_TCP_ADDR` — Bind TCP instead of Unix socket (e.g. `127.0.0.1:9800`)
//! - `SONGBIRD_SOCKET` — Override Songbird discovery socket

use wetspring_barracuda::ipc::handlers::CAPABILITIES;
use wetspring_barracuda::ipc::{Server, discover, songbird};

const PRIMAL: &str = wetspring_barracuda::PRIMAL_NAME;
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_version() {
    println!("{PRIMAL} {VERSION}");
}

fn print_help() {
    println!("{PRIMAL} {VERSION} — biomeOS science primal (BarraCuda-powered)");
    println!();
    println!("USAGE:");
    println!("  wetspring <COMMAND>");
    println!();
    println!("COMMANDS:");
    println!("  server      Start the IPC server (default)");
    println!("  status      Report health, capabilities, and socket info");
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
    for cap in CAPABILITIES {
        println!("  {cap}");
    }
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

    eprintln!("  Capabilities: {} methods registered", CAPABILITIES.len());
    eprintln!("  Ready.");

    server.run();
}

fn run_status() {
    println!("{PRIMAL} {VERSION}");
    println!();

    let bind_path = discover::resolve_bind_path("WETSPRING_SOCKET", PRIMAL);
    let socket_exists = bind_path.exists();
    println!("Socket:  {}", bind_path.display());
    println!("Exists:  {socket_exists}");
    println!();

    let songbird_status = songbird::discover_socket()
        .map_or_else(|| "not found".to_owned(), |p| p.display().to_string());
    println!("Songbird: {songbird_status}");
    println!();

    println!("Capabilities ({}):", CAPABILITIES.len());
    for cap in CAPABILITIES {
        println!("  {cap}");
    }
    println!();

    if socket_exists {
        println!("Status: RUNNING (socket present)");
    } else {
        println!("Status: STOPPED (no socket)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => print_help(),
        Some("--version" | "-V" | "version") => print_version(),
        Some("status") => run_status(),
        Some("server") | None => run_server(),
        Some(unknown) => {
            eprintln!("error: unknown command '{unknown}'");
            eprintln!("Run '{PRIMAL} help' for usage.");
            std::process::exit(2);
        }
    }
}
