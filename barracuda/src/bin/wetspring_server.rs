// SPDX-License-Identifier: AGPL-3.0-or-later
//! wetSpring IPC server — biomeOS science primal.
//!
//! Listens on a Unix socket and handles JSON-RPC 2.0 requests for
//! science capabilities (diversity, QS model, NCBI fetch, Anderson).
//! Registers with Songbird for capability-based discovery when available.
//!
//! # Usage
//!
//! ```text
//! cargo run --features ipc --bin wetspring_server
//! ```
//!
//! # Environment
//!
//! - `WETSPRING_SOCKET` — Override the default socket path
//! - `SONGBIRD_SOCKET` — Override Songbird discovery socket

use wetspring_barracuda::ipc::{Server, songbird};

fn main() {
    eprintln!("wetspring-server v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Science primal for biomeOS — BarraCuda-powered");

    let server = match Server::bind_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FATAL: cannot bind socket: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("  Socket: {}", server.socket_path().display());

    // Songbird registration (non-fatal — standalone mode if unavailable)
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
