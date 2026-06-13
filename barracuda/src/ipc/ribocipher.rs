// SPDX-License-Identifier: AGPL-3.0-or-later
//! riboCipher transport signal standard — wetSpring implementation.
//!
//! The riboCipher standard replaces fragile peek-and-guess connection
//! identification with intentional transport signaling. Every new IPC
//! connection (UDS or TCP) MUST begin with a 2-byte signal frame:
//!
//! | Byte 0 | Byte 1 | Meaning |
//! |--------|--------|---------|
//! | 0xEC   | 0x01   | Clear-tier (plaintext JSON-RPC) |
//! | 0xED   | 0x01   | Mito-tier (HKDF+HMAC authenticated) |
//! | 0xEE   | 0x01   | Nuclear-tier (ChaCha20-Poly1305 encrypted) |
//!
//! Wave 112 policy: unsignalled connections produce ERROR logs (was WARN
//! in Wave 111). Connections are still accepted — rejection starts Wave 113.

use std::io::Read;
use std::os::unix::net::UnixStream;

/// riboCipher tier byte: clear (plaintext JSON-RPC).
pub const TIER_CLEAR: u8 = 0xEC;

/// riboCipher protocol version.
pub const VERSION: u8 = 0x01;

/// Complete clear-tier signal frame (sent by clients on connect).
pub const CLEAR_SIGNAL: [u8; 2] = [TIER_CLEAR, VERSION];

/// Result of attempting to read the riboCipher signal from a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalResult {
    /// Valid riboCipher signal detected and consumed.
    Valid {
        /// Signal tier byte (0xEC = clear, 0xED = mito, 0xEE = nuclear).
        tier: u8,
        /// Protocol version byte.
        version: u8,
    },
    /// No signal detected — first bytes are not riboCipher.
    /// The `peeked` bytes were consumed and must be replayed.
    Unsignalled {
        /// The 2 bytes that were consumed from the stream.
        peeked: [u8; 2],
    },
    /// Stream closed or errored before 2 bytes could be read.
    Incomplete,
}

/// Attempt to detect a riboCipher signal at the start of a connection.
///
/// Reads exactly 2 bytes from the stream. If they match a known riboCipher
/// tier+version, returns `Valid` (bytes consumed). Otherwise returns
/// `Unsignalled` with the consumed bytes so the caller can prepend them.
pub fn detect_signal(stream: &UnixStream) -> SignalResult {
    let mut buf = [0u8; 2];
    let mut reader: &UnixStream = stream;
    match reader.read_exact(&mut buf) {
        Ok(()) => {
            if buf[0] == TIER_CLEAR && buf[1] == VERSION {
                SignalResult::Valid {
                    tier: buf[0],
                    version: buf[1],
                }
            } else {
                SignalResult::Unsignalled { peeked: buf }
            }
        }
        Err(_) => SignalResult::Incomplete,
    }
}

/// Prepend the riboCipher clear signal to an outgoing connection.
///
/// Called by client transport functions immediately after connect, before
/// any JSON-RPC data. This is what primalSpring expects on all IPC.
///
/// # Errors
///
/// Returns `Err` if the write fails.
pub fn send_clear_signal(stream: &UnixStream) -> std::io::Result<()> {
    use std::io::Write;
    (&*stream).write_all(&CLEAR_SIGNAL)
}

/// Prepend the riboCipher clear signal to a TCP stream.
///
/// # Errors
///
/// Returns `Err` if the write fails.
pub fn send_clear_signal_tcp(stream: &std::net::TcpStream) -> std::io::Result<()> {
    use std::io::Write;
    (&*stream).write_all(&CLEAR_SIGNAL)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::net::UnixStream as UdsStream;

    #[test]
    fn detect_valid_clear_signal() {
        let (server, client) = UdsStream::pair().unwrap();
        (&client).write_all(&CLEAR_SIGNAL).unwrap();
        let result = detect_signal(&server);
        assert_eq!(
            result,
            SignalResult::Valid {
                tier: TIER_CLEAR,
                version: VERSION
            }
        );
    }

    #[test]
    fn detect_unsignalled_json() {
        let (server, client) = UdsStream::pair().unwrap();
        (&client).write_all(b"{\"jsonrpc\":\"2.0\"}").unwrap();
        let result = detect_signal(&server);
        assert!(matches!(result, SignalResult::Unsignalled { peeked } if peeked == [b'{', b'"']));
    }

    #[test]
    fn detect_incomplete_on_empty() {
        let (server, client) = UdsStream::pair().unwrap();
        drop(client);
        server
            .set_read_timeout(Some(std::time::Duration::from_millis(50)))
            .unwrap();
        let result = detect_signal(&server);
        assert_eq!(result, SignalResult::Incomplete);
    }

    #[test]
    fn send_clear_signal_writes_prefix() {
        let (server, client) = UdsStream::pair().unwrap();
        send_clear_signal(&client).unwrap();
        let mut buf = [0u8; 2];
        (&server).read_exact(&mut buf).unwrap();
        assert_eq!(buf, CLEAR_SIGNAL);
    }

    #[test]
    fn constants_match_ecosystem_standard() {
        assert_eq!(TIER_CLEAR, 0xEC);
        assert_eq!(VERSION, 0x01);
        assert_eq!(CLEAR_SIGNAL, [0xEC, 0x01]);
    }
}
