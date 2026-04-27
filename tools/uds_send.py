#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# uds_send.py — Minimal Unix domain socket send/receive for JSON-RPC.
# Drop-in replacement for `socat - UNIX-CONNECT:$sock` when socat is
# unavailable. Reads stdin, sends to socket, prints response to stdout.
#
# Usage: echo '{"jsonrpc":"2.0",...}' | python3 uds_send.py /path/to.sock

import socket
import sys

def main():
    if len(sys.argv) < 2:
        print("usage: uds_send.py <socket_path>", file=sys.stderr)
        sys.exit(1)

    sock_path = sys.argv[1]
    payload = sys.stdin.read().strip()
    if not payload:
        sys.exit(0)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(5)
        s.connect(sock_path)
        s.sendall((payload + "\n").encode())
        chunks = []
        while True:
            try:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            except socket.timeout:
                break
        if chunks:
            print(b"".join(chunks).decode().strip())
    except (ConnectionRefusedError, FileNotFoundError, ConnectionResetError):
        pass
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
    finally:
        s.close()

if __name__ == "__main__":
    main()
