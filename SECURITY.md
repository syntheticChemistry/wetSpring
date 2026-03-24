# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| V137+   | Yes       |
| < V137  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in wetSpring, please report it
responsibly:

1. **Do NOT** open a public issue
2. Email: security@ecoprimals.org
3. Include: description, reproduction steps, impact assessment

We aim to acknowledge reports within 48 hours and provide a fix or
mitigation within 7 days for critical issues.

## Security Design

- `#![forbid(unsafe_code)]` enforced workspace-wide
- All cryptographic operations (vault module) use audited RustCrypto crates
- No C dependencies in default build (ecoBin compliant)
- Socket paths use platform-standard directories (XDG_RUNTIME_DIR)
- Consent-gated data access via Genomic Vault (ChaCha20-Poly1305 + Ed25519)
