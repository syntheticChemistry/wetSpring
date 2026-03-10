// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-discovered HTTP transport for NCBI queries.
//!
//! HTTP GET uses a capability-based transport chain — the first available
//! backend wins:
//!
//! 1. **`WETSPRING_HTTP_CMD`** — user-supplied command (e.g. `wget -qO-`)
//! 2. **System `curl`** — sovereign HTTPS without TLS crate deps
//! 3. **System `wget`** — common fallback on minimal containers
//!
//! # Evolution path
//!
//! | Phase | Strategy | Status |
//! |-------|----------|--------|
//! | Current | Capability-discovered system HTTP — zero compile deps | active |
//! | Phase 2 | metalForge HTTP substrate — route through forge dispatch | blocked on forge HTTP |
//! | Phase 3 | Sovereign Rust TLS (if HTTPS becomes pipeline-critical) | not needed for validation |

/// Discovered HTTP transport backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Custom,
    Curl,
    Wget,
}

/// Discover an available HTTP GET backend at runtime.
///
/// Returns the backend kind and the command name/path. Checks:
/// 1. `WETSPRING_HTTP_CMD` environment variable
/// 2. `curl` on `$PATH`
/// 3. `wget` on `$PATH`
fn discover_backend() -> Option<(Backend, String)> {
    let custom = std::env::var("WETSPRING_HTTP_CMD").ok();
    select_backend(
        custom.as_deref(),
        which_exists("curl"),
        which_exists("wget"),
    )
}

/// Pure-logic backend selection — no env or filesystem access.
fn select_backend(
    custom_cmd: Option<&str>,
    has_curl: bool,
    has_wget: bool,
) -> Option<(Backend, String)> {
    if let Some(cmd) = custom_cmd {
        if !cmd.trim().is_empty() {
            return Some((Backend::Custom, cmd.to_string()));
        }
    }

    if has_curl {
        return Some((Backend::Curl, "curl".to_string()));
    }

    if has_wget {
        return Some((Backend::Wget, "wget".to_string()));
    }

    None
}

/// Check whether a command exists on `$PATH` — pure Rust, no subprocess.
fn which_exists(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(cmd).is_file()))
}

use crate::error::Error;

/// HTTP GET via capability-discovered system transport.
///
/// Discovers the best available HTTP backend at runtime and uses it.
/// Returns the response body as a `String`.
///
/// # Errors
///
/// Returns `Err` if no HTTP transport is available, the request fails,
/// times out (30 s), or the response contains invalid UTF-8.
#[must_use = "HTTP response body is discarded if not used"]
pub fn get(url: &str) -> crate::error::Result<String> {
    let (backend, cmd) = discover_backend().ok_or_else(|| {
        Error::Ncbi(
            "no HTTP transport available (need curl or wget on PATH, or set WETSPRING_HTTP_CMD)"
                .to_string(),
        )
    })?;

    let output = match backend {
        Backend::Custom => {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            let (program, args) = parts
                .split_first()
                .ok_or_else(|| Error::Ncbi("WETSPRING_HTTP_CMD is empty".to_string()))?;
            let mut command = std::process::Command::new(program);
            command.args(args);
            command.arg(url);
            command
                .output()
                .map_err(|e| Error::Ncbi(format!("{cmd}: {e}")))?
        }
        Backend::Curl => std::process::Command::new("curl")
            .args(["-sfS", "-m", "30", url])
            .output()
            .map_err(|e| Error::Ncbi(format!("curl: {e}")))?,
        Backend::Wget => std::process::Command::new("wget")
            .args(["-qO-", "--timeout=30", url])
            .output()
            .map_err(|e| Error::Ncbi(format!("wget: {e}")))?,
    };

    interpret_output(output, &cmd)
}

/// Interpret the output of an HTTP subprocess.
///
/// Takes ownership to avoid cloning `stdout` on the success path.
fn interpret_output(output: std::process::Output, cmd: &str) -> crate::error::Result<String> {
    if output.status.success() {
        String::from_utf8(output.stdout).map_err(|e| Error::Ncbi(e.to_string()))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let preview = &stderr[..stderr.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN)];
        Err(Error::Ncbi(format!(
            "{cmd} failed (exit {:?}): {preview}",
            output.status.code()
        )))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_finds_curl_or_wget() {
        let result = discover_backend();
        if which_exists("curl") {
            let (backend, _) = result.unwrap();
            assert_eq!(backend, Backend::Curl);
        } else if which_exists("wget") {
            let (backend, _) = result.unwrap();
            assert_eq!(backend, Backend::Wget);
        }
    }

    #[test]
    fn which_exists_finds_sh() {
        assert!(which_exists("sh"));
    }

    #[test]
    fn which_exists_rejects_nonexistent() {
        assert!(!which_exists("__wetspring_nonexistent_binary_xyz__"));
    }

    #[test]
    fn get_localhost_refused() {
        let result = get("http://127.0.0.1:1/nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn which_exists_finds_ls() {
        assert!(which_exists("ls"));
    }

    #[test]
    fn which_exists_finds_echo() {
        assert!(which_exists("echo"));
    }

    #[test]
    fn select_backend_custom_wins() {
        let (backend, cmd) = select_backend(Some("myhttp --get"), true, true).unwrap();
        assert_eq!(backend, Backend::Custom);
        assert_eq!(cmd, "myhttp --get");
    }

    #[test]
    fn select_backend_empty_custom_ignored() {
        let (backend, _) = select_backend(Some(""), true, false).unwrap();
        assert_eq!(backend, Backend::Curl);
    }

    #[test]
    fn select_backend_curl_over_wget() {
        let (backend, _) = select_backend(None, true, true).unwrap();
        assert_eq!(backend, Backend::Curl);
    }

    #[test]
    fn select_backend_wget_fallback() {
        let (backend, _) = select_backend(None, false, true).unwrap();
        assert_eq!(backend, Backend::Wget);
    }

    #[test]
    fn select_backend_none_when_nothing() {
        assert!(select_backend(None, false, false).is_none());
    }

    #[test]
    fn select_backend_custom_only_no_system() {
        let (backend, cmd) = select_backend(Some("wget2"), false, false).unwrap();
        assert_eq!(backend, Backend::Custom);
        assert_eq!(cmd, "wget2");
    }

    #[test]
    fn interpret_output_success_returns_body() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: b"hello world".to_vec(),
            stderr: vec![],
        };
        let result = interpret_output(output, "test-cmd");
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn interpret_output_failure_returns_stderr_preview() {
        let output = std::process::Output {
            status: std::process::Command::new("false").status().unwrap(),
            stdout: vec![],
            stderr: b"connection refused".to_vec(),
        };
        let result = interpret_output(output, "curl");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("curl"));
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn interpret_output_failure_truncates_long_stderr() {
        let output = std::process::Output {
            status: std::process::Command::new("false").status().unwrap(),
            stdout: vec![],
            stderr: "x".repeat(500).into_bytes(),
        };
        let result = interpret_output(output, "wget");
        let err = result.unwrap_err();
        assert!(err.to_string().len() < 500);
    }

    #[test]
    fn interpret_output_success_empty_body() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: vec![],
            stderr: vec![],
        };
        assert_eq!(interpret_output(output, "cmd").unwrap(), "");
    }

    #[test]
    fn interpret_output_invalid_utf8() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: vec![0xFF, 0xFE],
            stderr: vec![],
        };
        assert!(interpret_output(output, "cmd").is_err());
    }

    #[test]
    fn select_backend_whitespace_only_custom_ignored() {
        assert!(select_backend(Some("   "), false, false).is_none());
    }

    #[test]
    fn select_backend_custom_with_args() {
        let (backend, cmd) = select_backend(Some("myhttp --silent -L"), false, false).unwrap();
        assert_eq!(backend, Backend::Custom);
        assert_eq!(cmd, "myhttp --silent -L");
    }

    #[test]
    #[cfg(unix)]
    fn interpret_output_failure_exit_code_none() {
        use std::os::unix::process::ExitStatusExt;
        let output = std::process::Output {
            status: std::process::ExitStatus::from_raw(0x80),
            stdout: vec![],
            stderr: b"killed by signal".to_vec(),
        };
        let result = interpret_output(output, "curl");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("curl"));
        assert!(msg.contains("killed") || msg.contains("exit"));
    }

    #[test]
    fn interpret_output_success_unicode_body() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: "café résumé".as_bytes().to_vec(),
            stderr: vec![],
        };
        let result = interpret_output(output, "cmd");
        assert_eq!(result.unwrap(), "café résumé");
    }

    #[test]
    fn select_backend_trimmed_custom() {
        let (backend, cmd) = select_backend(Some("  curl  "), false, false).unwrap();
        assert_eq!(backend, Backend::Custom);
        assert!(!cmd.is_empty());
    }

    #[test]
    fn select_backend_empty_after_trim_ignored() {
        let result = select_backend(Some("   "), true, false);
        assert!(result.is_some());
        let (backend, _) = result.unwrap();
        assert_eq!(backend, Backend::Curl);
    }
}
