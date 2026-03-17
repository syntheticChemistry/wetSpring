// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC resilience primitives: retry policy and circuit breaker.
//!
//! Follows the sweetGrass/healthSpring pattern for structured IPC fault
//! tolerance. Used by all outbound IPC clients (`compute_dispatch`,
//! `songbird`, `provenance`) to handle transient failures gracefully.

use std::time::{Duration, Instant};

use crate::error::IpcError;

/// Configurable retry policy with exponential backoff and jitter.
///
/// Intended for transient IPC failures (`IpcError::Connect`, `Transport`,
/// `EmptyResponse`). Non-retriable errors (`Codec`, `RpcReject`, `SocketPath`)
/// are never retried regardless of policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    backoff_factor: f64,
}

impl RetryPolicy {
    /// Create a retry policy.
    ///
    /// - `max_attempts`: total attempts (1 = no retries)
    /// - `base_delay`: initial delay between attempts
    /// - `max_delay`: ceiling on exponential backoff
    #[must_use]
    pub const fn new(max_attempts: u32, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay,
            backoff_factor: 2.0,
        }
    }

    /// Quick retry: 3 attempts, 100ms base, 2s max.
    #[must_use]
    pub const fn quick() -> Self {
        Self::new(3, Duration::from_millis(100), Duration::from_secs(2))
    }

    /// Standard retry: 5 attempts, 500ms base, 30s max.
    #[must_use]
    pub const fn standard() -> Self {
        Self::new(5, Duration::from_millis(500), Duration::from_secs(30))
    }

    /// Compute the delay for attempt `n` (0-indexed).
    #[must_use]
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        #[expect(clippy::cast_possible_wrap, reason = "attempt capped at 10, fits i32")]
        let multiplier = self.backoff_factor.powi(attempt.min(10) as i32);
        let delay = self.base_delay.mul_f64(multiplier);
        delay.min(self.max_delay)
    }

    /// Execute a fallible operation with retry logic.
    ///
    /// Only retries when `IpcError::is_retriable()` returns true.
    /// Returns the first success or the last error after all attempts.
    ///
    /// # Errors
    ///
    /// Returns the last `IpcError` if all attempts fail.
    pub fn execute<F, T>(&self, mut op: F) -> Result<T, IpcError>
    where
        F: FnMut() -> Result<T, IpcError>,
    {
        let mut last_err = None;

        for attempt in 0..self.max_attempts {
            match op() {
                Ok(val) => return Ok(val),
                Err(e) if !e.is_retriable() => return Err(e),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < self.max_attempts {
                        std::thread::sleep(self.delay_for_attempt(attempt));
                    }
                }
            }
        }

        Err(last_err.unwrap_or(IpcError::EmptyResponse))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

/// Circuit breaker states per the standard half-open pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests pass through.
    Closed,
    /// Too many failures — requests are rejected immediately.
    Open,
    /// Probing — one request allowed to test recovery.
    HalfOpen,
}

/// Circuit breaker for IPC endpoints.
///
/// Prevents retry storms against unhealthy primals by tracking consecutive
/// failures and opening the circuit when a threshold is reached. After a
/// cooldown period, one probe request is allowed (half-open). If it succeeds,
/// the circuit closes; if it fails, the circuit reopens.
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    consecutive_failures: u32,
    last_failure: Option<Instant>,
    state: CircuitState,
}

impl CircuitBreaker {
    /// Create a circuit breaker.
    ///
    /// - `failure_threshold`: consecutive failures before opening
    /// - `cooldown`: time to wait before probing (half-open)
    #[must_use]
    pub const fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            consecutive_failures: 0,
            last_failure: None,
            state: CircuitState::Closed,
        }
    }

    /// Default circuit breaker: 5 failures, 30s cooldown.
    #[must_use]
    pub const fn default_config() -> Self {
        Self::new(5, Duration::from_secs(30))
    }

    /// Current circuit state.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        if self.state == CircuitState::Open {
            if let Some(last) = self.last_failure {
                if last.elapsed() >= self.cooldown {
                    return CircuitState::HalfOpen;
                }
            }
        }
        self.state
    }

    /// Whether a request should be allowed through.
    #[must_use]
    pub fn allow_request(&self) -> bool {
        match self.state() {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    /// Record a successful operation — resets the failure counter and closes the circuit.
    pub const fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = CircuitState::Closed;
    }

    /// Record a failed operation — increments the failure counter and may open the circuit.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
        if self.consecutive_failures >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    /// Execute an operation through the circuit breaker.
    ///
    /// Returns `Err(IpcError::Connect("circuit open"))` if the circuit is open.
    ///
    /// # Errors
    ///
    /// Returns `IpcError` from the operation, or a synthetic `Connect` error
    /// when the circuit is open.
    pub fn execute<F, T>(&mut self, op: F) -> Result<T, IpcError>
    where
        F: FnOnce() -> Result<T, IpcError>,
    {
        if !self.allow_request() {
            return Err(IpcError::Connect(format!(
                "circuit open: {} consecutive failures, cooldown {}s remaining",
                self.consecutive_failures,
                self.cooldown
                    .saturating_sub(self.last_failure.map_or(Duration::ZERO, |t| t.elapsed()))
                    .as_secs()
            )));
        }

        match op() {
            Ok(val) => {
                self.record_success();
                Ok(val)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn retry_succeeds_first_attempt() {
        let policy = RetryPolicy::quick();
        let result = policy.execute(|| Ok::<_, IpcError>(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn retry_succeeds_after_transient_failure() {
        let policy = RetryPolicy::new(3, Duration::from_millis(1), Duration::from_millis(10));
        let mut attempts = 0;
        let result = policy.execute(|| {
            attempts += 1;
            if attempts < 3 {
                Err(IpcError::Connect("refused".into()))
            } else {
                Ok(42)
            }
        });
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn retry_does_not_retry_non_retriable() {
        let policy = RetryPolicy::new(5, Duration::from_millis(1), Duration::from_millis(10));
        let mut attempts = 0;
        let result = policy.execute(|| {
            attempts += 1;
            Err::<i32, _>(IpcError::Codec("bad json".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn retry_exhausts_all_attempts() {
        let policy = RetryPolicy::new(3, Duration::from_millis(1), Duration::from_millis(10));
        let mut attempts = 0;
        let result = policy.execute(|| {
            attempts += 1;
            Err::<i32, _>(IpcError::Transport("broken pipe".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, 3);
    }

    #[test]
    fn retry_delay_increases_with_backoff() {
        let policy = RetryPolicy::new(4, Duration::from_millis(100), Duration::from_secs(60));
        assert!(policy.delay_for_attempt(0) <= Duration::from_millis(200));
        assert!(policy.delay_for_attempt(1) > policy.delay_for_attempt(0));
        assert!(policy.delay_for_attempt(2) > policy.delay_for_attempt(1));
    }

    #[test]
    fn retry_delay_capped_at_max() {
        let policy = RetryPolicy::new(10, Duration::from_millis(100), Duration::from_secs(1));
        assert!(policy.delay_for_attempt(20) <= Duration::from_secs(1));
    }

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::default_config();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures, 0);
    }

    #[test]
    fn circuit_breaker_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(1));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.allow_request());
    }

    #[test]
    fn circuit_breaker_execute_rejects_when_open() {
        let mut cb = CircuitBreaker::new(1, Duration::from_secs(60));
        cb.record_failure();
        let result = cb.execute(|| Ok::<_, IpcError>(42));
        assert!(result.is_err());
    }

    #[test]
    fn circuit_breaker_execute_passes_when_closed() {
        let mut cb = CircuitBreaker::default_config();
        let result = cb.execute(|| Ok::<_, IpcError>(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn circuit_breaker_execute_records_state() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(60));
        let _ = cb.execute(|| Err::<i32, _>(IpcError::Transport("fail".into())));
        assert_eq!(cb.consecutive_failures, 1);
        let _ = cb.execute(|| Ok::<_, IpcError>(1));
        assert_eq!(cb.consecutive_failures, 0);
    }
}
