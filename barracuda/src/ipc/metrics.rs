// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structured metrics for Neural API pathway learning.
//!
//! Tracks per-capability execution time, success/failure counts, and
//! resource usage. Metrics feed biomeOS's pathway learner for
//! optimization of capability routing and substrate selection.
//!
//! All counters are lock-free atomics; per-method detail uses a `Mutex`
//! that is held only for microsecond-scale map updates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Thread-safe metrics collector for the wetSpring IPC server.
pub struct Metrics {
    /// Total calls received (success + error).
    pub total_calls: AtomicU64,
    /// Calls that returned a JSON-RPC result.
    pub success_count: AtomicU64,
    /// Calls that returned a JSON-RPC error.
    pub error_count: AtomicU64,
    /// Cumulative wall-clock time across all calls (microseconds).
    pub total_duration_us: AtomicU64,
    /// Per-method breakdown.
    method_timings: Mutex<HashMap<String, MethodMetrics>>,
}

/// Per-method timing and count data.
#[derive(Debug, Clone)]
struct MethodMetrics {
    calls: u64,
    successes: u64,
    errors: u64,
    total_us: u64,
    min_us: u64,
    max_us: u64,
}

impl Default for MethodMetrics {
    fn default() -> Self {
        Self {
            calls: 0,
            successes: 0,
            errors: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }
}

impl Metrics {
    /// Create a new, zeroed metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_duration_us: AtomicU64::new(0),
            method_timings: Mutex::new(HashMap::new()),
        }
    }

    /// Record a successful method call with its wall-clock duration.
    pub fn record_success(&self, method: &str, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_duration_us.fetch_add(us, Ordering::Relaxed);

        if let Ok(mut map) = self.method_timings.lock() {
            let entry = map.entry(method.to_string()).or_default();
            entry.calls += 1;
            entry.successes += 1;
            entry.total_us += us;
            if us < entry.min_us {
                entry.min_us = us;
            }
            if us > entry.max_us {
                entry.max_us = us;
            }
        }
    }

    /// Record a failed method call with its wall-clock duration.
    pub fn record_error(&self, method: &str, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.total_duration_us.fetch_add(us, Ordering::Relaxed);

        if let Ok(mut map) = self.method_timings.lock() {
            let entry = map.entry(method.to_string()).or_default();
            entry.calls += 1;
            entry.errors += 1;
            entry.total_us += us;
        }
    }

    /// Generate a structured metrics snapshot for Neural API reporting.
    ///
    /// The returned JSON follows the biomeOS primal metrics specification:
    /// per-method call counts, latency percentiles, and aggregate totals.
    #[must_use]
    pub fn snapshot(&self) -> serde_json::Value {
        let methods = self.method_timings.lock().map_or(
            serde_json::Value::Null,
            |map| {
                let mut methods = serde_json::Map::new();
                for (name, m) in &*map {
                    let avg_us = if m.calls > 0 {
                        m.total_us / m.calls
                    } else {
                        0
                    };
                    let min_us = if m.min_us == u64::MAX { 0 } else { m.min_us };
                    methods.insert(
                        name.clone(),
                        serde_json::json!({
                            "calls": m.calls,
                            "successes": m.successes,
                            "errors": m.errors,
                            "avg_us": avg_us,
                            "min_us": min_us,
                            "max_us": m.max_us,
                        }),
                    );
                }
                serde_json::Value::Object(methods)
            },
        );

        serde_json::json!({
            "primal": "wetspring",
            "total_calls": self.total_calls.load(Ordering::Relaxed),
            "success_count": self.success_count.load(Ordering::Relaxed),
            "error_count": self.error_count.load(Ordering::Relaxed),
            "total_duration_us": self.total_duration_us.load(Ordering::Relaxed),
            "methods": methods,
        })
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_zeroed() {
        let m = Metrics::new();
        assert_eq!(m.total_calls.load(Ordering::Relaxed), 0);
        assert_eq!(m.success_count.load(Ordering::Relaxed), 0);
        assert_eq!(m.error_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn record_success_increments() {
        let m = Metrics::new();
        m.record_success("health.check", Duration::from_micros(100));
        assert_eq!(m.total_calls.load(Ordering::Relaxed), 1);
        assert_eq!(m.success_count.load(Ordering::Relaxed), 1);
        assert_eq!(m.error_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn record_error_increments() {
        let m = Metrics::new();
        m.record_error("bogus.method", Duration::from_micros(50));
        assert_eq!(m.total_calls.load(Ordering::Relaxed), 1);
        assert_eq!(m.success_count.load(Ordering::Relaxed), 0);
        assert_eq!(m.error_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn snapshot_includes_method_detail() {
        let m = Metrics::new();
        m.record_success("science.diversity", Duration::from_micros(200));
        m.record_success("science.diversity", Duration::from_micros(400));
        m.record_error("science.diversity", Duration::from_micros(10));

        let snap = m.snapshot();
        let div = &snap["methods"]["science.diversity"];
        assert_eq!(div["calls"], 3);
        assert_eq!(div["successes"], 2);
        assert_eq!(div["errors"], 1);
        assert_eq!(div["min_us"], 200);
        assert_eq!(div["max_us"], 400);
    }

    #[test]
    fn snapshot_structure() {
        let m = Metrics::new();
        let snap = m.snapshot();
        assert_eq!(snap["primal"], "wetspring");
        assert_eq!(snap["total_calls"], 0);
        assert!(snap["methods"].is_object());
    }

    #[test]
    fn duration_accumulates() {
        let m = Metrics::new();
        m.record_success("a", Duration::from_micros(100));
        m.record_success("b", Duration::from_micros(200));
        assert_eq!(m.total_duration_us.load(Ordering::Relaxed), 300);
    }
}
