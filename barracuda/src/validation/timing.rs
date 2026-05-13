// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark timing helpers for validation binaries.

/// Run a closure and return `(result, elapsed_ms)`.
#[must_use]
pub fn bench<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    (result, ms)
}

/// Run a closure and return `(result, elapsed_microseconds)`.
#[must_use]
pub fn timed_us<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    #[expect(
        clippy::cast_precision_loss,
        reason = "precision: microsecond timing fits f64"
    )]
    let us = start.elapsed().as_micros() as f64;
    (result, us)
}

/// Run a closure `n` times and return `(last_result, mean_elapsed_microseconds)`.
///
/// Useful for amortized micro-benchmarks where a single run is too fast to measure.
#[must_use]
pub fn bench_n_us<T>(n: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let mut result = f();
    for _ in 1..n {
        result = f();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "precision: microsecond timing and iteration counts fit f64"
    )]
    let mean_us = start.elapsed().as_micros() as f64 / n as f64;
    (result, mean_us)
}

/// Print a timing summary table with columns: Name, CPU µs, GPU µs, Status.
pub fn print_timing_table(rows: &[(&str, f64, f64, &str)]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "\n  {:40} {:>12} {:>12} {:>8}",
        "Domain", "CPU (µs)", "GPU (µs)", "Status"
    );
    let _ = writeln!(out, "  {}", "─".repeat(76));
    for &(name, cpu, gpu, status) in rows {
        let _ = writeln!(out, "  {name:40} {cpu:12.0} {gpu:12.0} {status:>8}");
    }
}

/// Per-primitive benchmark row with label, origin, and elapsed time.
pub struct BenchRow {
    /// Primitive or operation name.
    pub label: &'static str,
    /// Provenance origin (e.g. "wetSpring→S58").
    pub origin: &'static str,
    /// Elapsed time in milliseconds.
    pub ms: f64,
}

/// Run a closure, print elapsed time, and return `(result, elapsed_ms)`.
///
/// Combines `bench` with stdout reporting — replaces per-binary `bench` helpers.
pub fn bench_print<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    use std::io::Write as _;
    let (result, ms) = bench(f);
    let _ = writeln!(std::io::stdout().lock(), "  {label}: {ms:.3} ms");
    (result, ms)
}

/// Per-primitive benchmark row with provenance evolution tracking.
///
/// Extends [`BenchRow`] with a `evolved` field for cross-spring evolution
/// narratives that track which session evolved each primitive.
pub struct BenchRowEvolved {
    /// Primitive or operation name.
    pub label: &'static str,
    /// Provenance origin (e.g. "wetSpring→S87").
    pub origin: &'static str,
    /// Evolution session that introduced this primitive.
    pub evolved: &'static str,
    /// Elapsed time in milliseconds.
    pub ms: f64,
}

/// Print a four-column box-drawing table from [`BenchRowEvolved`] data.
pub fn print_bench_evolved_table(rows: &[BenchRowEvolved]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  ┌────────────────────────────────┬────────────────────────┬────────────────┬──────────┐"
    );
    let _ = writeln!(
        out,
        "  │ Primitive                      │ Origin                 │ Evolved        │ Time     │"
    );
    let _ = writeln!(
        out,
        "  ├────────────────────────────────┼────────────────────────┼────────────────┼──────────┤"
    );
    for t in rows {
        let _ = writeln!(
            out,
            "  │ {:30} │ {:22} │ {:14} │ {:7.3}ms│",
            t.label, t.origin, t.evolved, t.ms
        );
    }
    let _ = writeln!(
        out,
        "  └────────────────────────────────┴────────────────────────┴────────────────┴──────────┘"
    );
}

/// Print a key-value summary box with box-drawing characters.
///
/// Avoids copy-paste of `╔═╗` / `║ metric │ value ║` blocks across
/// cross-spring validation binaries.
pub fn print_kv_box(title: &str, rows: &[(&str, &str)]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let width = 60;
    let _ = writeln!(out, "  ╔{}╗", "═".repeat(width));
    let _ = writeln!(out, "  ║ {title:^w$} ║", w = width - 2);
    let _ = writeln!(out, "  ╠{}╣", "═".repeat(width));
    for &(k, v) in rows {
        let _ = writeln!(out, "  ║ {k:24} │ {v:>w$} ║", w = width - 28);
    }
    let _ = writeln!(out, "  ╚{}╝", "═".repeat(width));
}

/// Per-domain CPU vs GPU timing comparison row.
///
/// Used by `validate_cpu_vs_gpu_*` and `benchmark_cross_spring_*` binaries.
pub struct CpuGpuRow {
    /// Domain or primitive name.
    pub name: &'static str,
    /// CPU elapsed time in microseconds.
    pub cpu_us: f64,
    /// GPU elapsed time in microseconds.
    pub gpu_us: f64,
    /// Pass/fail status label.
    pub status: &'static str,
}

/// Per-primitive cross-spring benchmark entry with provenance.
///
/// Used by `validate_cross_spring_*` binaries to track which spring
/// evolved each primitive and when.
pub struct CrossSpringEntry {
    /// Primitive or operation name.
    pub primitive: &'static str,
    /// Spring that evolved the primitive.
    pub evolved_by: &'static str,
    /// `ToadStool` session where it was absorbed.
    pub session: &'static str,
    /// CPU elapsed time in microseconds.
    pub cpu_us: f64,
    /// Problem description (e.g. "10×10 lattice").
    pub problem: &'static str,
    /// Number of validation checks performed.
    pub checks: u32,
}

/// Print a CPU-vs-GPU timing table from [`CpuGpuRow`] data.
pub fn print_cpu_gpu_table(rows: &[CpuGpuRow]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "\n  {:40} {:>12} {:>12} {:>8}",
        "Domain", "CPU (µs)", "GPU (µs)", "Status"
    );
    let _ = writeln!(out, "  {}", "─".repeat(76));
    for r in rows {
        let _ = writeln!(
            out,
            "  {:40} {:12.0} {:12.0} {:>8}",
            r.name, r.cpu_us, r.gpu_us, r.status
        );
    }
}

/// Print a cross-spring evolution summary table from [`CrossSpringEntry`] data.
pub fn print_cross_spring_table(rows: &[CrossSpringEntry]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  ┌──────────────────────────────┬──────────────┬────────┬──────────┬────────────────┬───────┐"
    );
    let _ = writeln!(
        out,
        "  │ Primitive                    │ Evolved By   │ Sess   │ CPU (µs) │ Problem        │ Chks  │"
    );
    let _ = writeln!(
        out,
        "  ├──────────────────────────────┼──────────────┼────────┼──────────┼────────────────┼───────┤"
    );
    for r in rows {
        let _ = writeln!(
            out,
            "  │ {:28} │ {:12} │ {:6} │ {:8.0} │ {:14} │ {:5} │",
            r.primitive, r.evolved_by, r.session, r.cpu_us, r.problem, r.checks
        );
    }
    let _ = writeln!(
        out,
        "  └──────────────────────────────┴──────────────┴────────┴──────────┴────────────────┴───────┘"
    );
}

/// Print a three-column box-drawing timing table from [`BenchRow`] data.
pub fn print_bench_table(rows: &[BenchRow]) {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  ┌────────────────────────────────┬────────────────────────┬──────────┐"
    );
    let _ = writeln!(
        out,
        "  │ Primitive                      │ Origin                 │ Time     │"
    );
    let _ = writeln!(
        out,
        "  ├────────────────────────────────┼────────────────────────┼──────────┤"
    );
    for t in rows {
        let _ = writeln!(
            out,
            "  │ {:30} │ {:22} │ {:7.3}ms│",
            t.label, t.origin, t.ms
        );
    }
    let _ = writeln!(
        out,
        "  └────────────────────────────────┴────────────────────────┴──────────┘"
    );
}
