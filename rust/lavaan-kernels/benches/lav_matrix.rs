use lavaan_kernels::{
    delta_a_delta,
    helpers::{
        benchmark_latency_stats, print_benchmark_summary_table, seeded_delta_a_delta_inputs,
        seeded_diag, seeded_square_matrix, BenchmarkTableRow,
    },
    lav_matrix_antidiag_idx, lav_matrix_diag_prepost, lav_matrix_diagh_idx,
};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug)]
struct BenchmarkSummary {
    label: String,
    iterations: usize,
    total_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    max_ms: f64,
    checksum: f64,
}

impl BenchmarkSummary {
    fn cells(&self) -> [String; 7] {
        [
            self.label.clone(),
            self.iterations.to_string(),
            format!("{:.3}", self.total_ms),
            format!("{:.3}", self.median_ms),
            format!("{:.3}", self.mean_ms),
            format!("{:.3}", self.max_ms),
            format!("{:.6}", self.checksum),
        ]
    }
}

impl BenchmarkTableRow for BenchmarkSummary {
    fn benchmark_table_cells(&self) -> [String; 7] {
        self.cells()
    }
}

fn run_benchmark<F>(label: &str, iterations: usize, mut benchmark_fn: F) -> BenchmarkSummary
where
    F: FnMut() -> f64,
{
    let mut iterations = iterations;
    if iterations == 0 {
        iterations = 1;
    }

    let mut samples = Vec::with_capacity(iterations);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        let contribution = benchmark_fn();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        checksum += contribution;
    }

    let benchmark_stats = benchmark_latency_stats(&samples);

    BenchmarkSummary {
        label: label.to_string(),
        iterations,
        total_ms: benchmark_stats.total_ms,
        mean_ms: benchmark_stats.mean_ms,
        median_ms: benchmark_stats.median_ms,
        max_ms: benchmark_stats.max_ms,
        checksum,
    }
}

fn benchmark_iterations_for_size(m: usize, n: usize) -> usize {
    let work = m.saturating_mul(n).max(1);
    (40_000 / work).clamp(500, 10_000)
}

fn bench_delta_a_delta() -> Vec<BenchmarkSummary> {
    let sizes: &[(usize, usize)] = &[
        (8, 2),
        (16, 4),
        (24, 6),
        (32, 8),
        (48, 12),
        (64, 16),
        (96, 24),
        (128, 32),
        (160, 40),
    ];

    sizes
        .iter()
        .map(|&(m, n)| {
            let (delta, a1) = seeded_delta_a_delta_inputs(m, n);
            let iterations = benchmark_iterations_for_size(m, n);

            run_benchmark(&format!("delta_a_delta {m}x{n}"), iterations, || {
                let result = delta_a_delta(black_box(&delta), black_box(&a1), m, n)
                    .expect("benchmark input dimensions should be valid");
                result[0]
            })
        })
        .collect()
}

fn bench_lav_matrix_diag_prepost() -> BenchmarkSummary {
    let dim: usize = 128;
    let iterations: usize = 10_000;
    let a = seeded_square_matrix(dim);
    let d = seeded_diag(dim);

    run_benchmark(&format!("diag_prepost {dim}"), iterations, || {
        let result = lav_matrix_diag_prepost(black_box(&a), black_box(&d));
        result[(0, 0)] + result[(dim - 1, dim - 1)]
    })
}

fn bench_lav_matrix_diagh_idx() -> BenchmarkSummary {
    let n: usize = 100;
    let iterations: usize = 10_000;

    run_benchmark(&format!("diagh_idx {n}"), iterations, || {
        let result = lav_matrix_diagh_idx(black_box(n));
        result.len() as f64
    })
}

fn bench_lav_matrix_antidiag_idx() -> BenchmarkSummary {
    let n: usize = 1_000;
    let iterations: usize = 10_000;

    run_benchmark(&format!("antidiag_idx {n}"), iterations, || {
        let result = lav_matrix_antidiag_idx(black_box(n));
        result.len() as f64
    })
}

fn main() {
    let delta_rows = bench_delta_a_delta();
    let helper_rows = vec![
        bench_lav_matrix_diag_prepost(),
        bench_lav_matrix_diagh_idx(),
        bench_lav_matrix_antidiag_idx(),
    ];

    print_benchmark_summary_table("delta_a_delta sweep", &delta_rows);
    print_benchmark_summary_table("other kernels", &helper_rows);
}
