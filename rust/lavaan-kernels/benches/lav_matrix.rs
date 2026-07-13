use lavaan_kernels::{
    delta_a_delta,
    helpers::{benchmark_latency_stats, seeded_delta_a_delta_inputs, seeded_diag, seeded_square_matrix},
    lav_matrix_diag_prepost, lav_matrix_diagh_idx, lav_matrix_antidiag_idx
};
use std::hint::black_box;
use std::time::Instant;

fn run_benchmark<F>(label: &str, iterations: usize, mut benchmark_fn: F)
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

    println!(
        "\n===RESULTS===\n{label} iterations={iterations} total_ms={total_ms:.3} checksum={checksum:.6}\nmin_ms={min_ms:.3} mean_ms={mean_ms:.3} median_ms={median_ms:.3} max_ms={max_ms:.3}\n===RESULTS===\n",
        total_ms = benchmark_stats.total_ms,
        min_ms = benchmark_stats.min_ms,
        mean_ms = benchmark_stats.mean_ms,
        median_ms = benchmark_stats.median_ms,
        max_ms = benchmark_stats.max_ms,
        checksum = checksum,
        iterations = iterations,
        label = label,
    );
}

fn bench_delta_a_delta() {
    let m: usize = 24;
    let n: usize = 12;
    let (delta, a1) = seeded_delta_a_delta_inputs(m, n);
    let iterations: usize = 10_000;

    run_benchmark("lav_matrix::delta_a_delta", iterations, || {
        let result = delta_a_delta(black_box(&delta), black_box(&a1), m, n)
            .expect("benchmark input dimensions should be valid");
        result[0]
    });
}

fn bench_lav_matrix_diag_prepost() {
    let dim: usize = 128;
    let iterations: usize = 10_000;
    let a = seeded_square_matrix(dim);
    let d = seeded_diag(dim);

    run_benchmark(
        &format!("lav_matrix::lav_matrix_diag_prepost dim={dim}"),
        iterations,
        || {
            let result = lav_matrix_diag_prepost(black_box(&a), black_box(&d));
            result[(0, 0)] + result[(dim - 1, dim - 1)]
        },
    );
}

fn bench_lav_matrix_diagh_idx() {
    let n: usize = 100;
    let iterations: usize = 10_000;

    run_benchmark(
        &format!("lav_matrix::lav_matrix_diagh_idx n={n}"),
        iterations,
        || {
            let result = lav_matrix_diagh_idx(black_box(n));
            result.len() as f64
        },
    );
}

fn bench_lav_matrix_antidiag_idx() {
    let n: usize = 1_000;
    let iterations: usize = 10_000;

    run_benchmark(
        &format!("lav_matrix::lav_matrix_antidiag_idx n={n}"),
        iterations,
        || {
            let result = lav_matrix_antidiag_idx(black_box(n));
            result.len() as f64
        },
    );
}

fn main() {
    bench_delta_a_delta();
    bench_lav_matrix_diag_prepost();
    bench_lav_matrix_diagh_idx();
    bench_lav_matrix_antidiag_idx();
}
