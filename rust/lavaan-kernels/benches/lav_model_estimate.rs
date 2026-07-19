use lavaan_kernels::{
    helpers::{benchmark_latency_stats, print_benchmark_summary_table, BenchmarkTableRow},
    lav_model_estimate_gradient_postprocess, lav_model_estimate_pack,
    lav_model_estimate_unpack_unscale,
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

impl BenchmarkTableRow for BenchmarkSummary {
    fn benchmark_table_cells(&self) -> [String; 7] {
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

fn run_benchmark<F>(label: &str, iterations: usize, mut benchmark_fn: F) -> BenchmarkSummary
where
    F: FnMut() -> f64,
{
    let iterations = iterations.max(1);
    let mut samples = Vec::with_capacity(iterations);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        checksum += benchmark_fn();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let stats = benchmark_latency_stats(&samples);
    BenchmarkSummary {
        label: label.to_string(),
        iterations,
        total_ms: stats.total_ms,
        mean_ms: stats.mean_ms,
        median_ms: stats.median_ms,
        max_ms: stats.max_ms,
        checksum,
    }
}

fn seeded_values(len: usize, offset: usize) -> Vec<f64> {
    (0..len)
        .map(|index| ((index * 37 + offset * 11) % 101) as f64 / 53.0 - 0.75)
        .collect()
}

fn seeded_positive_values(len: usize, offset: usize) -> Vec<f64> {
    seeded_values(len, offset)
        .into_iter()
        .map(|value| value.abs() + 0.5)
        .collect()
}

fn benchmark_iterations(nrow: usize, ncol: usize) -> usize {
    (1_000_000 / nrow.saturating_mul(ncol).max(1)).clamp(1_000, 20_000)
}

fn bench_size(nrow: usize, ncol: usize, offset: usize) -> Vec<BenchmarkSummary> {
    let iterations = benchmark_iterations(nrow, ncol);
    let k = seeded_values(nrow * ncol, offset + 1);
    let k0 = seeded_values(nrow, offset + 2);
    let values = seeded_values(nrow, offset + 3);
    let packed_values = seeded_values(ncol, offset + 4);
    let parscale = seeded_positive_values(nrow, offset + 5);

    let pack = run_benchmark(
        &format!("model_estimate pack {nrow}x{ncol}"),
        iterations,
        || {
            let result = lav_model_estimate_pack(
                black_box(&values),
                black_box(&k0),
                black_box(&k),
                nrow,
                ncol,
            )
            .expect("benchmark inputs should be valid");
            result[0] + result[result.len() - 1]
        },
    );

    let unpack_unscale = run_benchmark(
        &format!("model_estimate unpack_unscale {nrow}x{ncol}"),
        iterations,
        || {
            let result = lav_model_estimate_unpack_unscale(
                black_box(&packed_values),
                black_box(&k0),
                black_box(&k),
                black_box(&parscale),
                nrow,
                ncol,
            )
            .expect("benchmark inputs should be valid");
            result[0] + result[result.len() - 1]
        },
    );

    let gradient = run_benchmark(
        &format!("model_estimate gradient {nrow}x{ncol}"),
        iterations,
        || {
            let result = lav_model_estimate_gradient_postprocess(
                black_box(&values),
                black_box(&parscale),
                black_box(&k),
                nrow,
                ncol,
                120.0,
                true,
            )
            .expect("benchmark inputs should be valid");
            result[0] + result[result.len() - 1]
        },
    );

    vec![pack, unpack_unscale, gradient]
}

fn main() {
    let mut rows = Vec::new();
    rows.extend(bench_size(32, 16, 1));
    rows.extend(bench_size(96, 32, 2));
    rows.extend(bench_size(256, 80, 3));
    print_benchmark_summary_table("lav_model_estimate kernels", &rows);
}
