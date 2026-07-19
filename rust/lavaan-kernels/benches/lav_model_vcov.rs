use lavaan_kernels::{
    helpers::{benchmark_latency_stats, seeded_delta_a_delta_inputs, BenchmarkTableRow},
    lav_model_vcov_delta_a_delta, lav_model_vcov_jacobian_vcov_jacobian_t, lav_model_vcov_sandwich,
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

fn seeded_spd(dim: usize, diagonal: f64, offset: usize) -> Vec<f64> {
    let mut values = seeded_values(dim * dim, offset);
    for col in 0..dim {
        for row in 0..dim {
            let left = values[row + col * dim];
            let right = values[col + row * dim];
            values[row + col * dim] = (left + right) / 2.0;
        }
        values[col + col * dim] += diagonal;
    }
    values
}

fn bench_delta_a_delta() -> BenchmarkSummary {
    let n_moments = 96;
    let n_params = 24;
    let iterations = 1_000;
    let (delta, a1) = seeded_delta_a_delta_inputs(n_moments, n_params);

    run_benchmark("vcov delta_A_delta 96x24", iterations, || {
        let result =
            lav_model_vcov_delta_a_delta(black_box(&delta), black_box(&a1), n_moments, n_params)
                .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn bench_sandwich() -> BenchmarkSummary {
    let dim = 64;
    let iterations = 500;
    let left = seeded_spd(dim, 12.0, 1);
    let middle = seeded_spd(dim, 16.0, 2);

    run_benchmark("vcov sandwich 64", iterations, || {
        let result = lav_model_vcov_sandwich(black_box(&left), black_box(&middle), dim)
            .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn bench_jacobian_vcov_jacobian_t() -> BenchmarkSummary {
    let n_def = 24;
    let n_free = 80;
    let iterations = 500;
    let jac = seeded_values(n_def * n_free, 3);
    let vcov = seeded_spd(n_free, 14.0, 4);

    run_benchmark("vcov jac_VCOV_jac_t 24x80", iterations, || {
        let result = lav_model_vcov_jacobian_vcov_jacobian_t(
            black_box(&jac),
            black_box(&vcov),
            n_def,
            n_free,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn main() {
    let rows = vec![
        bench_delta_a_delta(),
        bench_sandwich(),
        bench_jacobian_vcov_jacobian_t(),
    ];
    lavaan_kernels::helpers::print_benchmark_summary_table("lav_model_vcov kernels", &rows);
}
