use lavaan_kernels::{
    helpers::{benchmark_latency_stats, print_benchmark_summary_table, BenchmarkTableRow},
    lav_model_gradient_conditional_x_sample_cache, lav_model_gradient_delta_post,
    lav_model_gradient_dwls, lav_model_gradient_group_weight, lav_model_gradient_ml_conditional,
    lav_model_gradient_ml_conditional_post, lav_model_gradient_ml_group,
    lav_model_gradient_ntrls_post, lav_model_gradient_omega_gls,
    lav_model_gradient_omega_missing_pattern, lav_model_gradient_omega_ml,
    lav_model_gradient_t_d1_delta, lav_model_gradient_wls,
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

fn bench_delta_post() -> BenchmarkSummary {
    let n_moments = 96;
    let n_params = 32;
    let iterations = 5_000;
    let delta = seeded_values(n_moments * n_params, 1);
    let post = seeded_values(n_moments, 2);

    run_benchmark("gradient delta_post 96x32", iterations, || {
        let result = lav_model_gradient_delta_post(
            black_box(&delta),
            n_moments,
            n_params,
            black_box(&post),
            -0.5,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[n_params - 1]
    })
}

fn bench_omega_ml() -> BenchmarkSummary {
    let nvar = 24;
    let iterations = 2_000;
    let sigma = seeded_spd(nvar, 14.0, 1);
    let sigma_inv = seeded_spd(nvar, 8.0, 2);
    let sample_cov = seeded_spd(nvar, 12.0, 3);
    let mean_diff = seeded_values(nvar, 4);

    run_benchmark("gradient omega ML 24", iterations, || {
        let result = lav_model_gradient_omega_ml(
            black_box(&sigma),
            black_box(&sigma_inv),
            black_box(&sample_cov),
            black_box(&mean_diff),
            true,
        )
        .expect("benchmark inputs should be valid");
        result.omega[(0, 0)] + result.omega[(nvar - 1, nvar - 1)] + result.omega_mu[0]
    })
}

fn bench_omega_gls() -> BenchmarkSummary {
    let nvar = 24;
    let iterations = 2_000;
    let sigma = seeded_spd(nvar, 14.0, 5);
    let weight_inv = seeded_spd(nvar, 8.0, 6);
    let sample_cov = seeded_spd(nvar, 12.0, 7);
    let mean_diff = seeded_values(nvar, 8);

    run_benchmark("gradient omega GLS 24", iterations, || {
        let result = lav_model_gradient_omega_gls(
            black_box(&sigma),
            black_box(&weight_inv),
            black_box(&sample_cov),
            black_box(&mean_diff),
            true,
            120.0,
        )
        .expect("benchmark inputs should be valid");
        result.omega[(0, 0)] + result.omega[(nvar - 1, nvar - 1)] + result.omega_mu[0]
    })
}

fn bench_omega_missing_pattern() -> BenchmarkSummary {
    let nvar = 32;
    let pattern_len = 18;
    let iterations = 2_000;
    let sigma_inv = seeded_spd(pattern_len, 8.0, 9);
    let sample_cov = seeded_spd(pattern_len, 12.0, 10);
    let mean_diff = seeded_values(pattern_len, 11);
    let var_idx: Vec<usize> = (0..pattern_len)
        .map(|index| index * 31 / pattern_len)
        .collect();

    run_benchmark("gradient omega missing pattern", iterations, || {
        let result = lav_model_gradient_omega_missing_pattern(
            black_box(&sigma_inv),
            black_box(&sample_cov),
            black_box(&mean_diff),
            black_box(&var_idx),
            nvar,
            0.15,
        )
        .expect("benchmark inputs should be valid");
        result.omega[(0, 0)] + result.omega[(nvar - 1, nvar - 1)] + result.omega_mu[0]
    })
}

fn bench_wls() -> BenchmarkSummary {
    let n_moments = 96;
    let n_params = 32;
    let iterations = 2_000;
    let delta = seeded_values(n_moments * n_params, 3);
    let wls_v = seeded_spd(n_moments, 16.0, 4);
    let diff = seeded_values(n_moments, 5);

    run_benchmark("gradient WLS 96x32", iterations, || {
        let result = lav_model_gradient_wls(
            black_box(&delta),
            n_moments,
            n_params,
            black_box(&wls_v),
            black_box(&diff),
            0.5,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[n_params - 1]
    })
}

fn bench_dwls() -> BenchmarkSummary {
    let n_moments = 96;
    let n_params = 32;
    let iterations = 5_000;
    let delta = seeded_values(n_moments * n_params, 6);
    let wls_vd = seeded_values(n_moments, 7);
    let diff = seeded_values(n_moments, 8);

    run_benchmark("gradient DWLS 96x32", iterations, || {
        let result = lav_model_gradient_dwls(
            black_box(&delta),
            n_moments,
            n_params,
            black_box(&wls_vd),
            black_box(&diff),
            0.5,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[n_params - 1]
    })
}

fn bench_ml_conditional_post() -> BenchmarkSummary {
    let nvar = 24;
    let nx = 8;
    let iterations = 1_000;
    let mean_x = seeded_values(nx, 9);
    let cov_x = seeded_spd(nx, 8.0, 10);
    let res_int = seeded_values(nvar, 11);
    let res_slopes = seeded_values(nvar * nx, 12);
    let cache =
        lav_model_gradient_conditional_x_sample_cache(&mean_x, &cov_x, &res_int, &res_slopes)
            .expect("benchmark sample cache inputs should be valid");
    let c3 = array_to_column_major_vec(&cache.c3);
    let obs = array_to_column_major_vec(&cache.obs);
    let mu = seeded_values(nvar, 13);
    let pi = seeded_values(nvar * nx, 14);
    let sigma_inv = seeded_spd(nvar, 12.0, 15);
    let res_cov = seeded_spd(nvar, 12.0, 16);

    run_benchmark("gradient ML conditional post", iterations, || {
        let result = lav_model_gradient_ml_conditional_post(
            black_box(&c3),
            black_box(&obs),
            nvar,
            nx,
            black_box(&mu),
            black_box(&pi),
            black_box(&sigma_inv),
            black_box(&res_cov),
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn bench_ml_conditional_gradient() -> BenchmarkSummary {
    let nvar = 24;
    let nx = 8;
    let post_len = (nx + 1) * nvar + nvar * (nvar + 1) / 2;
    let n_params = 40;
    let iterations = 2_000;
    let delta = seeded_values(post_len * n_params, 17);
    let post = seeded_values(post_len, 18);

    run_benchmark("gradient ML conditional chain", iterations, || {
        let result = lav_model_gradient_ml_conditional(
            black_box(&delta),
            post_len,
            n_params,
            black_box(&post),
            0.5,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[n_params - 1]
    })
}

fn bench_ml_group() -> BenchmarkSummary {
    let nvar = 24;
    let post_len = 1 + nvar + nvar * (nvar + 1) / 2;
    let n_params = 40;
    let iterations = 2_000;
    let delta = seeded_values(post_len * n_params, 19);
    let omega = seeded_values(nvar * nvar, 20);
    let omega_mu = seeded_values(nvar, 21);

    run_benchmark("gradient ML group 24", iterations, || {
        let result = lav_model_gradient_ml_group(
            black_box(&delta),
            post_len,
            n_params,
            black_box(&omega),
            black_box(&omega_mu),
            0.5,
            true,
            true,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[n_params - 1]
    })
}

fn bench_t_d1_delta() -> BenchmarkSummary {
    let n_moments = 96;
    let n_d1_cols = 24;
    let n_params = 32;
    let iterations = 2_000;
    let d1 = seeded_values(n_moments * n_d1_cols, 22);
    let delta = seeded_values(n_moments * n_params, 23);

    run_benchmark("gradient t(d1) delta 96x24x32", iterations, || {
        let result = lav_model_gradient_t_d1_delta(
            black_box(&d1),
            n_moments,
            n_d1_cols,
            black_box(&delta),
            n_moments,
            n_params,
            0.25,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn bench_ntrls_post() -> BenchmarkSummary {
    let nvar = 24;
    let iterations = 1_000;
    let sample_cov = seeded_spd(nvar, 12.0, 24);
    let model_cov = seeded_spd(nvar, 14.0, 25);
    let sigma_inv = seeded_spd(nvar, 8.0, 26);
    let mean_observed = seeded_values(nvar, 27);
    let mean_model = seeded_values(nvar, 28);

    run_benchmark("gradient NTRLS post 24", iterations, || {
        let result = lav_model_gradient_ntrls_post(
            black_box(&sample_cov),
            black_box(&model_cov),
            black_box(&sigma_inv),
            black_box(&mean_observed),
            black_box(&mean_model),
            true,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn bench_group_weight() -> BenchmarkSummary {
    let n_groups = 32;
    let iterations = 20_000;
    let log_group_weight = (0..n_groups)
        .map(|index| (20.0 + index as f64).ln())
        .collect::<Vec<_>>();
    let observed_group_weight = (0..n_groups)
        .map(|index| (index as f64 + 1.0) / 528.0)
        .collect::<Vec<_>>();

    run_benchmark("gradient group weight 32", iterations, || {
        let result = lav_model_gradient_group_weight(
            black_box(&log_group_weight),
            black_box(&observed_group_weight),
            500.0,
        )
        .expect("benchmark inputs should be valid");
        result[0] + result[result.len() - 1]
    })
}

fn array_to_column_major_vec(array: &ndarray::Array2<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(array.nrows() * array.ncols());
    for col in 0..array.ncols() {
        for row in 0..array.nrows() {
            out.push(array[(row, col)]);
        }
    }
    out
}

fn main() {
    let rows = vec![
        bench_omega_ml(),
        bench_omega_gls(),
        bench_omega_missing_pattern(),
        bench_delta_post(),
        bench_wls(),
        bench_dwls(),
        bench_ml_conditional_post(),
        bench_ml_conditional_gradient(),
        bench_ml_group(),
        bench_t_d1_delta(),
        bench_ntrls_post(),
        bench_group_weight(),
    ];
    print_benchmark_summary_table("lav_model_gradient kernels", &rows);
}
