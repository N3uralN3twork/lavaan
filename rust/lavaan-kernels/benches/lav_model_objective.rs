use lavaan_kernels::helpers::{benchmark_latency_stats, BenchmarkTableRow};
use lavaan_kernels::{
    lav_model_objective_dwls, lav_model_objective_gls, lav_model_objective_ml,
    lav_model_objective_ml_res, lav_model_objective_wls,
};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Summary {
    label: String,
    iterations: usize,
    total_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    max_ms: f64,
    checksum: f64,
}

impl BenchmarkTableRow for Summary {
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

fn run<F>(label: &str, iterations: usize, mut operation: F) -> Summary
where
    F: FnMut() -> f64,
{
    let mut samples = Vec::with_capacity(iterations);
    let mut checksum = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        checksum += operation();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let stats = benchmark_latency_stats(&samples);
    Summary {
        label: label.to_string(),
        iterations,
        total_ms: stats.total_ms,
        mean_ms: stats.mean_ms,
        median_ms: stats.median_ms,
        max_ms: stats.max_ms,
        checksum,
    }
}

fn main() {
    let nvar = 32;
    let sigma = seeded_spd(nvar, 18.0, 1);
    let sigma_inv = seeded_spd(nvar, 12.0, 2);
    let data_cov = seeded_spd(nvar, 16.0, 3);
    let mu = seeded_values(nvar, 4);
    let data_mean = seeded_values(nvar, 5);
    let res_int = seeded_values(nvar, 6);
    let res_slopes = seeded_values(nvar * 8, 7);
    let pi = seeded_values(nvar * 8, 8);
    let cov_x = seeded_spd(8, 10.0, 9);
    let mean_x = seeded_values(8, 10);
    let wls_est = seeded_values(96, 11);
    let wls_obs = seeded_values(96, 12);
    let wls_v = seeded_spd(96, 20.0, 13);
    let wls_vd: Vec<f64> = seeded_values(96, 14)
        .into_iter()
        .map(|value| value.abs() + 0.5)
        .collect();

    let rows = vec![
        run("objective ML 32", 1_000, || {
            lav_model_objective_ml(
                black_box(&sigma),
                black_box(&sigma_inv),
                black_box(&data_cov),
                black_box(&mu),
                black_box(&data_mean),
                1.25,
                1.05,
                true,
                nvar,
            )
            .unwrap()
        }),
        run("objective ML conditional 32x8", 500, || {
            lav_model_objective_ml_res(
                black_box(&sigma),
                black_box(&sigma_inv),
                black_box(&data_cov),
                black_box(&res_int),
                black_box(&res_slopes),
                black_box(&mu),
                black_box(&pi),
                black_box(&cov_x),
                black_box(&mean_x),
                1.25,
                1.05,
                nvar,
                8,
            )
            .unwrap()
        }),
        run("objective GLS 32", 1_000, || {
            lav_model_objective_gls(
                black_box(&sigma),
                black_box(&data_cov),
                black_box(&sigma_inv),
                black_box(&mu),
                black_box(&data_mean),
                true,
                false,
                nvar,
            )
            .unwrap()
        }),
        run("objective WLS 96", 200, || {
            lav_model_objective_wls(
                black_box(&wls_est),
                black_box(&wls_obs),
                black_box(&wls_v),
                96,
            )
            .unwrap()
        }),
        run("objective DWLS 96", 5_000, || {
            lav_model_objective_dwls(black_box(&wls_est), black_box(&wls_obs), black_box(&wls_vd))
                .unwrap()
        }),
    ];
    lavaan_kernels::helpers::print_benchmark_summary_table("lav_model_objective kernels", &rows);
}
