use faer::Mat;
use polars::prelude::*;

pub struct BenchmarkStats {
    pub total_ms: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

pub fn seeded_delta_a_delta_inputs(m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let delta_len = m
        .checked_mul(n)
        .expect("delta dimensions overflow usize");

    let delta: Vec<f64> = (0..delta_len)
        .map(|index| ((index % 17) as f64 + 1.0) / 17.0)
        .collect();

    let a1: Vec<f64> = (0..(m * m))
        .map(|index| {
            let row = index % m;
            let col = index / m;
            if row == col {
                2.0
            } else {
                ((row + col) % 7) as f64 / 100.0
            }
        })
        .collect();

    (delta, a1)
}

pub fn seeded_square_matrix(dim: usize) -> Mat<f64> {
    Mat::from_fn(dim, dim, |row, col| ((row * 31 + col * 17) as f64 + 1.0) / 101.0)
}

pub fn seeded_diag(dim: usize) -> Vec<f64> {
    (0..dim).map(|index| (index as f64 + 1.0) / (dim as f64 + 1.0)).collect()
}

pub fn benchmark_latency_stats(samples: &[f64]) -> BenchmarkStats {
    let elapsed_series = Series::new("elapsed_ms".into(), samples.to_vec());
    let elapsed = elapsed_series.f64().expect("elapsed times should be Float64");

    BenchmarkStats {
        total_ms: elapsed.sum().unwrap_or(0.0),
        min_ms: elapsed.min().unwrap_or(0.0),
        mean_ms: elapsed.mean().unwrap_or(0.0),
        median_ms: elapsed.median().unwrap_or(0.0),
        max_ms: elapsed.max().unwrap_or(0.0),
    }
}

pub fn square_matrix_from_column_major(values: &[f64], dim: usize) -> Result<Mat<f64>, String> {
    let expected_len = dim
        .checked_mul(dim)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    if values.len() != expected_len {
        return Err(format!(
            "matrix length mismatch: got {}, expected {} for {}x{}",
            values.len(),
            expected_len,
            dim,
            dim
        ));
    }
    Ok(Mat::from_fn(dim, dim, |row, col| values[row + col * dim]))
}

pub fn matrix_from_column_major(
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<Mat<f64>, String> {
    let expected_len = nrows
        .checked_mul(ncols)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    if values.len() != expected_len {
        return Err(format!(
            "matrix length mismatch: got {}, expected {} for {}x{}",
            values.len(),
            expected_len,
            nrows,
            ncols
        ));
    }
    Ok(Mat::from_fn(nrows, ncols, |row, col| values[row + col * nrows]))
}

pub fn assert_close_vec(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (left - right).abs();
        assert!(
            diff <= tolerance,
            "value {index} differs: actual={left}, expected={right}, diff={diff}"
        );
    }
}

pub fn assert_close_mat(actual: &Mat<f64>, expected: &Mat<f64>, tolerance: f64) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());

    for row in 0..actual.nrows() {
        for col in 0..actual.ncols() {
            let left = actual[(row, col)];
            let right = expected[(row, col)];
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "matrix value ({row}, {col}) differs: actual={left}, expected={right}, diff={diff}"
            );
        }
    }
}
