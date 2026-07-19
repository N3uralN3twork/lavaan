use ndarray::{Array2, ShapeBuilder};
use std::cmp::Ordering;

pub struct BenchmarkStats {
    pub total_ms: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

pub trait BenchmarkTableRow {
    fn benchmark_table_cells(&self) -> [String; 7];
}

pub fn seeded_delta_a_delta_inputs(m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let delta_len = m.checked_mul(n).expect("delta dimensions overflow usize");

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

pub fn seeded_square_matrix(dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim).f(), |(row, col)| {
        ((row * 31 + col * 17) as f64 + 1.0) / 101.0
    })
}

pub fn seeded_diag(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|index| (index as f64 + 1.0) / (dim as f64 + 1.0))
        .collect()
}

pub fn benchmark_latency_stats(samples: &[f64]) -> BenchmarkStats {
    if samples.is_empty() {
        return BenchmarkStats {
            total_ms: 0.0,
            min_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            max_ms: 0.0,
        };
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));

    let total_ms: f64 = sorted.iter().copied().sum();
    let min_ms = *sorted.first().unwrap_or(&0.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let mean_ms = total_ms / sorted.len() as f64;
    let middle = sorted.len() / 2;
    let median_ms = if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };

    BenchmarkStats {
        total_ms,
        min_ms,
        mean_ms,
        median_ms,
        max_ms,
    }
}

pub fn print_benchmark_summary_table<T>(title: &str, rows: &[T])
where
    T: BenchmarkTableRow,
{
    if rows.is_empty() {
        return;
    }

    let headers = [
        "name",
        "iters",
        "total_ms",
        "median_ms",
        "mean_ms",
        "max_ms",
        "checksum",
    ];

    let mut widths = headers.map(|header| header.len());
    let mut cells = Vec::with_capacity(rows.len());

    for row in rows {
        let row_cells = row.benchmark_table_cells();
        for (width, cell) in widths.iter_mut().zip(row_cells.iter()) {
            *width = (*width).max(cell.len());
        }
        cells.push(row_cells);
    }

    let border = |widths: &[usize]| -> String {
        let mut line = String::from("+");
        for width in widths {
            line.push_str(&"-".repeat(width + 2));
            line.push('+');
        }
        line
    };

    println!();
    println!("=== {title} ===");
    println!("{}", border(&widths));
    println!(
        "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$} |",
        headers[0],
        headers[1],
        headers[2],
        headers[3],
        headers[4],
        headers[5],
        headers[6],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
        w3 = widths[3],
        w4 = widths[4],
        w5 = widths[5],
        w6 = widths[6],
    );
    println!("{}", border(&widths));

    for row_cells in &cells {
        println!(
            "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$} |",
            row_cells[0],
            row_cells[1],
            row_cells[2],
            row_cells[3],
            row_cells[4],
            row_cells[5],
            row_cells[6],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
            w4 = widths[4],
            w5 = widths[5],
            w6 = widths[6],
        );
    }

    println!("{}", border(&widths));
}

pub fn square_matrix_from_column_major(values: &[f64], dim: usize) -> Result<Array2<f64>, String> {
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

    Array2::from_shape_vec((dim, dim).f(), values.to_vec())
        .map_err(|error| format!("matrix shape error: {error}"))
}

pub fn matrix_from_column_major(
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<Array2<f64>, String> {
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

    Array2::from_shape_vec((nrows, ncols).f(), values.to_vec())
        .map_err(|error| format!("matrix shape error: {error}"))
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

pub fn assert_close_mat(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
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
