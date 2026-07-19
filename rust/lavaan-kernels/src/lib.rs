//! Experimental Rust kernels for lavaan performance candidates.

pub mod helpers;
pub mod lav_model_estimate;
pub mod lav_model_gradient;
pub mod lav_model_objective;
pub mod lav_model_vcov;

use ndarray::{Array2, ArrayView2, ShapeBuilder};

pub use lav_model_estimate::{
    lav_model_estimate_gradient_postprocess, lav_model_estimate_pack, lav_model_estimate_unpack,
    lav_model_estimate_unpack_unscale,
};
pub use lav_model_gradient::{
    lav_model_gradient_conditional_x_sample_cache, lav_model_gradient_delta_post,
    lav_model_gradient_duplication_pre, lav_model_gradient_dwls, lav_model_gradient_group_weight,
    lav_model_gradient_ml_conditional, lav_model_gradient_ml_conditional_post,
    lav_model_gradient_ml_group, lav_model_gradient_ntrls_post, lav_model_gradient_omega_gls,
    lav_model_gradient_omega_missing_pattern, lav_model_gradient_omega_ml,
    lav_model_gradient_t_d1_delta, lav_model_gradient_wls, ConditionalXSampleCache, MlOmega,
};
pub use lav_model_objective::{
    lav_model_objective_dwls, lav_model_objective_gls, lav_model_objective_ml,
    lav_model_objective_ml_res, lav_model_objective_wls,
};
pub use lav_model_vcov::{
    lav_model_vcov_delta_a_delta, lav_model_vcov_jacobian_vcov_jacobian_t, lav_model_vcov_sandwich,
};

/// Compute `t(delta) %*% a1 %*% delta` for R-style column-major matrices.
///
/// `delta` is an `m x n` matrix and `a1` is an `m x m` matrix, both flattened
/// column-by-column as R stores numeric matrices. The returned vector is the
/// `n x n` result in the same column-major layout.
pub fn delta_a_delta(delta: &[f64], a1: &[f64], m: usize, n: usize) -> Result<Vec<f64>, String> {
    let delta_len = m
        .checked_mul(n)
        .ok_or_else(|| "delta dimensions overflow usize".to_string())?;
    if delta.len() != delta_len {
        return Err(format!(
            "delta length mismatch: got {}, expected {} for {}x{}",
            delta.len(),
            delta_len,
            m,
            n
        ));
    }

    let a1_len = m
        .checked_mul(m)
        .ok_or_else(|| "a1 dimensions overflow usize".to_string())?;
    if a1.len() != a1_len {
        return Err(format!(
            "a1 length mismatch: got {}, expected {} for {}x{}",
            a1.len(),
            a1_len,
            m,
            m
        ));
    }

    let delta_mat = ArrayView2::from_shape((m, n).f(), delta)
        .map_err(|err| format!("delta shape error: {err}"))?;
    let a1_mat =
        ArrayView2::from_shape((m, m).f(), a1).map_err(|err| format!("a1 shape error: {err}"))?;
    let left = delta_mat.t().dot(&a1_mat);
    let result = left.dot(&delta_mat);

    Ok(array_to_column_major_vec(&result))
}

fn array_to_column_major_vec(array: &Array2<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(array.nrows() * array.ncols());
    for col in 0..array.ncols() {
        out.extend(array.column(col).iter().copied());
    }
    out
}

/// Column/row scaling helper used by the validation R wrapper.
pub fn lav_matrix_diag_prepost(a: &Array2<f64>, d: &[f64]) -> Array2<f64> {
    if d.is_empty() {
        return a.clone();
    }

    debug_assert_eq!(a.nrows(), d.len(), "d must match A's row count");
    debug_assert_eq!(a.ncols(), d.len(), "d must match A's column count");

    Array2::from_shape_fn((a.nrows(), a.ncols()).f(), |(i, j)| a[(i, j)] * d[i] * d[j])
}

pub fn lav_matrix_diagh_idx(n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    let mut idx = Vec::with_capacity(n);
    let mut pos: usize = 0;
    idx.push(pos);

    for col_height in (2..=n).rev() {
        pos += col_height;
        idx.push(pos);
    }

    idx
}

/// Return the vector indices of the anti-diagonal elements of a symmetric
/// matrix of size `n`.
pub fn lav_matrix_antidiag_idx(n: usize) -> Vec<usize> {
    if n < 1 {
        return Vec::new();
    }

    (1..=n).map(|k| k * (n - 1)).collect()
}

pub fn lav_matrix_vec(values: &[f64]) -> Vec<f64> {
    values.to_vec()
}

pub fn lav_matrix_vecr(values: &[f64], nrow: usize, ncol: usize) -> Result<Vec<f64>, String> {
    let expected_len = nrow
        .checked_mul(ncol)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    if values.len() != expected_len {
        return Err(format!(
            "matrix length mismatch: got {}, expected {} for {}x{}",
            values.len(),
            expected_len,
            nrow,
            ncol
        ));
    }

    let mut out = Vec::with_capacity(expected_len);
    for row in 0..nrow {
        for col in 0..ncol {
            out.push(values[col * nrow + row]);
        }
    }
    Ok(out)
}

pub fn lav_matrix_diag_idx(n: usize) -> Vec<usize> {
    if n < 1 {
        return Vec::new();
    }

    (0..n).map(|i| i * (n + 1) + 1).collect()
}

pub fn lav_matrix_vech_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_indices(n, diagonal, TriangularOrder::ColumnLower)
}

pub fn lav_matrix_vech_row_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_row_col_indices(n, diagonal, TriangularOrder::ColumnLower, true)
}

pub fn lav_matrix_vech_col_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_row_col_indices(n, diagonal, TriangularOrder::ColumnLower, false)
}

pub fn lav_matrix_vechr_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_indices(n, diagonal, TriangularOrder::RowLower)
}

pub fn lav_matrix_vechu_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_indices(n, diagonal, TriangularOrder::ColumnUpper)
}

pub fn lav_matrix_vechru_idx(n: usize, diagonal: bool) -> Vec<usize> {
    triangular_indices(n, diagonal, TriangularOrder::RowUpper)
}

pub fn lav_matrix_vech(values: &[f64], n: usize, diagonal: bool) -> Result<Vec<f64>, String> {
    triangular_extract(values, n, diagonal, TriangularOrder::ColumnLower)
}

pub fn lav_matrix_vechr(values: &[f64], n: usize, diagonal: bool) -> Result<Vec<f64>, String> {
    triangular_extract(values, n, diagonal, TriangularOrder::RowLower)
}

pub fn lav_matrix_vechu(values: &[f64], n: usize, diagonal: bool) -> Result<Vec<f64>, String> {
    triangular_extract(values, n, diagonal, TriangularOrder::ColumnUpper)
}

pub fn lav_matrix_vechru(values: &[f64], n: usize, diagonal: bool) -> Result<Vec<f64>, String> {
    triangular_extract(values, n, diagonal, TriangularOrder::RowUpper)
}

pub fn lav_matrix_vech_reverse(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::ColumnLower)
}

pub fn lav_matrix_vechru_reverse(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::RowUpper)
}

pub fn lav_matrix_upper2full(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::ColumnUpper)
}

pub fn lav_matrix_vechr_reverse(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::RowLower)
}

pub fn lav_matrix_vechu_reverse(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::ColumnUpper)
}

pub fn lav_matrix_lower2full(values: &[f64], diagonal: bool) -> Result<Array2<f64>, String> {
    triangular_full(values, diagonal, TriangularOrder::ColumnLower)
}

fn triangular_indices(n: usize, diagonal: bool, order: TriangularOrder) -> Vec<usize> {
    triangular_row_col_indices(n, diagonal, order, false)
        .into_iter()
        .map(|index| index + 1)
        .collect()
}

fn triangular_row_col_indices(
    n: usize,
    diagonal: bool,
    order: TriangularOrder,
    rows_only: bool,
) -> Vec<usize> {
    let coords = triangular_coords(n, diagonal, order);
    let mut out = Vec::with_capacity(coords.len());
    for (row, col) in coords {
        let value = if rows_only { row + 1 } else { col + 1 };
        out.push(value);
    }
    out
}

fn triangular_extract(
    values: &[f64],
    n: usize,
    diagonal: bool,
    order: TriangularOrder,
) -> Result<Vec<f64>, String> {
    let expected_len = n
        .checked_mul(n)
        .ok_or_else(|| "matrix dimensions overflow usize".to_string())?;
    if values.len() != expected_len {
        return Err(format!(
            "matrix length mismatch: got {}, expected {} for {}x{}",
            values.len(),
            expected_len,
            n,
            n
        ));
    }

    let mut out = Vec::with_capacity(triangular_coords(n, diagonal, order).len());
    for (row, col) in triangular_coords(n, diagonal, order) {
        out.push(values[row + col * n]);
    }
    Ok(out)
}

fn triangular_full(
    values: &[f64],
    diagonal: bool,
    order: TriangularOrder,
) -> Result<Array2<f64>, String> {
    let expected_len = if diagonal {
        triangular_count_with_diagonal(values.len())?
    } else {
        triangular_count_without_diagonal(values.len())?
    };
    let n = expected_len.0;
    let needed_len = expected_len.1;
    if values.len() != needed_len {
        return Err(format!(
            "vector length mismatch: got {}, expected {} for n={}",
            values.len(),
            needed_len,
            n
        ));
    }

    Ok(Array2::from_shape_fn((n, n).f(), |(row, col)| {
        if let Some(pos) = triangular_position(row, col, n, diagonal, order) {
            values[pos]
        } else if let Some(pos) = triangular_position(col, row, n, diagonal, order) {
            values[pos]
        } else {
            0.0
        }
    }))
}

fn triangular_position(
    row: usize,
    col: usize,
    n: usize,
    diagonal: bool,
    order: TriangularOrder,
) -> Option<usize> {
    match order {
        TriangularOrder::ColumnLower => lower_column_position(row, col, n, diagonal),
        TriangularOrder::RowLower => lower_row_position(row, col, n, diagonal),
        TriangularOrder::ColumnUpper => upper_column_position(row, col, n, diagonal),
        TriangularOrder::RowUpper => upper_row_position(row, col, n, diagonal),
    }
}

fn triangular_coords(n: usize, diagonal: bool, order: TriangularOrder) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();
    match order {
        TriangularOrder::ColumnLower => {
            for col in 0..n {
                for row in 0..n {
                    if row > col || (diagonal && row == col) {
                        coords.push((row, col));
                    }
                }
            }
        }
        TriangularOrder::RowLower => {
            for row in 0..n {
                for col in 0..n {
                    if row > col || (diagonal && row == col) {
                        coords.push((row, col));
                    }
                }
            }
        }
        TriangularOrder::ColumnUpper => {
            for col in 0..n {
                for row in 0..n {
                    if row < col || (diagonal && row == col) {
                        coords.push((row, col));
                    }
                }
            }
        }
        TriangularOrder::RowUpper => {
            for row in 0..n {
                for col in 0..n {
                    if row < col || (diagonal && row == col) {
                        coords.push((row, col));
                    }
                }
            }
        }
    }
    coords
}

fn lower_column_position(row: usize, col: usize, n: usize, diagonal: bool) -> Option<usize> {
    if diagonal {
        if row < col {
            return None;
        }
        Some(col * n - (col * col.saturating_sub(1)) / 2 + (row - col))
    } else {
        if row <= col {
            return None;
        }
        Some(col * (2 * n - col - 1) / 2 + (row - col - 1))
    }
}

fn lower_row_position(row: usize, col: usize, _n: usize, diagonal: bool) -> Option<usize> {
    if diagonal {
        if row < col {
            return None;
        }
        Some(row * (row + 1) / 2 + col)
    } else {
        if row <= col {
            return None;
        }
        Some(row * row.saturating_sub(1) / 2 + col)
    }
}

fn upper_column_position(row: usize, col: usize, _n: usize, diagonal: bool) -> Option<usize> {
    if diagonal {
        if row > col {
            return None;
        }
        Some(col * (col + 1) / 2 + row)
    } else {
        if row >= col {
            return None;
        }
        Some(col * (col - 1) / 2 + row)
    }
}

fn upper_row_position(row: usize, col: usize, n: usize, diagonal: bool) -> Option<usize> {
    if diagonal {
        if row > col {
            return None;
        }
        Some(row * n - row * row.saturating_sub(1) / 2 + (col - row))
    } else {
        if row >= col {
            return None;
        }
        Some(row * (2 * n - row - 1) / 2 + (col - row - 1))
    }
}

fn triangular_count_with_diagonal(len: usize) -> Result<(usize, usize), String> {
    let disc = 1 + 8 * len;
    let root = (disc as f64).sqrt() as usize;
    let n = (root - 1) / 2;
    if n * (n + 1) / 2 != len {
        return Err(format!(
            "vector length {} is not triangular with diagonal",
            len
        ));
    }
    Ok((n, len))
}

fn triangular_count_without_diagonal(len: usize) -> Result<(usize, usize), String> {
    let disc = 1 + 8 * len;
    let root = (disc as f64).sqrt() as usize;
    let n = (root + 1) / 2;
    if n * (n - 1) / 2 != len {
        return Err(format!(
            "vector length {} is not triangular without diagonal",
            len
        ));
    }
    Ok((n, len))
}

#[derive(Clone, Copy)]
enum TriangularOrder {
    ColumnLower,
    RowLower,
    ColumnUpper,
    RowUpper,
}
