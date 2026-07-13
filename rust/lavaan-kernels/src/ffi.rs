use std::slice;

use crate::{
    delta_a_delta, lav_matrix_antidiag_idx, lav_matrix_diagh_idx, lav_matrix_diag_idx,
    lav_matrix_lower2full, lav_matrix_vech, lav_matrix_vech_idx, lav_matrix_vech_reverse,
    lav_matrix_vech_row_idx,
    lav_matrix_vech_col_idx, lav_matrix_vechru, lav_matrix_vechru_idx,
    lav_matrix_vechru_reverse, lav_matrix_vechr, lav_matrix_vechr_idx,
    lav_matrix_vechr_reverse, lav_matrix_vechu, lav_matrix_vechu_idx,
    lav_matrix_vechu_reverse, lav_matrix_upper2full, lav_matrix_vec, lav_matrix_vecr,
};
use faer::Mat;

fn copy_vec_to_out(values: &[f64], out: *mut f64) -> bool {
    if out.is_null() {
        return false;
    }

    unsafe {
        slice::from_raw_parts_mut(out, values.len()).copy_from_slice(values);
    }
    true
}

fn copy_i32_vec_to_out(values: &[usize], out: *mut i32) -> bool {
    if out.is_null() {
        return false;
    }

    let out_slice = unsafe { slice::from_raw_parts_mut(out, values.len()) };
    for (dest, value) in out_slice.iter_mut().zip(values.iter().copied()) {
        match i32::try_from(value) {
            Ok(v) => *dest = v,
            Err(_) => return false,
        }
    }
    true
}

fn copy_mat_to_out(mat: &Mat<f64>, out: *mut f64) -> bool {
    if out.is_null() {
        return false;
    }

    let nrow = mat.nrows();
    let ncol = mat.ncols();
    let out_slice = unsafe { slice::from_raw_parts_mut(out, nrow * ncol) };
    for col in 0..ncol {
        let start = col * nrow;
        let end = start + nrow;
        out_slice[start..end].copy_from_slice(mat.col_as_slice(col));
    }
    true
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vec(input: *const f64, len: usize, out: *mut f64) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    copy_vec_to_out(&lav_matrix_vec(input), out)
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vecr(
    input: *const f64,
    nrow: usize,
    ncol: usize,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let len = match nrow.checked_mul(ncol) {
        Some(value) => value,
        None => return false,
    };
    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_vecr(input, nrow, ncol) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_diag_prepost(
    input: *const f64,
    n: usize,
    d: *const f64,
    out: *mut f64,
) -> bool {
    if input.is_null() || d.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, n * n) };
    let d = unsafe { slice::from_raw_parts(d, n) };
    let mat = Mat::from_fn(n, n, |row, col| input[col * n + row] * d[row] * d[col]);
    copy_mat_to_out(&mat, out)
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_delta_a_delta(
    delta: *const f64,
    m: usize,
    n: usize,
    a1: *const f64,
    out: *mut f64,
) -> bool {
    if delta.is_null() || a1.is_null() {
        return false;
    }

    let delta = unsafe { slice::from_raw_parts(delta, m * n) };
    let a1 = unsafe { slice::from_raw_parts(a1, m * m) };
    match delta_a_delta(delta, a1, m, n) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_diagh_idx(n: usize, out: *mut i32) -> usize {
    let values = lav_matrix_diagh_idx(n);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_antidiag_idx(n: usize, out: *mut i32) -> usize {
    let values = lav_matrix_antidiag_idx(n);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_diag_idx(n: usize, out: *mut i32) -> usize {
    let values = lav_matrix_diag_idx(n);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vech_idx(n: usize, diagonal: bool, out: *mut i32) -> usize {
    let values = lav_matrix_vech_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vech_row_idx(
    n: usize,
    diagonal: bool,
    out: *mut i32,
) -> usize {
    let values = lav_matrix_vech_row_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vech_col_idx(
    n: usize,
    diagonal: bool,
    out: *mut i32,
) -> usize {
    let values = lav_matrix_vech_col_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechr_idx(n: usize, diagonal: bool, out: *mut i32) -> usize {
    let values = lav_matrix_vechr_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechu_idx(n: usize, diagonal: bool, out: *mut i32) -> usize {
    let values = lav_matrix_vechu_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechru_idx(
    n: usize,
    diagonal: bool,
    out: *mut i32,
) -> usize {
    let values = lav_matrix_vechru_idx(n, diagonal);
    if !copy_i32_vec_to_out(&values, out) {
        return 0;
    }
    values.len()
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vech(
    input: *const f64,
    n: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, n * n) };
    match lav_matrix_vech(input, n, diagonal) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechr(
    input: *const f64,
    n: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, n * n) };
    match lav_matrix_vechr(input, n, diagonal) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechu(
    input: *const f64,
    n: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, n * n) };
    match lav_matrix_vechu(input, n, diagonal) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechru(
    input: *const f64,
    n: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, n * n) };
    match lav_matrix_vechru(input, n, diagonal) {
        Ok(values) => copy_vec_to_out(&values, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vech_reverse(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_vech_reverse(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechru_reverse(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_vechru_reverse(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_upper2full(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_upper2full(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechr_reverse(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_vechr_reverse(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_vechu_reverse(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_vechu_reverse(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}

#[no_mangle]
pub extern "C" fn lavaan_lav_matrix_lower2full(
    input: *const f64,
    len: usize,
    diagonal: bool,
    out: *mut f64,
) -> bool {
    if input.is_null() {
        return false;
    }

    let input = unsafe { slice::from_raw_parts(input, len) };
    match lav_matrix_lower2full(input, diagonal) {
        Ok(mat) => copy_mat_to_out(&mat, out),
        Err(_) => false,
    }
}
