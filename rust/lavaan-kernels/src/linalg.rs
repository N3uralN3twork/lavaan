//! Small dense column-major operations; no R objects or external runtime libraries.
pub fn transpose(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for col in 0..cols {
        for row in 0..rows {
            out[col + row * cols] = x[row + col * rows];
        }
    }
    out
}

pub fn multiply(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);

    let mut out = vec![0.0; rows * cols];

    if rows == 0 || inner == 0 || cols == 0 {
        return out;
    }

    // Process four output columns at once.
    let full_cols = cols - cols % 4;

    for col in (0..full_cols).step_by(4) {
        let block = &mut out[col * rows..(col + 4) * rows];

        // Split into four independent mutable slices.
        let (o0, rest) = block.split_at_mut(rows);
        let (o1, rest) = rest.split_at_mut(rows);
        let (o2, o3) = rest.split_at_mut(rows);

        for (k, a_col) in a.chunks_exact(rows).enumerate() {
            // Load four B coefficients.
            let b0 = b[col * inner + k];
            let b1 = b[(col + 1) * inner + k];
            let b2 = b[(col + 2) * inner + k];
            let b3 = b[(col + 3) * inner + k];

            // Load each A element only once.
            for (r, &x) in a_col.iter().enumerate() {
                o0[r] += x * b0;
                o1[r] += x * b1;
                o2[r] += x * b2;
                o3[r] += x * b3;
            }
        }
    }

    // Process any remaining columns.
    for col in full_cols..cols {
        let out_col = &mut out[col * rows..(col + 1) * rows];
        let b_col = &b[col * inner..(col + 1) * inner];

        for (&scale, a_col) in b_col.iter().zip(a.chunks_exact(rows)) {
            for (dst, &x) in out_col.iter_mut().zip(a_col) {
                *dst += x * scale;
            }
        }
    }

    out
}

pub fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i + i * n] = 1.0;
    }
    out
}

pub fn spd_inverse(x: &[f64], n: usize) -> Option<(Vec<f64>, f64)> {
    let mut lower = vec![0.0; n * n];
    let mut logdet = 0.0;
    for col in 0..n {
        for row in col..n {
            let mut value = x[row + col * n];
            for k in 0..col {
                value -= lower[row + k * n] * lower[col + k * n];
            }
            if row == col {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                lower[row + col * n] = value.sqrt();
                logdet += value.ln();
            } else {
                lower[row + col * n] = value / lower[col + col * n];
            }
        }
    }
    let mut inverse_lower = vec![0.0; n * n];
    for col in 0..n {
        for row in col..n {
            let mut value = if row == col { 1.0 } else { 0.0 };
            for k in col..row {
                value -= lower[row + k * n] * inverse_lower[k + col * n];
            }
            inverse_lower[row + col * n] = value / lower[row + row * n];
        }
    }
    let inv = multiply(&transpose(&inverse_lower, n, n), &inverse_lower, n, n, n);
    Some((inv, logdet))
}

pub fn inverse(x: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut work = x.to_vec();
    let mut out = identity(n);
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| work[a + col * n].abs().total_cmp(&work[b + col * n].abs()))?;
        let value = work[pivot + col * n];
        if !value.is_finite() || value.abs() < f64::EPSILON {
            return None;
        }
        for j in 0..n {
            work.swap(col + j * n, pivot + j * n);
            out.swap(col + j * n, pivot + j * n);
        }
        let value = work[col + col * n];
        for j in 0..n {
            work[col + j * n] /= value;
            out[col + j * n] /= value;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = work[row + col * n];
            for j in 0..n {
                work[row + j * n] -= factor * work[col + j * n];
                out[row + j * n] -= factor * out[col + j * n];
            }
        }
    }
    Some(out)
}
