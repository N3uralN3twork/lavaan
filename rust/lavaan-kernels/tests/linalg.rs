use lavaan_kernels::linalg::{identity, inverse, multiply, spd_inverse, transpose};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }
}

#[test]
fn transpose_rectangular_matrix_uses_column_major_order() {
    // Rows [1, 2, 3] and [4, 5, 6], stored one column at a time.
    let matrix = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(transpose(&matrix, 2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_handles_vectors_and_empty_matrices() {
    let vector = [1.0, -2.0, 3.0];
    assert_eq!(transpose(&vector, 1, 3), vector);
    assert_eq!(transpose(&vector, 3, 1), vector);
    assert!(transpose(&[], 0, 3).is_empty());
    assert!(transpose(&[], 3, 0).is_empty());
}

#[test]
fn identity_has_ones_only_on_the_diagonal() {
    assert_eq!(identity(0), Vec::<f64>::new());
    assert_eq!(identity(1), [1.0]);
    assert_eq!(identity(3), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn multiply_rectangular_matrices_across_column_block_boundaries() {
    // A has rows [1, 2, 3] and [4, 5, 6]. Each group of three in B is a column.
    let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = [
        7.0, 10.0, 13.0, 8.0, 11.0, 14.0, 9.0, 12.0, 15.0, 10.0, 13.0, 16.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0,
    ];
    let expected = [
        66.0, 156.0, 72.0, 171.0, 78.0, 186.0, 84.0, 201.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.0, 2.0,
        0.0, 3.0,
    ];

    // Covers leftover columns, exact blocks of four, and multiple blocks.
    for cols in 1..=9 {
        assert_eq!(
            multiply(&a, &b[..3 * cols], 2, 3, cols),
            expected[..2 * cols],
            "incorrect product with {cols} output columns"
        );
    }
}

#[test]
fn multiply_column_by_row_produces_outer_product() {
    assert_eq!(
        multiply(&[-1.0, 2.0, 0.5], &[2.0, -3.0], 3, 1, 2),
        [-2.0, 4.0, 1.0, 3.0, -6.0, -1.5]
    );
}

#[test]
fn multiply_handles_zero_dimensions() {
    assert!(multiply(&[], &[1.0; 15], 0, 3, 5).is_empty());
    assert!(multiply(&[1.0; 6], &[], 2, 3, 0).is_empty());
    assert_eq!(multiply(&[], &[], 2, 0, 5), [0.0; 10]);
}

#[test]
#[should_panic]
fn multiply_rejects_wrong_left_buffer_length() {
    multiply(&[1.0; 5], &[1.0; 3], 2, 3, 1);
}

#[test]
#[should_panic]
fn multiply_rejects_wrong_right_buffer_length() {
    multiply(&[1.0; 6], &[1.0; 2], 2, 3, 1);
}

#[test]
fn inverse_matches_known_nonsymmetric_matrix_inverse() {
    // Rows [1, 2, 3], [0, 1, 4], [5, 6, 0]; determinant is 1.
    let matrix = [1.0, 0.0, 5.0, 2.0, 1.0, 6.0, 3.0, 4.0, 0.0];
    let expected = [-24.0, 20.0, -5.0, 18.0, -15.0, 4.0, 5.0, -4.0, 1.0];
    assert_close(&inverse(&matrix, 3).unwrap(), &expected);
}

#[test]
fn inverse_swaps_rows_when_first_pivot_is_zero() {
    // Rows [0, 2] and [1, 3].
    assert_close(
        &inverse(&[0.0, 1.0, 2.0, 3.0], 2).unwrap(),
        &[-1.5, 0.5, 1.0, 0.0],
    );
}

#[test]
fn inverse_rejects_singular_matrices() {
    assert!(inverse(&[1.0, 2.0, 2.0, 4.0], 2).is_none());
    assert!(inverse(&[1.0, 0.0, 0.0, 0.0], 2).is_none());
}

#[test]
fn inverse_rejects_nonfinite_pivots() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(inverse(&[value, 0.0, 0.0, 1.0], 2).is_none());
    }
}

#[test]
fn spd_inverse_matches_known_inverse_and_log_determinant() {
    // Symmetric tridiagonal matrix with determinant 4.
    let matrix = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
    let expected = [0.75, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.75];
    let (actual, logdet) = spd_inverse(&matrix, 3).unwrap();
    assert_close(&actual, &expected);
    assert_close(&[logdet], &[4.0_f64.ln()]);
}

#[test]
fn inverses_handle_single_element_matrices() {
    assert_close(&inverse(&[-4.0], 1).unwrap(), &[-0.25]);
    let (actual, logdet) = spd_inverse(&[4.0], 1).unwrap();
    assert_close(&actual, &[0.25]);
    assert_close(&[logdet], &[4.0_f64.ln()]);
}

#[test]
fn spd_inverse_rejects_matrices_that_are_not_positive_definite() {
    for matrix in [
        [1.0, 2.0, 2.0, 1.0],   // Indefinite.
        [1.0, 1.0, 1.0, 1.0],   // Singular.
        [-1.0, 0.0, 0.0, -1.0], // Negative definite.
        [0.0; 4],
    ] {
        assert!(spd_inverse(&matrix, 2).is_none(), "accepted {matrix:?}");
    }
}

#[test]
fn spd_inverse_rejects_nonfinite_values_in_the_lower_triangle() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(spd_inverse(&[value, 0.0, 0.0, 1.0], 2).is_none());
        assert!(spd_inverse(&[1.0, value, value, 1.0], 2).is_none());
    }
}
