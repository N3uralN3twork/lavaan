use lavaan_kernels::{
    lav_model_vcov_delta_a_delta, lav_model_vcov_jacobian_vcov_jacobian_t, lav_model_vcov_sandwich,
};

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn delta_a_delta_matches_r_crossprod_formula() {
    let delta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a1 = [2.0, 0.5, 1.0, 0.5, 3.0, -1.0, 1.0, -1.0, 4.0];

    let actual = lav_model_vcov_delta_a_delta(&delta, &a1, 3, 2).unwrap();

    assert_close(&actual, &[46.0, 107.5, 107.5, 259.0], 1e-12);
}

#[test]
fn delta_a_delta_keeps_r_column_major_layout() {
    let delta = [1.0, -1.0, 2.0, 0.5];
    let a1 = [3.0, 0.25, 0.25, 2.0];

    let actual = lav_model_vcov_delta_a_delta(&delta, &a1, 2, 2).unwrap();

    assert_close(&actual, &[4.5, 4.625, 4.625, 13.0], 1e-12);
}

#[test]
fn delta_a_delta_rejects_delta_length_mismatch() {
    let err =
        lav_model_vcov_delta_a_delta(&[1.0, 2.0, 3.0], &[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap_err();

    assert!(
        err.contains("delta length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn delta_a_delta_rejects_a1_length_mismatch() {
    let err =
        lav_model_vcov_delta_a_delta(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0], 2, 2).unwrap_err();

    assert!(
        err.contains("a1 length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn sandwich_matches_r_matrix_product_order() {
    let left = [1.0, 3.0, 2.0, 4.0];
    let middle = [2.0, 0.5, 0.5, 3.0];

    let actual = lav_model_vcov_sandwich(&left, &middle, 2).unwrap();

    assert_close(&actual, &[22.5, 48.5, 32.0, 70.0], 1e-12);
}

#[test]
fn sandwich_rejects_middle_length_mismatch() {
    let err = lav_model_vcov_sandwich(&[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0], 2).unwrap_err();

    assert!(
        err.contains("middle length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn jacobian_vcov_jacobian_t_matches_r_delta_method_product() {
    let jac = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vcov = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];

    let actual = lav_model_vcov_jacobian_vcov_jacobian_t(&jac, &vcov, 2, 3).unwrap();

    assert_close(&actual, &[129.0, 160.0, 160.0, 200.0], 1e-12);
}

#[test]
fn jacobian_vcov_jacobian_t_rejects_jac_length_mismatch() {
    let err =
        lav_model_vcov_jacobian_vcov_jacobian_t(&[1.0, 2.0, 3.0], &[1.0, 0.0, 0.0, 1.0], 2, 2)
            .unwrap_err();

    assert!(
        err.contains("jac length mismatch"),
        "unexpected error: {err}"
    );
}
