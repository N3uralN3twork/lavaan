use lavaan_kernels::{
    lav_model_estimate_gradient_postprocess, lav_model_estimate_pack,
    lav_model_estimate_unpack_unscale,
};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-12,
            "value {index} differs: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn pack_matches_r_row_vector_matrix_product() {
    let actual = lav_model_estimate_pack(
        &[3.0, 5.0, 7.0],
        &[1.0, 1.0, 1.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
    )
    .unwrap();
    assert_close(&actual, &[28.0, 64.0]);
}

#[test]
fn unpack_unscale_matches_r_matrix_vector_product() {
    let actual = lav_model_estimate_unpack_unscale(
        &[2.0, 3.0],
        &[1.0, -1.0, 0.5],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2.0, 4.0, 5.0],
        3,
        2,
    )
    .unwrap();
    assert_close(&actual, &[7.5, 4.5, 4.9]);
}

#[test]
fn gradient_postprocess_matches_r_order_of_operations() {
    let actual = lav_model_estimate_gradient_postprocess(
        &[2.0, 4.0, 8.0],
        &[2.0, 4.0, 8.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
        10.0,
        true,
    )
    .unwrap();
    assert_close(&actual, &[0.6, 1.5]);
}

#[test]
fn no_constraints_still_unscales() {
    let actual =
        lav_model_estimate_unpack_unscale(&[2.0, 4.0], &[], &[], &[2.0, 4.0], 0, 0).unwrap();
    assert_close(&actual, &[1.0, 1.0]);
}

#[test]
fn invalid_dimensions_are_rejected() {
    assert!(lav_model_estimate_pack(
        &[1.0, 2.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
    )
    .is_err());
    assert!(lav_model_estimate_unpack_unscale(
        &[1.0, 2.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1.0, 2.0],
        3,
        2,
    )
    .is_err());
}
