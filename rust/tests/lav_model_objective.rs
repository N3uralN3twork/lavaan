use lavaan_kernels::{
    lav_model_objective_dwls, lav_model_objective_gls, lav_model_objective_ml,
    lav_model_objective_ml_res, lav_model_objective_wls,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn ml_objective_matches_covariance_formula() {
    let sigma = [1.0, 0.0, 0.0, 1.0];
    let sigma_inv = [1.0, 0.0, 0.0, 1.0];
    let data_cov = [2.0, 0.5, 0.5, 3.0];

    assert_close(
        lav_model_objective_ml(&sigma, &sigma_inv, &data_cov, &[], &[], 1.2, 0.7, false, 2)
            .unwrap(),
        3.5,
    );
}

#[test]
fn ml_objective_matches_mean_structure_formula() {
    let sigma = [1.0, 0.0, 0.0, 1.0];
    let sigma_inv = [1.0, 0.0, 0.0, 1.0];
    let data_cov = [2.0, 0.5, 0.5, 3.0];

    assert_close(
        lav_model_objective_ml(
            &sigma,
            &sigma_inv,
            &data_cov,
            &[1.0, 1.0],
            &[2.0, 3.0],
            1.2,
            0.7,
            true,
            2,
        )
        .unwrap(),
        8.5,
    );
}

#[test]
fn conditional_ml_objective_matches_matrix_formula() {
    let result = lav_model_objective_ml_res(
        &[1.0, 0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[2.0, 0.0, 0.0, 3.0],
        &[2.0, 3.0],
        &[4.0, 5.0],
        &[1.0, 1.0],
        &[3.0, 4.0],
        &[2.0],
        &[0.5],
        1.2,
        0.7,
        2,
        1,
    )
    .unwrap();
    assert_close(result, 16.0);
}

#[test]
fn gls_objective_supports_correlation_adjustment() {
    let sigma = [1.0, 0.0, 0.0, 1.0];
    let data_cov = [2.0, 0.0, 0.0, 3.0];
    let data_cov_inv = [1.0, 0.0, 0.0, 1.0];

    assert_close(
        lav_model_objective_gls(&sigma, &data_cov, &data_cov_inv, &[], &[], false, false, 2)
            .unwrap(),
        2.5,
    );
    assert_close(
        lav_model_objective_gls(&sigma, &data_cov, &data_cov_inv, &[], &[], false, true, 2)
            .unwrap(),
        7.0 / 6.0,
    );
}

#[test]
fn gls_objective_preserves_nonsymmetric_matrix_product_order() {
    assert_close(
        lav_model_objective_gls(
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 3.0, 2.0, 4.0],
            &[5.0, 7.0, 6.0, 8.0],
            &[],
            &[],
            false,
            false,
            2,
        )
        .unwrap(),
        2376.5,
    );
}

#[test]
fn wls_and_dwls_objectives_match_quadratic_forms() {
    assert_close(
        lav_model_objective_wls(&[0.0, 0.0], &[1.0, 2.0], &[2.0, 0.0, 0.0, 3.0], 2).unwrap(),
        14.0,
    );
    assert_close(
        lav_model_objective_dwls(&[0.0, 0.0], &[1.0, 2.0], &[0.5, 3.0]).unwrap(),
        12.5,
    );
}
