use lavaan_kernels::{
    lav_model_gradient_conditional_x_sample_cache, lav_model_gradient_delta_post,
    lav_model_gradient_duplication_pre, lav_model_gradient_dwls, lav_model_gradient_group_weight,
    lav_model_gradient_ml_conditional, lav_model_gradient_ml_conditional_post,
    lav_model_gradient_ml_group, lav_model_gradient_ntrls_post, lav_model_gradient_omega_gls,
    lav_model_gradient_omega_missing_pattern, lav_model_gradient_omega_ml,
    lav_model_gradient_t_d1_delta, lav_model_gradient_wls,
};
use ndarray::Array2;

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

fn column_major_vec(array: &Array2<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(array.nrows() * array.ncols());
    for col in 0..array.ncols() {
        for row in 0..array.nrows() {
            out.push(array[(row, col)]);
        }
    }
    out
}

#[test]
fn conditional_x_sample_cache_matches_r_layout() {
    let cache = lav_model_gradient_conditional_x_sample_cache(
        &[1.0, 2.0],
        &[4.0, 1.0, 1.0, 9.0],
        &[10.0, 20.0, 30.0],
        &[0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
    )
    .unwrap();

    assert_close(
        &column_major_vec(&cache.c3),
        &[1.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 13.0],
        1e-12,
    );
    assert_close(
        &column_major_vec(&cache.obs),
        &[10.0, 0.1, 1.1, 20.0, 0.2, 1.2, 30.0, 0.3, 1.3],
        1e-12,
    );
}

#[test]
fn omega_ml_matches_reference_values_with_meanstructure() {
    let actual = lav_model_gradient_omega_ml(
        &[1.5, 0.2, 0.2, 1.1],
        &[
            0.6832298136645962,
            -0.12422360248447208,
            -0.12422360248447208,
            0.9316770186335404,
        ],
        &[2.0, 0.3, 0.3, 1.4],
        &[0.5, -0.5],
        true,
    )
    .unwrap();
    assert_close(
        &column_major_vec(&actual.omega),
        &[
            0.38405154122140356,
            -0.22510705605493614,
            -0.22510705605493614,
            0.5237066471200955,
        ],
        1e-12,
    );
    assert_close(
        &actual.omega_mu,
        &[0.4037267080745342, -0.5279503105590061],
        1e-12,
    );
}

#[test]
fn omega_ml_matches_reference_values_without_meanstructure() {
    let actual = lav_model_gradient_omega_ml(
        &[1.5, 0.2, 0.2, 1.1],
        &[
            0.6832298136645962,
            -0.12422360248447208,
            -0.12422360248447208,
            0.9316770186335404,
        ],
        &[2.0, 0.3, 0.3, 1.4],
        &[],
        false,
    )
    .unwrap();
    assert_close(
        &column_major_vec(&actual.omega),
        &[
            0.2210562864087034,
            -0.011959415146020592,
            -0.011959415146020596,
            0.24497511670074437,
        ],
        1e-12,
    );
    assert!(actual.omega_mu.is_empty());
}

#[test]
fn omega_gls_matches_reference_values_with_meanstructure() {
    let actual = lav_model_gradient_omega_gls(
        &[1.5, 0.2, 0.2, 1.1],
        &[1.2, 0.1, 0.1, 0.9],
        &[2.0, 0.3, 0.3, 1.4],
        &[0.5, -0.5],
        true,
        10.0,
    )
    .unwrap();
    assert_close(
        &column_major_vec(&actual.omega),
        &[
            0.6723,
            0.17639999999999997,
            0.17639999999999997,
            0.23939999999999992,
        ],
        1e-12,
    );
    assert_close(&actual.omega_mu, &[0.55, -0.4], 1e-12);
}

#[test]
fn group_weight_gradient_matches_reference_values() {
    let actual = lav_model_gradient_group_weight(
        &[3.4011973816621555, 3.6888794541139363, 2.995732273553991],
        &[0.3, 0.5, 0.2],
        100.0,
    )
    .unwrap();
    assert_close(&actual, &[0.0, -0.1, 0.0], 1e-12);
}

#[test]
fn omega_missing_pattern_matches_scattered_reference_values() {
    let actual = lav_model_gradient_omega_missing_pattern(
        &[1.2, 0.1, 0.1, 0.8],
        &[2.0, 0.3, 0.3, 1.4],
        &[0.5, -0.25],
        &[0, 2],
        4,
        0.3,
    )
    .unwrap();
    assert_close(
        &column_major_vec(&actual.omega),
        &[
            0.6289874999999998,
            0.0,
            0.13702499999999998,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.13702499999999998,
            0.0,
            0.055949999999999965,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        1e-12,
    );
    assert_close(&actual.omega_mu, &[0.1725, 0.0, -0.045, 0.0], 1e-12);
}

#[test]
fn delta_post_matches_scaled_crossprod() {
    let actual = lav_model_gradient_delta_post(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
        &[0.5, -1.0, 2.0],
        -0.5,
    )
    .unwrap();
    assert_close(&actual, &[-2.25, -4.5], 1e-12);
}

#[test]
fn wls_and_dwls_match_r_crossprod_formulas() {
    let delta = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0];
    let diff = [0.25, -0.5, 1.0];
    let wls_v = [2.0, 0.1, 0.3, 0.2, 3.0, 0.4, 0.5, 0.6, 4.0];

    let wls = lav_model_gradient_wls(&delta, 3, 2, &wls_v, &diff, 0.25).unwrap();
    assert_close(&wls, &[-2.53125, -4.18125], 1e-12);

    let dwls = lav_model_gradient_dwls(&delta, 3, 2, &[2.0, 3.0, 4.0], &diff, 0.25).unwrap();
    assert_close(&dwls, &[-2.375, -4.4375], 1e-12);
}

#[test]
fn duplication_pre_matches_lavaan_rule() {
    let actual = lav_model_gradient_duplication_pre(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
    assert_close(&actual, &[1.0, 5.0, 4.0], 1e-12);
}

#[test]
fn ml_conditional_post_matches_reference_values() {
    let cache =
        lav_model_gradient_conditional_x_sample_cache(&[0.5], &[2.0], &[1.0, -0.5], &[0.25, -0.75])
            .unwrap();
    let c3 = column_major_vec(&cache.c3);
    let obs = column_major_vec(&cache.obs);
    let post = lav_model_gradient_ml_conditional_post(
        &c3,
        &obs,
        2,
        1,
        &[0.2, -0.1],
        &[0.05, 0.15],
        &[1.2, -0.2, -0.2, 0.8],
        &[1.1, 0.3, 0.3, 0.9],
    )
    .unwrap();

    assert_close(
        &post,
        &[2.5, 2.93, -1.72, -3.9, 2.1913, -3.2428, 1.6188],
        1e-12,
    );
}

#[test]
fn ml_conditional_gradient_applies_lavaan_half_factor() {
    let actual =
        lav_model_gradient_ml_conditional(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[0.25, -0.75], 0.4)
            .unwrap();
    assert_close(&actual, &[0.25, 0.45], 1e-12);
}

#[test]
fn ml_group_matches_reference_with_mean_and_group_weight_padding() {
    let actual = lav_model_gradient_ml_group(
        &[
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25,
        ],
        6,
        2,
        &[1.0, 2.0, 3.0, 4.0],
        &[0.25, -0.5],
        0.75,
        true,
        true,
    )
    .unwrap();
    assert_close(&actual, &[-5.671875, -11.015625], 1e-12);
}

#[test]
fn ntrls_post_matches_reference_values_with_meanstructure() {
    let actual = lav_model_gradient_ntrls_post(
        &[2.0, 0.3, 0.3, 1.4],
        &[1.5, 0.2, 0.2, 1.1],
        &[
            0.6832298136645962,
            -0.12422360248447208,
            -0.12422360248447208,
            0.9316770186335404,
        ],
        &[0.7, -0.4],
        &[0.2, 0.1],
        true,
    )
    .unwrap();
    assert_close(
        &actual,
        &[
            0.8074534161490683,
            -1.0559006211180122,
            0.45645021267439223,
            -0.442872167007206,
            0.5887633734704177,
        ],
        1e-12,
    );
}

#[test]
fn ntrls_post_matches_reference_values_without_meanstructure() {
    let actual = lav_model_gradient_ntrls_post(
        &[2.0, 0.3, 0.3, 1.4],
        &[1.5, 0.2, 0.2, 1.1],
        &[
            0.6832298136645962,
            -0.12422360248447208,
            -0.12422360248447208,
            0.9316770186335404,
        ],
        &[],
        &[],
        false,
    )
    .unwrap();
    assert_close(
        &actual,
        &[
            0.2934549578616921,
            -0.016576885189374967,
            0.31003184305106674,
        ],
        1e-12,
    );
}

#[test]
fn t_d1_delta_matches_scaled_r_matrix_product() {
    let actual = lav_model_gradient_t_d1_delta(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3,
        2,
        &[0.5, -1.0, 2.0, 1.5, 0.25, -0.75],
        3,
        2,
        0.5,
    )
    .unwrap();
    assert_close(&actual, &[2.25, 4.5, -0.125, 1.375], 1e-12);
}
