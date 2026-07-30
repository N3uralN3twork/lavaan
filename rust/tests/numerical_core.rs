use lavaan_kernels::lav_lavaan_baseline_utils::{baseline_fiml_loglik, baseline_fiml_moments};
use lavaan_kernels::lav_model_objective::model_objective_ml;
use lavaan_kernels::lav_model_objective_gradient::model_objective_gradient_ml;
use lavaan_kernels::lav_model_objective_gradient_full::{
    full_lisrel_ml_objective_gradient, full_lisrel_ml_objective_gradient_with_diagnostics,
    full_lisrel_ml_objective_gradient_with_plan, FullMlPlan,
};

#[test]
fn baseline_fiml_moments_and_loglik_match_hand_calculation() {
    let lengths = [2, 1];
    let indices = [0, 1, 0];
    let frequencies = [2.0, 1.0];
    let means = [1.0, 4.0, 3.0];
    let sy = [0.5, 2.0, 0.25];
    let moments = baseline_fiml_moments(&lengths, &indices, &frequencies, &means, &sy, 2).unwrap();
    assert!((moments.mean[0] - 5.0 / 3.0).abs() < 1e-14);
    assert!((moments.mean[1] - 4.0).abs() < 1e-14);
    assert!(moments.var.iter().all(|value| *value > 0.0));
    assert!(baseline_fiml_loglik(
        &lengths,
        &indices,
        &frequencies,
        &means,
        &sy,
        &moments.mean,
        &moments.var
    )
    .unwrap()
    .is_finite());
}

#[test]
fn model_objective_preserves_column_major_trace() {
    let inverse = [0.5, 0.0, 0.0, 0.25];
    let sample_cov = [2.0, 0.0, 0.0, 4.0];
    let fx = model_objective_ml(&inverse, &sample_cov, &[], &[], 1.0, 0.5, false).unwrap();
    assert!((fx - 0.5).abs() < 1e-14);
}

#[test]
fn combined_ml_objective_gradient_matches_one_variable_hand_calculation() {
    let result = model_objective_gradient_ml(
        &[2.0],
        &[0.5],
        &[3.0],
        &[],
        &[],
        &[2.0_f64.ln()],
        &[3.0_f64.ln()],
        &[1.0],
        &[1],
        &[1],
        &[1],
        &[1.0],
        false,
    )
    .unwrap();
    let expected_raw = 2.0_f64.ln() + 1.5 - 3.0_f64.ln() - 1.0;
    assert!((result.objective - 0.5 * expected_raw).abs() < 1e-14);
    assert!((result.group_objectives[0] - 0.5 * expected_raw).abs() < 1e-14);
    assert!((result.gradient[0] + 0.125).abs() < 1e-14);
}

#[test]
fn full_lisrel_ml_objective_gradient_matches_one_factor_hand_calculation() {
    let result = full_lisrel_ml_objective_gradient(
        &[2.0],
        &[1.0],
        &[3.0],
        &[],
        &[],
        &[],
        &[14.0],
        &[],
        &[1],
        &[1],
        &[false],
        &[1.0],
        1.0,
        &[0],
        &[0],
        &[0],
        &[0],
        1,
        false,
    )
    .unwrap();
    let expected_raw = 13.0_f64.ln() + 14.0 / 13.0 - 14.0_f64.ln() - 1.0;
    assert!((result.objective - 0.5 * expected_raw).abs() < 1e-14);
    assert!((result.gradient[0] + 6.0 / 169.0).abs() < 1e-14);
}

#[test]
fn full_lisrel_diagnostics_preserve_the_numerical_result() {
    let diagnostics = full_lisrel_ml_objective_gradient_with_diagnostics(
        &[2.0],
        &[1.0],
        &[3.0],
        &[],
        &[],
        &[],
        &[14.0],
        &[],
        &[1],
        &[1],
        &[false],
        &[1.0],
        1.0,
        &[0],
        &[0],
        &[0],
        &[0],
        1,
        false,
    )
    .unwrap();
    let plain = full_lisrel_ml_objective_gradient(
        &[2.0],
        &[1.0],
        &[3.0],
        &[],
        &[],
        &[],
        &[14.0],
        &[],
        &[1],
        &[1],
        &[false],
        &[1.0],
        1.0,
        &[0],
        &[0],
        &[0],
        &[0],
        1,
        false,
    )
    .unwrap();
    assert!((diagnostics.result.objective - plain.objective).abs() < 1e-14);
    assert_eq!(diagnostics.result.group_objectives, plain.group_objectives);
    assert_eq!(diagnostics.result.gradient, plain.gradient);
}

#[test]
fn cached_full_lisrel_plan_preserves_objective_and_gradient() {
    let plan = FullMlPlan::new(
        vec![14.0],
        vec![],
        vec![1],
        vec![1],
        vec![false],
        vec![1.0],
        1.0,
        vec![0],
        vec![0],
        vec![0],
        vec![0],
        1,
        false,
    )
    .unwrap();
    let cached =
        full_lisrel_ml_objective_gradient_with_plan(&plan, &[2.0], &[1.0], &[3.0], &[], &[], &[])
            .unwrap();
    let plain = full_lisrel_ml_objective_gradient(
        &[2.0],
        &[1.0],
        &[3.0],
        &[],
        &[],
        &[],
        &[14.0],
        &[],
        &[1],
        &[1],
        &[false],
        &[1.0],
        1.0,
        &[0],
        &[0],
        &[0],
        &[0],
        1,
        false,
    )
    .unwrap();
    assert!((cached.objective - plain.objective).abs() < 1e-14);
    assert_eq!(cached.group_objectives, plain.group_objectives);
    assert_eq!(cached.gradient, plain.gradient);
}
