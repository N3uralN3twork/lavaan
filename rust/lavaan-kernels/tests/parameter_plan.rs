use lavaan_kernels::lisrel::FullMlPlan;

fn fixture() -> (FullMlPlan, [Vec<f64>; 6], Vec<f64>) {
    let plan = FullMlPlan::new(
        &[2.0, 0.3, 0.3, 1.8],
        &[1.0, 2.0],
        &[2],
        &[1],
        &[true],
        &[100.0],
        100.0,
        &[0; 10],
        &[0, 1, 1, 2, 4, 4, 5, 1, 1, 3],
        &[1, 0, 3, 0, 0, 1, 0, 1, 2, 0],
        &[0, 1, 2, 3, 4, 5, 6, 7, 7, 8],
        9,
        true,
    )
    .unwrap();
    let matrices = [
        vec![1.0, 0.7],
        vec![0.4, 0.1, 0.1, 0.8],
        vec![1.3],
        vec![0.1],
        vec![0.1, 0.2],
        vec![0.3],
    ];
    (
        plan,
        matrices,
        vec![0.7, 0.4, 0.8, 1.3, 0.1, 0.2, 0.3, 0.1, 0.1],
    )
}

#[test]
fn parameters_match_matrices_and_numerical_derivatives() {
    let (plan, matrices, x) = fixture();
    let reference = plan
        .evaluate(std::array::from_fn(|i| matrices[i].as_slice()))
        .unwrap();
    plan.set_templates(matrices).unwrap();
    assert_eq!(plan.evaluate_parameters(&x).unwrap(), reference);
    for i in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        let derivative = (plan.evaluate_parameters(&plus).unwrap()[0]
            - plan.evaluate_parameters(&minus).unwrap()[0])
            / 2e-6;
        assert!(
            (derivative - reference[2 + i]).abs() < 2e-7,
            "parameter {i}: numerical {derivative}, analytical {}",
            reference[2 + i]
        );
    }
}

#[test]
fn repeated_rejected_and_invalid_evaluations_preserve_fixed_values() {
    let (plan, matrices, x) = fixture();
    plan.set_templates(matrices.clone()).unwrap();
    let first = plan.evaluate_parameters(&x).unwrap();
    assert!(plan.evaluate_parameters(&x[..8]).is_err());
    let mut bad = x.clone();
    bad[0] = f64::NAN;
    assert!(plan.evaluate_parameters(&bad).is_err());
    bad = x.clone();
    bad[1] = -100.0;
    assert!(plan.evaluate_parameters(&bad).unwrap()[0].is_infinite());
    let mut other = x.clone();
    other[0] += 0.15;
    assert_ne!(plan.evaluate_parameters(&other).unwrap(), first);
    assert_eq!(plan.evaluate_parameters(&x).unwrap(), first);
    assert!(plan.set_templates(matrices).is_err());
}

#[test]
fn group_offsets_and_shared_parameter_accumulation_match_single_group() {
    let (single, matrices, x) = fixture();
    single.set_templates(matrices.clone()).unwrap();
    let first = single.evaluate_parameters(&x).unwrap();
    let maps = [0, 1, 2, 3, 4, 5, 6, 7, 7, 8];
    let plan = FullMlPlan::new(
        &[2.0, 0.3, 0.3, 1.8, 2.0, 0.3, 0.3, 1.8],
        &[1.0, 2.0, 1.0, 2.0],
        &[2, 2],
        &[1, 1],
        &[true, true],
        &[60.0, 40.0],
        100.0,
        &[vec![0; 10], vec![1; 10]].concat(),
        &[0, 1, 1, 2, 4, 4, 5, 1, 1, 3].repeat(2),
        &[1, 0, 3, 0, 0, 1, 0, 1, 2, 0].repeat(2),
        &maps.repeat(2),
        9,
        true,
    )
    .unwrap();
    plan.set_templates(matrices.map(|x| x.repeat(2))).unwrap();
    let result = plan.evaluate_parameters(&x).unwrap();
    assert!((result[0] - first[0]).abs() < 1e-14);
    assert_eq!(result[1], result[2]);
    for (a, b) in result[3..].iter().zip(&first[2..]) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn malformed_contracts_fail_without_indexing() {
    assert!(FullMlPlan::new(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        1.0,
        &[],
        &[],
        &[],
        &[],
        0,
        false
    )
    .is_err());
    let (plan, mut matrices, x) = fixture();
    assert!(plan.evaluate_parameters(&x).is_err());
    matrices[1].pop();
    assert!(plan.set_templates(matrices).is_err());
    assert!(FullMlPlan::new(
        &[2.0, 0.3, 0.3, 1.8],
        &[],
        &[2],
        &[1],
        &[false],
        &[100.0],
        100.0,
        &[0],
        &[1],
        &[100],
        &[0],
        1,
        false
    )
    .is_err());
}
