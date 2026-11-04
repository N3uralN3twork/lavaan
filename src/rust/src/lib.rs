//! Thin extendr boundary. Numerical state is owned by the pure Rust core.
// Existing packed interfaces keep their R argument lists for compatibility.
#![allow(clippy::too_many_arguments)]
use extendr_api::prelude::*;
use lavaan_kernels::linalg::{inverse, multiply, transpose};
use lavaan_kernels::lisrel::FullMlPlan;
use std::time::Instant;
type Result<T> = std::result::Result<T, Error>;
type MissingBuffers = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>);

fn numeric(x: &Robj) -> Result<Vec<f64>> {
    if let Some(v) = x.as_real_slice() {
        return Ok(v.to_vec());
    }
    if let Some(v) = x.as_integer_slice() {
        return Ok(v.iter().map(|&v| v as f64).collect());
    }
    Err(Error::Other("expected a numeric buffer".into()))
}
fn indices(x: &Robj) -> Result<Vec<usize>> {
    x.as_integer_slice()
        .ok_or_else(|| Error::Other("expected integer indices".into()))?
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| Error::Other("indices must be nonnegative".into()))
        })
        .collect()
}
fn shape(x: &Robj) -> Result<(usize, usize)> {
    let dim = x
        .get_attrib("dim")
        .ok_or_else(|| Error::Other("expected a matrix".into()))?;
    let dims = indices(&dim)?;
    if dims.len() != 2 {
        return Err(Error::Other("expected two matrix dimensions".into()));
    }
    Ok((dims[0], dims[1]))
}
fn matrix(values: Vec<f64>, rows: usize, cols: usize) -> Result<Robj> {
    let mut out = r!(values);
    out.set_attrib("dim", [rows as i32, cols as i32])?;
    Ok(out)
}
fn checked_plan(plan: &Robj) -> Result<&ExternalPtr<FullMlPlan>> {
    // Check the tag before extendr downcasts Box<dyn Any>. This also rejects
    // foreign pointers and serialized/finalized handles without dereferencing.
    if !plan.check_external_ptr_type::<FullMlPlan>() {
        return Err(Error::Other(
            "invalid lavaan LISREL native plan handle".into(),
        ));
    }
    <&ExternalPtr<FullMlPlan>>::try_from(plan)
        .map_err(|_| Error::Other("invalid lavaan LISREL native plan handle".into()))
}
#[allow(clippy::too_many_arguments)]
fn plan_from_inputs(
    sample_cov: &Robj,
    sample_mean: &Robj,
    nvar: &Robj,
    nfac: &Robj,
    beta_present: &Robj,
    nobs: &Robj,
    ntotal: f64,
    free_group: &Robj,
    free_kind: &Robj,
    free_element: &Robj,
    free_parameter: &Robj,
    nfree: i32,
    meanstructure: bool,
) -> Result<FullMlPlan> {
    let nfree =
        usize::try_from(nfree).map_err(|_| Error::Other("nfree must be positive".into()))?;
    let beta = indices(beta_present)?;
    if beta.iter().any(|&b| b > 1) {
        return Err(Error::Other("beta_present must contain zero or one".into()));
    }
    FullMlPlan::new(
        &numeric(sample_cov)?,
        &numeric(sample_mean)?,
        &indices(nvar)?,
        &indices(nfac)?,
        &beta.iter().map(|&b| b == 1).collect::<Vec<_>>(),
        &numeric(nobs)?,
        ntotal,
        &indices(free_group)?,
        &indices(free_kind)?,
        &indices(free_element)?,
        &indices(free_parameter)?,
        nfree,
        meanstructure,
    )
    .map_err(Error::Other)
}

/// Returns whether the package-native Rust backend is loaded.
/// @export
#[extendr]
fn lav_rust_backend_ping() -> bool {
    true
}

/// Construct immutable sample statistics and free-element maps once per fit.
/// @export
#[extendr]
fn lav_rust_lisrel_ml_plan_new(
    sample_cov: Robj,
    sample_mean: Robj,
    nvar: Robj,
    nfac: Robj,
    beta_present: Robj,
    nobs: Robj,
    ntotal: f64,
    free_group: Robj,
    free_kind: Robj,
    free_element: Robj,
    free_parameter: Robj,
    nfree: i32,
    meanstructure: bool,
) -> Result<ExternalPtr<FullMlPlan>> {
    Ok(ExternalPtr::new(plan_from_inputs(
        &sample_cov,
        &sample_mean,
        &nvar,
        &nfac,
        &beta_present,
        &nobs,
        ntotal,
        &free_group,
        &free_kind,
        &free_element,
        &free_parameter,
        nfree,
        meanstructure,
    )?))
}

/// Evaluate packed model matrices using a persistent native plan.
/// @export
#[extendr]
fn lav_rust_model_objective_gradient_lisrel_ml_with_plan(
    plan: Robj,
    lambda: Robj,
    theta: Robj,
    psi: Robj,
    beta: Robj,
    nu: Robj,
    alpha: Robj,
) -> Result<Vec<f64>> {
    let plan = checked_plan(&plan)?;
    let values = [
        numeric(&lambda)?,
        numeric(&theta)?,
        numeric(&psi)?,
        numeric(&beta)?,
        numeric(&nu)?,
        numeric(&alpha)?,
    ];
    plan.evaluate(std::array::from_fn(|i| values[i].as_slice()))
        .map_err(Error::Other)
}

/// Attach owned matrix templates once, before optimizer evaluations begin.
#[extendr]
fn lav_rust_lisrel_ml_plan_set_templates(
    plan: Robj,
    lambda: Robj,
    theta: Robj,
    psi: Robj,
    beta: Robj,
    nu: Robj,
    alpha: Robj,
) -> Result<bool> {
    checked_plan(&plan)?
        .set_templates([
            numeric(&lambda)?,
            numeric(&theta)?,
            numeric(&psi)?,
            numeric(&beta)?,
            numeric(&nu)?,
            numeric(&alpha)?,
        ])
        .map_err(Error::Other)?;
    Ok(true)
}

/// Evaluate from the free-parameter vector using model matrices owned by Rust.
#[extendr]
fn lav_rust_model_objective_gradient_lisrel_ml_parameters(
    plan: Robj,
    parameters: Robj,
) -> Result<Vec<f64>> {
    let plan = checked_plan(&plan)?;
    let x = parameters
        .as_real_slice()
        .ok_or_else(|| Error::Other("expected a numeric parameter vector".into()))?;
    plan.evaluate_parameters(x).map_err(Error::Other)
}

/// Fully numerical continuous LISREL ML objective and gradient.
/// @export
#[extendr]
fn lav_rust_model_objective_gradient_lisrel_ml(
    lambda: Robj,
    theta: Robj,
    psi: Robj,
    beta: Robj,
    nu: Robj,
    alpha: Robj,
    sample_cov: Robj,
    sample_mean: Robj,
    nvar: Robj,
    nfac: Robj,
    beta_present: Robj,
    nobs: Robj,
    ntotal: f64,
    free_group: Robj,
    free_kind: Robj,
    free_element: Robj,
    free_parameter: Robj,
    nfree: i32,
    meanstructure: bool,
) -> Result<Vec<f64>> {
    let plan = plan_from_inputs(
        &sample_cov,
        &sample_mean,
        &nvar,
        &nfac,
        &beta_present,
        &nobs,
        ntotal,
        &free_group,
        &free_kind,
        &free_element,
        &free_parameter,
        nfree,
        meanstructure,
    )?;
    let values = [
        numeric(&lambda)?,
        numeric(&theta)?,
        numeric(&psi)?,
        numeric(&beta)?,
        numeric(&nu)?,
        numeric(&alpha)?,
    ];
    plan.evaluate(std::array::from_fn(|i| values[i].as_slice()))
        .map_err(Error::Other)
}

/// Same result plus six diagnostic stage timings (milliseconds).
/// @export
#[extendr]
fn lav_rust_model_objective_gradient_lisrel_ml_diagnostics(
    lambda: Robj,
    theta: Robj,
    psi: Robj,
    beta: Robj,
    nu: Robj,
    alpha: Robj,
    sample_cov: Robj,
    sample_mean: Robj,
    nvar: Robj,
    nfac: Robj,
    beta_present: Robj,
    nobs: Robj,
    ntotal: f64,
    free_group: Robj,
    free_kind: Robj,
    free_element: Robj,
    free_parameter: Robj,
    nfree: i32,
    meanstructure: bool,
) -> Result<Vec<f64>> {
    let start = Instant::now();
    let values = [
        numeric(&lambda)?,
        numeric(&theta)?,
        numeric(&psi)?,
        numeric(&beta)?,
        numeric(&nu)?,
        numeric(&alpha)?,
    ];
    let conversion_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let plan = plan_from_inputs(
        &sample_cov,
        &sample_mean,
        &nvar,
        &nfac,
        &beta_present,
        &nobs,
        ntotal,
        &free_group,
        &free_kind,
        &free_element,
        &free_parameter,
        nfree,
        meanstructure,
    )?;
    let plan_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let mut out = plan
        .evaluate(std::array::from_fn(|i| values[i].as_slice()))
        .map_err(Error::Other)?;
    // The recreated core fuses the numerical stages. Report their combined time
    // in factorization/objective instead of inventing individual measurements.
    out.extend_from_slice(&[
        conversion_ms,
        plan_ms,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
        0.0,
        0.0,
    ]);
    Ok(out)
}

/// Standard ML discrepancy (before lavaan's one-half scaling).
/// @export
#[extendr]
fn lav_rust_model_objective_ml(
    sigma_hat: Robj,
    sigma_hat_inv: Robj,
    data_cov: Robj,
    mu_hat: Robj,
    data_mean: Robj,
    sigma_hat_log_det: f64,
    data_cov_log_det: f64,
    meanstructure: bool,
) -> Result<f64> {
    let (p, _) = shape(&sigma_hat)?;
    let si = numeric(&sigma_hat_inv)?;
    let sample = numeric(&data_cov)?;
    if si.len() != p * p || sample.len() != p * p {
        return Err(Error::Other("invalid ML matrix lengths".into()));
    }
    let mut value = sigma_hat_log_det - data_cov_log_det - p as f64
        + si.iter().zip(sample).map(|(a, b)| a * b).sum::<f64>();
    if meanstructure {
        let m = numeric(&mu_hat)?;
        let y = numeric(&data_mean)?;
        if m.len() != p || y.len() != p {
            return Err(Error::Other("invalid ML mean lengths".into()));
        }
        let diff: Vec<_> = y.iter().zip(m).map(|(y, m)| y - m).collect();
        let q = multiply(&si, &diff, p, p, 1);
        value += diff.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
    }
    Ok(if value.is_finite() {
        value.max(0.0)
    } else {
        value
    })
}

#[extendr]
fn lav_rust_model_estimate_pack(values: Robj, k0: Robj, k: Robj) -> Result<Vec<f64>> {
    let mut values = numeric(&values)?;
    let offset = numeric(&k0)?;
    if offset.is_empty() {
        return Ok(values);
    }
    if values.len() != offset.len() {
        return Err(Error::Other("invalid parameter offset length".into()));
    }
    for (x, b) in values.iter_mut().zip(offset) {
        *x -= b;
    }
    let (rows, cols) = shape(&k)?;
    if rows != values.len() {
        return Err(Error::Other("invalid parameter map dimensions".into()));
    }
    Ok(multiply(&values, &numeric(&k)?, 1, rows, cols))
}

#[extendr]
fn lav_rust_model_objective_ml_res(
    sigma_hat: Robj,
    sigma_hat_inv: Robj,
    res_cov: Robj,
    res_int: Robj,
    res_slopes: Robj,
    mu_hat: Robj,
    pi0: Robj,
    cov_x: Robj,
    mean_x: Robj,
    sigma_hat_log_det: f64,
    res_cov_log_det: f64,
) -> Result<f64> {
    let (p, _) = shape(&sigma_hat)?;
    let (rows, q) = shape(&res_slopes)?;
    if rows != p || shape(&pi0)? != (p, q) || shape(&cov_x)? != (q, q) {
        return Err(Error::Other("invalid conditional ML dimensions".into()));
    }
    let si = numeric(&sigma_hat_inv)?;
    let sample = numeric(&res_cov)?;
    let intercept = numeric(&res_int)?;
    let mu = numeric(&mu_hat)?;
    let slopes = numeric(&res_slopes)?;
    let pi = numeric(&pi0)?;
    let cx = numeric(&cov_x)?;
    let mx = numeric(&mean_x)?;
    if si.len() != p * p
        || sample.len() != p * p
        || intercept.len() != p
        || mu.len() != p
        || mx.len() != q
    {
        return Err(Error::Other("invalid conditional ML buffer lengths".into()));
    }
    let n = q + 1;
    let mut c3 = vec![0.0; n * n];
    c3[0] = 1.0;
    for j in 0..q {
        c3[j + 1] = mx[j];
        c3[(j + 1) * n] = mx[j];
        for i in 0..q {
            c3[i + 1 + (j + 1) * n] = cx[i + j * q] + mx[i] * mx[j];
        }
    }
    let mut diff = vec![0.0; n * p];
    for j in 0..p {
        diff[j * n] = intercept[j] - mu[j];
        for i in 0..q {
            diff[i + 1 + j * n] = slopes[j + i * p] - pi[j + i * p];
        }
    }
    let left = multiply(&c3, &diff, n, n, p);
    let right = multiply(&diff, &si, n, p, p);
    Ok((sigma_hat_log_det - res_cov_log_det - p as f64
        + si.iter().zip(sample).map(|(a, b)| a * b).sum::<f64>()
        + left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>())
    .max(0.0))
}

/// Combined objective and delta-method gradient from packed implied moments.
#[extendr]
fn lav_rust_model_objective_gradient_ml(
    sigma: Robj,
    sigma_inv: Robj,
    data_cov: Robj,
    mu_hat: Robj,
    data_mean: Robj,
    sigma_log_det: Robj,
    data_cov_log_det: Robj,
    delta: Robj,
    nvar: Robj,
    delta_rows: Robj,
    delta_cols: Robj,
    group_weights: Robj,
    meanstructure: bool,
) -> Result<Vec<f64>> {
    let p = indices(&nvar)?;
    let dr = indices(&delta_rows)?;
    let dc = indices(&delta_cols)?;
    let ng = p.len();
    if ng == 0 || dr.len() != ng || dc.len() != ng || dc.iter().any(|&c| c != dc[0]) {
        return Err(Error::Other("invalid delta group dimensions".into()));
    }
    let s = numeric(&sigma)?;
    let si = numeric(&sigma_inv)?;
    let sample = numeric(&data_cov)?;
    let mu = numeric(&mu_hat)?;
    let mean = numeric(&data_mean)?;
    let ld = numeric(&sigma_log_det)?;
    let sd = numeric(&data_cov_log_det)?;
    let d = numeric(&delta)?;
    let weights = numeric(&group_weights)?;
    let ns: usize = p.iter().map(|p| p * p).sum();
    let nm: usize = if meanstructure { p.iter().sum() } else { 0 };
    if s.len() != ns
        || si.len() != ns
        || sample.len() != ns
        || mu.len() != nm
        || mean.len() != nm
        || ld.len() != ng
        || sd.len() != ng
        || weights.len() != ng
        || d.len() != dr.iter().zip(&dc).map(|(a, b)| a * b).sum::<usize>()
    {
        return Err(Error::Other(
            "invalid objective/gradient buffer lengths".into(),
        ));
    }
    let mut out = vec![0.0; 1 + ng + dc[0]];
    let (mut off, mut moff, mut doff) = (0, 0, 0);
    for g in 0..ng {
        let n = p[g];
        let inv = &si[off..off + n * n];
        let cov = &sample[off..off + n * n];
        let diff: Vec<_> = if meanstructure {
            (0..n).map(|i| mu[moff + i] - mean[moff + i]).collect()
        } else {
            vec![0.0; n]
        };
        let q = multiply(inv, &diff, n, n, 1);
        let fx = 0.5
            * (ld[g] - sd[g] - n as f64
                + inv.iter().zip(cov).map(|(a, b)| a * b).sum::<f64>()
                + diff.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>());
        out[g + 1] = fx.max(0.0);
        out[0] += weights[g] * out[g + 1];
        let sisi = multiply(&multiply(inv, cov, n, n, n), inv, n, n, n);
        let score: Vec<_> = (0..n * n)
            .map(|i| 0.5 * (inv[i] - sisi[i] - q[i % n] * q[i / n]))
            .collect();
        let mut post = if meanstructure { q } else { Vec::new() };
        for col in 0..n {
            for row in col..n {
                post.push(if row == col {
                    score[row + col * n]
                } else {
                    score[row + col * n] + score[col + row * n]
                });
            }
        }
        if post.len() != dr[g] {
            return Err(Error::Other("invalid delta row count".into()));
        }
        for j in 0..dc[g] {
            out[ng + 1 + j] += weights[g]
                * post
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * d[doff + i + j * dr[g]])
                    .sum::<f64>();
        }
        off += n * n;
        if meanstructure {
            moff += n;
        }
        doff += dr[g] * dc[g];
    }
    Ok(out)
}

fn missing_buffers(
    lengths: &Robj,
    observed: &Robj,
    freq: &Robj,
    means: &Robj,
    diagonals: &Robj,
    n: usize,
) -> Result<MissingBuffers> {
    let lengths = indices(lengths)?;
    let obs = indices(observed)?;
    let f = numeric(freq)?;
    let m = numeric(means)?;
    let sy = numeric(diagonals)?;
    if lengths.len() != f.len()
        || obs.len() != lengths.iter().sum::<usize>()
        || obs.len() != m.len()
        || obs.len() != sy.len()
        || obs.iter().any(|&i| i >= n)
        || f.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || m.iter().chain(&sy).any(|v| !v.is_finite())
    {
        return Err(Error::Other("invalid missing-data moment buffers".into()));
    }
    Ok((lengths, obs, f, m, sy))
}
#[extendr]
fn lav_rust_baseline_fiml_moments(
    pattern_lengths: Robj,
    observed_indices: Robj,
    frequencies: Robj,
    means: Robj,
    sy_diagonals: Robj,
    nvar: i32,
) -> Result<Vec<f64>> {
    let n = usize::try_from(nvar).map_err(|_| Error::Other("nvar must be positive".into()))?;
    if n == 0 {
        return Err(Error::Other("nvar must be positive".into()));
    }
    let (lengths, obs, f, m, sy) = missing_buffers(
        &pattern_lengths,
        &observed_indices,
        &frequencies,
        &means,
        &sy_diagonals,
        n,
    )?;
    let mut count = vec![0.0; n];
    let mut mean = vec![0.0; n];
    let mut off = 0;
    for (g, &len) in lengths.iter().enumerate() {
        for i in off..off + len {
            count[obs[i]] += f[g];
            mean[obs[i]] += f[g] * m[i];
        }
        off += len;
    }
    if count.iter().any(|&c| c <= 0.0) {
        return Err(Error::Other("a variable has no observed moments".into()));
    }
    for i in 0..n {
        mean[i] /= count[i];
    }
    let mut var = vec![0.0; n];
    off = 0;
    for (g, &len) in lengths.iter().enumerate() {
        for i in off..off + len {
            let diff = m[i] - mean[obs[i]];
            var[obs[i]] += f[g] * (sy[i] + diff * diff);
        }
        off += len;
    }
    for i in 0..n {
        var[i] /= count[i];
    }
    if var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::Other("invalid missing-data variances".into()));
    }
    mean.extend(var);
    Ok(mean)
}
#[extendr]
fn lav_rust_baseline_fiml_loglik(
    pattern_lengths: Robj,
    observed_indices: Robj,
    frequencies: Robj,
    means: Robj,
    sy_diagonals: Robj,
    mean: Robj,
    var: Robj,
) -> Result<f64> {
    let mean = numeric(&mean)?;
    let var = numeric(&var)?;
    let n = mean.len();
    if var.len() != n || var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::Other("invalid missing-data variances".into()));
    }
    let (lengths, obs, f, m, sy) = missing_buffers(
        &pattern_lengths,
        &observed_indices,
        &frequencies,
        &means,
        &sy_diagonals,
        n,
    )?;
    let mut out = 0.0;
    let mut off = 0;
    for (g, &len) in lengths.iter().enumerate() {
        for i in off..off + len {
            let j = obs[i];
            let diff = m[i] - mean[j];
            out -= 0.5
                * f[g]
                * ((2.0 * std::f64::consts::PI).ln()
                    + var[j].ln()
                    + (sy[i] + diff * diff) / var[j]);
        }
        off += len;
    }
    Ok(out)
}

#[extendr]
fn lav_rust_model_objective_wls(wls_est: Robj, wls_obs: Robj, wls_v: Robj) -> Result<f64> {
    let est = numeric(&wls_est)?;
    let obs = numeric(&wls_obs)?;
    if est.len() != obs.len() || shape(&wls_v)? != (est.len(), est.len()) {
        return Err(Error::Other("invalid WLS dimensions".into()));
    }
    let d: Vec<_> = obs.iter().zip(est).map(|(a, b)| a - b).collect();
    let wd = multiply(&numeric(&wls_v)?, &d, d.len(), d.len(), 1);
    Ok(d.iter().zip(wd).map(|(a, b)| a * b).sum::<f64>().max(0.0))
}
#[extendr]
fn lav_rust_model_objective_dwls(wls_est: Robj, wls_obs: Robj, wls_vd: Robj) -> Result<f64> {
    let est = numeric(&wls_est)?;
    let obs = numeric(&wls_obs)?;
    let w = numeric(&wls_vd)?;
    if est.len() != obs.len() || est.len() != w.len() {
        return Err(Error::Other("invalid DWLS lengths".into()));
    }
    Ok(obs
        .iter()
        .zip(est)
        .zip(w)
        .map(|((a, b), w)| (a - b) * (a - b) * w)
        .sum::<f64>()
        .max(0.0))
}
#[extendr]
fn lav_rust_model_objective_gls(
    sigma_hat: Robj,
    data_cov: Robj,
    data_cov_inv: Robj,
    mu_hat: Robj,
    data_mean: Robj,
    meanstructure: bool,
    correlation: bool,
) -> Result<f64> {
    let (p, _) = shape(&sigma_hat)?;
    if shape(&data_cov)? != (p, p) || shape(&data_cov_inv)? != (p, p) {
        return Err(Error::Other("invalid GLS dimensions".into()));
    }
    let sigma = numeric(&sigma_hat)?;
    let s = numeric(&data_cov)?;
    let si = numeric(&data_cov_inv)?;
    let d: Vec<_> = s.iter().zip(sigma).map(|(a, b)| a - b).collect();
    let t = multiply(&si, &d, p, p, p);
    let mut fx = 0.0;
    for j in 0..p {
        for i in 0..p {
            fx += 0.5 * t[i + j * p] * t[j + i * p];
        }
    }
    if correlation {
        let mut tt: Vec<_> = s.iter().zip(&si).map(|(a, b)| a * b).collect();
        for i in 0..p {
            tt[i + i * p] += 1.0;
        }
        let tt_inv = inverse(&tt, p)
            .ok_or_else(|| Error::Other("singular GLS correlation correction".into()))?;
        let diag: Vec<_> = (0..p).map(|i| t[i + i * p]).collect();
        fx -= diag
            .iter()
            .zip(multiply(&tt_inv, &diag, p, p, 1))
            .map(|(a, b)| a * b)
            .sum::<f64>();
    }
    if meanstructure {
        let mu = numeric(&mu_hat)?;
        let y = numeric(&data_mean)?;
        if mu.len() != p || y.len() != p {
            return Err(Error::Other("invalid GLS mean lengths".into()));
        }
        let diff: Vec<_> = mu.iter().zip(y).map(|(a, b)| a - b).collect();
        fx += diff
            .iter()
            .zip(multiply(&si, &diff, p, p, 1))
            .map(|(a, b)| a * b)
            .sum::<f64>();
    }
    Ok(fx.max(0.0))
}
#[extendr]
fn lav_rust_model_estimate_unpack_unscale(
    values: Robj,
    k0: Robj,
    k: Robj,
    parscale: Robj,
) -> Result<Vec<f64>> {
    let values = numeric(&values)?;
    let offset = numeric(&k0)?;
    let mut out = if offset.is_empty() {
        values
    } else {
        let (rows, cols) = shape(&k)?;
        if cols != values.len() || rows != offset.len() {
            return Err(Error::Other("invalid parameter map dimensions".into()));
        }
        let mut out = multiply(&numeric(&k)?, &values, rows, cols, 1);
        for (x, b) in out.iter_mut().zip(offset) {
            *x += b;
        }
        out
    };
    let scale = numeric(&parscale)?;
    if out.len() != scale.len() {
        return Err(Error::Other("invalid parameter scale length".into()));
    }
    for (x, s) in out.iter_mut().zip(scale) {
        *x /= s;
    }
    Ok(out)
}
#[extendr]
fn lav_rust_model_estimate_gradient_postprocess(
    dx: Robj,
    parscale: Robj,
    k: Robj,
    total_nobs: f64,
    pml: bool,
) -> Result<Vec<f64>> {
    let mut dx = numeric(&dx)?;
    let scale = numeric(&parscale)?;
    if dx.len() != scale.len() {
        return Err(Error::Other("invalid gradient scale length".into()));
    }
    for (x, s) in dx.iter_mut().zip(scale) {
        *x /= s;
    }
    let (rows, cols) = shape(&k)?;
    if rows > 0 {
        if dx.len() != rows {
            return Err(Error::Other("invalid gradient map dimensions".into()));
        }
        dx = multiply(&dx, &numeric(&k)?, 1, rows, cols);
    }
    if pml {
        for x in &mut dx {
            *x /= total_nobs;
        }
    }
    Ok(dx)
}

#[extendr]
fn lav_rust_model_vcov_delta_a_delta(delta: Robj, a1: Robj) -> Result<Robj> {
    let (rows, cols) = shape(&delta)?;
    if shape(&a1)? != (rows, rows) {
        return Err(Error::Other("invalid weight dimensions".into()));
    }
    let d = numeric(&delta)?;
    matrix(
        multiply(
            &transpose(&d, rows, cols),
            &multiply(&numeric(&a1)?, &d, rows, rows, cols),
            cols,
            rows,
            cols,
        ),
        cols,
        cols,
    )
}
#[extendr]
fn lav_rust_model_vcov_sandwich(left: Robj, middle: Robj) -> Result<Robj> {
    let (n, c) = shape(&left)?;
    if n != c || shape(&middle)? != (n, n) {
        return Err(Error::Other("invalid sandwich dimensions".into()));
    }
    let left = numeric(&left)?;
    matrix(
        multiply(
            &multiply(&left, &numeric(&middle)?, n, n, n),
            &left,
            n,
            n,
            n,
        ),
        n,
        n,
    )
}
#[extendr]
fn lav_rust_model_vcov_jacobian_vcov_jacobian_t(jac: Robj, vcov: Robj) -> Result<Robj> {
    let (rows, cols) = shape(&jac)?;
    if shape(&vcov)? != (cols, cols) {
        return Err(Error::Other("invalid covariance dimensions".into()));
    }
    let jac = numeric(&jac)?;
    matrix(
        multiply(
            &multiply(&jac, &numeric(&vcov)?, rows, cols, cols),
            &transpose(&jac, rows, cols),
            rows,
            cols,
            rows,
        ),
        rows,
        rows,
    )
}

extendr_module! {
    mod lavaan;
    fn lav_rust_backend_ping;
    fn lav_rust_lisrel_ml_plan_new;
    fn lav_rust_lisrel_ml_plan_set_templates;
    fn lav_rust_model_objective_gradient_lisrel_ml_parameters;
    fn lav_rust_model_objective_gradient_lisrel_ml;
    fn lav_rust_model_objective_gradient_lisrel_ml_with_plan;
    fn lav_rust_model_objective_gradient_lisrel_ml_diagnostics;
    fn lav_rust_model_objective_ml;
    fn lav_rust_model_objective_gradient_ml;
    fn lav_rust_baseline_fiml_moments;
    fn lav_rust_baseline_fiml_loglik;
    fn lav_rust_model_objective_ml_res;
    fn lav_rust_model_objective_gls;
    fn lav_rust_model_objective_wls;
    fn lav_rust_model_objective_dwls;
    fn lav_rust_model_estimate_pack;
    fn lav_rust_model_estimate_unpack_unscale;
    fn lav_rust_model_estimate_gradient_postprocess;
    fn lav_rust_model_vcov_delta_a_delta;
    fn lav_rust_model_vcov_sandwich;
    fn lav_rust_model_vcov_jacobian_vcov_jacobian_t;
}
