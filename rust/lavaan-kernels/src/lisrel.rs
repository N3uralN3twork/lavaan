use crate::linalg::{identity, inverse, multiply, spd_inverse, transpose};
use std::cell::RefCell;

pub type KernelResult<T> = Result<T, String>;

#[derive(Clone, Debug)]
pub struct FreeElement {
    pub kind: usize,
    pub element: usize,
    pub parameter: usize,
}

#[derive(Clone, Debug)]
pub struct GroupPlan {
    pub p: usize,
    pub k: usize,
    pub beta_present: bool,
    pub sample_cov: Vec<f64>,
    pub sample_mean: Vec<f64>,
    pub sample_logdet: f64,
    pub weight: f64,
    pub free: Vec<FreeElement>,
}

#[derive(Clone, Debug)]
struct ParameterMatrices {
    values: [Vec<f64>; 6],
    // Flattened offsets are resolved once, rather than on every evaluation.
    assignments: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct FullMlPlan {
    pub groups: Vec<GroupPlan>,
    pub nfree: usize,
    pub means: bool,
    parameters: RefCell<Option<ParameterMatrices>>,
}

impl FullMlPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_cov: &[f64],
        sample_mean: &[f64],
        nvar: &[usize],
        nfac: &[usize],
        beta_present: &[bool],
        nobs: &[f64],
        ntotal: f64,
        free_group: &[usize],
        free_kind: &[usize],
        free_element: &[usize],
        free_parameter: &[usize],
        nfree: usize,
        means: bool,
    ) -> KernelResult<Self> {
        if nfree == 0 {
            return Err("nfree must be positive".into());
        }
        let ng = nvar.len();
        if ng == 0
            || [nfac.len(), beta_present.len(), nobs.len()]
                .iter()
                .any(|&n| n != ng)
        {
            return Err("group dimensions have incompatible lengths".into());
        }
        if !ntotal.is_finite() || ntotal <= 0.0 || nobs.iter().any(|n| !n.is_finite() || *n <= 0.0)
        {
            return Err("sample sizes must be finite and positive".into());
        }
        if [free_kind.len(), free_element.len(), free_parameter.len()]
            .iter()
            .any(|&n| n != free_group.len())
        {
            return Err("free-map vectors have incompatible lengths".into());
        }
        let expected_cov: usize = nvar.iter().map(|p| p * p).sum();
        let expected_mean: usize = if means { nvar.iter().sum() } else { 0 };
        if sample_cov.len() != expected_cov || sample_mean.len() != expected_mean {
            return Err("sample moment buffers have invalid lengths".into());
        }
        let (mut cov_offset, mut mean_offset) = (0, 0);
        let mut groups = Vec::with_capacity(ng);
        for g in 0..ng {
            let (p, k) = (nvar[g], nfac[g]);
            if p == 0 {
                return Err("nvar must be positive".into());
            }
            let covariance = sample_cov[cov_offset..cov_offset + p * p].to_vec();
            let (_, logdet) =
                spd_inverse(&covariance, p).ok_or("sample covariance is not positive definite")?;
            let mean = if means {
                sample_mean[mean_offset..mean_offset + p].to_vec()
            } else {
                Vec::new()
            };
            groups.push(GroupPlan {
                p,
                k,
                beta_present: beta_present[g],
                sample_cov: covariance,
                sample_mean: mean,
                sample_logdet: logdet,
                weight: nobs[g] / ntotal,
                free: Vec::new(),
            });
            cov_offset += p * p;
            if means {
                mean_offset += p;
            }
        }
        for i in 0..free_group.len() {
            let group = groups
                .get_mut(free_group[i])
                .ok_or("free group index is out of range")?;
            let lengths = group.matrix_lengths(means);
            if free_kind[i] >= 6
                || free_element[i] >= lengths[free_kind[i]]
                || free_parameter[i] >= nfree
            {
                return Err("free-map index is out of range".into());
            }
            group.free.push(FreeElement {
                kind: free_kind[i],
                element: free_element[i],
                parameter: free_parameter[i],
            });
        }
        Ok(Self {
            groups,
            nfree,
            means,
            parameters: RefCell::new(None),
        })
    }

    /// Own the fixed entries and reusable matrix buffers for this optimizer run.
    /// Free maps already contain both entries of symmetric matrices, matching
    /// lavaan's x2glist assignment semantics exactly.
    pub fn set_templates(&self, values: [Vec<f64>; 6]) -> KernelResult<()> {
        let mut expected = [0; 6];
        let mut assignments = Vec::new();
        for group in &self.groups {
            for free in &group.free {
                assignments.push((
                    free.kind,
                    expected[free.kind] + free.element,
                    free.parameter,
                ));
            }
            for (n, size) in expected.iter_mut().zip(group.matrix_lengths(self.means)) {
                *n += size;
            }
        }
        if values.iter().zip(expected).any(|(x, n)| x.len() != n) {
            return Err("model matrix templates have invalid lengths".into());
        }
        let mut storage = self
            .parameters
            .try_borrow_mut()
            .map_err(|_| "native plan is already in use")?;
        if storage.is_some() {
            return Err("native plan templates are already initialized".into());
        }
        *storage = Some(ParameterMatrices {
            values,
            assignments,
        });
        Ok(())
    }

    /// The only changing input is the free-parameter vector. No R storage is
    /// retained; fixed entries stay in owned buffers and every free entry is
    /// overwritten, so evaluation order and rejected trials cannot leak state.
    pub fn evaluate_parameters(&self, x: &[f64]) -> KernelResult<Vec<f64>> {
        if x.len() != self.nfree {
            return Err("parameter vector has an invalid length".into());
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err("parameter vector must contain finite values".into());
        }
        let mut storage = self
            .parameters
            .try_borrow_mut()
            .map_err(|_| "native plan is already in use")?;
        let matrices = storage
            .as_mut()
            .ok_or("native plan has no matrix templates")?;
        for &(kind, element, parameter) in &matrices.assignments {
            matrices.values[kind][element] = x[parameter];
        }
        self.evaluate(std::array::from_fn(|i| matrices.values[i].as_slice()))
    }

    pub fn evaluate(&self, matrices: [&[f64]; 6]) -> KernelResult<Vec<f64>> {
        let mut expected = [0; 6];
        for group in &self.groups {
            for (n, size) in expected.iter_mut().zip(group.matrix_lengths(self.means)) {
                *n += size;
            }
        }
        if matrices.iter().zip(expected).any(|(x, n)| x.len() != n) {
            return Err("model matrix buffers have invalid lengths".into());
        }
        let mut out = vec![0.0; 1 + self.groups.len() + self.nfree];
        let mut offsets = [0; 6];
        for (g, group) in self.groups.iter().enumerate() {
            let sizes = group.matrix_lengths(self.means);
            let local = std::array::from_fn(|i| &matrices[i][offsets[i]..offsets[i] + sizes[i]]);
            match group.evaluate(local, self.means) {
                Some((fx, scores)) => {
                    out[g + 1] = fx;
                    out[0] += group.weight * fx;
                    for free in &group.free {
                        out[1 + self.groups.len() + free.parameter] +=
                            group.weight * scores[free.kind][free.element];
                    }
                }
                None => {
                    out[0] = f64::INFINITY;
                    out[g + 1] = f64::INFINITY;
                    out[1 + self.groups.len()..].fill(f64::NAN);
                }
            }
            for i in 0..6 {
                offsets[i] += sizes[i];
            }
        }
        Ok(out)
    }
}

impl GroupPlan {
    pub fn matrix_lengths(&self, means: bool) -> [usize; 6] {
        [
            self.p * self.k,
            self.p * self.p,
            self.k * self.k,
            if self.beta_present {
                self.k * self.k
            } else {
                0
            },
            if means { self.p } else { 0 },
            if means { self.k } else { 0 },
        ]
    }

    fn evaluate(&self, matrices: [&[f64]; 6], means: bool) -> Option<(f64, [Vec<f64>; 6])> {
        let [lambda, theta, psi, beta, nu, alpha] = matrices;
        let (p, k) = (self.p, self.k);
        // A = (I - beta)^-1, C = lambda A;
        // Sigma = C psi C' + theta and mu = nu + C alpha.
        let a = if self.beta_present {
            let mut ib = identity(k);
            for i in 0..k * k {
                ib[i] -= beta[i];
            }
            inverse(&ib, k)?
        } else {
            identity(k)
        };
        let c = multiply(lambda, &a, p, k, k);
        let ct = transpose(&c, p, k);
        let cp = multiply(&c, psi, p, k, k);
        let mut sigma = multiply(&cp, &ct, p, k, p);
        for i in 0..p * p {
            sigma[i] += theta[i];
        }
        let (si, logdet) = spd_inverse(&sigma, p)?;
        let mut diff = vec![0.0; p];
        if means {
            diff = multiply(&c, alpha, p, k, 1);
            for i in 0..p {
                diff[i] += nu[i] - self.sample_mean[i];
            }
        }
        let q = multiply(&si, &diff, p, p, 1);
        // The objective is lavaan's half-scaled ML discrepancy. Its derivatives
        // are dF/dmu = q and dF/dSigma = (Si - Si S Si - q q') / 2.
        let fx = 0.5
            * (logdet - self.sample_logdet - p as f64
                + si.iter()
                    .zip(&self.sample_cov)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                + diff.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>());
        let sis = multiply(&si, &self.sample_cov, p, p, p);
        let sisi = multiply(&sis, &si, p, p, p);
        let mut gs = vec![0.0; p * p];
        for col in 0..p {
            for row in 0..p {
                let i = row + col * p;
                gs[i] = 0.5 * (si[i] - sisi[i] - q[row] * q[col]);
            }
        }
        let mut h = multiply(&gs, &cp, p, p, k);
        // Differentiate through C once, sharing covariance and mean work:
        // H = dF/dC = 2 (dF/dSigma) C psi + q alpha'.
        for col in 0..k {
            for row in 0..p {
                h[row + col * p] =
                    2.0 * h[row + col * p] + if means { q[row] * alpha[col] } else { 0.0 };
            }
        }
        let at = transpose(&a, k, k);
        // Chain rule for lambda, psi, beta and alpha. Summing the free-element
        // map handles mirrored symmetric entries and shared parameters.
        let gl = multiply(&h, &at, p, k, k);
        let gp = multiply(&multiply(&ct, &gs, k, p, p), &c, k, p, k);
        let gb = if self.beta_present {
            multiply(&multiply(&ct, &h, k, p, k), &at, k, k, k)
        } else {
            Vec::new()
        };
        let ga = if means {
            multiply(&ct, &q, k, p, 1)
        } else {
            Vec::new()
        };
        Some((
            fx.max(0.0),
            [gl, gs, gp, gb, if means { q } else { Vec::new() }, ga],
        ))
    }
}
