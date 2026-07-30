# Coarse-grained standard-ML objective/gradient evaluation.
#
# Model construction and implied moments remain in R. We pack the resulting
# numerical state once, call the native kernel once, and let the optimizer cache
# reuse both objective and gradient for the same parameter vector.
lav_model_objective_gradient_ml <- function(lavmodel = NULL,
                                            glist = NULL,
                                            lavsamplestats = NULL,
                                            lavdata = NULL,
                                            implied = NULL,
                                            ceq_simple = FALSE) {
  if (is.null(glist)) glist <- lavmodel@GLIST

  if (lavmodel@estimator != "ML" ||
      lavdata@nlevels != 1L ||
      lavmodel@categorical ||
      lavmodel@conditional.x ||
      lavmodel@group.w.free ||
      lavsamplestats@missing.flag ||
      lavsamplestats@ridge > 0.0 ||
      !lav_rust_kernel_enabled("model_objective")) {
    return(NULL)
  }

  meanstructure <- lavmodel@meanstructure
  implied_fast <- lav_model_implied_fast_state(
    lavmodel = lavmodel, glist = glist, implied = implied,
    need_sigma = TRUE, need_mu = meanstructure, extra = TRUE
  )
  sigma_hat <- implied_fast$sigma
  if (!all(vapply(sigma_hat, function(x) isTRUE(attr(x, "po")), logical(1)))) {
    return(NULL)
  }

  # Delta is deliberately kept on its reference R implementation: this helper
  # combines the objective and final delta contraction, it does not change the
  # model-derivative contract.
  delta <- lav_model_gradient_delta_reference(lavmodel = lavmodel, glist = glist)
  nvar <- vapply(sigma_hat, ncol, integer(1))
  delta_rows <- vapply(delta, nrow, integer(1))
  delta_cols <- vapply(delta, ncol, integer(1))
  if (length(unique(delta_cols)) != 1L || any(delta_cols < 1L)) return(NULL)

  mu_hat <- if (meanstructure) unlist(implied_fast$mu, use.names = FALSE) else numeric(0L)
  data_mean <- if (meanstructure) unlist(lavsamplestats@mean, use.names = FALSE) else numeric(0L)
  group_weights <- unlist(lavsamplestats@nobs) / lavsamplestats@ntotal
  result <- try(lav_native_lav_model_objective_gradient_ml(
    sigma = unlist(sigma_hat, use.names = FALSE),
    sigma_inv = unlist(lapply(sigma_hat, attr, which = "inv"), use.names = FALSE),
    data_cov = unlist(lavsamplestats@cov, use.names = FALSE),
    mu_hat = mu_hat,
    data_mean = data_mean,
    sigma_log_det = vapply(sigma_hat, attr, numeric(1), which = "log.det"),
    data_cov_log_det = lavsamplestats@cov.log.det,
    delta = unlist(delta, use.names = FALSE),
    nvar = nvar,
    delta_rows = delta_rows,
    delta_cols = delta_cols,
    group_weights = group_weights,
    meanstructure = meanstructure
  ), silent = TRUE)
  if (inherits(result, "try-error")) return(NULL)

  ngroups <- lavsamplestats@ngroups
  nfree <- delta_cols[[1L]]
  if (length(result) != 1L + ngroups + nfree) return(NULL)
  gradient <- as.numeric(result[(ngroups + 2L):length(result)])
  if (lavmodel@ceq.simple.only && ceq_simple) {
    gradient <- lav_model_gradient_ceq_simple_dx(lavmodel@ceq.simple.K, gradient)
  }
  list(
    objective = as.numeric(result[[1L]]),
    fx.group = as.numeric(result[seq_len(ngroups) + 1L]),
    gradient = gradient
  )
}
