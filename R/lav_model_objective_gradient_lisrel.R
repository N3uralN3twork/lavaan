# Fully numerical continuous-LISREL ML objective/gradient bridge.
#
# R owns S4 access and one-shot wire packing. Rust receives only typed numeric
# values in R/Fortran column-major order and performs the numerical work.
lav_model_objective_gradient_lisrel_wire <- function(lavmodel, glist,
                                                      lavsamplestats,
                                                      lavdata,
                                                      plan = NULL) {
  if (is.null(glist)) glist <- lavmodel@GLIST
  if (lavmodel@estimator != "ML" || lavdata@nlevels != 1L ||
      lavmodel@representation != "LISREL" || lavmodel@categorical ||
      lavmodel@conditional.x || lavmodel@composites ||
      lavmodel@group.w.free || lavmodel@ceq.simple.only ||
      lavmodel@eq.constraints || lavsamplestats@missing.flag ||
      lavsamplestats@ridge > 0.0 ||
      !lav_rust_kernel_enabled("model_objective")) {
    return(NULL)
  }

  if (!is.null(plan) && !inherits(plan, "lavaan_lisrel_ml_plan")) {
    stop("lavaan LISREL objective/gradient plan has an invalid class", call. = FALSE)
  }
  use_plan <- !is.null(plan)
  mm_idx <- lav_model_get_mm_idx(lavmodel)
  matrix_names <- c("lambda", "theta", "psi", "beta", "nu", "alpha")
  lambda <- theta <- psi <- beta <- nu <- alpha <- numeric(0L)
  nvar <- if (use_plan) plan$nvar else integer(lavmodel@nblocks)
  nfac <- if (use_plan) plan$nfac else integer(lavmodel@nblocks)
  beta_present <- if (use_plan) plan$beta_present else integer(lavmodel@nblocks)
  free_group <- if (use_plan) plan$free_group else integer(0L)
  free_kind <- if (use_plan) plan$free_kind else integer(0L)
  free_element <- if (use_plan) plan$free_element else integer(0L)
  free_parameter <- if (use_plan) plan$free_parameter else integer(0L)

  for (g in seq_len(lavmodel@nblocks)) {
    indices <- mm_idx[[g]]
    mlist <- glist[indices]
    if (is.null(mlist$lambda) || is.null(mlist$theta) || is.null(mlist$psi)) {
      return(NULL)
    }
    p <- nrow(mlist$lambda)
    k <- ncol(mlist$lambda)
    if (nrow(mlist$theta) != p || ncol(mlist$theta) != p ||
        nrow(mlist$psi) != k || ncol(mlist$psi) != k) return(NULL)
    if (use_plan && (p != nvar[g] || k != nfac[g] ||
        as.integer(!is.null(mlist$beta)) != beta_present[g])) {
      stop("lavaan LISREL objective/gradient plan is incompatible with model structure",
           call. = FALSE)
    }
    if (!use_plan) {
      nvar[g] <- p
      nfac[g] <- k
    }
    lambda <- c(lambda, as.numeric(mlist$lambda))
    theta <- c(theta, as.numeric(mlist$theta))
    psi <- c(psi, as.numeric(mlist$psi))
    if (!use_plan) beta_present[g] <- as.integer(!is.null(mlist$beta))
    if (beta_present[g] == 1L) {
      if (nrow(mlist$beta) != k || ncol(mlist$beta) != k) return(NULL)
      beta <- c(beta, as.numeric(mlist$beta))
    }
    if (lavmodel@meanstructure) {
      if (is.null(mlist$nu) || is.null(mlist$alpha) ||
          length(mlist$nu) != p || length(mlist$alpha) != k) return(NULL)
      nu <- c(nu, as.numeric(mlist$nu))
      alpha <- c(alpha, as.numeric(mlist$alpha))
    }

    if (!use_plan) {
      for (kind in seq_along(matrix_names)) {
        position <- match(matrix_names[[kind]], names(mlist))
        if (is.na(position)) next
        model_matrix <- indices[[position]]
        m_free <- lavmodel@m.free.idx[[model_matrix]]
        x_free <- lavmodel@x.free.idx[[model_matrix]]
        keep <- x_free > 0L
        if (length(m_free) > 0L && any(keep)) {
          free_group <- c(free_group, rep.int(g - 1L, sum(keep)))
          free_kind <- c(free_kind, rep.int(kind - 1L, sum(keep)))
          free_element <- c(free_element, m_free[keep] - 1L)
          free_parameter <- c(free_parameter, x_free[keep] - 1L)
        }
      }
    }
  }

  list(
    lambda = lambda, theta = theta, psi = psi, beta = beta, nu = nu,
    alpha = alpha,
    sample_cov = if (use_plan) plan$sample_cov else unlist(lavsamplestats@cov, use.names = FALSE),
    sample_mean = if (use_plan) {
      plan$sample_mean
    } else if (lavmodel@meanstructure) {
      unlist(lavsamplestats@mean, use.names = FALSE)
    } else {
      numeric(0L)
    },
    nvar = nvar, nfac = nfac, beta_present = beta_present,
    nobs = if (use_plan) plan$nobs else unlist(lavsamplestats@nobs),
    ntotal = if (use_plan) plan$ntotal else lavsamplestats@ntotal,
    free_group = free_group, free_kind = free_kind,
    free_element = free_element, free_parameter = free_parameter,
    nfree = if (use_plan) plan$nfree else lavmodel@nx.free,
    meanstructure = if (use_plan) plan$meanstructure else lavmodel@meanstructure
  )
}

# Build once per optimizer run. The external pointer owns only copied immutable
# numerical state, so it cannot retain R-managed memory after a native call.
lav_model_objective_gradient_lisrel_plan <- function(lavmodel,
                                                      lavsamplestats,
                                                      lavdata) {
  wire <- lav_model_objective_gradient_lisrel_wire(
    lavmodel, lavmodel@GLIST, lavsamplestats, lavdata
  )
  if (is.null(wire)) return(NULL)
  wire$native <- lav_native_lisrel_ml_plan_new(
    sample_cov = wire$sample_cov, sample_mean = wire$sample_mean,
    nvar = wire$nvar, nfac = wire$nfac, beta_present = wire$beta_present,
    nobs = wire$nobs, ntotal = wire$ntotal, free_group = wire$free_group,
    free_kind = wire$free_kind, free_element = wire$free_element,
    free_parameter = wire$free_parameter, nfree = wire$nfree,
    meanstructure = wire$meanstructure
  )
  wire[c("lambda", "theta", "psi", "beta", "nu", "alpha")] <- NULL
  structure(wire, class = "lavaan_lisrel_ml_plan")
}

lav_model_objective_gradient_lisrel_result <- function(result, ngroups, nfree) {
  if (length(result) != 1L + ngroups + nfree) {
    stop("lavaan Rust numerical kernel returned an invalid LISREL objective/gradient result",
         call. = FALSE)
  }
  list(
    objective = as.numeric(result[[1L]]),
    fx.group = as.numeric(result[seq_len(ngroups) + 1L]),
    gradient = as.numeric(result[(ngroups + 2L):length(result)])
  )
}

lav_model_objective_gradient_lisrel_timed <- function(fun) {
  value <- NULL
  elapsed <- bench::system_time(value <- fun())
  list(
    value = value,
    milliseconds = as.numeric(elapsed[["real"]], units = "secs") * 1000
  )
}

lav_model_objective_gradient_lisrel_ml <- function(lavmodel = NULL,
                                                   glist = NULL,
                                                   lavsamplestats = NULL,
                                                   lavdata = NULL,
                                                   plan = NULL) {
  wire <- lav_model_objective_gradient_lisrel_wire(
    lavmodel, glist, lavsamplestats, lavdata, plan = plan
  )
  if (is.null(wire)) return(NULL)
  result <- if (!is.null(plan)) {
    lav_native_lav_model_objective_gradient_lisrel_ml_with_plan(
      plan$native, lambda = wire$lambda, theta = wire$theta, psi = wire$psi,
      beta = wire$beta, nu = wire$nu, alpha = wire$alpha
    )
  } else {
    lav_native_lav_model_objective_gradient_lisrel_ml(
      lambda = wire$lambda, theta = wire$theta, psi = wire$psi, beta = wire$beta,
      nu = wire$nu, alpha = wire$alpha, sample_cov = wire$sample_cov,
      sample_mean = wire$sample_mean, nvar = wire$nvar, nfac = wire$nfac,
      beta_present = wire$beta_present, nobs = wire$nobs, ntotal = wire$ntotal,
      free_group = wire$free_group, free_kind = wire$free_kind,
      free_element = wire$free_element, free_parameter = wire$free_parameter,
      nfree = wire$nfree, meanstructure = wire$meanstructure
    )
  }
  lav_model_objective_gradient_lisrel_result(
    result, lavsamplestats@ngroups, lavmodel@nx.free
  )
}

# This is deliberately opt-in. It uses the same fitted state as the normal
# path, but returns stage timings for diagnosis rather than optimizer use.
lav_model_objective_gradient_lisrel_ml_diagnostics <- function(
    lavmodel = NULL, glist = NULL, lavsamplestats = NULL, lavdata = NULL) {
  if (!requireNamespace("bench", quietly = TRUE)) {
    stop("LISREL objective/gradient diagnostics requires the 'bench' package",
         call. = FALSE)
  }
  packing <- lav_model_objective_gradient_lisrel_timed(function() {
    lav_model_objective_gradient_lisrel_wire(
      lavmodel, glist, lavsamplestats, lavdata
    )
  })
  wire <- packing$value
  if (is.null(wire)) return(NULL)

  bridge <- lav_model_objective_gradient_lisrel_timed(function() {
    lav_native_lav_model_objective_gradient_lisrel_ml_diagnostics(
      lambda = wire$lambda, theta = wire$theta, psi = wire$psi, beta = wire$beta,
      nu = wire$nu, alpha = wire$alpha, sample_cov = wire$sample_cov,
      sample_mean = wire$sample_mean, nvar = wire$nvar, nfac = wire$nfac,
      beta_present = wire$beta_present, nobs = wire$nobs, ntotal = wire$ntotal,
      free_group = wire$free_group, free_kind = wire$free_kind,
      free_element = wire$free_element, free_parameter = wire$free_parameter,
      nfree = wire$nfree, meanstructure = wire$meanstructure
    )
  })
  result <- bridge$value
  result_length <- 1L + lavsamplestats@ngroups + lavmodel@nx.free
  if (length(result) != result_length + 6L) {
    stop("lavaan Rust numerical kernel returned invalid LISREL diagnostic data",
         call. = FALSE)
  }
  native_times <- as.numeric(result[seq.int(result_length + 1L, length(result))])
  names(native_times) <- c(
    "bridge_conversion_ms", "model_plan_free_map_ms", "implied_moments_ms",
    "factorization_objective_ms", "direct_score_gradient_ms",
    "result_buffer_ms"
  )
  timings <- c(
    r_packing_ms = packing$milliseconds,
    r_bridge_call_ms = bridge$milliseconds,
    native_times,
    r_ffi_and_result_conversion_ms = max(0, bridge$milliseconds - sum(native_times))
  )
  list(
    result = lav_model_objective_gradient_lisrel_result(
      result[seq_len(result_length)], lavsamplestats@ngroups, lavmodel@nx.free
    ),
    timings = timings
  )
}
