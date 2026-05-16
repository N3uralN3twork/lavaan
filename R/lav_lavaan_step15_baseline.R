lav_lavaan_step15_baseline_fast <- function(lavoptions = NULL,
                                            lavsamplestats = NULL,
                                            lavdata = NULL,
                                            lavpartable = NULL) {
  if (!identical(lavoptions$test, "standard") ||
      !identical(lavoptions$estimator, "ML") ||
      !(lavoptions$likelihood %in% c("normal", "wishart")) ||
      (!is.null(lavoptions$baseline.type) &&
       !identical(lavoptions$baseline.type, "independence")) ||
      isTRUE(lavoptions$conditional.x) ||
      isTRUE(lavoptions$correlation) ||
      isTRUE(lavoptions$group.w.free) ||
      lavdata@nlevels != 1L ||
      !identical(lavdata@missing, "listwise") ||
      isTRUE(lavsamplestats@missing.flag) ||
      length(lavdata@ordered) > 0L ||
      any(lavdata@ov$type != "numeric") ||
      any(lengths(lavdata@ov.names.x) > 0L)) {
    return(NULL)
  }

  ngroups <- lavdata@ngroups
  if (length(lavsamplestats@cov) != ngroups ||
      length(lavsamplestats@cov.log.det) != ngroups) {
    return(NULL)
  }

  observed_var <- vector("list", ngroups)
  ov_names <- vector("list", ngroups)
  sample_log_det <- numeric(ngroups)
  for (g in seq_len(ngroups)) {
    sample_cov <- lavsamplestats@cov[[g]]
    if (!is.matrix(sample_cov) || anyNA(sample_cov)) {
      return(NULL)
    }

    observed_var[[g]] <- diag(sample_cov)
    if (length(observed_var[[g]]) == 0L ||
        any(!is.finite(observed_var[[g]])) ||
        any(observed_var[[g]] <= 0)) {
      return(NULL)
    }

    sample_log_det[[g]] <- lavsamplestats@cov.log.det[[g]]
    if (!is.finite(sample_log_det[[g]])) {
      return(NULL)
    }

    ov_names[[g]] <- lavdata@ov.names[[g]]
    if (length(ov_names[[g]]) != length(observed_var[[g]])) {
      ov_names[[g]] <- colnames(sample_cov)
    }
    if (is.null(ov_names[[g]]) || anyNA(ov_names[[g]]) ||
        any(!nzchar(ov_names[[g]]))) {
      return(NULL)
    }
  }

  # Meanstructure independence models still have closed-form ML estimates when
  # the baseline only adds free observed means. If observed variables are used
  # as regressors, their covariance terms are also needed, so use the full path.
  if (isTRUE(lavoptions$meanstructure)) {
    rhs_is_ov <- lavpartable$op == "~" &
      lavpartable$rhs %in% unlist(ov_names, use.names = FALSE)
    if (any(rhs_is_ov)) {
      return(NULL)
    }
    if (length(lavsamplestats@mean) != ngroups) {
      return(NULL)
    }
    sample_mean <- vector("list", ngroups)
    for (g in seq_len(ngroups)) {
      sample_mean[[g]] <- lavsamplestats@mean[[g]]
      if (is.null(sample_mean[[g]]) ||
          length(sample_mean[[g]]) != length(observed_var[[g]]) ||
          any(!is.finite(sample_mean[[g]]))) {
        return(NULL)
      }
    }
  } else {
    sample_mean <- vector("list", ngroups)
  }

  lhs <- rhs <- op <- label <- character(0L)
  block <- group <- free <- exo <- integer(0L)
  start_est <- lower <- upper <- numeric(0L)
  for (g in seq_len(ngroups)) {
    nvar <- length(observed_var[[g]])
    nmean <- length(sample_mean[[g]])
    npar_g <- nvar + nmean
    mean_names <- if (nmean > 0L) ov_names[[g]] else character(0L)

    lhs <- c(lhs, ov_names[[g]], mean_names)
    op <- c(op, rep("~~", nvar), rep("~1", nmean))
    rhs <- c(rhs, ov_names[[g]], rep("", nmean))
    block <- c(block, rep(g, npar_g))
    group <- c(group, rep(g, npar_g))
    free <- c(free, seq_len(npar_g))
    exo <- c(exo, rep(0L, npar_g))
    label <- c(label, rep("", npar_g))
    start_est <- c(start_est, observed_var[[g]], sample_mean[[g]])
    if (!is.null(lavoptions$optim.bounds)) {
      lower <- c(lower, rep(0, nvar), rep(-Inf, nmean))
      upper <- c(upper, rep(Inf, npar_g))
    }
  }
  free <- seq_along(free)
  partable <- list(
    id = seq_along(lhs),
    lhs = lhs,
    op = op,
    rhs = rhs,
    user = rep(1L, length(lhs)),
    block = block,
    group = group,
    free = free,
    ustart = start_est,
    exo = exo,
    label = label
  )

  if (!is.null(lavoptions$optim.bounds)) {
    partable$lower <- lower
    partable$upper <- upper
  }
  partable$start <- start_est
  partable$est <- start_est

  fx_group <- numeric(ngroups)
  for (g in seq_len(ngroups)) {
    fx_group[[g]] <- 0.5 * (sum(log(observed_var[[g]])) - sample_log_det[[g]])
    if (is.finite(fx_group[[g]]) && fx_group[[g]] < 0.0) {
      fx_group[[g]] <- 0.0
    }
  }
  nfac <- 2 * unlist(lavsamplestats@nobs)
  if (identical(lavoptions$likelihood, "wishart")) {
    nfac <- 2 * (nfac / 2 - 1)
  }
  stat_group <- fx_group * nfac
  stat <- sum(stat_group)
  df <- sum(vapply(
    observed_var,
    function(var) length(var) * (length(var) - 1L) / 2L,
    numeric(1L)
  ))
  pvalue <- if (df == 0L) {
    as.numeric(NA)
  } else {
    1 - pchisq(stat, df)
  }

  test <- list(standard = list(
    test = "standard",
    stat = stat,
    stat.group = stat_group,
    df = as.integer(df),
    refdistr = "chisq",
    pvalue = pvalue
  ))
  attr(test, "info") <- list(
    ngroups = lavdata@ngroups,
    group.label = lavdata@group.label,
    information = lavoptions$information,
    h1.information = lavoptions$h1.information,
    observed.information = lavoptions$observed.information
  )

  list(partable = partable, test = test)
}

lav_lavaan_step15_baseline <- function(lavoptions = NULL,
                                       lavsamplestats = NULL,
                                       lavdata = NULL,
                                       lavcache = NULL,
                                       lavh1 = NULL,
                                       lavpartable = NULL) {
  # # # # # # # # # # #
  # #  15. baseline # #  (since 0.6-5)
  # # # # # # # # # # #

  # if options$do.fit and options$test not "none" and options$baseline = TRUE
  #   try fit.indep <- lav_object_independence(...)
  #   if not successful or not converged
  #     ** warning **
  #     lavbaseline < list()
  #   else
  #     lavbaseline <- list with partable and test of fit.indep
  lavbaseline <- list()
  if (lavoptions$do.fit &&
    !("none" %in% lavoptions$test) &&
    is.logical(lavoptions$baseline) && lavoptions$baseline) {
    if (lav_verbose()) {
      cat("lavbaseline ...")
    }
    lavbaseline <- lav_lavaan_step15_baseline_fast(
      lavoptions = lavoptions,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata,
      lavpartable = lavpartable
    )
    if (!is.null(lavbaseline)) {
      if (lav_verbose()) {
        cat(" done.\n")
      }
      return(lavbaseline)
    }
    current_verbose <- lav_verbose()
    lav_verbose(FALSE)
    fit_indep <- try(lav_object_independence(
      object = NULL,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata,
      lavcache = lavcache,
      lavoptions = lavoptions,
      lavpartable = lavpartable,
      lavh1 = lavh1
    ), silent = TRUE)
    lav_verbose(current_verbose)
    if (inherits(fit_indep, "try-error") || !fit_indep@optim$converged) {
      lav_msg_warn(gettext("estimation of the baseline model failed."))
      lavbaseline <- list()
      if (lav_verbose()) {
        cat(" FAILED.\n")
      }
    } else {
      # store relevant information
      lavbaseline <- list(
        partable = fit_indep@ParTable,
        test = fit_indep@test
      )
      if (lav_verbose()) {
        cat(" done.\n")
      }
    }
  }

  lavbaseline
}
