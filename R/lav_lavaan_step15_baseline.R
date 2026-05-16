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
      lavdata@ngroups != 1L ||
      !identical(lavdata@missing, "listwise") ||
      isTRUE(lavsamplestats@missing.flag) ||
      length(lavdata@ordered) > 0L ||
      any(lavdata@ov$type != "numeric") ||
      length(lavdata@ov.names.x[[1L]]) > 0L) {
    return(NULL)
  }

  sample_cov <- lavsamplestats@cov[[1L]]
  if (!is.matrix(sample_cov) || anyNA(sample_cov)) {
    return(NULL)
  }

  observed_var <- diag(sample_cov)
  if (length(observed_var) == 0L || any(!is.finite(observed_var)) ||
      any(observed_var <= 0)) {
    return(NULL)
  }

  sample_log_det <- lavsamplestats@cov.log.det[[1L]]
  if (!is.finite(sample_log_det)) {
    return(NULL)
  }

  ov_names <- lavdata@ov.names[[1L]]
  if (length(ov_names) != length(observed_var)) {
    ov_names <- colnames(sample_cov)
  }
  if (is.null(ov_names) || anyNA(ov_names) || any(!nzchar(ov_names))) {
    return(NULL)
  }

  # Meanstructure independence models still have closed-form ML estimates when
  # the baseline only adds free observed means. If observed variables are used
  # as regressors, their covariance terms are also needed, so use the full path.
  if (isTRUE(lavoptions$meanstructure)) {
    rhs_is_ov <- lavpartable$op == "~" & lavpartable$rhs %in% ov_names
    if (any(rhs_is_ov)) {
      return(NULL)
    }
    sample_mean <- lavsamplestats@mean[[1L]]
    if (is.null(sample_mean) || length(sample_mean) != length(observed_var) ||
        any(!is.finite(sample_mean))) {
      return(NULL)
    }
  } else {
    sample_mean <- NULL
  }

  nvar <- length(observed_var)
  nmean <- length(sample_mean)
  npar <- nvar + nmean
  mean_names <- if (nmean > 0L) ov_names else character(0L)
  partable <- list(
    id = seq_len(npar),
    lhs = c(ov_names, mean_names),
    op = c(rep("~~", nvar), rep("~1", nmean)),
    rhs = c(ov_names, rep("", nmean)),
    user = rep(1L, npar),
    block = rep(1L, npar),
    group = rep(1L, npar),
    free = seq_len(npar),
    ustart = c(observed_var, sample_mean),
    exo = rep(0L, npar),
    label = rep("", npar)
  )

  if (!is.null(lavoptions$optim.bounds)) {
    partable$lower <- c(rep(0, nvar), rep(-Inf, nmean))
    partable$upper <- rep(Inf, npar)
  }
  partable$start <- c(observed_var, sample_mean)
  partable$est <- c(observed_var, sample_mean)

  fx_group <- 0.5 * (sum(log(observed_var)) - sample_log_det)
  if (is.finite(fx_group) && fx_group < 0.0) {
    fx_group <- 0.0
  }
  nfac <- 2 * unlist(lavsamplestats@nobs)[[1L]]
  if (identical(lavoptions$likelihood, "wishart")) {
    nfac <- 2 * (nfac / 2 - 1)
  }
  stat <- fx_group * nfac
  df <- nvar * (nvar - 1L) / 2L
  pvalue <- if (df == 0L) {
    as.numeric(NA)
  } else {
    1 - pchisq(stat, df)
  }

  test <- list(standard = list(
    test = "standard",
    stat = stat,
    stat.group = stat,
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
