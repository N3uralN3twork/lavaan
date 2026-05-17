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
      any(lavdata@ov$type != "numeric")) {
    return(NULL)
  }

  ngroups <- lavdata@ngroups
  if (length(lavsamplestats@cov) != ngroups ||
      length(lavsamplestats@cov.log.det) != ngroups) {
    return(NULL)
  }

  sample_covs <- vector("list", ngroups)
  observed_var <- vector("list", ngroups)
  ov_names <- vector("list", ngroups)
  ov_names_x <- vector("list", ngroups)
  ov_names_nox <- vector("list", ngroups)
  exo_idx <- vector("list", ngroups)
  sample_log_det <- numeric(ngroups)
  model_log_det <- numeric(ngroups)
  for (g in seq_len(ngroups)) {
    sample_cov <- lavsamplestats@cov[[g]]
    if (!is.matrix(sample_cov) || anyNA(sample_cov)) {
      return(NULL)
    }
    sample_covs[[g]] <- sample_cov

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

    ov_names_x[[g]] <- lavdata@ov.names.x[[g]]
    if (is.null(ov_names_x[[g]])) {
      ov_names_x[[g]] <- character(0L)
    }
    if (length(ov_names_x[[g]]) > 0L) {
      if (!isTRUE(lavoptions$fixed.x) ||
          !isTRUE(lavoptions$baseline.fixed.x.free.cov) ||
          any(!ov_names_x[[g]] %in% ov_names[[g]])) {
        return(NULL)
      }
    }
    ov_names_nox[[g]] <- ov_names[[g]][!ov_names[[g]] %in% ov_names_x[[g]]]
    exo_idx[[g]] <- match(ov_names_x[[g]], ov_names[[g]])

    model_log_det[[g]] <- sum(log(observed_var[[g]][
      !ov_names[[g]] %in% ov_names_x[[g]]
    ]))
    if (length(exo_idx[[g]]) > 0L) {
      sample_cov_x <- sample_cov[exo_idx[[g]], exo_idx[[g]], drop = FALSE]
      sample_log_det_x <- determinant(
        sample_cov_x,
        logarithm = TRUE
      )$modulus
      sample_log_det_x <- as.numeric(sample_log_det_x)
      if (!is.finite(sample_log_det_x)) {
        return(NULL)
      }
      model_log_det[[g]] <- model_log_det[[g]] + sample_log_det_x
    }
  }

  # Meanstructure independence models still have closed-form ML estimates when
  # the baseline only adds observed means. Observed fixed.x regressors are okay:
  # their sample moments are fixed in the baseline partable.
  if (isTRUE(lavoptions$meanstructure)) {
    rhs_is_ov <- lavpartable$op == "~" &
      lavpartable$rhs %in% unlist(ov_names, use.names = FALSE)
    fixed_ov_x <- unlist(ov_names_x, use.names = FALSE)
    rhs_is_free_ov <- rhs_is_ov & !lavpartable$rhs %in% fixed_ov_x
    if (any(rhs_is_free_ov)) {
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
    mean_names <- if (nmean > 0L) ov_names[[g]] else character(0L)
    ov_is_x <- ov_names[[g]] %in% ov_names_x[[g]]
    mean_is_x <- mean_names %in% ov_names_x[[g]]

    lhs <- c(lhs, ov_names[[g]], mean_names)
    op <- c(op, rep("~~", nvar), rep("~1", nmean))
    rhs <- c(rhs, ov_names[[g]], rep("", nmean))
    block <- c(block, rep(g, nvar + nmean))
    group <- c(group, rep(g, nvar + nmean))
    free <- c(free, as.integer(!ov_is_x), as.integer(!mean_is_x))
    exo <- c(exo, as.integer(ov_is_x), as.integer(mean_is_x))
    label <- c(label, rep("", nvar + nmean))
    start_est <- c(start_est, observed_var[[g]], sample_mean[[g]])
    if (!is.null(lavoptions$optim.bounds)) {
      lower <- c(
        lower,
        ifelse(ov_is_x, observed_var[[g]], 0),
        ifelse(mean_is_x, sample_mean[[g]], -Inf)
      )
      upper <- c(
        upper,
        ifelse(ov_is_x, observed_var[[g]], Inf),
        ifelse(mean_is_x, sample_mean[[g]], Inf)
      )
    }

    nx <- length(ov_names_x[[g]])
    if (nx > 1L) {
      tmp <- utils::combn(ov_names_x[[g]], 2L)
      cov_start <- sample_covs[[g]][
        cbind(match(tmp[1L, ], ov_names[[g]]), match(tmp[2L, ], ov_names[[g]]))
      ]
      if (any(!is.finite(cov_start))) {
        return(NULL)
      }
      ncov <- length(cov_start)
      lhs <- c(lhs, tmp[1L, ])
      op <- c(op, rep("~~", ncov))
      rhs <- c(rhs, tmp[2L, ])
      block <- c(block, rep(g, ncov))
      group <- c(group, rep(g, ncov))
      free <- c(free, rep(0L, ncov))
      exo <- c(exo, rep(1L, ncov))
      label <- c(label, rep("", ncov))
      start_est <- c(start_est, cov_start)
      if (!is.null(lavoptions$optim.bounds)) {
        lower <- c(lower, cov_start)
        upper <- c(upper, cov_start)
      }
    }
  }
  free[free > 0L] <- seq_len(sum(free > 0L))
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
    fx_group[[g]] <- 0.5 * (model_log_det[[g]] - sample_log_det[[g]])
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
    seq_len(ngroups),
    function(g) {
      nvar <- length(observed_var[[g]])
      nx <- length(ov_names_x[[g]])
      nvar * (nvar - 1L) / 2L - nx * (nx - 1L) / 2L
    },
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
