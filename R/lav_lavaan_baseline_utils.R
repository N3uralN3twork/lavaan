lav_lavaan_baseline_supports_ml <- function(lavoptions = NULL,
                                            lavsamplestats = NULL,
                                            lavdata = NULL,
                                            likelihood = c("normal", "wishart"),
                                            conditional_x = FALSE,
                                            missing = "listwise",
                                            missing_flag = FALSE,
                                            meanstructure = NULL,
                                            no_observed_x = FALSE,
                                            no_weights = FALSE) {
  if (!is.null(meanstructure) &&
      !identical(isTRUE(lavoptions$meanstructure), meanstructure)) {
    return(FALSE)
  }
  if (no_observed_x && !all(lengths(lavdata@ov.names.x) == 0L)) {
    return(FALSE)
  }
  if (no_weights && !all(vapply(lavdata@weights, is.null, logical(1L)))) {
    return(FALSE)
  }

  identical(lavoptions$test, "standard") &&
    identical(lavoptions$estimator, "ML") &&
    (lavoptions$likelihood %in% likelihood) &&
    (is.null(lavoptions$baseline.type) ||
      identical(lavoptions$baseline.type, "independence")) &&
    identical(isTRUE(lavoptions$conditional.x), conditional_x) &&
    !isTRUE(lavoptions$correlation) &&
    !isTRUE(lavoptions$group.w.free) &&
    lavdata@nlevels == 1L &&
    identical(lavdata@missing, missing) &&
    identical(isTRUE(lavsamplestats@missing.flag), missing_flag) &&
    length(lavdata@ordered) == 0L &&
    all(lavdata@ov$type == "numeric")
}

lav_lavaan_baseline_supports_standard_ml <- function(lavoptions = NULL,
                                                      lavsamplestats = NULL,
                                                      lavdata = NULL,
                                                      conditional_x = FALSE) {
  lav_lavaan_baseline_supports_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    likelihood = c("normal", "wishart"),
    conditional_x = conditional_x,
    missing = "listwise",
    missing_flag = FALSE
  )
}

lav_lavaan_baseline_supports_fiml_ml <- function(lavoptions = NULL,
                                                 lavsamplestats = NULL,
                                                 lavdata = NULL) {
  lav_lavaan_baseline_supports_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    likelihood = "normal",
    conditional_x = FALSE,
    missing = "ml",
    missing_flag = TRUE,
    meanstructure = TRUE,
    no_observed_x = TRUE,
    no_weights = TRUE
  )
}

lav_lavaan_baseline_standard_test <- function(fx_group = NULL,
                                              stat_group = NULL,
                                              df_group = NULL,
                                              lavoptions = NULL,
                                              lavsamplestats = NULL,
                                              lavdata = NULL) {
  if (is.null(stat_group)) {
    nfac <- 2 * unlist(lavsamplestats@nobs)
    if (identical(lavoptions$likelihood, "wishart")) {
      nfac <- 2 * (nfac / 2 - 1)
    }
    stat_group <- fx_group * nfac
  }
  stat <- sum(stat_group)
  df <- sum(df_group)
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

  test
}

lav_lavaan_baseline_has_optim_bounds <- function(bounds = NULL) {
  if (is.null(bounds)) {
    return(FALSE)
  }
  if (!is.list(bounds)) {
    return(length(bounds) > 0L)
  }
  any(lengths(bounds) > 0L)
}

lav_lavaan_baseline_context <- function(lavoptions = NULL,
                                        lavsamplestats = NULL,
                                        lavdata = NULL,
                                        lavpartable = NULL,
                                        lavh1 = NULL) {
  list(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    lavpartable = lavpartable,
    lavh1 = lavh1
  )
}

lav_lavaan_baseline_payload_order <- function() {
  c("conditional_x", "simple", "fiml", "generated_independence")
}

lav_lavaan_baseline_payloads <- function() {
  payloads <- list(
    conditional_x = function(context) {
      lav_lavaan_baseline_conditional_x_payload(
        lavoptions = context$lavoptions,
        lavsamplestats = context$lavsamplestats,
        lavdata = context$lavdata
      )
    },
    simple = function(context) {
      lav_lavaan_baseline_simple_payload(
        lavoptions = context$lavoptions,
        lavsamplestats = context$lavsamplestats,
        lavdata = context$lavdata,
        lavpartable = context$lavpartable
      )
    },
    fiml = function(context) {
      lav_lavaan_baseline_fiml_payload(
        lavoptions = context$lavoptions,
        lavsamplestats = context$lavsamplestats,
        lavdata = context$lavdata,
        lavpartable = context$lavpartable,
        lavh1 = context$lavh1
      )
    },
    generated_independence = function(context) {
      lav_lavaan_baseline_generated_independence_payload(
        lavoptions = context$lavoptions,
        lavsamplestats = context$lavsamplestats,
        lavdata = context$lavdata,
        lavpartable = context$lavpartable,
        lavh1 = context$lavh1
      )
    }
  )
  payloads[lav_lavaan_baseline_payload_order()]
}

lav_lavaan_baseline_payload_is_valid <- function(payload = NULL) {
  is.list(payload) &&
    !is.null(payload$partable) &&
    !is.null(payload$test)
}

lav_lavaan_baseline_try_payloads <- function(context = NULL,
                                             payloads = NULL) {
  if (is.null(payloads)) {
    payloads <- lav_lavaan_baseline_payloads()
  }
  payload_names <- names(payloads)
  for (i in seq_along(payloads)) {
    payload <- payloads[[i]]
    payload_name <- payload_names[[i]]
    if (is.null(payload_name) || !nzchar(payload_name)) {
      payload_name <- paste0("payload_", i)
    }
    out <- payload(context)
    if (!is.null(out)) {
      if (!lav_lavaan_baseline_payload_is_valid(out)) {
        lav_msg_stop(gettextf(
          "baseline fast payload '%s' returned a malformed payload.",
          payload_name
        ))
      }
      return(out)
    }
  }
  NULL
}

lav_lavaan_baseline_finite_vector <- function(x = NULL,
                                              n = NULL,
                                              positive = FALSE) {
  if (is.null(x) || anyNA(x) || any(!is.finite(x))) {
    return(FALSE)
  }
  if (!is.null(n) && length(x) != n) {
    return(FALSE)
  }
  if (positive && any(x <= 0)) {
    return(FALSE)
  }
  TRUE
}

lav_lavaan_baseline_finite_matrix <- function(x = NULL,
                                              nrow = NULL,
                                              ncol = NULL) {
  if (!is.matrix(x) || anyNA(x) || any(!is.finite(x))) {
    return(FALSE)
  }
  if (!is.null(nrow) && base::nrow(x) != nrow) {
    return(FALSE)
  }
  if (!is.null(ncol) && base::ncol(x) != ncol) {
    return(FALSE)
  }
  TRUE
}

lav_lavaan_baseline_valid_names <- function(names = NULL, n = NULL) {
  !is.null(names) &&
    length(names) == n &&
    !anyNA(names) &&
    all(nzchar(names))
}

lav_lavaan_baseline_ov_names <- function(lavdata = NULL,
                                         sample_cov = NULL,
                                         group = 1L) {
  ov_names <- lavdata@ov.names[[group]]
  nvar <- ncol(sample_cov)
  if (length(ov_names) != nvar) {
    ov_names <- colnames(sample_cov)
  }
  if (!lav_lavaan_baseline_valid_names(ov_names, nvar)) {
    return(NULL)
  }
  ov_names
}

lav_lavaan_baseline_ov_x_names <- function(lavdata = NULL,
                                           group = 1L) {
  ov_names_x <- lavdata@ov.names.x[[group]]
  if (is.null(ov_names_x)) {
    ov_names_x <- character(0L)
  }
  ov_names_x
}

lav_lavaan_baseline_cov_moment <- function(lavsamplestats = NULL,
                                           group = 1L) {
  sample_cov <- lavsamplestats@cov[[group]]
  if (!lav_lavaan_baseline_finite_matrix(sample_cov) ||
      nrow(sample_cov) != ncol(sample_cov)) {
    return(NULL)
  }
  sample_cov
}

lav_lavaan_baseline_mean_moment <- function(lavsamplestats = NULL,
                                            group = 1L,
                                            nvar = NULL) {
  sample_mean <- lavsamplestats@mean[[group]]
  if (!lav_lavaan_baseline_finite_vector(sample_mean, n = nvar)) {
    return(NULL)
  }
  sample_mean
}

lav_lavaan_baseline_res_cov_moment <- function(lavsamplestats = NULL,
                                               group = 1L,
                                               nvar = NULL) {
  res_cov <- lavsamplestats@res.cov[[group]]
  if (!lav_lavaan_baseline_finite_matrix(res_cov, nrow = nvar, ncol = nvar)) {
    return(NULL)
  }
  res_cov
}

lav_lavaan_baseline_res_int_moment <- function(lavsamplestats = NULL,
                                               group = 1L,
                                               nvar = NULL) {
  res_int <- lavsamplestats@res.int[[group]]
  if (!lav_lavaan_baseline_finite_vector(res_int, n = nvar)) {
    return(NULL)
  }
  res_int
}

lav_lavaan_baseline_res_slopes_moment <- function(lavsamplestats = NULL,
                                                  group = 1L,
                                                  nvar = NULL,
                                                  nexo = NULL) {
  res_slopes <- lavsamplestats@res.slopes[[group]]
  if (!lav_lavaan_baseline_finite_matrix(res_slopes,
    nrow = nvar,
    ncol = nexo
  )) {
    return(NULL)
  }
  res_slopes
}

lav_lavaan_baseline_cov_x_moment <- function(lavsamplestats = NULL,
                                             group = 1L,
                                             nexo = NULL) {
  cov_x <- lavsamplestats@cov.x[[group]]
  if (!lav_lavaan_baseline_finite_matrix(cov_x, nrow = nexo, ncol = nexo)) {
    return(NULL)
  }
  cov_x
}

lav_lavaan_baseline_mean_x_moment <- function(lavsamplestats = NULL,
                                              group = 1L,
                                              nexo = NULL) {
  mean_x <- lavsamplestats@mean.x[[group]]
  if (!lav_lavaan_baseline_finite_vector(mean_x, n = nexo)) {
    return(NULL)
  }
  mean_x
}

lav_lavaan_baseline_exo_block_names <- function(lavpartable = NULL,
                                                ov_names = NULL,
                                                ov_names_x = NULL) {
  if (length(ov_names_x) > 0L) {
    return(ov_names_x[ov_names_x %in% ov_names])
  }

  if (is.null(lavpartable) ||
      is.null(lavpartable$lhs) ||
      is.null(lavpartable$op) ||
      is.null(lavpartable$rhs)) {
    return(character(0L))
  }

  regression_idx <- which(lavpartable$op == "~")
  rhs_names <- lavpartable$rhs[regression_idx]
  lhs_observed <- lavpartable$lhs[regression_idx]
  lhs_observed <- lhs_observed[lhs_observed %in% ov_names]
  rhs_names <- rhs_names[rhs_names %in% ov_names]
  rhs_names <- rhs_names[!rhs_names %in% lhs_observed]
  rhs_names <- ov_names[ov_names %in% unique(rhs_names)]

  if (length(rhs_names) > 1L) {
    include_block <- TRUE
    for (i in seq_len(length(rhs_names) - 1L)) {
      for (j in seq.int(i + 1L, length(rhs_names))) {
        lhs <- rhs_names[[i]]
        rhs <- rhs_names[[j]]
        cov_idx <- which(
          lavpartable$op == "~~" &
            ((lavpartable$lhs == lhs & lavpartable$rhs == rhs) |
              (lavpartable$lhs == rhs & lavpartable$rhs == lhs))
        )
        if (length(cov_idx) != 1L || lavpartable$user[[cov_idx]] != 0L) {
          include_block <- FALSE
          break
        }
      }
      if (!include_block) {
        break
      }
    }
    if (!include_block) {
      return(character(0L))
    }
  }

  rhs_names
}

lav_lavaan_baseline_new_partable <- function() {
  list(
    id = integer(0L),
    lhs = character(0L),
    op = character(0L),
    rhs = character(0L),
    user = integer(0L),
    block = integer(0L),
    group = integer(0L),
    free = integer(0L),
    ustart = numeric(0L),
    exo = integer(0L),
    label = character(0L),
    start = numeric(0L),
    est = numeric(0L)
  )
}

lav_lavaan_baseline_add_partable_row <- function(partable = NULL,
                                                 lhs = "",
                                                 op = "",
                                                 rhs = "",
                                                 group = 1L,
                                                 free = 0L,
                                                 ustart = NA_real_,
                                                 exo = 0L,
                                                 est = NA_real_) {
  id <- length(partable$id) + 1L
  partable$id <- c(partable$id, id)
  partable$lhs <- c(partable$lhs, lhs)
  partable$op <- c(partable$op, op)
  partable$rhs <- c(partable$rhs, rhs)
  partable$user <- c(partable$user, 1L)
  partable$block <- c(partable$block, group)
  partable$group <- c(partable$group, group)
  partable$free <- c(partable$free, free)
  partable$ustart <- c(partable$ustart, ustart)
  partable$exo <- c(partable$exo, exo)
  partable$label <- c(partable$label, "")
  partable$start <- c(partable$start, if (is.na(ustart)) 0.0 else ustart)
  partable$est <- c(partable$est, est)
  partable
}

lav_lavaan_baseline_append_partable_rows <- function(partable = NULL,
                                                    rows = NULL,
                                                    free_idx = 0L) {
  if (length(rows) == 0L) {
    return(list(partable = partable, free_idx = free_idx))
  }
  for (row in rows) {
    if (isTRUE(row$free)) {
      free_idx <- free_idx + 1L
      free <- free_idx
    } else {
      free <- 0L
    }
    partable <- lav_lavaan_baseline_add_partable_row(
      partable = partable,
      lhs = row$lhs,
      op = row$op,
      rhs = row$rhs,
      group = row$group,
      free = free,
      ustart = row$ustart,
      exo = row$exo,
      est = row$est
    )
  }
  list(partable = partable, free_idx = free_idx)
}

lav_lavaan_baseline_row <- function(lhs = "",
                                    op = "",
                                    rhs = "",
                                    group = 1L,
                                    free = FALSE,
                                    ustart = NA_real_,
                                    exo = 0L,
                                    est = NA_real_) {
  list(
    lhs = lhs,
    op = op,
    rhs = rhs,
    group = group,
    free = free,
    ustart = ustart,
    exo = exo,
    est = est
  )
}

lav_lavaan_baseline_variance_rows <- function(names = NULL,
                                              values = NULL,
                                              free_names = names,
                                              exo_names = character(0L),
                                              group = 1L) {
  rows <- vector("list", length(names))
  for (i in seq_along(names)) {
    lhs <- names[[i]]
    rows[[i]] <- lav_lavaan_baseline_row(
      lhs = lhs,
      op = "~~",
      rhs = lhs,
      group = group,
      free = lhs %in% free_names,
      ustart = values[[i]],
      exo = as.integer(lhs %in% exo_names),
      est = values[[i]]
    )
  }
  rows
}

lav_lavaan_baseline_mean_rows <- function(names = NULL,
                                          values = NULL,
                                          free_names = names,
                                          exo_names = character(0L),
                                          group = 1L) {
  rows <- vector("list", length(names))
  for (i in seq_along(names)) {
    lhs <- names[[i]]
    rows[[i]] <- lav_lavaan_baseline_row(
      lhs = lhs,
      op = "~1",
      rhs = "",
      group = group,
      free = lhs %in% free_names,
      ustart = values[[i]],
      exo = as.integer(lhs %in% exo_names),
      est = values[[i]]
    )
  }
  rows
}

lav_lavaan_baseline_covariance_rows <- function(names = NULL,
                                                cov = NULL,
                                                moment_names = names,
                                                free = FALSE,
                                                fixed_ustart = TRUE,
                                                exo = 0L,
                                                group = 1L) {
  rows <- list()
  if (length(names) <= 1L) {
    return(rows)
  }
  row_idx <- 0L
  for (i in seq_len(length(names) - 1L)) {
    for (j in seq.int(i + 1L, length(names))) {
      lhs <- names[[i]]
      rhs <- names[[j]]
      lhs_idx <- match(lhs, moment_names)
      rhs_idx <- match(rhs, moment_names)
      if (is.na(lhs_idx) || is.na(rhs_idx)) {
        return(NULL)
      }
      value <- cov[lhs_idx, rhs_idx]
      if (!is.finite(value)) {
        return(NULL)
      }
      row_idx <- row_idx + 1L
      rows[[row_idx]] <- lav_lavaan_baseline_row(
        lhs = lhs,
        op = "~~",
        rhs = rhs,
        group = group,
        free = free,
        ustart = if (fixed_ustart) value else as.numeric(NA),
        exo = exo,
        est = value
      )
    }
  }
  rows
}

lav_lavaan_baseline_slope_rows <- function(lhs_names = NULL,
                                           rhs_names = NULL,
                                           slopes = NULL,
                                           group = 1L) {
  rows <- list()
  row_idx <- 0L
  for (rhs in rhs_names) {
    rhs_idx <- match(rhs, rhs_names)
    for (lhs in lhs_names) {
      lhs_idx <- match(lhs, lhs_names)
      value <- slopes[lhs_idx, rhs_idx]
      if (!is.finite(value)) {
        return(NULL)
      }
      row_idx <- row_idx + 1L
      rows[[row_idx]] <- lav_lavaan_baseline_row(
        lhs = lhs,
        op = "~",
        rhs = rhs,
        group = group,
        free = TRUE,
        ustart = value,
        exo = 1L,
        est = value
      )
    }
  }
  rows
}

lav_lavaan_baseline_model_log_det <- function(model_cov = NULL) {
  model_log_det <- as.numeric(determinant(model_cov, logarithm = TRUE)$modulus)
  if (!is.finite(model_log_det)) {
    return(NULL)
  }
  model_log_det
}

lav_lavaan_baseline_fx_group <- function(model_log_det = NULL,
                                         sample_log_det = NULL) {
  fx_group <- 0.5 * (model_log_det - sample_log_det)
  if (is.finite(fx_group) && fx_group < 0.0) {
    fx_group <- 0.0
  }
  fx_group
}

lav_lavaan_baseline_sy_diag <- function(sy = NULL,
                                        n = NULL) {
  sy_diag <- if (is.matrix(sy)) {
    diag(sy)
  } else {
    as.numeric(sy)
  }
  if (!is.null(n) && length(sy_diag) == 1L && n > 1L &&
      isTRUE(all.equal(sy_diag, 0.0))) {
    sy_diag <- numeric(n)
  }
  sy_diag
}

lav_lavaan_baseline_fiml_moments <- function(missing = NULL,
                                             nvar = NULL) {
  if (is.null(missing) || length(missing) == 0L || is.null(nvar)) {
    return(NULL)
  }

  count <- sum_y <- numeric(nvar)
  for (p in seq_along(missing)) {
    pat <- missing[[p]]
    var_idx <- pat$var.idx
    freq <- pat$freq
    my <- pat$MY
    if (!is.logical(var_idx) || length(var_idx) != nvar ||
        !is.finite(freq) || freq <= 0L) {
      return(NULL)
    }
    obs_idx <- which(var_idx)
    if (length(obs_idx) != length(my) || anyNA(my) || any(!is.finite(my))) {
      return(NULL)
    }
    count[obs_idx] <- count[obs_idx] + freq
    sum_y[obs_idx] <- sum_y[obs_idx] + freq * my
  }
  if (any(count <= 0.0)) {
    return(NULL)
  }

  mean <- sum_y / count
  ss <- numeric(nvar)
  for (p in seq_along(missing)) {
    pat <- missing[[p]]
    obs_idx <- which(pat$var.idx)
    sy_diag <- lav_lavaan_baseline_sy_diag(pat$SY, n = length(obs_idx))
    if (length(sy_diag) != length(obs_idx) ||
        anyNA(sy_diag) || any(!is.finite(sy_diag))) {
      return(NULL)
    }
    diff <- pat$MY - mean[obs_idx]
    ss[obs_idx] <- ss[obs_idx] + pat$freq * (sy_diag + diff * diff)
  }

  var <- ss / count
  if (!lav_lavaan_baseline_finite_vector(mean, n = nvar) ||
      !lav_lavaan_baseline_finite_vector(var, n = nvar, positive = TRUE)) {
    return(NULL)
  }

  list(mean = mean, var = var)
}

lav_lavaan_baseline_fiml_loglik <- function(missing = NULL,
                                            mean = NULL,
                                            var = NULL) {
  log_2pi <- log(2 * pi)
  log_var <- log(var)
  loglik <- 0.0

  for (p in seq_along(missing)) {
    pat <- missing[[p]]
    obs_idx <- which(pat$var.idx)
    sy_diag <- lav_lavaan_baseline_sy_diag(pat$SY, n = length(obs_idx))
    diff <- pat$MY - mean[obs_idx]
    loglik <- loglik - 0.5 * pat$freq * (
      length(obs_idx) * log_2pi +
        sum(log_var[obs_idx]) +
        sum((sy_diag + diff * diff) / var[obs_idx])
    )
  }

  loglik
}

lav_lavaan_baseline_simple_payload <- function(lavoptions = NULL,
                                                lavsamplestats = NULL,
                                                lavdata = NULL,
                                               lavpartable = NULL) {
  if (!lav_lavaan_baseline_supports_standard_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    conditional_x = FALSE
  ) ||
      lavdata@ngroups != 1L ||
      lav_lavaan_baseline_has_optim_bounds(lavoptions$optim.bounds)) {
    return(NULL)
  }

  sample_cov <- lav_lavaan_baseline_cov_moment(lavsamplestats, group = 1L)
  if (is.null(sample_cov)) {
    return(NULL)
  }
  sample_log_det <- lavsamplestats@cov.log.det[[1L]]
  if (!is.finite(sample_log_det)) {
    return(NULL)
  }

  observed_var <- diag(sample_cov)
  if (!lav_lavaan_baseline_finite_vector(observed_var, positive = TRUE)) {
    return(NULL)
  }
  ov_names <- lav_lavaan_baseline_ov_names(lavdata, sample_cov, group = 1L)
  if (is.null(ov_names)) {
    return(NULL)
  }

  if (isTRUE(lavoptions$meanstructure)) {
    sample_mean <- lav_lavaan_baseline_mean_moment(
      lavsamplestats = lavsamplestats,
      group = 1L,
      nvar = length(observed_var)
    )
    if (is.null(sample_mean)) {
      return(NULL)
    }
  } else {
    sample_mean <- numeric(0L)
  }

  ov_names_x <- lav_lavaan_baseline_ov_x_names(lavdata, group = 1L)
  exo_block_names <- lav_lavaan_baseline_exo_block_names(
    lavpartable = lavpartable,
    ov_names = ov_names,
    ov_names_x = ov_names_x
  )
  if (is.null(exo_block_names)) {
    return(NULL)
  }

  if (isTRUE(lavoptions$fixed.x) &&
      length(exo_block_names) > 0L &&
      !isTRUE(lavoptions$baseline.fixed.x.free.cov)) {
    return(NULL)
  }

  fixed_x_names <- if (isTRUE(lavoptions$fixed.x)) ov_names_x else character(0L)
  free_ov_names <- ov_names[!ov_names %in% fixed_x_names]

  rows <- lav_lavaan_baseline_variance_rows(
    names = ov_names,
    values = observed_var,
    free_names = free_ov_names,
    exo_names = fixed_x_names
  )
  if (isTRUE(lavoptions$meanstructure)) {
    rows <- c(rows, lav_lavaan_baseline_mean_rows(
      names = ov_names,
      values = sample_mean,
      free_names = free_ov_names,
      exo_names = fixed_x_names
    ))
  }

  exo_cov_rows <- lav_lavaan_baseline_covariance_rows(
    names = exo_block_names,
    cov = sample_cov,
    moment_names = ov_names,
    free = !all(exo_block_names %in% fixed_x_names),
    fixed_ustart = all(exo_block_names %in% fixed_x_names),
    exo = as.integer(all(exo_block_names %in% fixed_x_names))
  )
  if (is.null(exo_cov_rows)) {
    return(NULL)
  }
  rows <- c(rows, exo_cov_rows)

  partable_out <- lav_lavaan_baseline_append_partable_rows(
    partable = lav_lavaan_baseline_new_partable(),
    rows = rows,
    free_idx = 0L
  )
  partable <- partable_out$partable

  model_cov <- diag(observed_var, nrow = length(observed_var))
  dimnames(model_cov) <- list(ov_names, ov_names)
  if (length(exo_block_names) > 1L) {
    for (i in seq_len(length(exo_block_names) - 1L)) {
      for (j in seq.int(i + 1L, length(exo_block_names))) {
        lhs_idx <- match(exo_block_names[[i]], ov_names)
        rhs_idx <- match(exo_block_names[[j]], ov_names)
        value <- sample_cov[lhs_idx, rhs_idx]
        model_cov[lhs_idx, rhs_idx] <- value
        model_cov[rhs_idx, lhs_idx] <- value
      }
    }
  }

  model_log_det <- lav_lavaan_baseline_model_log_det(model_cov)
  if (is.null(model_log_det)) {
    return(NULL)
  }
  offdiag_cov <- length(exo_cov_rows)
  nvar <- length(observed_var)
  df_group <- as.integer(nvar * (nvar - 1L) / 2L - offdiag_cov)

  list(
    partable = partable,
    test = lav_lavaan_baseline_standard_test(
      fx_group = lav_lavaan_baseline_fx_group(model_log_det, sample_log_det),
      df_group = df_group,
      lavoptions = lavoptions,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata
    )
  )
}

lav_lavaan_baseline_fiml_payload <- function(lavoptions = NULL,
                                             lavsamplestats = NULL,
                                             lavdata = NULL,
                                             lavpartable = NULL,
                                             lavh1 = NULL) {
  if (!lav_lavaan_baseline_supports_fiml_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata
  )) {
    return(NULL)
  }

  ngroups <- lavdata@ngroups
  if (length(lavsamplestats@missing) != ngroups ||
      length(lavsamplestats@cov) != ngroups ||
      is.null(lavh1$logl$loglik.group) ||
      length(lavh1$logl$loglik.group) != ngroups) {
    return(NULL)
  }

  moments <- vector("list", ngroups)
  ov_names <- vector("list", ngroups)
  start_var <- vector("list", ngroups)
  start_mean <- vector("list", ngroups)
  h0_loglik <- numeric(ngroups)
  df_group <- integer(ngroups)
  for (g in seq_len(ngroups)) {
    nvar <- ncol(lavsamplestats@cov[[g]])
    ov_names[[g]] <- lav_lavaan_baseline_ov_names(
      lavdata = lavdata,
      sample_cov = lavsamplestats@cov[[g]],
      group = g
    )
    if (is.null(ov_names[[g]])) {
      return(NULL)
    }
    start_var[[g]] <- diag(lavsamplestats@cov[[g]])
    if (!lav_lavaan_baseline_finite_vector(start_var[[g]], n = nvar,
                                           positive = TRUE)) {
      return(NULL)
    }
    start_mean[[g]] <- lav_lavaan_baseline_mean_moment(
      lavsamplestats = lavsamplestats,
      group = g,
      nvar = nvar
    )
    if (is.null(start_mean[[g]])) {
      return(NULL)
    }
    moments[[g]] <- lav_lavaan_baseline_fiml_moments(
      missing = lavsamplestats@missing[[g]],
      nvar = nvar
    )
    if (is.null(moments[[g]])) {
      return(NULL)
    }
    h0_loglik[[g]] <- lav_lavaan_baseline_fiml_loglik(
      missing = lavsamplestats@missing[[g]],
      mean = moments[[g]]$mean,
      var = moments[[g]]$var
    )
    if (!is.finite(h0_loglik[[g]])) {
      return(NULL)
    }
    df_group[[g]] <- as.integer(nvar * (nvar - 1L) / 2L)
  }

  rows <- list()
  for (g in seq_len(ngroups)) {
    rows <- c(rows, lav_lavaan_baseline_variance_rows(
      names = ov_names[[g]],
      values = start_var[[g]],
      group = g
    ))
    rows <- c(rows, lav_lavaan_baseline_mean_rows(
      names = ov_names[[g]],
      values = start_mean[[g]],
      group = g
    ))
  }
  partable_out <- lav_lavaan_baseline_append_partable_rows(
    partable = lav_lavaan_baseline_new_partable(),
    rows = rows,
    free_idx = 0L
  )
  partable <- partable_out$partable

  if (!is.null(lavoptions$optim.bounds)) {
    partable$lower <- rep(-Inf, length(partable$lhs))
    partable$upper <- rep(Inf, length(partable$lhs))
    var_rows <- which(partable$op == "~~" & partable$lhs == partable$rhs)
    partable$lower[var_rows] <- 0.0
  }

  for (g in seq_len(ngroups)) {
    group_rows <- which(partable$group == g)
    for (i in group_rows) {
      lhs_idx <- match(partable$lhs[[i]], ov_names[[g]])
      if (is.na(lhs_idx)) {
        return(NULL)
      }
      if (partable$op[[i]] == "~~" && partable$lhs[[i]] == partable$rhs[[i]]) {
        value <- moments[[g]]$var[[lhs_idx]]
      } else if (partable$op[[i]] == "~1") {
        value <- moments[[g]]$mean[[lhs_idx]]
      } else {
        return(NULL)
      }
      partable$est[[i]] <- value
    }
  }

  stat_group <- 2 * (lavh1$logl$loglik.group - h0_loglik)
  stat_group[abs(stat_group) < .Machine$double.eps^(1 / 2)] <- 0.0
  if (any(!is.finite(stat_group)) || any(stat_group < 0.0)) {
    return(NULL)
  }

  list(
    partable = partable,
    test = lav_lavaan_baseline_standard_test(
      stat_group = stat_group,
      df_group = df_group,
      lavoptions = lavoptions,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata
    )
  )
}

lav_lavaan_baseline_conditional_x_payload <- function(lavoptions = NULL,
                                                      lavsamplestats = NULL,
                                                      lavdata = NULL) {
  if (!lav_lavaan_baseline_supports_standard_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    conditional_x = TRUE
  ) ||
      !isTRUE(lavoptions$fixed.x) ||
      lavdata@ngroups != 1L ||
      lav_lavaan_baseline_has_optim_bounds(lavoptions$optim.bounds)) {
    return(NULL)
  }

  ov_names <- lavdata@ov.names[[1L]]
  ov_names_x <- lav_lavaan_baseline_ov_x_names(lavdata, group = 1L)
  if (length(ov_names_x) == 0L ||
      !lav_lavaan_baseline_valid_names(ov_names, length(ov_names)) ||
      !lav_lavaan_baseline_valid_names(ov_names_x, length(ov_names_x))) {
    return(NULL)
  }

  nvar <- length(ov_names)
  nexo <- length(ov_names_x)
  res_cov <- lav_lavaan_baseline_res_cov_moment(lavsamplestats, 1L, nvar)
  res_slopes <- lav_lavaan_baseline_res_slopes_moment(
    lavsamplestats, 1L, nvar, nexo
  )
  cov_x <- lav_lavaan_baseline_cov_x_moment(lavsamplestats, 1L, nexo)
  if (is.null(res_cov) || is.null(res_slopes) || is.null(cov_x)) {
    return(NULL)
  }
  sample_log_det <- lavsamplestats@res.cov.log.det[[1L]]
  if (!is.finite(sample_log_det)) {
    return(NULL)
  }

  observed_var <- diag(res_cov)
  if (!lav_lavaan_baseline_finite_vector(observed_var, positive = TRUE)) {
    return(NULL)
  }
  x_var <- diag(cov_x)
  if (!lav_lavaan_baseline_finite_vector(x_var, n = nexo)) {
    return(NULL)
  }

  if (isTRUE(lavoptions$meanstructure)) {
    res_int <- lav_lavaan_baseline_res_int_moment(lavsamplestats, 1L, nvar)
    mean_x <- lav_lavaan_baseline_mean_x_moment(lavsamplestats, 1L, nexo)
    if (is.null(res_int) || is.null(mean_x)) {
      return(NULL)
    }
  } else {
    res_int <- numeric(0L)
    mean_x <- numeric(0L)
  }

  rows <- lav_lavaan_baseline_variance_rows(
    names = ov_names,
    values = observed_var
  )
  if (isTRUE(lavoptions$meanstructure)) {
    rows <- c(rows, lav_lavaan_baseline_mean_rows(
      names = ov_names,
      values = res_int
    ))
  }
  rows <- c(rows, lav_lavaan_baseline_variance_rows(
    names = ov_names_x,
    values = x_var,
    free_names = character(0L),
    exo_names = ov_names_x
  ))
  cov_x_rows <- lav_lavaan_baseline_covariance_rows(
    names = ov_names_x,
    cov = cov_x,
    free = FALSE,
    fixed_ustart = TRUE,
    exo = 1L
  )
  if (is.null(cov_x_rows)) {
    return(NULL)
  }
  rows <- c(rows, cov_x_rows)
  if (isTRUE(lavoptions$meanstructure)) {
    rows <- c(rows, lav_lavaan_baseline_mean_rows(
      names = ov_names_x,
      values = mean_x,
      free_names = character(0L),
      exo_names = ov_names_x
    ))
  }
  slope_rows <- lav_lavaan_baseline_slope_rows(
    lhs_names = ov_names,
    rhs_names = ov_names_x,
    slopes = res_slopes
  )
  if (is.null(slope_rows)) {
    return(NULL)
  }
  rows <- c(rows, slope_rows)

  partable_out <- lav_lavaan_baseline_append_partable_rows(
    partable = lav_lavaan_baseline_new_partable(),
    rows = rows,
    free_idx = 0L
  )
  model_log_det <- sum(log(observed_var))
  df_group <- as.integer(nvar * (nvar - 1L) / 2L)

  list(
    partable = partable_out$partable,
    test = lav_lavaan_baseline_standard_test(
      fx_group = lav_lavaan_baseline_fx_group(model_log_det, sample_log_det),
      df_group = df_group,
      lavoptions = lavoptions,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata
    )
  )
}

lav_lavaan_baseline_generated_independence_payload <- function(lavoptions = NULL,
                                                               lavsamplestats = NULL,
                                                               lavdata = NULL,
                                                               lavpartable = NULL,
                                                               lavh1 = NULL) {
  if (!lav_lavaan_baseline_supports_standard_ml(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    conditional_x = FALSE
  )) {
    return(NULL)
  }

  ngroups <- lavdata@ngroups
  if (length(lavsamplestats@cov) != ngroups ||
      length(lavsamplestats@cov.log.det) != ngroups) {
    return(NULL)
  }

  observed_var <- vector("list", ngroups)
  ov_names <- vector("list", ngroups)
  ov_names_x <- vector("list", ngroups)
  sample_log_det <- numeric(ngroups)
  sample_mean <- vector("list", ngroups)
  for (g in seq_len(ngroups)) {
    sample_cov <- lav_lavaan_baseline_cov_moment(lavsamplestats, group = g)
    if (is.null(sample_cov)) {
      return(NULL)
    }
    observed_var[[g]] <- diag(sample_cov)
    if (!lav_lavaan_baseline_finite_vector(observed_var[[g]], positive = TRUE)) {
      return(NULL)
    }
    sample_log_det[[g]] <- lavsamplestats@cov.log.det[[g]]
    if (!is.finite(sample_log_det[[g]])) {
      return(NULL)
    }
    ov_names[[g]] <- lav_lavaan_baseline_ov_names(lavdata, sample_cov, group = g)
    if (is.null(ov_names[[g]])) {
      return(NULL)
    }
    ov_names_x[[g]] <- lav_lavaan_baseline_ov_x_names(lavdata, group = g)
    if (length(ov_names_x[[g]]) > 0L) {
      if (!isTRUE(lavoptions$fixed.x) ||
          !isTRUE(lavoptions$baseline.fixed.x.free.cov) ||
          any(!ov_names_x[[g]] %in% ov_names[[g]])) {
        return(NULL)
      }
    }
    if (isTRUE(lavoptions$meanstructure)) {
      sample_mean[[g]] <- lav_lavaan_baseline_mean_moment(
        lavsamplestats = lavsamplestats,
        group = g,
        nvar = length(observed_var[[g]])
      )
      if (is.null(sample_mean[[g]])) {
        return(NULL)
      }
    }
  }

  lavpta <- lav_partable_attributes(lavpartable)
  partable <- lav_partable_indep_or_unrestricted(
    lavobject = NULL,
    lavdata = lavdata,
    lavpta = lavpta,
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavh1 = lavh1,
    independent = TRUE
  )

  if (lav_lavaan_baseline_has_optim_bounds(lavoptions$optim.bounds)) {
    lavoptions_bounds <- lavoptions
    lavoptions_bounds$bounds <- "doe.maar"
    lavoptions_bounds$effect.coding <- ""
    lavoptions_bounds$optim.bounds <- list(lower = "ov.var")
    partable <- lav_partable_add_bounds(
      partable = partable,
      lavh1 = lavh1,
      lavdata = lavdata,
      lavsamplestats = lavsamplestats,
      lavoptions = lavoptions_bounds
    )
  }
  if (is.null(partable$label)) {
    partable$label <- rep("", length(partable$lhs))
  }
  partable$start <- ifelse(is.na(partable$ustart), 0.0, partable$ustart)
  partable$est <- numeric(length(partable$lhs))

  fx_group <- numeric(ngroups)
  df_group <- integer(ngroups)
  for (g in seq_len(ngroups)) {
    sample_cov <- lavsamplestats@cov[[g]]
    model_cov <- diag(observed_var[[g]], nrow = length(observed_var[[g]]))
    dimnames(model_cov) <- list(ov_names[[g]], ov_names[[g]])
    group_rows <- which(partable$group == g)
    offdiag_cov <- 0L
    for (i in group_rows) {
      if (partable$op[[i]] == "~~") {
        lhs_idx <- match(partable$lhs[[i]], ov_names[[g]])
        rhs_idx <- match(partable$rhs[[i]], ov_names[[g]])
        if (is.na(lhs_idx) || is.na(rhs_idx)) {
          return(NULL)
        }
        value <- sample_cov[lhs_idx, rhs_idx]
        if (!is.finite(value)) {
          return(NULL)
        }
        partable$est[[i]] <- value
        if (lhs_idx != rhs_idx) {
          model_cov[lhs_idx, rhs_idx] <- value
          model_cov[rhs_idx, lhs_idx] <- value
          offdiag_cov <- offdiag_cov + 1L
        }
      } else if (partable$op[[i]] == "~1") {
        lhs_idx <- match(partable$lhs[[i]], ov_names[[g]])
        if (is.na(lhs_idx) ||
            length(sample_mean[[g]]) != length(ov_names[[g]])) {
          return(NULL)
        }
        value <- sample_mean[[g]][[lhs_idx]]
        if (!is.finite(value)) {
          return(NULL)
        }
        partable$est[[i]] <- value
      } else {
        return(NULL)
      }
    }
    model_log_det <- lav_lavaan_baseline_model_log_det(model_cov)
    if (is.null(model_log_det)) {
      return(NULL)
    }
    fx_group[[g]] <- lav_lavaan_baseline_fx_group(
      model_log_det = model_log_det,
      sample_log_det = sample_log_det[[g]]
    )
    nvar <- length(observed_var[[g]])
    df_group[[g]] <- as.integer(nvar * (nvar - 1L) / 2L - offdiag_cov)
  }

  list(
    partable = partable,
    test = lav_lavaan_baseline_standard_test(
      fx_group = fx_group,
      df_group = df_group,
      lavoptions = lavoptions,
      lavsamplestats = lavsamplestats,
      lavdata = lavdata
    )
  )
}
