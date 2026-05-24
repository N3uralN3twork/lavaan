lav_lavaan_step15_baseline_fast <- function(lavoptions = NULL,
                                            lavsamplestats = NULL,
                                            lavdata = NULL,
                                            lavpartable = NULL,
                                            lavh1 = NULL) {
  conditional_x_payload <- lav_lavaan_baseline_conditional_x_payload(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata
  )
  if (!is.null(conditional_x_payload)) {
    return(conditional_x_payload)
  }

  simple_payload <- lav_lavaan_baseline_simple_payload(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    lavpartable = lavpartable
  )
  if (!is.null(simple_payload)) {
    return(simple_payload)
  }

  lav_lavaan_baseline_generated_independence_payload(
    lavoptions = lavoptions,
    lavsamplestats = lavsamplestats,
    lavdata = lavdata,
    lavpartable = lavpartable,
    lavh1 = lavh1
  )
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
      lavpartable = lavpartable,
      lavh1 = lavh1
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
