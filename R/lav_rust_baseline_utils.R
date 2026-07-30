# Rust adapters for the pure numerical FIML helpers in
# `lav_lavaan_baseline_utils.R`. S4/data orchestration remains in R.

lav_rust_baseline_flatten_missing <- function(missing = NULL, nvar = NULL) {
  if (is.null(missing) || length(missing) == 0L || is.null(nvar) ||
      length(nvar) != 1L || is.na(nvar) || nvar < 1L) {
    return(NULL)
  }

  pattern_lengths <- integer(length(missing))
  frequencies <- numeric(length(missing))
  observed_indices <- means <- sy_diagonals <- numeric(0L)
  for (p in seq_along(missing)) {
    pat <- missing[[p]]
    var_idx <- pat$var.idx
    frequency <- pat$freq
    my <- pat$MY
    if (!is.logical(var_idx) || length(var_idx) != nvar ||
        !is.finite(frequency) || frequency <= 0.0) {
      return(NULL)
    }
    obs_idx <- which(var_idx)
    sy_diag <- lav_lavaan_baseline_sy_diag(pat$SY, n = length(obs_idx))
    if (length(obs_idx) != length(my) || length(sy_diag) != length(obs_idx) ||
        anyNA(my) || any(!is.finite(my)) || anyNA(sy_diag) ||
        any(!is.finite(sy_diag))) {
      return(NULL)
    }
    pattern_lengths[[p]] <- length(obs_idx)
    frequencies[[p]] <- frequency
    observed_indices <- c(observed_indices, obs_idx - 1L)
    means <- c(means, my)
    sy_diagonals <- c(sy_diagonals, sy_diag)
  }

  list(
    pattern_lengths = pattern_lengths,
    observed_indices = as.integer(observed_indices),
    frequencies = frequencies,
    means = means,
    sy_diagonals = sy_diagonals
  )
}

lav_native_lav_lavaan_baseline_fiml_moments <- function(missing = NULL,
                                                         nvar = NULL) {
  flattened <- lav_rust_baseline_flatten_missing(missing, nvar)
  if (is.null(flattened)) {
    return(NULL)
  }
  # The R reference treats infeasible moment data as a declined fast-path
  # (`NULL`), not as an exception. The Rust kernel validates the same contract;
  # this handler only restores that documented `NULL` result for those inputs.
  values <- tryCatch(lav_rust_baseline_fiml_moments(
    flattened$pattern_lengths,
    flattened$observed_indices,
    flattened$frequencies,
    flattened$means,
    flattened$sy_diagonals,
    as.integer(nvar)
  ), error = function(e) NULL)
  if (is.null(values)) {
    return(NULL)
  }
  list(
    mean = values[seq_len(nvar)],
    var = values[nvar + seq_len(nvar)]
  )
}

lav_native_lav_lavaan_baseline_fiml_loglik <- function(missing = NULL,
                                                        mean = NULL,
                                                        var = NULL) {
  flattened <- lav_rust_baseline_flatten_missing(missing, length(mean))
  if (is.null(flattened)) {
    return(NULL)
  }
  lav_rust_baseline_fiml_loglik(
    flattened$pattern_lengths,
    flattened$observed_indices,
    flattened$frequencies,
    flattened$means,
    flattened$sy_diagonals,
    as.numeric(mean),
    as.numeric(var)
  )
}
