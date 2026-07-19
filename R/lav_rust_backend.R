# Internal Rust bridge helpers.
#
# These functions keep the public `lavaan/R` implementation stable while
# selected Rust kernels under `lavaan/rust/lavaan-kernels/` are called through
# rextendr-generated package wrappers. Rust is built during package
# compilation, not from hot R call sites.
#
# Backend selection controls whether the native bridge may be used at all:
# - default: Rust bridge available
# - override for benchmarks: options(lavaan.backend = "r") or
#   Sys.setenv(LAVAAN_BACKEND = "r")
#
# Kernel selection controls which Rust kernels are actually used. All available
# experimental kernels are enabled by default. Override with
# options(lavaan.rust.kernels = "none"), a specific kernel list, or
# Sys.setenv(LAVAAN_RUST_KERNELS = "none").

lav_backend_mode <- function() {
  env_mode <- tolower(Sys.getenv("LAVAAN_BACKEND", unset = ""))
  opt_mode <- tolower(getOption("lavaan.backend", default = ""))

  mode <- if (nzchar(env_mode)) {
    env_mode
  } else if (nzchar(opt_mode)) {
    opt_mode
  } else {
    "rust"
  }

  match.arg(mode, c("rust", "r", "auto"))
}

lav_backend_use <- function(mode = c("rust", "r", "auto")) {
  mode <- match.arg(mode)
  options(lavaan.backend = mode)
  invisible(mode)
}

lav_backend_reset <- function() {
  options(lavaan.backend = NULL)
  invisible(NULL)
}


lav_native_backend_available <- function() {
  isTRUE(
    tryCatch(
      lav_rust_backend_ping(),
      error = function(e) FALSE
    )
  )
}

lav_rust_prepare_backend <- function() {
  invisible(lav_native_backend_available())
}

lav_backend_available <- function() {
  lav_rust_backend_available()
}

lav_rust_backend_available <- function() {
  if (identical(lav_backend_mode(), "r")) {
    return(FALSE)
  }

  lav_native_backend_available()
}

lav_rust_kernel_enabled <- function(kernel) {
  kernel <- match.arg(kernel, c("matrix", "model_gradient", "model_vcov", "model_estimate"))
  env_kernels <- Sys.getenv("LAVAAN_RUST_KERNELS", unset = "")
  opt_kernels <- getOption("lavaan.rust.kernels", default = NULL)

  enabled <- if (!nzchar(env_kernels) && is.null(opt_kernels)) {
    "all"
  } else if (nzchar(env_kernels)) {
    trimws(strsplit(tolower(env_kernels), ",", fixed = TRUE)[[1L]])
  } else if (!is.null(opt_kernels)) {
    trimws(tolower(as.character(opt_kernels)))
  } else {
    character(0L)
  }

  if ("all" %in% enabled) {
    return(lav_rust_backend_available())
  }
  if ("none" %in% enabled) {
    return(FALSE)
  }

  if (!(kernel %in% enabled)) {
    return(FALSE)
  }

  lav_rust_backend_available()
}

lav_rust_try_model_gradient <- function(expr, ...) {
  if (!lav_rust_kernel_enabled("model_gradient")) {
    return(NULL)
  }

  try(expr, silent = TRUE)
}

lav_native_lav_matrix_vec <- function(A) {
  lav_rust_matrix_vec(A)
}

lav_native_lav_matrix_vecr <- function(A) {
  lav_rust_matrix_vecr(A)
}

lav_native_lav_matrix_diag_prepost <- function(A, d) {
  lav_rust_matrix_diag_prepost(A, d)
}

lav_native_lav_matrix_delta_A_delta <- function(delta, a1) {
  lav_rust_matrix_delta_a_delta(delta, a1)
}

lav_native_lav_model_vcov_delta_A_delta <- function(delta, a1) {
  lav_rust_model_vcov_delta_a_delta(delta, a1)
}

lav_native_lav_model_vcov_sandwich <- function(left, middle) {
  lav_rust_model_vcov_sandwich(left, middle)
}

lav_native_lav_model_vcov_jacobian_vcov_jacobian_t <- function(jac, VCOV) {
  lav_rust_model_vcov_jacobian_vcov_jacobian_t(jac, VCOV)
}

lav_native_lav_model_gradient_conditional_x_sample_cache <- function(mean_x, cov_x, res_int, res_slopes) {
  lav_rust_model_gradient_conditional_x_sample_cache(mean_x, cov_x, res_int, res_slopes)
}

lav_native_lav_model_gradient_omega_ml <- function(sigma, sigma_inv, sample_cov,
                                                   mean_diff = numeric(0L),
                                                   meanstructure = FALSE) {
  lav_rust_model_gradient_omega_ml(
    sigma, sigma_inv, sample_cov, as.numeric(mean_diff),
    as.logical(meanstructure)
  )
}

lav_native_lav_model_gradient_omega_gls <- function(sigma, weight_inv, sample_cov,
                                                    mean_diff = numeric(0L),
                                                    meanstructure = FALSE,
                                                    nobs = 1.0) {
  lav_rust_model_gradient_omega_gls(
    sigma, weight_inv, sample_cov, as.numeric(mean_diff),
    as.logical(meanstructure), as.numeric(nobs)
  )
}

lav_native_lav_model_gradient_omega_missing_pattern <- function(sigma_inv, sample_cov,
                                                                mean_diff, var_idx,
                                                                nvar, weight) {
  lav_rust_model_gradient_omega_missing_pattern(
    sigma_inv, sample_cov, as.numeric(mean_diff), as.integer(var_idx),
    as.integer(nvar), as.numeric(weight)
  )
}

lav_native_lav_model_gradient_delta_post <- function(delta, post, scale = 1.0) {
  lav_rust_model_gradient_delta_post(delta, post, as.numeric(scale))
}

lav_native_lav_model_gradient_t_d1_delta <- function(d1, delta, scale = 1.0) {
  lav_rust_model_gradient_t_d1_delta(d1, delta, as.numeric(scale))
}

lav_native_lav_model_gradient_group_weight <- function(log_group_weight,
                                                       observed_group_weight,
                                                       total_nobs) {
  lav_rust_model_gradient_group_weight(
    as.numeric(log_group_weight), as.numeric(observed_group_weight),
    as.numeric(total_nobs)
  )
}

lav_native_lav_model_gradient_wls <- function(delta, wls_v, diff, group_weight = 1.0) {
  lav_rust_model_gradient_wls(delta, wls_v, diff, as.numeric(group_weight))
}

lav_native_lav_model_gradient_dwls <- function(delta, wls_vd, diff, group_weight = 1.0) {
  lav_rust_model_gradient_dwls(delta, wls_vd, diff, as.numeric(group_weight))
}

lav_native_lav_model_gradient_ml_conditional_post <- function(c3, obs, mu, pi, sigma_inv, res_cov) {
  lav_rust_model_gradient_ml_conditional_post(c3, obs, mu, pi, sigma_inv, res_cov)
}

lav_native_lav_model_gradient_ml_conditional <- function(delta, post, group_weight = 1.0) {
  lav_rust_model_gradient_ml_conditional(delta, post, as.numeric(group_weight))
}

lav_native_lav_model_gradient_ml_group <- function(delta, omega,
                                                   omega_mu = numeric(0L),
                                                   group_weight = 1.0,
                                                   meanstructure = FALSE,
                                                   group_weight_free = FALSE) {
  lav_rust_model_gradient_ml_group(
    delta, as.numeric(omega), as.numeric(omega_mu), as.numeric(group_weight),
    as.logical(meanstructure), as.logical(group_weight_free)
  )
}

lav_native_lav_model_gradient_ntrls_post <- function(sample_cov, model_cov, sigma_inv,
                                                     mean_observed = numeric(0L),
                                                     mean_model = numeric(0L),
                                                     meanstructure = FALSE) {
  lav_rust_model_gradient_ntrls_post(
    sample_cov, model_cov, sigma_inv, mean_observed, mean_model,
    as.logical(meanstructure)
  )
}

lav_native_lav_model_estimate_pack <- function(values, k0, K) {
  lav_rust_model_estimate_pack(values, k0, K)
}

lav_native_lav_model_estimate_unpack_unscale <- function(values, k0, K, parscale) {
  lav_rust_model_estimate_unpack_unscale(values, k0, K, parscale)
}

lav_native_lav_model_estimate_gradient_postprocess <- function(dx, parscale, K,
                                                                total_nobs, pml) {
  lav_rust_model_estimate_gradient_postprocess(
    dx, parscale, K, as.numeric(total_nobs), as.logical(pml)
  )
}

lav_native_lav_matrix_diagh_idx <- function(n) {
  lav_rust_matrix_diagh_idx(as.integer(n))
}

lav_native_lav_matrix_antidiag_idx <- function(n) {
  lav_rust_matrix_antidiag_idx(as.integer(n))
}

lav_native_lav_matrix_diag_idx <- function(n) {
  lav_rust_matrix_diag_idx(as.integer(n))
}

lav_native_lav_matrix_vech_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vech_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vech_row_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vech_row_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vech_col_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vech_col_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vechr_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vechr_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vechu_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vechu_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vechru_idx <- function(n, diagonal = TRUE) {
  lav_rust_matrix_vechru_idx(as.integer(n), as.logical(diagonal))
}

lav_native_lav_matrix_vech <- function(S, diagonal = TRUE) {
  lav_rust_matrix_vech(S, as.logical(diagonal))
}

lav_native_lav_matrix_vechr <- function(S, diagonal = TRUE) {
  lav_rust_matrix_vechr(S, as.logical(diagonal))
}

lav_native_lav_matrix_vechu <- function(S, diagonal = TRUE) {
  lav_rust_matrix_vechu(S, as.logical(diagonal))
}

lav_native_lav_matrix_vechru <- function(S, diagonal = TRUE) {
  lav_rust_matrix_vechru(S, as.logical(diagonal))
}

lav_native_lav_matrix_vech_reverse <- function(x, diagonal = TRUE) {
  lav_rust_matrix_vech_reverse(x, as.logical(diagonal))
}

lav_native_lav_matrix_vechru_reverse <- function(x, diagonal = TRUE) {
  lav_rust_matrix_vechru_reverse(x, as.logical(diagonal))
}

lav_native_lav_matrix_upper2full <- function(x, diagonal = TRUE) {
  lav_rust_matrix_upper2full(x, as.logical(diagonal))
}

lav_native_lav_matrix_vechr_reverse <- function(x, diagonal = TRUE) {
  lav_rust_matrix_vechr_reverse(x, as.logical(diagonal))
}

lav_native_lav_matrix_vechu_reverse <- function(x, diagonal = TRUE) {
  lav_rust_matrix_vechu_reverse(x, as.logical(diagonal))
}

lav_native_lav_matrix_lower2full <- function(x, diagonal = TRUE) {
  lav_rust_matrix_lower2full(x, as.logical(diagonal))
}

lav_matrix_vec <- function(A) {
  if (lav_rust_kernel_enabled("matrix") && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(lav_native_lav_matrix_vec(A), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  as.vector(A)
}

lav_matrix_vecr <- function(A) {
  if (lav_rust_kernel_enabled("matrix") && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(lav_native_lav_matrix_vecr(A), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  lav_matrix_vec(t(A))
}

lav_matrix_diag_idx <- function(n = 1L) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_diag_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  1L + (seq_len(n) - 1L) * (n + 1L)
}

lav_matrix_diagh_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && diagonal && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_diagh_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  if (n == 1L) {
    return(1L)
  }
  c(1L, cumsum(n:2L) + 1L)
}

lav_matrix_antidiag_idx <- function(n = 1L) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_antidiag_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  1L + seq_len(n) * (n - 1L)
}

lav_matrix_vech_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vech_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  if (diagonal) {
    sequence(n:1L) + rep((seq_len(n) - 1L) * (n + 1L), n:1L)
  } else {
    if (n == 1L) {
      return(integer(0L))
    }
    (sequence((n - 1L):1L) +
      rep((seq_len(n - 1L) - 1L) * (n + 1L) + 1L, (n - 1L):1L))
  }
}

lav_matrix_vech_row_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vech_row_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (diagonal) {
    sequence(n:1L) + rep(seq_len(n) - 1L, n:1L)
  } else {
    if (n <= 1L) {
      return(integer(0L))
    }
    sequence((n - 1L):1L) + rep(seq_len(n - 1L), (n - 1L):1L)
  }
}

lav_matrix_vech_col_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vech_col_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (!diagonal) {
    n <- n - 1L
  }
  rep.int(seq_len(n), times = rev(seq_len(n)))
}

lav_matrix_vechr_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vechr_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  if (diagonal) {
    row_idx <- rep(seq_len(n), seq_len(n))
    col_idx <- sequence(seq_len(n))
  } else {
    if (n == 1L) {
      return(integer(0L))
    }
    row_idx <- rep(seq_len(n)[-1L], seq_len(n - 1L))
    col_idx <- sequence(seq_len(n - 1L))
  }
  (col_idx - 1L) * n + row_idx
}

lav_matrix_vechu_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vechu_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  if (diagonal) {
    sequence(seq_len(n)) + rep((seq_len(n) - 1L) * n, seq_len(n))
  } else {
    if (n == 1L) {
      return(integer(0L))
    }
    sequence(seq_len(n - 1L)) + rep(seq_len(n - 1L) * n, seq_len(n - 1L))
  }
}

lav_matrix_vechru_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_kernel_enabled("matrix") && n >= 1L) {
    rust_result <- try(lav_native_lav_matrix_vechru_idx(n, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  if (diagonal) {
    row_idx <- rep(seq_len(n), n:1L)
    col_idx <- sequence(n:1L) + rep(seq_len(n) - 1L, n:1L)
  } else {
    if (n == 1L) {
      return(integer(0L))
    }
    row_idx <- rep(seq_len(n - 1L), (n - 1L):1L)
    col_idx <- sequence((n - 1L):1L) + rep(seq_len(n - 1L), (n - 1L):1L)
  }
  (col_idx - 1L) * n + row_idx
}

lav_matrix_vech <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_kernel_enabled("matrix") && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vech(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  row_1 <- row(S)
  col_1 <- col(S)
  if (diagonal) S[row_1 >= col_1] else S[row_1 > col_1]
}

lav_matrix_vechr <- function(S, diagonal = TRUE) {      # nolint
  if (lav_rust_kernel_enabled("matrix") && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechr(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechr_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechu <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_kernel_enabled("matrix") && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechu(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechu_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechru <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_kernel_enabled("matrix") && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechru(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechru_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vech_reverse <- lav_matrix_vechru_reverse <- lav_matrix_upper2full <- function(x, 
    diagonal = TRUE) {
  if (lav_rust_kernel_enabled("matrix") && is.numeric(x)) {
    rust_command <- "upper2full"
    if (identical(match.call()$FUN, as.name("lav_matrix_vech_reverse"))) {
      rust_command <- "vech_reverse"
    } else if (identical(match.call()$FUN, as.name("lav_matrix_vechru_reverse"))) {
      rust_command <- "vechru_reverse"
    }

    rust_result <- try(
      switch(
        rust_command,
        vech_reverse = lav_native_lav_matrix_vech_reverse(x, diagonal),
        vechru_reverse = lav_native_lav_matrix_vechru_reverse(x, diagonal),
        upper2full = lav_native_lav_matrix_upper2full(x, diagonal)
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (diagonal) {
    p <- (sqrt(1 + 8 * length(x)) - 1)/2
  }
  else {
    p <- (sqrt(1 + 8 * length(x)) + 1)/2
  }
  stopifnot(p == round(p, 0))
  s <- numeric(p * p)
  s[lav_matrix_vech_idx(p, diagonal = diagonal)] <- x
  s[lav_matrix_vechru_idx(p, diagonal = diagonal)] <- x
  attr(s, "dim") <- c(p, p)
  s
}

lav_matrix_vechr_reverse <- lav_matrix_vechu_reverse <- lav_matrix_lower2full <- function(x, 
    diagonal = TRUE) {
  if (lav_rust_kernel_enabled("matrix") && is.numeric(x)) {
    rust_command <- "lower2full"
    if (identical(match.call()$FUN, as.name("lav_matrix_vechr_reverse"))) {
      rust_command <- "vechr_reverse"
    } else if (identical(match.call()$FUN, as.name("lav_matrix_vechu_reverse"))) {
      rust_command <- "vechu_reverse"
    }

    rust_result <- try(
      switch(
        rust_command,
        vechr_reverse = lav_native_lav_matrix_vechr_reverse(x, diagonal),
        vechu_reverse = lav_native_lav_matrix_vechu_reverse(x, diagonal),
        lower2full = lav_native_lav_matrix_lower2full(x, diagonal)
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (diagonal) {
    p <- (sqrt(1 + 8 * length(x)) - 1)/2
  }
  else {
    p <- (sqrt(1 + 8 * length(x)) + 1)/2
  }
  stopifnot(p == round(p, 0))
  s <- numeric(p * p)
  s[lav_matrix_vechr_idx(p, diagonal = diagonal)] <- x
  s[lav_matrix_vechu_idx(p, diagonal = diagonal)] <- x
  attr(s, "dim") <- c(p, p)
  s
}

lav_matrix_delta_A_delta <- function(delta, a1) {
  if (lav_rust_kernel_enabled("matrix") && is.matrix(delta) && is.matrix(a1)) {
    rust_result <- try(lav_native_lav_matrix_delta_A_delta(delta, a1), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  crossprod(delta, a1) %*% delta
}

lav_model_vcov_delta_A_delta <- function(delta, a1) {
  if (
    lav_rust_kernel_enabled("model_vcov") &&
      is.matrix(delta) &&
      is.matrix(a1) &&
      is.numeric(delta) &&
      is.numeric(a1)
  ) {
    rust_result <- try(lav_native_lav_model_vcov_delta_A_delta(delta, a1), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  crossprod(delta, a1) %*% delta
}

lav_model_vcov_sandwich <- function(left, middle) {
  if (
    lav_rust_kernel_enabled("model_vcov") &&
      is.matrix(left) &&
      is.matrix(middle) &&
      is.numeric(left) &&
      is.numeric(middle)
  ) {
    rust_result <- try(lav_native_lav_model_vcov_sandwich(left, middle), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  left %*% middle %*% left
}

lav_model_vcov_jacobian_vcov_jacobian_t <- function(jac, VCOV) {
  if (
    lav_rust_kernel_enabled("model_vcov") &&
      is.matrix(jac) &&
      is.matrix(VCOV) &&
      is.numeric(jac) &&
      is.numeric(VCOV)
  ) {
    rust_result <- try(lav_native_lav_model_vcov_jacobian_vcov_jacobian_t(jac, VCOV),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  jac %*% VCOV %*% t(jac)
}

lav_matrix_diag_prepost <- function(A, d) {
  d <- as.vector(d)
  if (length(d) == 0L) {
    return(A)
  }

  if (
    lav_rust_kernel_enabled("matrix") &&
      is.matrix(A) &&
      is.numeric(A) &&
      nrow(A) == ncol(A) &&
      length(d) == nrow(A)
  ) {
    rust_result <- try(lav_native_lav_matrix_diag_prepost(A, d), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  A <- A * d
  t(t(A) * d)
}


