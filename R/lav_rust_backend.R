# Internal Rust bridge helpers.
#
# These functions keep `lavaan/R` as the reference path while the Rust kernels
# under `lavaan/rust/lavaan-kernels/` are being validated. They call the Rust
# CLI directly, so the exported R API stays unchanged.
#
# Backend selection:
# - default: Rust
# - override for benchmarks: options(lavaan.backend = "r") or
#   Sys.setenv(LAVAAN_BACKEND = "r")

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

lav_rust_find_repo_root <- function() {
  override <- getOption("lavaan.rust.repo_root", default = NULL)
  if (!is.null(override)) {
    root <- normalizePath(override, winslash = "/", mustWork = TRUE)
    return(root)
  }

  current <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  repeat {
    candidate <- file.path(current, "lavaan", "rust", "lavaan-kernels", "Cargo.toml")
    if (file.exists(candidate)) {
      return(current)
    }

    parent <- dirname(current)
    if (identical(parent, current)) {
      break
    }
    current <- parent
  }

  stop(
    "Unable to locate the lavaan Rust workspace. ",
    "Set options(lavaan.rust.repo_root = '<repo-root>') or run from the repo."
  )
}

lav_rust_manifest_path <- function() {
  file.path(lav_rust_find_repo_root(), "lavaan", "rust", "lavaan-kernels", "Cargo.toml")
}

lav_rust_cargo_exe <- function() {
  cargo <- Sys.which("cargo")
  if (!nzchar(cargo)) {
    stop("Rust backend is unavailable because `cargo` is not on PATH.")
  }
  cargo
}

lav_rust_binary_exe <- function() {
  override <- getOption("lavaan.rust.binary", default = NULL)
  if (is.null(override) || !nzchar(override)) {
    return("")
  }

  binary <- normalizePath(override, winslash = "/", mustWork = TRUE)
  binary
}

lav_rust_backend_available <- function() {
  if (identical(lav_backend_mode(), "r")) {
    return(FALSE)
  }

  isTRUE(
    tryCatch(
      {
        binary <- lav_rust_binary_exe()
        if (nzchar(binary)) {
          return(TRUE)
        }

        manifest <- lav_rust_manifest_path()
        nzchar(Sys.which("cargo")) && file.exists(manifest)
      },
      error = function(e) FALSE
    )
  )
}

lav_rust_run_lav_matrix <- function(command, input) {
  binary <- lav_rust_binary_exe()

  if (nzchar(binary)) {
    cmd <- binary
    args <- command
  } else {
    repo_root <- lav_rust_find_repo_root()
    old_wd <- getwd()
    on.exit(setwd(old_wd), add = TRUE)
    setwd(repo_root)

    manifest <- file.path("lavaan", "rust", "lavaan-kernels", "Cargo.toml")
    cargo <- lav_rust_cargo_exe()
    cmd <- cargo
    args <- c("run", "--manifest-path", manifest, "--quiet", "--bin", "lav_matrix", "--", command)
  }

  output <- system2(cmd, args = args, input = input, stdout = TRUE, stderr = TRUE)

  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(output, collapse = "\n"))
  }

  scan(text = paste(output, collapse = " "), quiet = TRUE)
}

lav_rust_lav_matrix_diag_prepost <- function(A, d) {
  A <- unname(as.matrix(A))
  d <- as.vector(d)

  if (length(d) == 0L) {
    return(A)
  }

  values <- lav_rust_run_lav_matrix(
    "diag_prepost",
    paste(c(nrow(A), as.vector(A), d), collapse = " ")
  )

  dim_n <- as.integer(values[1])
  data <- values[-1]
  out <- matrix(data, nrow = dim_n, ncol = dim_n)
  dimnames(out) <- dimnames(A)
  out
}

lav_rust_lav_matrix_delta_A_delta <- function(delta, a1) {
  delta <- unname(as.matrix(delta))
  a1 <- unname(as.matrix(a1))

  values <- lav_rust_run_lav_matrix(
    "delta_a_delta",
    paste(c(nrow(delta), ncol(delta), as.vector(delta), as.vector(a1)), collapse = " ")
  )

  dim_nrow <- as.integer(values[1])
  dim_ncol <- as.integer(values[2])
  data <- values[-c(1L, 2L)]
  out <- matrix(data, nrow = dim_nrow, ncol = dim_ncol)
  dn <- dimnames(delta)
  if (!is.null(dn)) {
    dimnames(out) <- list(dn[[2]], dn[[2]])
  }
  out
}

lav_rust_lav_matrix_diagh_idx <- function(n) {
  values <- lav_rust_run_lav_matrix("diagh_idx", as.character(as.integer(n)))
  if (length(values) == 0L) {
    return(integer(0L))
  }

  out_len <- as.integer(values[1])
  as.integer(values[-1][seq_len(out_len)])
}

lav_rust_lav_matrix_antidiag_idx <- function(n) {
  values <- lav_rust_run_lav_matrix("antidiag_idx", as.character(as.integer(n)))
  if (length(values) == 0L) {
    return(integer(0L))
  }

  out_len <- as.integer(values[1])
  as.integer(values[-1][seq_len(out_len)])
}

lav_matrix_diagh_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_backend_available() && diagonal && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_diagh_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result + 1L)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_antidiag_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result + 1L)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  1L + seq_len(n) * (n - 1L)
}

lav_rust_decode_vector <- function(values, as_integer = FALSE) {
  if (length(values) == 0L) {
    return(if (as_integer) integer(0L) else numeric(0L))
  }

  out_len <- as.integer(values[1])
  out <- values[-1][seq_len(out_len)]
  if (as_integer) {
    as.integer(out)
  } else {
    as.numeric(out)
  }
}

lav_rust_decode_matrix <- function(values, dimnames = NULL) {
  if (length(values) < 2L) {
    return(matrix(numeric(0L), 0L, 0L))
  }

  nrow <- as.integer(values[1])
  ncol <- as.integer(values[2])
  data <- values[-c(1L, 2L)]
  out <- matrix(data, nrow = nrow, ncol = ncol)
  if (!is.null(dimnames)) {
    dimnames(out) <- dimnames
  }
  out
}

lav_matrix_vec <- function(A) {
  if (lav_rust_backend_available() && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix("vec", paste(c(length(A), as.vector(A)), collapse = " "))
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  as.vector(A)
}

lav_matrix_vecr <- function(A) {
  if (lav_rust_backend_available() && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vecr",
          paste(c(nrow(A), ncol(A), as.vector(A)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  lav_matrix_vec(t(A))
}

lav_matrix_diag_idx <- function(n = 1L) {
  n <- as.integer(n)
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_diag_idx(n), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  if (n < 1L) {
    return(integer(0L))
  }
  1L + (seq_len(n) - 1L) * (n + 1L)
}

lav_matrix_vech_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_row_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_col_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechr_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechu_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechru_idx(n, diagonal), silent = TRUE)
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

lav_native_backend_available <- function() {
  dll_path <- tryCatch(lav_rust_dll_path(), error = function(e) "")
  nzchar(dll_path) && file.exists(dll_path)
}

lav_rust_dll_path <- function() {
  dll_name <- if (.Platform$OS.type == "windows") {
    "lavaan_kernels.dll"
  } else if (Sys.info()[["sysname"]] == "Darwin") {
    "liblavaan_kernels.dylib"
  } else {
    "liblavaan_kernels.so"
  }
  file.path(lav_rust_find_repo_root(), "target", "release", dll_name)
}

lav_rust_prepare_backend <- function() {
  dll_path <- tryCatch(lav_rust_dll_path(), error = function(e) "")
  if (nzchar(dll_path) && file.exists(dll_path)) {
    Sys.setenv(LAVAAN_RUST_DLL = normalizePath(dll_path, winslash = "/", mustWork = TRUE))
  }
  invisible(dll_path)
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

lav_native_lav_matrix_vec <- function(A) {
  .Call("C_lav_matrix_vec", A, PACKAGE = "lavaan")
}

lav_native_lav_matrix_vecr <- function(A) {
  .Call("C_lav_matrix_vecr", A, PACKAGE = "lavaan")
}

lav_native_lav_matrix_diag_prepost <- function(A, d) {
  .Call("C_lav_matrix_diag_prepost", A, d, PACKAGE = "lavaan")
}

lav_native_lav_matrix_delta_A_delta <- function(delta, a1) {
  .Call("C_lav_matrix_delta_A_delta", delta, a1, PACKAGE = "lavaan")
}

lav_native_lav_matrix_diagh_idx <- function(n) {
  .Call("C_lav_matrix_diagh_idx", as.integer(n), PACKAGE = "lavaan")
}

lav_native_lav_matrix_antidiag_idx <- function(n) {
  .Call("C_lav_matrix_antidiag_idx", as.integer(n), PACKAGE = "lavaan")
}

lav_native_lav_matrix_diag_idx <- function(n) {
  .Call("C_lav_matrix_diag_idx", as.integer(n), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vech_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vech_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vech_row_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vech_row_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vech_col_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vech_col_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechr_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vechr_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechu_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vechu_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechru_idx <- function(n, diagonal = TRUE) {
  .Call("C_lav_matrix_vechru_idx", as.integer(n), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vech <- function(S, diagonal = TRUE) {
  .Call("C_lav_matrix_vech", S, as.integer(NCOL(S)), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechr <- function(S, diagonal = TRUE) {
  .Call("C_lav_matrix_vechr", S, as.integer(NCOL(S)), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechu <- function(S, diagonal = TRUE) {
  .Call("C_lav_matrix_vechu", S, as.integer(NCOL(S)), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechru <- function(S, diagonal = TRUE) {
  .Call("C_lav_matrix_vechru", S, as.integer(NCOL(S)), as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vech_reverse <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_vech_reverse", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechru_reverse <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_vechru_reverse", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_upper2full <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_upper2full", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechr_reverse <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_vechr_reverse", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_vechu_reverse <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_vechu_reverse", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_native_lav_matrix_lower2full <- function(x, diagonal = TRUE) {
  .Call("C_lav_matrix_lower2full", x, as.logical(diagonal), PACKAGE = "lavaan")
}

lav_matrix_vec <- function(A) {
  if (lav_rust_backend_available() && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(lav_native_lav_matrix_vec(A), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  as.vector(A)
}

lav_matrix_vecr <- function(A) {
  if (lav_rust_backend_available() && is.matrix(A) && is.numeric(A)) {
    rust_result <- try(lav_native_lav_matrix_vecr(A), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  lav_matrix_vec(t(A))
}

lav_matrix_diag_idx <- function(n = 1L) {
  n <- as.integer(n)
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && diagonal && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && n >= 1L) {
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
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
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
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechr(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechr_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechu <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechu(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechu_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechru <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(lav_native_lav_matrix_vechru(S, diagonal), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechru_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vech_reverse <- lav_matrix_vechru_reverse <- lav_matrix_upper2full <- function(x, 
    diagonal = TRUE) {
  if (lav_rust_backend_available() && is.numeric(x)) {
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
  if (lav_rust_backend_available() && is.numeric(x)) {
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
  if (lav_rust_backend_available() && is.matrix(delta) && is.matrix(a1)) {
    rust_result <- try(lav_native_lav_matrix_delta_A_delta(delta, a1), silent = TRUE)
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  crossprod(delta, a1) %*% delta
}

lav_matrix_diag_prepost <- function(A, d) {
  d <- as.vector(d)
  if (length(d) == 0L) {
    return(A)
  }

  if (
    lav_rust_backend_available() &&
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

lav_matrix_vech <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vech",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  row_1 <- row(S)
  col_1 <- col(S)
  if (diagonal) S[row_1 >= col_1] else S[row_1 > col_1]
}

lav_matrix_vechr <- function(S, diagonal = TRUE) {      # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechr",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechr_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechu <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechu",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechu_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechru <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechru",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechru_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vech_reverse <- lav_matrix_vechru_reverse <- lav_matrix_upper2full <- function(x, 
    diagonal = TRUE) {
  if (lav_rust_backend_available() && is.numeric(x)) {
    rust_command <- "upper2full"
    if (identical(match.call()$FUN, as.name("lav_matrix_vech_reverse"))) {
      rust_command <- "vech_reverse"
    } else if (identical(match.call()$FUN, as.name("lav_matrix_vechru_reverse"))) {
      rust_command <- "vechru_reverse"
    }

    rust_result <- try(
      lav_rust_decode_matrix(
        lav_rust_run_lav_matrix(
          rust_command,
          paste(c(as.integer(diagonal), x), collapse = " ")
        )
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
  if (lav_rust_backend_available() && is.numeric(x)) {
    rust_command <- "lower2full"
    if (identical(match.call()$FUN, as.name("lav_matrix_vechr_reverse"))) {
      rust_command <- "vechr_reverse"
    } else if (identical(match.call()$FUN, as.name("lav_matrix_vechu_reverse"))) {
      rust_command <- "vechu_reverse"
    }

    rust_result <- try(
      lav_rust_decode_matrix(
        lav_rust_run_lav_matrix(
          rust_command,
          paste(c(as.integer(diagonal), x), collapse = " ")
        )
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

lav_matrix_vech <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vech",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  row_1 <- row(S)
  col_1 <- col(S)
  if (diagonal) S[row_1 >= col_1] else S[row_1 > col_1]
}

lav_matrix_vechr <- function(S, diagonal = TRUE) {      # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechr",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechr_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechu <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechu",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechu_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vechru <- function(S, diagonal = TRUE) {     # nolint
  if (lav_rust_backend_available() && is.matrix(S) && is.numeric(S)) {
    rust_result <- try(
      lav_rust_decode_vector(
        lav_rust_run_lav_matrix(
          "vechru",
          paste(c(nrow(S), as.vector(S), as.integer(diagonal)), collapse = " ")
        )
      ),
      silent = TRUE
    )
    if (!inherits(rust_result, "try-error")) {
      return(rust_result)
    }
  }

  S[lav_matrix_vechru_idx(n = NCOL(S), diagonal = diagonal)]
}

lav_matrix_vech_idx <- function(n = 1L, diagonal = TRUE) {
  n <- as.integer(n)
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_row_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vech_col_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechr_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechu_idx(n, diagonal), silent = TRUE)
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
  if (lav_rust_backend_available() && n >= 1L) {
    rust_result <- try(lav_rust_lav_matrix_vechru_idx(n, diagonal), silent = TRUE)
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
