# Private wrappers for package-native C/C++ routines.

lav_cpp_vecr <- function(A) {
  .Call(C_lav_cpp_vecr, A)
}

lav_cpp_delta_A_delta <- function(delta, a1) {
  .Call(C_lav_cpp_delta_A_delta, delta, a1)
}

lav_cpp_duplication_pre <- function(A) {
  .Call(C_lav_cpp_duplication_pre, A)
}

lav_cpp_duplication_post <- function(A) {
  .Call(C_lav_cpp_duplication_post, A)
}

lav_cpp_duplication_pre_post <- function(A) {
  .Call(C_lav_cpp_duplication_pre_post, A)
}

lav_cpp_duplication_cor_pre_post <- function(A) {
  .Call(C_lav_cpp_duplication_cor_pre_post, A)
}

lav_cpp_duplication_ginv_pre <- function(A) {
  .Call(C_lav_cpp_duplication_ginv_pre, A)
}

lav_cpp_duplication_ginv_post <- function(A) {
  .Call(C_lav_cpp_duplication_ginv_post, A)
}

lav_cpp_duplication_ginv_pre_post <- function(A) {
  .Call(C_lav_cpp_duplication_ginv_pre_post, A)
}

lav_cpp_bdiag <- function(...) {
  if (nargs() == 0L) {
    return(matrix(0, 0, 0))
  }
  dots <- list(...)
  if (is.list(dots[[1L]])) {
    mlist <- dots[[1L]]
  } else {
    mlist <- dots
  }
  .Call(C_lav_cpp_bdiag, mlist)
}

lav_cpp_crossprod_na <- function(a, m_b) {
  .Call(C_lav_cpp_crossprod_na, a, m_b)
}

lav_cpp_diag_prepost <- function(A, d) {
  .Call(C_lav_cpp_diag_prepost, A, d)
}
