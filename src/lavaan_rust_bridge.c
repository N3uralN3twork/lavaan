#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

static void open_rust_backend(void);

static void get_matrix_dims(SEXP x, int *nrow, int *ncol, const char *arg) {
  open_rust_backend();
  if (!isReal(x) || !isMatrix(x)) {
    Rf_error("%s must be a numeric matrix", arg);
  }

  SEXP dims = getAttrib(x, R_DimSymbol);
  if (TYPEOF(dims) != INTSXP || LENGTH(dims) != 2) {
    Rf_error("%s must have matrix dimensions", arg);
  }

  *nrow = INTEGER(dims)[0];
  *ncol = INTEGER(dims)[1];
}

static void ensure_real_vector(SEXP x, const char *arg) {
  open_rust_backend();
  if (!isReal(x) || isMatrix(x)) {
    Rf_error("%s must be a numeric vector", arg);
  }
}

static void ensure_real_matrix(SEXP x, const char *arg) {
  open_rust_backend();
  if (!isReal(x) || !isMatrix(x)) {
    Rf_error("%s must be a numeric matrix", arg);
  }
}

static void open_rust_backend(void);

typedef bool (*rust_vec_fn)(const double *, size_t, double *);
typedef bool (*rust_vecr_fn)(const double *, size_t, size_t, double *);
typedef bool (*rust_diag_prepost_fn)(const double *, size_t, const double *, double *);
typedef bool (*rust_delta_a_delta_fn)(const double *, size_t, size_t, const double *, double *);
typedef size_t (*rust_idx_fn)(size_t, int *);
typedef size_t (*rust_tri_idx_fn)(size_t, bool, int *);
typedef bool (*rust_tri_extract_fn)(const double *, size_t, bool, double *);
typedef bool (*rust_tri_reverse_fn)(const double *, size_t, bool, double *);

static void *rust_handle = NULL;
static rust_vec_fn rust_lav_matrix_vec = NULL;
static rust_vecr_fn rust_lav_matrix_vecr = NULL;
static rust_diag_prepost_fn rust_lav_matrix_diag_prepost = NULL;
static rust_delta_a_delta_fn rust_lav_matrix_delta_a_delta = NULL;
static rust_idx_fn rust_lav_matrix_diagh_idx = NULL;
static rust_idx_fn rust_lav_matrix_antidiag_idx = NULL;
static rust_idx_fn rust_lav_matrix_diag_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vech_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vech_row_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vech_col_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vechr_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vechu_idx = NULL;
static rust_tri_idx_fn rust_lav_matrix_vechru_idx = NULL;
static rust_tri_extract_fn rust_lav_matrix_vech = NULL;
static rust_tri_extract_fn rust_lav_matrix_vechr = NULL;
static rust_tri_extract_fn rust_lav_matrix_vechu = NULL;
static rust_tri_extract_fn rust_lav_matrix_vechru = NULL;
static rust_tri_reverse_fn rust_lav_matrix_vech_reverse = NULL;
static rust_tri_reverse_fn rust_lav_matrix_vechru_reverse = NULL;
static rust_tri_reverse_fn rust_lav_matrix_upper2full = NULL;
static rust_tri_reverse_fn rust_lav_matrix_vechr_reverse = NULL;
static rust_tri_reverse_fn rust_lav_matrix_vechu_reverse = NULL;
static rust_tri_reverse_fn rust_lav_matrix_lower2full = NULL;

#define lavaan_lav_matrix_vec rust_lav_matrix_vec
#define lavaan_lav_matrix_vecr rust_lav_matrix_vecr
#define lavaan_lav_matrix_diag_prepost rust_lav_matrix_diag_prepost
#define lavaan_lav_matrix_delta_a_delta rust_lav_matrix_delta_a_delta
#define lavaan_lav_matrix_diagh_idx rust_lav_matrix_diagh_idx
#define lavaan_lav_matrix_antidiag_idx rust_lav_matrix_antidiag_idx
#define lavaan_lav_matrix_diag_idx rust_lav_matrix_diag_idx
#define lavaan_lav_matrix_vech_idx rust_lav_matrix_vech_idx
#define lavaan_lav_matrix_vech_row_idx rust_lav_matrix_vech_row_idx
#define lavaan_lav_matrix_vech_col_idx rust_lav_matrix_vech_col_idx
#define lavaan_lav_matrix_vechr_idx rust_lav_matrix_vechr_idx
#define lavaan_lav_matrix_vechu_idx rust_lav_matrix_vechu_idx
#define lavaan_lav_matrix_vechru_idx rust_lav_matrix_vechru_idx
#define lavaan_lav_matrix_vech rust_lav_matrix_vech
#define lavaan_lav_matrix_vechr rust_lav_matrix_vechr
#define lavaan_lav_matrix_vechu rust_lav_matrix_vechu
#define lavaan_lav_matrix_vechru rust_lav_matrix_vechru
#define lavaan_lav_matrix_vech_reverse rust_lav_matrix_vech_reverse
#define lavaan_lav_matrix_vechru_reverse rust_lav_matrix_vechru_reverse
#define lavaan_lav_matrix_upper2full rust_lav_matrix_upper2full
#define lavaan_lav_matrix_vechr_reverse rust_lav_matrix_vechr_reverse
#define lavaan_lav_matrix_vechu_reverse rust_lav_matrix_vechu_reverse
#define lavaan_lav_matrix_lower2full rust_lav_matrix_lower2full

static void *load_rust_symbol(const char *symbol) {
#ifdef _WIN32
  return (void *) GetProcAddress((HMODULE) rust_handle, symbol);
#else
  return NULL;
#endif
}

static void open_rust_backend(void) {
  if (rust_handle != NULL) {
    return;
  }

#ifndef _WIN32
  Rf_error("Rust backend loader is only implemented for Windows in this branch.");
#else
  const char *dll_path = getenv("LAVAAN_RUST_DLL");
  if (dll_path == NULL || dll_path[0] == '\0') {
    dll_path = "../../target/release/lavaan_kernels.dll";
  }

  rust_handle = (void *) LoadLibraryA(dll_path);
  if (rust_handle == NULL) {
    Rf_error("Unable to load Rust backend DLL at %s", dll_path);
  }

#define LOAD_RUST_SYM(name, type)                                                      \
  do {                                                                                 \
    rust_##name = (type) load_rust_symbol("lavaan_" #name);                            \
    if (rust_##name == NULL) {                                                         \
      Rf_error("Unable to resolve Rust symbol lavaan_%s", #name);                     \
    }                                                                                  \
  } while (0)

  LOAD_RUST_SYM(lav_matrix_vec, rust_vec_fn);
  LOAD_RUST_SYM(lav_matrix_vecr, rust_vecr_fn);
  LOAD_RUST_SYM(lav_matrix_diag_prepost, rust_diag_prepost_fn);
  LOAD_RUST_SYM(lav_matrix_delta_a_delta, rust_delta_a_delta_fn);
  LOAD_RUST_SYM(lav_matrix_diagh_idx, rust_idx_fn);
  LOAD_RUST_SYM(lav_matrix_antidiag_idx, rust_idx_fn);
  LOAD_RUST_SYM(lav_matrix_diag_idx, rust_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vech_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vech_row_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vech_col_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vechr_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vechu_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vechru_idx, rust_tri_idx_fn);
  LOAD_RUST_SYM(lav_matrix_vech, rust_tri_extract_fn);
  LOAD_RUST_SYM(lav_matrix_vechr, rust_tri_extract_fn);
  LOAD_RUST_SYM(lav_matrix_vechu, rust_tri_extract_fn);
  LOAD_RUST_SYM(lav_matrix_vechru, rust_tri_extract_fn);
  LOAD_RUST_SYM(lav_matrix_vech_reverse, rust_tri_reverse_fn);
  LOAD_RUST_SYM(lav_matrix_vechru_reverse, rust_tri_reverse_fn);
  LOAD_RUST_SYM(lav_matrix_upper2full, rust_tri_reverse_fn);
  LOAD_RUST_SYM(lav_matrix_vechr_reverse, rust_tri_reverse_fn);
  LOAD_RUST_SYM(lav_matrix_vechu_reverse, rust_tri_reverse_fn);
  LOAD_RUST_SYM(lav_matrix_lower2full, rust_tri_reverse_fn);

#undef LOAD_RUST_SYM
#endif
}

SEXP C_lav_matrix_vec(SEXP A) {
  open_rust_backend();
  if (!isReal(A)) {
    Rf_error("A must be a numeric matrix or vector");
  }
  SEXP out = PROTECT(allocVector(REALSXP, XLENGTH(A)));
  if (!lavaan_lav_matrix_vec(REAL(A), (size_t) XLENGTH(A), REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vec()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vecr(SEXP A) {
  int nrow = 0;
  int ncol = 0;
  get_matrix_dims(A, &nrow, &ncol, "A");
  SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t) nrow * (R_xlen_t) ncol));
  if (!lavaan_lav_matrix_vecr(REAL(A), (size_t) nrow, (size_t) ncol, REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vecr()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_diag_prepost(SEXP A, SEXP d) {
  int nrow = 0;
  int ncol = 0;
  get_matrix_dims(A, &nrow, &ncol, "A");
  ensure_real_vector(d, "d");

  if (nrow != ncol) {
    Rf_error("A must be a square matrix");
  }
  if (XLENGTH(d) != (R_xlen_t) nrow) {
    Rf_error("d must have the same length as the dimensions of A");
  }

  SEXP out = PROTECT(allocMatrix(REALSXP, nrow, ncol));
  if (!lavaan_lav_matrix_diag_prepost(
        REAL(A),
        (size_t) nrow,
        REAL(d),
        REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_diag_prepost()");
  }

  SEXP dimnames = PROTECT(getAttrib(A, R_DimNamesSymbol));
  if (dimnames != R_NilValue) {
    setAttrib(out, R_DimNamesSymbol, dimnames);
  }

  UNPROTECT(2);
  return out;
}

SEXP C_lav_matrix_delta_A_delta(SEXP delta, SEXP a1) {
  int delta_nrow = 0;
  int delta_ncol = 0;
  int a1_nrow = 0;
  int a1_ncol = 0;
  get_matrix_dims(delta, &delta_nrow, &delta_ncol, "delta");
  get_matrix_dims(a1, &a1_nrow, &a1_ncol, "a1");

  if (a1_nrow != a1_ncol) {
    Rf_error("a1 must be a square matrix");
  }
  if (a1_nrow != delta_nrow) {
    Rf_error("a1 must have the same number of rows as delta");
  }

  SEXP out = PROTECT(allocMatrix(REALSXP, delta_ncol, delta_ncol));
  if (!lavaan_lav_matrix_delta_a_delta(
        REAL(delta),
        (size_t) delta_nrow,
        (size_t) delta_ncol,
        REAL(a1),
        REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_delta_A_delta()");
  }

  SEXP dimnames = PROTECT(getAttrib(delta, R_DimNamesSymbol));
  if (dimnames != R_NilValue) {
    SEXP out_dimnames = PROTECT(allocVector(VECSXP, 2));
    SEXP delta_dimnames = dimnames;
    SET_VECTOR_ELT(out_dimnames, 0, VECTOR_ELT(delta_dimnames, 1));
    SET_VECTOR_ELT(out_dimnames, 1, VECTOR_ELT(delta_dimnames, 1));
    setAttrib(out, R_DimNamesSymbol, out_dimnames);
    UNPROTECT(2);
  } else {
    UNPROTECT(1);
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_diagh_idx(SEXP n) {
  int value = asInteger(n);
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  open_rust_backend();

  SEXP out = PROTECT(allocVector(INTSXP, value));
  size_t len = lavaan_lav_matrix_diagh_idx((size_t) value, INTEGER(out));
  for (size_t i = 0; i < len; i++) {
    INTEGER(out)[i] += 1;
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_antidiag_idx(SEXP n) {
  int value = asInteger(n);
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  open_rust_backend();

  SEXP out = PROTECT(allocVector(INTSXP, value));
  size_t len = lavaan_lav_matrix_antidiag_idx((size_t) value, INTEGER(out));
  for (size_t i = 0; i < len; i++) {
    INTEGER(out)[i] += 1;
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_diag_idx(SEXP n) {
  int value = asInteger(n);
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  open_rust_backend();

  SEXP out = PROTECT(allocVector(INTSXP, value));
  if (!lavaan_lav_matrix_diag_idx((size_t) value, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_diag_idx()");
  }
  UNPROTECT(1);
  return out;
}

static SEXP alloc_idx_result(size_t len) {
  open_rust_backend();
  if (len == 0) {
    return allocVector(INTSXP, 0);
  }
  return PROTECT(allocVector(INTSXP, (R_xlen_t) len));
}

SEXP C_lav_matrix_vech_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vech_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vech_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vech_row_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vech_row_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vech_row_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vech_col_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vech_col_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vech_col_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechr_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vechr_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechr_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechu_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vechu_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechu_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechru_idx(SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  if (value < 1) {
    return allocVector(INTSXP, 0);
  }
  size_t len = with_diagonal
    ? (size_t) value * ((size_t) value + 1) / 2
    : (size_t) value * ((size_t) value - 1) / 2;
  SEXP out = alloc_idx_result(len);
  if (len == 0) {
    return out;
  }
  if (!lavaan_lav_matrix_vechru_idx((size_t) value, with_diagonal, INTEGER(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechru_idx()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vech(SEXP input, SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  ensure_real_matrix(input, "input");
  if (XLENGTH(input) != (R_xlen_t) value * (R_xlen_t) value) {
    Rf_error("input must contain an n x n matrix in column-major order");
  }

  SEXP out = PROTECT(allocVector(REALSXP, with_diagonal
    ? (R_xlen_t) value * ((R_xlen_t) value + 1) / 2
    : (R_xlen_t) value * ((R_xlen_t) value - 1) / 2));
  if (!lavaan_lav_matrix_vech(REAL(input), (size_t) value, with_diagonal, REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vech()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechr(SEXP input, SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  ensure_real_matrix(input, "input");
  if (XLENGTH(input) != (R_xlen_t) value * (R_xlen_t) value) {
    Rf_error("input must contain an n x n matrix in column-major order");
  }

  SEXP out = PROTECT(allocVector(REALSXP, with_diagonal
    ? (R_xlen_t) value * ((R_xlen_t) value + 1) / 2
    : (R_xlen_t) value * ((R_xlen_t) value - 1) / 2));
  if (!lavaan_lav_matrix_vechr(REAL(input), (size_t) value, with_diagonal, REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechr()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechu(SEXP input, SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  ensure_real_matrix(input, "input");
  if (XLENGTH(input) != (R_xlen_t) value * (R_xlen_t) value) {
    Rf_error("input must contain an n x n matrix in column-major order");
  }

  SEXP out = PROTECT(allocVector(REALSXP, with_diagonal
    ? (R_xlen_t) value * ((R_xlen_t) value + 1) / 2
    : (R_xlen_t) value * ((R_xlen_t) value - 1) / 2));
  if (!lavaan_lav_matrix_vechu(REAL(input), (size_t) value, with_diagonal, REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechu()");
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vechru(SEXP input, SEXP n, SEXP diagonal) {
  int value = asInteger(n);
  bool with_diagonal = asLogical(diagonal) == TRUE;
  ensure_real_matrix(input, "input");
  if (XLENGTH(input) != (R_xlen_t) value * (R_xlen_t) value) {
    Rf_error("input must contain an n x n matrix in column-major order");
  }

  SEXP out = PROTECT(allocVector(REALSXP, with_diagonal
    ? (R_xlen_t) value * ((R_xlen_t) value + 1) / 2
    : (R_xlen_t) value * ((R_xlen_t) value - 1) / 2));
  if (!lavaan_lav_matrix_vechru(REAL(input), (size_t) value, with_diagonal, REAL(out))) {
    UNPROTECT(1);
    Rf_error("Rust backend failed in lav_matrix_vechru()");
  }
  UNPROTECT(1);
  return out;
}

static SEXP triangular_reverse_common(
    SEXP input,
    SEXP diagonal,
    bool (*fn)(const double *, size_t, bool, double *),
    const char *error_message) {
  open_rust_backend();
  ensure_real_vector(input, "input");
  bool with_diagonal = asLogical(diagonal) == TRUE;

  if (XLENGTH(input) == 0) {
    return allocMatrix(REALSXP, 0, 0);
  }

  double len = (double) XLENGTH(input);
  double n = with_diagonal
    ? (sqrt(1 + 8 * len) - 1.0) / 2.0
    : (sqrt(1 + 8 * len) + 1.0) / 2.0;
  int p = (int) (n + 0.5);
  if (p < 0 || (with_diagonal ? (p * (p + 1) / 2) : (p * (p - 1) / 2)) != (int) XLENGTH(input)) {
    Rf_error("input length does not match a triangular matrix");
  }

  SEXP out = PROTECT(allocMatrix(REALSXP, p, p));
  if (!fn(REAL(input), (size_t) XLENGTH(input), with_diagonal, REAL(out))) {
    UNPROTECT(1);
    Rf_error("%s", error_message);
  }
  UNPROTECT(1);
  return out;
}

SEXP C_lav_matrix_vech_reverse(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_vech_reverse,
    "Rust backend failed in lav_matrix_vech_reverse()");
}

SEXP C_lav_matrix_vechru_reverse(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_vechru_reverse,
    "Rust backend failed in lav_matrix_vechru_reverse()");
}

SEXP C_lav_matrix_upper2full(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_upper2full,
    "Rust backend failed in lav_matrix_upper2full()");
}

SEXP C_lav_matrix_vechr_reverse(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_vechr_reverse,
    "Rust backend failed in lav_matrix_vechr_reverse()");
}

SEXP C_lav_matrix_vechu_reverse(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_vechu_reverse,
    "Rust backend failed in lav_matrix_vechu_reverse()");
}

SEXP C_lav_matrix_lower2full(SEXP input, SEXP diagonal) {
  return triangular_reverse_common(
    input,
    diagonal,
    lavaan_lav_matrix_lower2full,
    "Rust backend failed in lav_matrix_lower2full()");
}
