#include <R_ext/Rdynload.h>
#include <Rinternals.h>

extern SEXP C_lav_matrix_vec(SEXP A);
extern SEXP C_lav_matrix_vecr(SEXP A);
extern SEXP C_lav_matrix_diag_prepost(SEXP A, SEXP d);
extern SEXP C_lav_matrix_delta_A_delta(SEXP delta, SEXP a1);
extern SEXP C_lav_matrix_diagh_idx(SEXP n);
extern SEXP C_lav_matrix_antidiag_idx(SEXP n);
extern SEXP C_lav_matrix_diag_idx(SEXP n);
extern SEXP C_lav_matrix_vech_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vech_row_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vech_col_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechr_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechu_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechru_idx(SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vech(SEXP input, SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechr(SEXP input, SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechu(SEXP input, SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vechru(SEXP input, SEXP n, SEXP diagonal);
extern SEXP C_lav_matrix_vech_reverse(SEXP input, SEXP diagonal);
extern SEXP C_lav_matrix_vechru_reverse(SEXP input, SEXP diagonal);
extern SEXP C_lav_matrix_upper2full(SEXP input, SEXP diagonal);
extern SEXP C_lav_matrix_vechr_reverse(SEXP input, SEXP diagonal);
extern SEXP C_lav_matrix_vechu_reverse(SEXP input, SEXP diagonal);
extern SEXP C_lav_matrix_lower2full(SEXP input, SEXP diagonal);

static const R_CallMethodDef CallEntries[] = {
    {"C_lav_matrix_vec", (DL_FUNC) &C_lav_matrix_vec, 1},
    {"C_lav_matrix_vecr", (DL_FUNC) &C_lav_matrix_vecr, 1},
    {"C_lav_matrix_diag_prepost", (DL_FUNC) &C_lav_matrix_diag_prepost, 2},
    {"C_lav_matrix_delta_A_delta", (DL_FUNC) &C_lav_matrix_delta_A_delta, 2},
    {"C_lav_matrix_diagh_idx", (DL_FUNC) &C_lav_matrix_diagh_idx, 1},
    {"C_lav_matrix_antidiag_idx", (DL_FUNC) &C_lav_matrix_antidiag_idx, 1},
    {"C_lav_matrix_diag_idx", (DL_FUNC) &C_lav_matrix_diag_idx, 1},
    {"C_lav_matrix_vech_idx", (DL_FUNC) &C_lav_matrix_vech_idx, 2},
    {"C_lav_matrix_vech_row_idx", (DL_FUNC) &C_lav_matrix_vech_row_idx, 2},
    {"C_lav_matrix_vech_col_idx", (DL_FUNC) &C_lav_matrix_vech_col_idx, 2},
    {"C_lav_matrix_vechr_idx", (DL_FUNC) &C_lav_matrix_vechr_idx, 2},
    {"C_lav_matrix_vechu_idx", (DL_FUNC) &C_lav_matrix_vechu_idx, 2},
    {"C_lav_matrix_vechru_idx", (DL_FUNC) &C_lav_matrix_vechru_idx, 2},
    {"C_lav_matrix_vech", (DL_FUNC) &C_lav_matrix_vech, 3},
    {"C_lav_matrix_vechr", (DL_FUNC) &C_lav_matrix_vechr, 3},
    {"C_lav_matrix_vechu", (DL_FUNC) &C_lav_matrix_vechu, 3},
    {"C_lav_matrix_vechru", (DL_FUNC) &C_lav_matrix_vechru, 3},
    {"C_lav_matrix_vech_reverse", (DL_FUNC) &C_lav_matrix_vech_reverse, 2},
    {"C_lav_matrix_vechru_reverse", (DL_FUNC) &C_lav_matrix_vechru_reverse, 2},
    {"C_lav_matrix_upper2full", (DL_FUNC) &C_lav_matrix_upper2full, 2},
    {"C_lav_matrix_vechr_reverse", (DL_FUNC) &C_lav_matrix_vechr_reverse, 2},
    {"C_lav_matrix_vechu_reverse", (DL_FUNC) &C_lav_matrix_vechu_reverse, 2},
    {"C_lav_matrix_lower2full", (DL_FUNC) &C_lav_matrix_lower2full, 2},
    {NULL, NULL, 0}
};

void R_init_lavaan(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
