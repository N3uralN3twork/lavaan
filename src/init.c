#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern SEXP lav_cpp_vecr(SEXP a);
extern SEXP lav_cpp_crossprod_na(SEXP a, SEXP b);
extern SEXP lav_cpp_commutation(SEXP m, SEXP n);
extern SEXP lav_cpp_commutation_pre(SEXP a);
extern SEXP lav_cpp_commutation_post(SEXP a);
extern SEXP lav_cpp_commutation_pre_post(SEXP a);
extern SEXP lav_cpp_commutation_mn_pre(SEXP a, SEXP m, SEXP n);
extern SEXP lav_cpp_delta_A_delta(SEXP delta, SEXP a);
extern SEXP lav_cpp_duplication_pre(SEXP a);
extern SEXP lav_cpp_duplication_post(SEXP a);
extern SEXP lav_cpp_duplication_pre_post(SEXP a);
extern SEXP lav_cpp_duplication_cor_pre_post(SEXP a);
extern SEXP lav_cpp_duplication_ginv_pre(SEXP a);
extern SEXP lav_cpp_duplication_ginv_post(SEXP a);
extern SEXP lav_cpp_duplication_ginv_pre_post(SEXP a);
extern SEXP lav_cpp_bdiag(SEXP matrices);
extern SEXP lav_cpp_diag_prepost(SEXP a, SEXP d);
extern SEXP lav_cpp_lisrel_sigma_fast(SEXP lambda, SEXP psi, SEXP theta,
                                      SEXP delta);

static const R_CallMethodDef CallEntries[] = {
    {"lav_cpp_vecr", (DL_FUNC) &lav_cpp_vecr, 1},
    {"lav_cpp_crossprod_na", (DL_FUNC) &lav_cpp_crossprod_na, 2},
    {"lav_cpp_commutation", (DL_FUNC) &lav_cpp_commutation, 2},
    {"lav_cpp_commutation_pre", (DL_FUNC) &lav_cpp_commutation_pre, 1},
    {"lav_cpp_commutation_post", (DL_FUNC) &lav_cpp_commutation_post, 1},
    {"lav_cpp_commutation_pre_post", (DL_FUNC) &lav_cpp_commutation_pre_post, 1},
    {"lav_cpp_commutation_mn_pre", (DL_FUNC) &lav_cpp_commutation_mn_pre, 3},
    {"lav_cpp_delta_A_delta", (DL_FUNC) &lav_cpp_delta_A_delta, 2},
    {"lav_cpp_duplication_pre", (DL_FUNC) &lav_cpp_duplication_pre, 1},
    {"lav_cpp_duplication_post", (DL_FUNC) &lav_cpp_duplication_post, 1},
    {"lav_cpp_duplication_pre_post", (DL_FUNC) &lav_cpp_duplication_pre_post, 1},
    {"lav_cpp_duplication_cor_pre_post", (DL_FUNC) &lav_cpp_duplication_cor_pre_post, 1},
    {"lav_cpp_duplication_ginv_pre", (DL_FUNC) &lav_cpp_duplication_ginv_pre, 1},
    {"lav_cpp_duplication_ginv_post", (DL_FUNC) &lav_cpp_duplication_ginv_post, 1},
    {"lav_cpp_duplication_ginv_pre_post", (DL_FUNC) &lav_cpp_duplication_ginv_pre_post, 1},
    {"lav_cpp_bdiag", (DL_FUNC) &lav_cpp_bdiag, 1},
    {"lav_cpp_diag_prepost", (DL_FUNC) &lav_cpp_diag_prepost, 2},
    {"lav_cpp_lisrel_sigma_fast", (DL_FUNC) &lav_cpp_lisrel_sigma_fast, 4},
    {NULL, NULL, 0}
};

void R_init_lavaan(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
