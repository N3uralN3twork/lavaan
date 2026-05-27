#include "lavaan_cpp/matrix.hpp"

#include <R.h>
#include <Rinternals.h>

#include <cmath>
#include <exception>
#include <limits>
#include <span>

namespace {

struct MatrixDimensions {
    std::size_t rows;
    std::size_t columns;
};

std::size_t checked_xlength_to_size(const R_xlen_t length)
{
    if (length < 0 ||
        static_cast<unsigned long long>(length) > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        Rf_error("matrix is too large for this platform");
    }

    return static_cast<std::size_t>(length);
}

int checked_size_to_r_int(const std::size_t value, const char* name)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        Rf_error("%s is too large for an R matrix dimension", name);
    }

    return static_cast<int>(value);
}

MatrixDimensions checked_numeric_matrix(SEXP matrix, const char* name)
{
    if (!Rf_isMatrix(matrix)) {
        Rf_error("%s must be a matrix", name);
    }
    if (TYPEOF(matrix) != REALSXP) {
        Rf_error("%s must be a numeric matrix", name);
    }

    SEXP dim = Rf_getAttrib(matrix, R_DimSymbol);
    const int rows = INTEGER(dim)[0];
    const int columns = INTEGER(dim)[1];
    if (rows < 0 || columns < 0) {
        Rf_error("%s dimensions must be non-negative", name);
    }

    return {
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(columns)
    };
}

std::size_t checked_symmetric_square_root_dimension(const std::size_t size)
{
    const auto root = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(size))));
    if (root * root != size) {
        Rf_error("matrix dimension must be a perfect square");
    }

    return root;
}

std::size_t checked_nstar(const std::size_t n)
{
    if (n == 0) {
        return 0;
    }
    if (n > (std::numeric_limits<std::size_t>::max() - 1) / n) {
        Rf_error("matrix dimension is too large");
    }

    return n * (n + 1) / 2;
}

std::size_t checked_offdiag_nstar(const std::size_t n)
{
    if (n <= 1) {
        return 0;
    }
    if (n > std::numeric_limits<std::size_t>::max() / (n - 1)) {
        Rf_error("matrix dimension is too large");
    }

    return n * (n - 1) / 2;
}

}  // namespace

extern "C" SEXP lav_cpp_vecr(SEXP a)
{
    if (!Rf_isMatrix(a)) {
        Rf_error("A must be a matrix");
    }
    if (TYPEOF(a) != REALSXP) {
        Rf_error("A must be a numeric matrix");
    }

    SEXP dim = Rf_getAttrib(a, R_DimSymbol);
    const int rows = INTEGER(dim)[0];
    const int columns = INTEGER(dim)[1];
    if (rows < 0 || columns < 0) {
        Rf_error("matrix dimensions must be non-negative");
    }

    const R_xlen_t length = XLENGTH(a);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, length));

    try {
        const std::size_t size = checked_xlength_to_size(length);
        const std::span<const double> input(REAL(a), size);
        std::span<double> output(REAL(out), size);

        lavaan::cpp::vecr(input,
                          output,
                          static_cast<std::size_t>(rows),
                          static_cast<std::size_t>(columns));
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_vecr");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_delta_A_delta(SEXP delta, SEXP a)
{
    const MatrixDimensions delta_dim = checked_numeric_matrix(delta, "delta");
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    if (delta_dim.rows != a_dim.rows) {
        Rf_error("nrow(delta) must equal nrow(A)");
    }

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(delta_dim.columns, "ncol(delta)"),
        checked_size_to_r_int(delta_dim.columns, "ncol(delta)")));

    try {
        const std::size_t delta_size = checked_xlength_to_size(XLENGTH(delta));
        const std::size_t a_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> delta_view(REAL(delta), delta_size);
        const std::span<const double> a_view(REAL(a), a_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::delta_a_delta(delta_view,
                                   a_view,
                                   output,
                                   delta_dim.rows,
                                   delta_dim.columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_delta_A_delta");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_pre(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nstar, "nstar"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_pre(input, output, n, a_dim.columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_pre");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.columns);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(nstar, "nstar")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_post(input, output, n, a_dim.rows);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_pre_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nstar, "nstar"),
        checked_size_to_r_int(nstar, "nstar")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_pre_post(input, output, n);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_pre_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_cor_pre_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);
    const std::size_t nstar = checked_offdiag_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nstar, "nstar"),
        checked_size_to_r_int(nstar, "nstar")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_cor_pre_post(input, output, n);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_cor_pre_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_ginv_pre(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nstar, "nstar"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_ginv_pre(input, output, n, a_dim.columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_ginv_pre");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_ginv_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.columns);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(nstar, "nstar")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_ginv_post(input, output, n, a_dim.rows);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_ginv_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_duplication_ginv_pre_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);
    const std::size_t nstar = checked_nstar(n);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nstar, "nstar"),
        checked_size_to_r_int(nstar, "nstar")));

    try {
        const std::size_t input_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> input(REAL(a), input_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::duplication_ginv_pre_post(input, output, n);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_duplication_ginv_pre_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_bdiag(SEXP matrices)
{
    if (!Rf_isVectorList(matrices)) {
        Rf_error("matrices must be a list");
    }

    Rf_error("lav_cpp_bdiag scaffold is not implemented");
}

extern "C" SEXP lav_cpp_crossprod_na(SEXP a, SEXP b)
{
    if (!Rf_isMatrix(a) || !Rf_isMatrix(b)) {
        Rf_error("a and b must be matrices");
    }
    if (TYPEOF(a) != REALSXP || TYPEOF(b) != REALSXP) {
        Rf_error("a and b must be numeric matrices");
    }

    SEXP a_dim = Rf_getAttrib(a, R_DimSymbol);
    SEXP b_dim = Rf_getAttrib(b, R_DimSymbol);
    const int rows = INTEGER(a_dim)[0];
    const int lhs_columns = INTEGER(a_dim)[1];
    const int rhs_rows = INTEGER(b_dim)[0];
    const int rhs_columns = INTEGER(b_dim)[1];
    if (rows < 0 || lhs_columns < 0 || rhs_rows < 0 || rhs_columns < 0) {
        Rf_error("matrix dimensions must be non-negative");
    }
    if (rows != rhs_rows) {
        Rf_error("a and b must have the same number of rows");
    }

    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, lhs_columns, rhs_columns));

    try {
        const std::size_t lhs_size = checked_xlength_to_size(XLENGTH(a));
        const std::size_t rhs_size = checked_xlength_to_size(XLENGTH(b));
        const std::size_t output_size = checked_xlength_to_size(XLENGTH(out));
        const std::span<const double> lhs(REAL(a), lhs_size);
        const std::span<const double> rhs(REAL(b), rhs_size);
        std::span<double> output(REAL(out), output_size);

        lavaan::cpp::crossprod_pairwise(lhs,
                                        rhs,
                                        output,
                                        static_cast<std::size_t>(rows),
                                        static_cast<std::size_t>(lhs_columns),
                                        static_cast<std::size_t>(rhs_columns));
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_crossprod_na");
    }

    SEXP a_dimnames = Rf_getAttrib(a, R_DimNamesSymbol);
    SEXP b_dimnames = Rf_getAttrib(b, R_DimNamesSymbol);
    SEXP row_names = R_NilValue;
    SEXP column_names = R_NilValue;
    if (a_dimnames != R_NilValue) {
        row_names = VECTOR_ELT(a_dimnames, 1);
    }
    if (b_dimnames != R_NilValue) {
        column_names = VECTOR_ELT(b_dimnames, 1);
    }
    if (row_names != R_NilValue || column_names != R_NilValue) {
        SEXP dimnames = PROTECT(Rf_allocVector(VECSXP, 2));
        SET_VECTOR_ELT(dimnames, 0, row_names);
        SET_VECTOR_ELT(dimnames, 1, column_names);
        Rf_setAttrib(out, R_DimNamesSymbol, dimnames);
        UNPROTECT(1);
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_diag_prepost(SEXP a, SEXP d)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    if (TYPEOF(d) != REALSXP || Rf_isMatrix(d)) {
        Rf_error("d must be a numeric vector");
    }

    const std::size_t d_size = checked_xlength_to_size(XLENGTH(d));
    if (d_size != a_dim.rows) {
        Rf_error("length(d) must equal nrow(A)");
    }

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::span<const double> input(
            REAL(a), checked_xlength_to_size(XLENGTH(a)));
        const std::span<const double> diagonal(REAL(d), d_size);
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));

        lavaan::cpp::diag_prepost(input, diagonal, output, a_dim.rows);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_diag_prepost");
    }

    UNPROTECT(1);
    return out;
}
