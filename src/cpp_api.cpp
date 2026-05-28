#include "lavaan_cpp/matrix.hpp"

#include <R.h>
#include <Rinternals.h>

#include <cmath>
#include <exception>
#include <limits>
#include <span>
#include <vector>

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

MatrixDimensions checked_numeric_matrix_or_vector(SEXP value, const char* name)
{
    if (Rf_isMatrix(value)) {
        return checked_numeric_matrix(value, name);
    }
    if (TYPEOF(value) != REALSXP) {
        Rf_error("%s must contain numeric matrices or vectors", name);
    }
    const std::size_t length = checked_xlength_to_size(XLENGTH(value));
    return {length, 1};
}

std::size_t checked_size_add(const std::size_t lhs, const std::size_t rhs)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        Rf_error("matrix dimensions are too large");
    }

    return lhs + rhs;
}

std::size_t checked_size_multiply(const std::size_t lhs, const std::size_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        Rf_error("matrix dimensions are too large");
    }

    return lhs * rhs;
}

std::size_t checked_positive_size(SEXP value, const char* name)
{
    const int integer_value = Rf_asInteger(value);
    if (integer_value == NA_INTEGER || integer_value < 1) {
        Rf_error("%s must be a positive integer", name);
    }

    return static_cast<std::size_t>(integer_value);
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

extern "C" SEXP lav_cpp_commutation(SEXP m, SEXP n)
{
    const std::size_t rows = checked_positive_size(m, "m");
    const std::size_t columns = checked_positive_size(n, "n");
    const std::size_t size = checked_size_multiply(rows, columns);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(size, "m * n"),
        checked_size_to_r_int(size, "m * n")));

    try {
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::commutation_matrix(output, rows, columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_commutation");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_commutation_pre(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::span<const double> input(
            REAL(a), checked_xlength_to_size(XLENGTH(a)));
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::commutation_pre(input, output, n, a_dim.columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_commutation_pre");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_commutation_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.columns);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::span<const double> input(
            REAL(a), checked_xlength_to_size(XLENGTH(a)));
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::commutation_post(input, output, a_dim.rows, n);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_commutation_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_commutation_pre_post(SEXP a)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    if (a_dim.rows != a_dim.columns) {
        Rf_error("A must be square");
    }
    const std::size_t n = checked_symmetric_square_root_dimension(a_dim.rows);

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::span<const double> input(
            REAL(a), checked_xlength_to_size(XLENGTH(a)));
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::commutation_pre_post(input, output, n);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_commutation_pre_post");
    }

    UNPROTECT(1);
    return out;
}

extern "C" SEXP lav_cpp_commutation_mn_pre(SEXP a, SEXP m, SEXP n)
{
    const MatrixDimensions a_dim = checked_numeric_matrix(a, "A");
    const std::size_t rows = checked_positive_size(m, "m");
    const std::size_t columns = checked_positive_size(n, "n");
    if (a_dim.rows != checked_size_multiply(rows, columns)) {
        Rf_error("nrow(A) must equal m * n");
    }

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(a_dim.rows, "nrow(A)"),
        checked_size_to_r_int(a_dim.columns, "ncol(A)")));

    try {
        const std::span<const double> input(
            REAL(a), checked_xlength_to_size(XLENGTH(a)));
        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::commutation_mn_pre(
            input, output, rows, columns, a_dim.columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_commutation_mn_pre");
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

    const R_xlen_t matrix_count_xlen = XLENGTH(matrices);
    if (matrix_count_xlen == 0) {
        return Rf_allocMatrix(REALSXP, 0, 0);
    }
    if (static_cast<unsigned long long>(matrix_count_xlen) >
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        Rf_error("matrix list is too large");
    }

    const std::size_t matrix_count = static_cast<std::size_t>(matrix_count_xlen);
    std::vector<SEXP> matrix_objects;
    std::vector<MatrixDimensions> dimensions;
    std::vector<std::size_t> rows;
    std::vector<std::size_t> columns;
    std::vector<std::span<const double>> inputs;
    matrix_objects.reserve(matrix_count);
    dimensions.reserve(matrix_count);
    rows.reserve(matrix_count);
    columns.reserve(matrix_count);
    inputs.reserve(matrix_count);

    std::size_t total_rows = 0;
    std::size_t total_columns = 0;
    for (std::size_t index = 0; index < matrix_count; ++index) {
        SEXP matrix = VECTOR_ELT(matrices, static_cast<R_xlen_t>(index));
        const MatrixDimensions dim =
            checked_numeric_matrix_or_vector(matrix, "matrices");
        const std::size_t storage_size = checked_xlength_to_size(XLENGTH(matrix));
        if (storage_size != checked_size_multiply(dim.rows, dim.columns)) {
            Rf_error("matrix storage size does not match dimensions");
        }

        matrix_objects.push_back(matrix);
        dimensions.push_back(dim);
        rows.push_back(dim.rows);
        columns.push_back(dim.columns);
        total_rows = checked_size_add(total_rows, dim.rows);
        total_columns = checked_size_add(total_columns, dim.columns);
    }

    SEXP out = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(total_rows, "nrow(output)"),
        checked_size_to_r_int(total_columns, "ncol(output)")));

    try {
        for (std::size_t index = 0; index < matrix_count; ++index) {
            inputs.push_back(std::span<const double>(
                REAL(matrix_objects[index]),
                checked_xlength_to_size(XLENGTH(matrix_objects[index]))));
        }

        std::span<double> output(
            REAL(out), checked_xlength_to_size(XLENGTH(out)));
        lavaan::cpp::block_diagonal(inputs,
                                    output,
                                    rows,
                                    columns,
                                    total_rows,
                                    total_columns);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_bdiag");
    }

    UNPROTECT(1);
    return out;
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

extern "C" SEXP lav_cpp_lisrel_sigma_fast(SEXP lambda,
                                           SEXP psi,
                                           SEXP theta,
                                           SEXP delta)
{
    const MatrixDimensions lambda_dim =
        checked_numeric_matrix(lambda, "lambda");
    const MatrixDimensions psi_dim = checked_numeric_matrix(psi, "psi");
    const MatrixDimensions theta_dim = checked_numeric_matrix(theta, "theta");

    const std::size_t nvar = lambda_dim.rows;
    const std::size_t nfac = lambda_dim.columns;
    if (psi_dim.rows != nfac || psi_dim.columns != nfac) {
        Rf_error("psi dimensions must match ncol(lambda)");
    }
    if (theta_dim.rows != nvar || theta_dim.columns != nvar) {
        Rf_error("theta dimensions must match nrow(lambda)");
    }

    const double* delta_data = nullptr;
    if (delta != R_NilValue) {
        if (TYPEOF(delta) != REALSXP || Rf_isMatrix(delta)) {
            Rf_error("delta must be a numeric vector or NULL");
        }
        const std::size_t delta_size = checked_xlength_to_size(XLENGTH(delta));
        if (delta_size != nvar) {
            Rf_error("length(delta) must equal nrow(lambda)");
        }
        delta_data = REAL(delta);
    }

    SEXP sigma = PROTECT(Rf_allocMatrix(
        REALSXP,
        checked_size_to_r_int(nvar, "nrow(sigma)"),
        checked_size_to_r_int(nvar, "ncol(sigma)")));

    try {
        const std::span<const double> lambda_span(
            REAL(lambda), checked_xlength_to_size(XLENGTH(lambda)));
        const std::span<const double> psi_span(
            REAL(psi), checked_xlength_to_size(XLENGTH(psi)));
        const std::span<const double> theta_span(
            REAL(theta), checked_xlength_to_size(XLENGTH(theta)));
        const std::span<const double> delta_span(
            delta_data, delta_data == nullptr ? 0 : nvar);
        std::span<double> output(
            REAL(sigma), checked_xlength_to_size(XLENGTH(sigma)));
        lavaan::cpp::lisrel_sigma(lambda_span,
                                  psi_span,
                                  theta_span,
                                  delta_span,
                                  output,
                                  nvar,
                                  nfac);
    } catch (const std::exception& ex) {
        UNPROTECT(1);
        Rf_error("%s", ex.what());
    } catch (...) {
        UNPROTECT(1);
        Rf_error("unknown C++ error in lav_cpp_lisrel_sigma_fast");
    }

    UNPROTECT(1);
    return sigma;
}
