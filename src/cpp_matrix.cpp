#include "lavaan_cpp/matrix.hpp"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lavaan::cpp {

namespace {

using ColumnMajorMatrixView =
    utils::detail::md::mdspan<double,
                                      utils::detail::MatrixExtents,
                                      utils::detail::md::layout_left>;
using ConstColumnMajorMatrixView =
    utils::detail::md::mdspan<const double,
                                      utils::detail::MatrixExtents,
                                      utils::detail::md::layout_left>;

std::size_t checked_matrix_size(const std::size_t rows, const std::size_t columns)
{
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::length_error("matrix dimensions are too large");
    }

    return rows * columns;
}

bool copy_without_nan(std::span<const double> input, std::span<double> output)
{
    bool has_infinite = false;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const double value = input[index];
        if (std::isnan(value)) {
            output[index] = 0.0;
        } else {
            output[index] = value;
            has_infinite = has_infinite || std::isinf(value);
        }
    }

    return has_infinite;
}

void crossprod_linalg(std::span<const double> lhs,
                      std::span<const double> rhs,
                      std::span<double> output,
                      const std::size_t rows,
                      const std::size_t lhs_columns,
                      const std::size_t rhs_columns)
{
    const ConstColumnMajorMatrixView lhs_view(lhs.data(), rows, lhs_columns);
    const ConstColumnMajorMatrixView rhs_view(rhs.data(), rows, rhs_columns);
    ColumnMajorMatrixView output_view(output.data(), lhs_columns, rhs_columns);

    utils::detail::linalg::matrix_product(
        utils::detail::linalg::transposed(lhs_view),
        rhs_view,
        output_view);
}

void copy_valid_pairwise_terms(std::span<const double> lhs,
                               std::span<const double> rhs,
                               std::span<double> lhs_work,
                               std::span<double> rhs_work,
                               const std::size_t rows,
                               const std::size_t lhs_column,
                               const std::size_t rhs_column)
{
    for (std::size_t row = 0; row < rows; ++row) {
        const double lhs_value = lhs[row + (lhs_column * rows)];
        const double rhs_value = rhs[row + (rhs_column * rows)];

        if (std::isnan(lhs_value * rhs_value)) {
            lhs_work[row] = 0.0;
            rhs_work[row] = 0.0;
        } else {
            lhs_work[row] = lhs_value;
            rhs_work[row] = rhs_value;
        }
    }
}

struct SymmetricMatrixPair {
    std::size_t lower_index;
    std::size_t upper_index;
    bool diagonal;
};

std::size_t checked_triangular_size(const std::size_t n, const bool diagonal)
{
    if (diagonal) {
        if (n > std::numeric_limits<std::size_t>::max() - 1) {
            throw std::length_error("matrix dimensions are too large");
        }
        const std::size_t factor = n + 1;
        if (factor != 0 && n > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::length_error("matrix dimensions are too large");
        }
        return (n * factor) / 2;
    }

    if (n <= 1) {
        return 0;
    }
    const std::size_t factor = n - 1;
    if (factor != 0 && n > std::numeric_limits<std::size_t>::max() / factor) {
        throw std::length_error("matrix dimensions are too large");
    }
    return (n * factor) / 2;
}

std::vector<SymmetricMatrixPair> symmetric_matrix_pairs(const std::size_t n,
                                                        const bool diagonal)
{
    std::vector<SymmetricMatrixPair> pairs;
    pairs.reserve(checked_triangular_size(n, diagonal));

    for (std::size_t column = 0; column < n; ++column) {
        const std::size_t first_row = diagonal ? column : column + 1;
        for (std::size_t row = first_row; row < n; ++row) {
            pairs.push_back({
                row + (column * n),
                column + (row * n),
                row == column
            });
        }
    }

    return pairs;
}

void duplication_pre_impl(std::span<const double> input,
                          std::span<double> output,
                          const std::size_t n,
                          const std::size_t input_columns,
                          const bool diagonal,
                          const bool average_pairs)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    const std::vector<SymmetricMatrixPair> pairs =
        symmetric_matrix_pairs(n, diagonal);
    const std::size_t pair_count = pairs.size();

    if (input.size() != checked_matrix_size(n2, input_columns) ||
        output.size() != checked_matrix_size(pair_count, input_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < input_columns; ++column) {
        const std::size_t input_column_offset = column * n2;
        const std::size_t output_column_offset = column * pair_count;

        for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
            const SymmetricMatrixPair& pair = pairs[pair_index];
            double value = input[pair.lower_index + input_column_offset] +
                           input[pair.upper_index + input_column_offset];
            if (average_pairs || pair.diagonal) {
                value *= 0.5;
            }

            output[pair_index + output_column_offset] = value;
        }
    }
}

void duplication_post_impl(std::span<const double> input,
                           std::span<double> output,
                           const std::size_t n,
                           const std::size_t input_rows,
                           const bool diagonal,
                           const bool average_pairs)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    const std::vector<SymmetricMatrixPair> pairs =
        symmetric_matrix_pairs(n, diagonal);
    const std::size_t pair_count = pairs.size();

    if (input.size() != checked_matrix_size(input_rows, n2) ||
        output.size() != checked_matrix_size(input_rows, pair_count)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
        const SymmetricMatrixPair& pair = pairs[pair_index];
        const std::size_t lower_column_offset = pair.lower_index * input_rows;
        const std::size_t upper_column_offset = pair.upper_index * input_rows;
        const std::size_t output_column_offset = pair_index * input_rows;

        for (std::size_t row = 0; row < input_rows; ++row) {
            double value = input[row + lower_column_offset] +
                           input[row + upper_column_offset];
            if (average_pairs || pair.diagonal) {
                value *= 0.5;
            }

            output[row + output_column_offset] = value;
        }
    }
}

void duplication_pre_post_impl(std::span<const double> input,
                               std::span<double> output,
                               const std::size_t n,
                               const bool diagonal,
                               const bool average_pairs)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    const std::vector<SymmetricMatrixPair> pairs =
        symmetric_matrix_pairs(n, diagonal);
    const std::size_t pair_count = pairs.size();

    if (input.size() != checked_matrix_size(n2, n2) ||
        output.size() != checked_matrix_size(pair_count, pair_count)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    NumericVector work(checked_matrix_size(pair_count, n2));
    duplication_pre_impl(input, work, n, n2, diagonal, average_pairs);
    duplication_post_impl(work, output, n, pair_count, diagonal, average_pairs);
}

}  // namespace

void vecr(std::span<const double> input,
          std::span<double> output,
          const std::size_t rows,
          const std::size_t columns)
{
    const std::size_t size = checked_matrix_size(rows, columns);
    if (input.size() != size || output.size() != size) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[(row * columns) + column] = input[row + (column * rows)];
        }
    }
}

NumericVector vecr(std::span<const double> input,
                   const std::size_t rows,
                   const std::size_t columns)
{
    NumericVector output(checked_matrix_size(rows, columns));
    std::span<double> output_view(output.data(), output.size());
    vecr(input, output_view, rows, columns);
    return output;
}

void crossprod_pairwise(std::span<const double> lhs,
                        std::span<const double> rhs,
                        std::span<double> output,
                        const std::size_t rows,
                        const std::size_t lhs_columns,
                        const std::size_t rhs_columns)
{
    if (lhs.size() != checked_matrix_size(rows, lhs_columns) ||
        rhs.size() != checked_matrix_size(rows, rhs_columns) ||
        output.size() != checked_matrix_size(lhs_columns, rhs_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    if (output.empty()) {
        return;
    }

    if (rows == 0) {
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }

    NumericVector lhs_clean(lhs.size());
    NumericVector rhs_clean(rhs.size());
    const bool lhs_has_infinite = copy_without_nan(lhs, lhs_clean);
    const bool rhs_has_infinite = copy_without_nan(rhs, rhs_clean);
    if (!lhs_has_infinite && !rhs_has_infinite) {
        crossprod_linalg(lhs_clean, rhs_clean, output, rows, lhs_columns, rhs_columns);
        return;
    }

    NumericVector lhs_work(rows);
    NumericVector rhs_work(rows);
    for (std::size_t rhs_column = 0; rhs_column < rhs_columns; ++rhs_column) {
        for (std::size_t lhs_column = 0; lhs_column < lhs_columns; ++lhs_column) {
            copy_valid_pairwise_terms(lhs,
                                      rhs,
                                      lhs_work,
                                      rhs_work,
                                      rows,
                                      lhs_column,
                                      rhs_column);
            output[lhs_column + (rhs_column * lhs_columns)] =
                utils::crossprod(lhs_work, rhs_work);
        }
    }
}

void block_diagonal(const std::vector<std::span<const double>>& inputs,
                    std::span<double> output,
                    const std::vector<std::size_t>& rows,
                    const std::vector<std::size_t>& columns,
                    const std::size_t total_rows,
                    const std::size_t total_columns)
{
    const std::size_t matrix_count = inputs.size();
    if (rows.size() != matrix_count || columns.size() != matrix_count ||
        output.size() != checked_matrix_size(total_rows, total_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    std::fill(output.begin(), output.end(), 0.0);

    std::size_t row_offset = 0;
    std::size_t column_offset = 0;
    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        const std::size_t block_rows = rows[matrix_index];
        const std::size_t block_columns = columns[matrix_index];
        const std::span<const double> block = inputs[matrix_index];
        if (block.size() != checked_matrix_size(block_rows, block_columns)) {
            throw std::invalid_argument("matrix storage size does not match dimensions");
        }

        for (std::size_t column = 0; column < block_columns; ++column) {
            for (std::size_t row = 0; row < block_rows; ++row) {
                output[(row_offset + row) +
                       ((column_offset + column) * total_rows)] =
                    block[row + (column * block_rows)];
            }
        }

        row_offset += block_rows;
        column_offset += block_columns;
    }
}

void commutation_matrix(std::span<double> output,
                        const std::size_t rows,
                        const std::size_t columns)
{
    const std::size_t size = checked_matrix_size(rows, columns);
    if (output.size() != checked_matrix_size(size, size)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    std::fill(output.begin(), output.end(), 0.0);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t input_index = row + (column * rows);
            const std::size_t output_index = column + (row * columns);
            output[output_index + (input_index * size)] = 1.0;
        }
    }
}

void commutation_pre(std::span<const double> input,
                     std::span<double> output,
                     const std::size_t n,
                     const std::size_t input_columns)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    if (input.size() != checked_matrix_size(n2, input_columns) ||
        output.size() != checked_matrix_size(n2, input_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < input_columns; ++column) {
        const std::size_t column_offset = column * n2;
        for (std::size_t row = 0; row < n2; ++row) {
            const std::size_t source_row = (row % n) * n + (row / n);
            output[row + column_offset] = input[source_row + column_offset];
        }
    }
}

void commutation_post(std::span<const double> input,
                      std::span<double> output,
                      const std::size_t input_rows,
                      const std::size_t n)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    if (input.size() != checked_matrix_size(input_rows, n2) ||
        output.size() != checked_matrix_size(input_rows, n2)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < n2; ++column) {
        const std::size_t source_column = (column % n) * n + (column / n);
        const std::size_t source_offset = source_column * input_rows;
        const std::size_t output_offset = column * input_rows;
        for (std::size_t row = 0; row < input_rows; ++row) {
            output[row + output_offset] = input[row + source_offset];
        }
    }
}

void commutation_pre_post(std::span<const double> input,
                          std::span<double> output,
                          const std::size_t n)
{
    const std::size_t n2 = checked_matrix_size(n, n);
    if (input.size() != checked_matrix_size(n2, n2) ||
        output.size() != checked_matrix_size(n2, n2)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < n2; ++column) {
        const std::size_t source_column = (column % n) * n + (column / n);
        for (std::size_t row = 0; row < n2; ++row) {
            const std::size_t source_row = (row % n) * n + (row / n);
            output[row + (column * n2)] =
                input[source_row + (source_column * n2)];
        }
    }
}

void commutation_mn_pre(std::span<const double> input,
                        std::span<double> output,
                        const std::size_t m,
                        const std::size_t n,
                        const std::size_t input_columns)
{
    const std::size_t mn = checked_matrix_size(m, n);
    if (input.size() != checked_matrix_size(mn, input_columns) ||
        output.size() != checked_matrix_size(mn, input_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < input_columns; ++column) {
        const std::size_t column_offset = column * mn;
        for (std::size_t row = 0; row < mn; ++row) {
            const std::size_t source_row = (row % n) * m + (row / n);
            output[row + column_offset] = input[source_row + column_offset];
        }
    }
}

void delta_a_delta(std::span<const double> delta,
                   std::span<const double> a,
                   std::span<double> output,
                   const std::size_t delta_rows,
                   const std::size_t delta_columns)
{
    if (delta.size() != checked_matrix_size(delta_rows, delta_columns) ||
        a.size() != checked_matrix_size(delta_rows, delta_rows) ||
        output.size() != checked_matrix_size(delta_columns, delta_columns)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
        }

    if (output.empty()) {
        return;
    }
    if (delta_rows == 0) {
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }

    const ConstColumnMajorMatrixView delta_view(delta.data(), delta_rows, delta_columns);
    const ConstColumnMajorMatrixView a_view(a.data(), delta_rows, delta_rows);

    NumericVector delta_t_a(checked_matrix_size(delta_columns, delta_rows));
    const ColumnMajorMatrixView delta_t_a_view(delta_t_a.data(), delta_columns, delta_rows);
    const ColumnMajorMatrixView output_view(output.data(), delta_columns, delta_columns);

    utils::detail::linalg::matrix_product(
        utils::detail::linalg::transposed(delta_view),
        a_view,
        delta_t_a_view);

    utils::detail::linalg::matrix_product(
        delta_t_a_view,
        delta_view,
        output_view);
}

void diag_prepost(std::span<const double> input,
                  std::span<const double> diagonal,
                  std::span<double> output,
                  const std::size_t n)
{
    if (input.size() != checked_matrix_size(n, n) ||
        diagonal.size() != n || output.size() != checked_matrix_size(n, n))
    {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }

    for (std::size_t column = 0; column < n; ++column)
    {
        const double column_scale = diagonal[column];
        for (std::size_t row = 0; row < n; ++row)
        {
            const std::size_t index = row + (column * n);
            output[index] = (input[index] * diagonal[row]) * column_scale;
        }
    }
}

void lisrel_sigma(std::span<const double> lambda,
                  std::span<const double> psi,
                  std::span<const double> theta,
                  std::span<const double> delta,
                  std::span<double> output,
                  const std::size_t nvar,
                  const std::size_t nfac)
{
    if (lambda.size() != checked_matrix_size(nvar, nfac) ||
        psi.size() != checked_matrix_size(nfac, nfac) ||
        theta.size() != checked_matrix_size(nvar, nvar) ||
        output.size() != checked_matrix_size(nvar, nvar)) {
        throw std::invalid_argument("matrix storage size does not match dimensions");
    }
    if (!delta.empty() && delta.size() != nvar) {
        throw std::invalid_argument("delta length must equal nrow(lambda)");
    }

    const ConstColumnMajorMatrixView lambda_view(lambda.data(), nvar, nfac);
    const ConstColumnMajorMatrixView psi_view(psi.data(), nfac, nfac);
    ColumnMajorMatrixView output_view(output.data(), nvar, nvar);

    NumericVector lambda_psi(checked_matrix_size(nvar, nfac));
    const ColumnMajorMatrixView lambda_psi_view(lambda_psi.data(), nvar, nfac);
    utils::detail::linalg::matrix_product(
        lambda_view,
        psi_view,
        lambda_psi_view);
    utils::detail::linalg::matrix_product(
        lambda_psi_view,
        utils::detail::linalg::transposed(lambda_view),
        output_view);

    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] += theta[index];
    }

    if (!delta.empty()) {
        for (std::size_t column = 0; column < nvar; ++column) {
            const double column_scale = delta[column];
            const std::size_t column_offset = column * nvar;
            for (std::size_t row = 0; row < nvar; ++row) {
                output[row + column_offset] *= delta[row] * column_scale;
            }
        }
    }
}

void duplication_pre(std::span<const double> input,
                     std::span<double> output,
                     const std::size_t n,
                     const std::size_t input_columns)
{
    duplication_pre_impl(input, output, n, input_columns, true, false);
}

void duplication_post(std::span<const double> input,
                      std::span<double> output,
                      const std::size_t n,
                      const std::size_t input_rows)
{
    duplication_post_impl(input, output, n, input_rows, true, false);
}

void duplication_pre_post(std::span<const double> input,
                          std::span<double> output,
                          const std::size_t n)
{
    duplication_pre_post_impl(input, output, n, true, false);
}

void duplication_cor_pre_post(std::span<const double> input,
                              std::span<double> output,
                              const std::size_t n)
{
    duplication_pre_post_impl(input, output, n, false, false);
}

void duplication_ginv_pre(std::span<const double> input,
                          std::span<double> output,
                          const std::size_t n,
                          const std::size_t input_columns)
{
    duplication_pre_impl(input, output, n, input_columns, true, true);
}

void duplication_ginv_post(std::span<const double> input,
                           std::span<double> output,
                           const std::size_t n,
                           const std::size_t input_rows)
{
    duplication_post_impl(input, output, n, input_rows, true, true);
}

void duplication_ginv_pre_post(std::span<const double> input,
                               std::span<double> output,
                               const std::size_t n)
{
    duplication_pre_post_impl(input, output, n, true, true);
}

}  // namespace lavaan::cpp
