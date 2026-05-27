#ifndef LAVAAN_UTILS_H
#define LAVAAN_UTILS_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#if __has_include(<linalg>) && __has_include(<mdspan>)
#include <mdspan>
#include <linalg>
#define LAVAAN_UTILS_HAS_NATIVE_LINALG 1
#elif __has_include(<experimental/linalg>) && __has_include(<mdspan/mdspan.hpp>)
#ifndef MDSPAN_USE_PAREN_OPERATOR
#define MDSPAN_USE_PAREN_OPERATOR 1
#endif
#include <mdspan/mdspan.hpp>
#include <experimental/linalg>
#define LAVAAN_UTILS_HAS_NATIVE_LINALG 0
#else
#error "lavaan utilities require C++26 <linalg> or the configured stdBLAS implementation"
#endif

namespace lavaan::utils {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

namespace detail {

#if LAVAAN_UTILS_HAS_NATIVE_LINALG
namespace md = std;
namespace linalg = std::linalg;
#else
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace linalg = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::linalg;
#endif

using VectorExtents = md::extents<std::size_t, md::dynamic_extent>;
using MatrixExtents = md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent>;
using ConstVectorView = md::mdspan<const double, VectorExtents>;
using MatrixView = md::mdspan<double, MatrixExtents>;
using ConstMatrixView = md::mdspan<const double, MatrixExtents>;

}  // namespace detail

[[nodiscard]] inline Matrix transpose(const Matrix& matrix)
{
    const std::size_t rows = matrix.size();
    if (rows == 0) {
        return {};
    }

    const std::size_t columns = matrix.front().size();
    Vector input_storage(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        if (matrix[row].size() != columns) {
            throw std::invalid_argument("matrix rows must all have the same length");
        }

        for (std::size_t column = 0; column < columns; ++column) {
            input_storage[(row * columns) + column] = matrix[row][column];
        }
    }

    Vector output_storage(rows * columns);
    const detail::ConstMatrixView input(input_storage.data(), rows, columns);
    detail::MatrixView output(output_storage.data(), columns, rows);

    detail::linalg::copy(detail::linalg::transposed(input), output);

    Matrix result(columns, Vector(rows));
    for (std::size_t row = 0; row < columns; ++row) {
        for (std::size_t column = 0; column < rows; ++column) {
            result[row][column] = output_storage[(row * rows) + column];
        }
    }

    return result;
}

[[nodiscard]] inline double crossprod(std::span<const double> lhs, std::span<const double> rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vectors must have the same length");
    }

    const detail::ConstVectorView lhs_view(lhs.data(), lhs.size());
    const detail::ConstVectorView rhs_view(rhs.data(), rhs.size());
    return detail::linalg::dot(lhs_view, rhs_view, 0.0);
}

[[nodiscard]] inline Matrix crossprod(const Matrix& matrix)
{
    const std::size_t rows = matrix.size();
    if (rows == 0) {
        return {};
    }

    const std::size_t columns = matrix.front().size();
    Vector input_storage(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        if (matrix[row].size() != columns) {
            throw std::invalid_argument("matrix rows must all have the same length");
        }

        for (std::size_t column = 0; column < columns; ++column) {
            input_storage[(row * columns) + column] = matrix[row][column];
        }
    }

    Vector output_storage(columns * columns, 0.0);
    const detail::ConstMatrixView input(input_storage.data(), rows, columns);
    detail::MatrixView output(output_storage.data(), columns, columns);

    detail::linalg::matrix_product(detail::linalg::transposed(input), input, output);

    Matrix result(columns, Vector(columns));
    for (std::size_t row = 0; row < columns; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row][column] = output_storage[(row * columns) + column];
        }
    }

    return result;
}

}  // namespace lavaan::utils

#endif // LAVAAN_UTILS_H
