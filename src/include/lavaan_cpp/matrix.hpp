#ifndef LAVAAN_CPP_MATRIX_HPP
#define LAVAAN_CPP_MATRIX_HPP

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace lavaan::cpp {

using NumericVector = std::vector<double>;

void vecr(std::span<const double> input,
          std::span<double> output,
          std::size_t rows,
          std::size_t columns);

[[nodiscard]] NumericVector vecr(std::span<const double> input,
                                 std::size_t rows,
                                 std::size_t columns);

void crossprod_pairwise(std::span<const double> lhs,
                        std::span<const double> rhs,
                        std::span<double> output,
                        std::size_t rows,
                        std::size_t lhs_columns,
                        std::size_t rhs_columns);

void delta_a_delta(std::span<const double> delta,
                   std::span<const double> a,
                   std::span<double> output,
                   std::size_t delta_rows,
                   std::size_t delta_columns);

void duplication_pre(std::span<const double> input,
                     std::span<double> output,
                     std::size_t n,
                     std::size_t input_columns);

void duplication_post(std::span<const double> input,
                      std::span<double> output,
                      std::size_t n,
                      std::size_t input_rows);

void duplication_pre_post(std::span<const double> input,
                          std::span<double> output,
                          std::size_t n);

void duplication_cor_pre_post(std::span<const double> input,
                              std::span<double> output,
                              std::size_t n);

void duplication_ginv_pre(std::span<const double> input,
                          std::span<double> output,
                          std::size_t n,
                          std::size_t input_columns);

void duplication_ginv_post(std::span<const double> input,
                           std::span<double> output,
                           std::size_t n,
                           std::size_t input_rows);

void duplication_ginv_pre_post(std::span<const double> input,
                               std::span<double> output,
                               std::size_t n);

void diag_prepost(std::span<const double> input,
    std::span<const double> diagonal,
    std::span<double> output,
    std::size_t n);

}  // namespace lavaan::cpp

#endif  // LAVAAN_CPP_MATRIX_HPP
