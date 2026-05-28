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

void block_diagonal(const std::vector<std::span<const double>>& inputs,
                    std::span<double> output,
                    const std::vector<std::size_t>& rows,
                    const std::vector<std::size_t>& columns,
                    std::size_t total_rows,
                    std::size_t total_columns);

void commutation_matrix(std::span<double> output,
                        std::size_t rows,
                        std::size_t columns);

void commutation_pre(std::span<const double> input,
                     std::span<double> output,
                     std::size_t n,
                     std::size_t input_columns);

void commutation_post(std::span<const double> input,
                      std::span<double> output,
                      std::size_t input_rows,
                      std::size_t n);

void commutation_pre_post(std::span<const double> input,
                          std::span<double> output,
                          std::size_t n);

void commutation_mn_pre(std::span<const double> input,
                        std::span<double> output,
                        std::size_t m,
                        std::size_t n,
                        std::size_t input_columns);

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

void lisrel_sigma(std::span<const double> lambda,
                  std::span<const double> psi,
                  std::span<const double> theta,
                  std::span<const double> delta,
                  std::span<double> output,
                  std::size_t nvar,
                  std::size_t nfac);

}  // namespace lavaan::cpp

#endif  // LAVAAN_CPP_MATRIX_HPP
