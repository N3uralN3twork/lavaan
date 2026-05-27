#include "library.h"
#include <iostream>
#include <vector>

void hello()
{
    std::cout << "Hello, World!" << std::endl;
}

void create_empty_vector()
{
    std::vector<float> vec;
}

void transpose_vector()
{
    const lavaan::utils::Matrix matrix{{1.0, 2.0}, {3.0, 4.0}};
    const auto transposed = lavaan::utils::transpose(matrix);
    (void)transposed;
}

void crossprod_vector()
{
    const std::vector<double> left = {1.0, 2.0, 3.0, 4.0};
    const auto crossproduct = lavaan::utils::crossprod(left, left);
    (void)crossproduct;
}