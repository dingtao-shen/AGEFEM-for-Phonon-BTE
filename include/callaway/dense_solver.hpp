#pragma once

#include <vector>

namespace callaway
{

void FactorDenseMatrixInPlace(double *matrix, int *pivots, int size);

void SolveDenseFactoredSystem(const double *lu,
                              const int *pivots,
                              int size,
                              std::vector<double> &rhs);

void SolveDenseLinearSystem(std::vector<double> &matrix,
                            std::vector<double> &rhs,
                            int size);

} // namespace callaway
