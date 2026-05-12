#include "callaway/dense_solver.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace callaway
{

void FactorDenseMatrixInPlace(double *matrix, int *pivots, int size)
{
   if (size <= 0)
   {
      throw std::runtime_error("Dense LU factorization requires a positive matrix size.");
   }

   constexpr double pivot_tolerance = 1.0e-14;

   for (int col = 0; col < size; ++col)
   {
      int pivot = col;
      double pivot_abs = std::abs(matrix[col * size + col]);
      for (int row = col + 1; row < size; ++row)
      {
         const double candidate_abs = std::abs(matrix[row * size + col]);
         if (candidate_abs > pivot_abs)
         {
            pivot_abs = candidate_abs;
            pivot = row;
         }
      }

      if (pivot_abs <= pivot_tolerance)
      {
         std::ostringstream os;
         os << "Dense linear solve found a singular local matrix at column " << col << ".";
         throw std::runtime_error(os.str());
      }

      pivots[col] = pivot;
      if (pivot != col)
      {
         for (int j = 0; j < size; ++j)
         {
            std::swap(matrix[col * size + j], matrix[pivot * size + j]);
         }
      }

      const double diagonal = matrix[col * size + col];
      for (int row = col + 1; row < size; ++row)
      {
         const double factor = matrix[row * size + col] / diagonal;
         matrix[row * size + col] = factor;
         for (int j = col + 1; j < size; ++j)
         {
            matrix[row * size + j] -= factor * matrix[col * size + j];
         }
      }
   }
}

void SolveDenseFactoredSystem(const double *lu,
                              const int *pivots,
                              int size,
                              std::vector<double> &rhs)
{
   if (size <= 0)
   {
      throw std::runtime_error("Dense factored solve requires a positive matrix size.");
   }
   if (static_cast<int>(rhs.size()) != size)
   {
      throw std::runtime_error("Dense factored solve received an inconsistent RHS dimension.");
   }

   for (int col = 0; col < size; ++col)
   {
      if (pivots[col] != col)
      {
         std::swap(rhs[static_cast<std::size_t>(col)],
                   rhs[static_cast<std::size_t>(pivots[col])]);
      }
   }

   for (int row = 0; row < size; ++row)
   {
      double value = rhs[static_cast<std::size_t>(row)];
      for (int col = 0; col < row; ++col)
      {
         value -= lu[row * size + col] * rhs[static_cast<std::size_t>(col)];
      }
      rhs[static_cast<std::size_t>(row)] = value;
   }

   for (int row = size - 1; row >= 0; --row)
   {
      double value = rhs[static_cast<std::size_t>(row)];
      for (int col = row + 1; col < size; ++col)
      {
         value -= lu[row * size + col] * rhs[static_cast<std::size_t>(col)];
      }
      rhs[static_cast<std::size_t>(row)] =
         value / lu[row * size + row];
   }
}

void SolveDenseLinearSystem(std::vector<double> &matrix,
                            std::vector<double> &rhs,
                            int size)
{
   if (static_cast<int>(matrix.size()) != size * size ||
       static_cast<int>(rhs.size()) != size)
   {
      throw std::runtime_error("Dense linear solve received inconsistent dimensions.");
   }

   std::vector<int> pivots(static_cast<std::size_t>(size), 0);
   FactorDenseMatrixInPlace(matrix.data(), pivots.data(), size);
   SolveDenseFactoredSystem(matrix.data(), pivots.data(), size, rhs);
}

} // namespace callaway
