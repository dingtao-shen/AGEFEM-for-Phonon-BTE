#include "callaway/dense_solver.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace
{

void CheckClose(double actual, double expected)
{
   assert(std::abs(actual - expected) <= 1.0e-13);
}

} // namespace

int main()
{
   std::vector<double> matrix = {
      0.0, 2.0,
      1.0, 1.0
   };
   std::vector<double> rhs = {4.0, 3.0};
   callaway::SolveDenseLinearSystem(matrix, rhs, 2);
   CheckClose(rhs[0], 1.0);
   CheckClose(rhs[1], 2.0);

   std::vector<double> lu = {
      2.0, 1.0, -1.0,
      -3.0, -1.0, 2.0,
      -2.0, 1.0, 2.0
   };
   std::vector<int> pivots(3, 0);
   callaway::FactorDenseMatrixInPlace(lu.data(), pivots.data(), 3);

   std::vector<double> rhs2 = {8.0, -11.0, -3.0};
   callaway::SolveDenseFactoredSystem(lu.data(), pivots.data(), 3, rhs2);
   CheckClose(rhs2[0], 2.0);
   CheckClose(rhs2[1], 3.0);
   CheckClose(rhs2[2], -1.0);

   std::vector<double> rhs3 = {-3.0, 5.0, 2.0};
   callaway::SolveDenseFactoredSystem(lu.data(), pivots.data(), 3, rhs3);
   CheckClose(rhs3[0], 1.0);
   CheckClose(rhs3[1], -2.0);
   CheckClose(rhs3[2], 3.0);

   return 0;
}
