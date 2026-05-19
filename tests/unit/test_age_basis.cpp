#include "callaway/age_basis.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/geometry/circular_arc.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

void CheckClose(double actual, double expected, double tol, const char *what)
{
   if (!(std::abs(actual - expected) <= tol))
   {
      std::cerr << "FAIL " << what << ": got " << actual << ", expected " << expected
                << ", diff " << std::abs(actual - expected) << "\n";
      std::abort();
   }
}

// Build a quarter-disk AGE element: interior vertex at the origin, curve a
// CCW unit-circle arc from (1, 0) to (0, 1), so the parameter interval is
// [0, 0.25]. Local face 1 is the curved one (face 0 from interior to (1,0)
// is straight, face 1 from (1,0) to (0,1) is the arc, face 2 from (0,1)
// back to interior is straight).
struct QuarterDiskFixture
{
   std::unique_ptr<callaway::CircularArc> arc;
   callaway::AgeElementGeometry geom;

   QuarterDiskFixture()
   {
      arc = std::make_unique<callaway::CircularArc>(
         callaway::CurvePoint{{0.0, 0.0}}, 1.0, +1);
      geom.element = 0;
      geom.curved_local_face = 1;
      geom.curve = arc.get();
      geom.parameter_interval = {0.0, 0.25};
      geom.interior_vertex = {{0.0, 0.0}};
      geom.curve_begin = {{1.0, 0.0}};
      geom.curve_end = {{0.0, 1.0}};
   }
};

void TestKroneckerDelta(int order)
{
   QuarterDiskFixture fix;
   const callaway::AgeElementBasis basis(order, fix.geom);
   const int n = basis.dofs();
   for (int l = 0; l < n; ++l)
   {
      for (int i = 0; i < n; ++i)
      {
         const auto &node = basis.nodes()[i];
         const double v = basis.Evaluate(l, node[0], node[1]);
         const double expected = (l == i) ? 1.0 : 0.0;
         CheckClose(v, expected, 1.0e-10, "kronecker delta");
      }
   }
}

void TestPartitionOfUnity(int order)
{
   QuarterDiskFixture fix;
   const callaway::AgeElementBasis basis(order, fix.geom);
   // Sample at a few interior points of the quarter-disk.
   const double samples[][2] = {
      {0.3, 0.3},   // interior
      {0.5, 0.2},
      {0.2, 0.5},
      {0.1, 0.1},
      {0.6, 0.3},
      {0.4, 0.5},
      {0.0, 0.0},   // interior vertex (a node, but partition of unity must hold)
      {1.0, 0.0},   // curve_begin (also a node)
      {0.0, 1.0}    // curve_end
   };
   for (const auto &s : samples)
   {
      const auto values = basis.EvaluateAll(s[0], s[1]);
      double sum = 0.0;
      for (double v : values) { sum += v; }
      CheckClose(sum, 1.0, 1.0e-9, "partition of unity");
   }
}

void TestGradientFiniteDifference(int order)
{
   QuarterDiskFixture fix;
   const callaway::AgeElementBasis basis(order, fix.geom);
   const double h = 1.0e-5;
   // Test at a few points that are clearly inside the element.
   const double samples[][2] = {
      {0.3, 0.3}, {0.5, 0.2}, {0.2, 0.5}, {0.4, 0.4}
   };
   for (const auto &s : samples)
   {
      for (int l = 0; l < basis.dofs(); ++l)
      {
         const auto grad = basis.EvaluateGradient(l, s[0], s[1]);
         const double fd_x = (basis.Evaluate(l, s[0] + h, s[1]) -
                              basis.Evaluate(l, s[0] - h, s[1])) / (2.0 * h);
         const double fd_y = (basis.Evaluate(l, s[0], s[1] + h) -
                              basis.Evaluate(l, s[0], s[1] - h)) / (2.0 * h);
         // FD tolerance scales with order: at high order, basis values can be large in magnitude.
         const double tol = 1.0e-5 * (1.0 + std::abs(fd_x) + std::abs(grad[0]));
         CheckClose(grad[0], fd_x, tol, "gradient dx vs FD");
         CheckClose(grad[1], fd_y, tol, "gradient dy vs FD");
      }
   }
}

void TestNodeCount(int order)
{
   QuarterDiskFixture fix;
   const callaway::AgeElementBasis basis(order, fix.geom);
   const int expected = (order + 1) * (order + 2) / 2;
   if (basis.dofs() != expected || static_cast<int>(basis.nodes().size()) != expected)
   {
      std::cerr << "FAIL node count for order " << order << "\n";
      std::abort();
   }
   // The two endpoints of the curved edge (i=order, j=0) and (i=0, j=order)
   // map to curve_begin and curve_end respectively. The interior vertex is
   // at (i=0, j=0) -> first node in the lexicographic enumeration.
   CheckClose(basis.nodes()[0][0], 0.0, 1.0e-15, "interior vertex node x");
   CheckClose(basis.nodes()[0][1], 0.0, 1.0e-15, "interior vertex node y");
   // The node at reference (i=order, j=0) is the (order)-th node in the
   // i-major ordering (j=0 row); that maps to curve_begin.
   const int curve_begin_idx = order;
   CheckClose(basis.nodes()[curve_begin_idx][0], 1.0, 1.0e-12, "curve_begin node x");
   CheckClose(basis.nodes()[curve_begin_idx][1], 0.0, 1.0e-12, "curve_begin node y");
   // The node at reference (i=0, j=order) is the last node.
   const int curve_end_idx = expected - 1;
   CheckClose(basis.nodes()[curve_end_idx][0], 0.0, 1.0e-12, "curve_end node x");
   CheckClose(basis.nodes()[curve_end_idx][1], 1.0, 1.0e-12, "curve_end node y");
}

} // namespace

int main()
{
   for (int order = 1; order <= 4; ++order)
   {
      TestNodeCount(order);
      TestKroneckerDelta(order);
      TestPartitionOfUnity(order);
      TestGradientFiniteDifference(order);
   }
   std::cout << "test_age_basis: all checks passed.\n";
   return 0;
}
