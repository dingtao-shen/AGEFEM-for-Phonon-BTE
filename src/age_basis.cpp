#include "callaway/age_basis.hpp"

#include "callaway/dense_solver.hpp"
#include "callaway/nodal_basis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace callaway
{
namespace
{

// Psi_n: map a reference-triangle node (xi, eta) in [0,1] x [0,1] with
// xi + eta <= 1 to the physical AGE element. Reference vertices map as
//   (0, 0)  ->  interior_vertex (x_0)
//   (1, 0)  ->  curve(lambda_1) = curve_begin (x_1)
//   (0, 1)  ->  curve(lambda_2) = curve_end   (x_2)
// For an interior reference node (s = xi + eta in (0, 1]), the physical
// position is the blend of x_0 with the curve evaluated at the parameter
// (xi * lambda_1 + eta * lambda_2) / s, weighted by (1 - s) and s
// respectively. This reproduces the affine map on the two straight edges
// (where one of xi, eta is zero) and lies on the curve when s = 1.
CurvePoint EvaluatePsiN(double xi, double eta, const AgeElementGeometry &geom)
{
   const double s = xi + eta;
   if (s <= 0.0)
   {
      return geom.interior_vertex;
   }
   const double lam_1 = geom.parameter_interval.begin;
   const double lam_2 = geom.parameter_interval.end;
   const double lambda = (xi * lam_1 + eta * lam_2) / s;
   const CurvePoint c = geom.curve->Point(lambda);
   return {(1.0 - s) * geom.interior_vertex[0] + s * c[0],
           (1.0 - s) * geom.interior_vertex[1] + s * c[1]};
}

} // namespace

AgeElementBasis::AgeElementBasis(int order, const AgeElementGeometry &geometry)
   : order_(order),
     dofs_(NodalBasis::TriangleDofs(order))
{
   if (order_ < 1 || order_ > 4)
   {
      throw std::runtime_error("AgeElementBasis: only orders 1 through 4 are supported.");
   }
   if (geometry.curve == nullptr)
   {
      throw std::runtime_error("AgeElementBasis: AgeElementGeometry has no bound curve.");
   }

   // 1. Build Psi_n-placed interpolation nodes. Same lexicographic order
   //    as NodalBasis::triangle_nodes_ so that nodal-ordering downstream
   //    code stays consistent.
   nodes_.reserve(static_cast<std::size_t>(dofs_));
   for (int j = 0; j <= order_; ++j)
   {
      for (int i = 0; i <= order_ - j; ++i)
      {
         const double xi  = static_cast<double>(i) / static_cast<double>(order_);
         const double eta = static_cast<double>(j) / static_cast<double>(order_);
         nodes_.push_back(EvaluatePsiN(xi, eta, geometry));
      }
   }

   // 2. Centroid + characteristic length for the shifted-scaled monomial basis.
   centroid_ = {0.0, 0.0};
   for (const CurvePoint &p : nodes_)
   {
      centroid_[0] += p[0];
      centroid_[1] += p[1];
   }
   centroid_[0] /= static_cast<double>(dofs_);
   centroid_[1] /= static_cast<double>(dofs_);

   double max_dist = 0.0;
   for (const CurvePoint &p : nodes_)
   {
      const double dx = p[0] - centroid_[0];
      const double dy = p[1] - centroid_[1];
      max_dist = std::max(max_dist, std::hypot(dx, dy));
   }
   length_scale_ = (max_dist > 0.0) ? max_dist : 1.0;

   // 3. Build the Vandermonde matrix M_{i, m} = monomial_m at node i, in
   //    shifted-scaled coordinates (u, v) = ((x - xc)/h, (y - yc)/h).
   //    Monomial order matches NodalBasis::MonomialIndex: 1, x, y, x^2, xy, y^2, ...
   const std::size_t n2 = static_cast<std::size_t>(dofs_) * static_cast<std::size_t>(dofs_);
   std::vector<double> vandermonde(n2, 0.0);
   for (int i = 0; i < dofs_; ++i)
   {
      const double u = (nodes_[static_cast<std::size_t>(i)][0] - centroid_[0]) / length_scale_;
      const double v = (nodes_[static_cast<std::size_t>(i)][1] - centroid_[1]) / length_scale_;
      int m = 0;
      for (int total = 0; total <= order_; ++total)
      {
         for (int yp = 0; yp <= total; ++yp)
         {
            const int xp = total - yp;
            vandermonde[static_cast<std::size_t>(i * dofs_ + m)] =
               std::pow(u, xp) * std::pow(v, yp);
            ++m;
         }
      }
   }

   // 4. Solve  V * c_l = e_l  for each basis l, where c_l is the row of
   //    coefficients for basis function l. Factor once, solve dofs_ times.
   //    The row-major Vandermonde V (i = row = node, m = col = monomial)
   //    matches the layout FactorDenseMatrixInPlace expects (row-major
   //    n x n stored as size n*n).
   coefficients_.assign(n2, 0.0);
   std::vector<int> pivots(static_cast<std::size_t>(dofs_), 0);
   FactorDenseMatrixInPlace(vandermonde.data(), pivots.data(), dofs_);

   std::vector<double> rhs(static_cast<std::size_t>(dofs_), 0.0);
   for (int l = 0; l < dofs_; ++l)
   {
      std::fill(rhs.begin(), rhs.end(), 0.0);
      rhs[static_cast<std::size_t>(l)] = 1.0;
      SolveDenseFactoredSystem(vandermonde.data(), pivots.data(), dofs_, rhs);
      for (int m = 0; m < dofs_; ++m)
      {
         coefficients_[static_cast<std::size_t>(l * dofs_ + m)] = rhs[static_cast<std::size_t>(m)];
      }
   }
}

double AgeElementBasis::Evaluate(int basis, double x, double y) const
{
   const double u = (x - centroid_[0]) / length_scale_;
   const double v = (y - centroid_[1]) / length_scale_;
   double val = 0.0;
   int m = 0;
   for (int total = 0; total <= order_; ++total)
   {
      for (int yp = 0; yp <= total; ++yp)
      {
         const int xp = total - yp;
         val += coefficients_[static_cast<std::size_t>(basis * dofs_ + m)] *
                std::pow(u, xp) * std::pow(v, yp);
         ++m;
      }
   }
   return val;
}

std::array<double, 2> AgeElementBasis::EvaluateGradient(int basis, double x, double y) const
{
   const double u = (x - centroid_[0]) / length_scale_;
   const double v = (y - centroid_[1]) / length_scale_;
   double du = 0.0;
   double dv = 0.0;
   int m = 0;
   for (int total = 0; total <= order_; ++total)
   {
      for (int yp = 0; yp <= total; ++yp)
      {
         const int xp = total - yp;
         const double c = coefficients_[static_cast<std::size_t>(basis * dofs_ + m)];
         if (xp > 0)
         {
            du += c * static_cast<double>(xp) * std::pow(u, xp - 1) * std::pow(v, yp);
         }
         if (yp > 0)
         {
            dv += c * std::pow(u, xp) * static_cast<double>(yp) * std::pow(v, yp - 1);
         }
         ++m;
      }
   }
   // Chain rule: d/dx = (1/h) d/du, d/dy = (1/h) d/dv.
   return {du / length_scale_, dv / length_scale_};
}

std::vector<double> AgeElementBasis::EvaluateAll(double x, double y) const
{
   std::vector<double> values(static_cast<std::size_t>(dofs_), 0.0);
   for (int l = 0; l < dofs_; ++l)
   {
      values[static_cast<std::size_t>(l)] = Evaluate(l, x, y);
   }
   return values;
}

std::vector<std::array<double, 2>> AgeElementBasis::EvaluateGradientAll(double x, double y) const
{
   std::vector<std::array<double, 2>> grads(static_cast<std::size_t>(dofs_));
   for (int l = 0; l < dofs_; ++l)
   {
      grads[static_cast<std::size_t>(l)] = EvaluateGradient(l, x, y);
   }
   return grads;
}

double AgeElementBasis::Coefficient(int basis, int monomial) const
{
   return coefficients_.at(static_cast<std::size_t>(basis * dofs_ + monomial));
}

std::vector<AgeElementBasis> BuildAgeElementBases(const AgeMesh &age_mesh, int order)
{
   std::vector<AgeElementBasis> result;
   result.reserve(static_cast<std::size_t>(age_mesh.age_element_count()));
   for (const AgeElementGeometry &geom : age_mesh.age_elements())
   {
      result.emplace_back(order, geom);
   }
   return result;
}

} // namespace callaway
