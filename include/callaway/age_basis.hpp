#pragma once

#include "callaway/age_mesh.hpp"
#include "callaway/geometry/boundary_curve.hpp"

#include <array>
#include <vector>

namespace callaway
{

// Physical-coordinate nodal Lagrangian basis for a single AGE element.
//
// Unlike the shared reference-triangle NodalBasis used for straight
// elements, an AGE element's basis is element-specific: it is defined
// directly on the curved physical element as
//
//     psi_l(x, y) = sum_m c_{l,m} * monomial_m(u, v),
//
// where (u, v) = ((x - xc)/h, (y - yc)/h) is the shifted-and-scaled local
// coordinate (xc = element centroid, h = characteristic length). The
// shift+scale spans the same polynomial space but yields a far better
// conditioned interpolation Vandermonde at orders 3-4 than raw physical
// monomials, addressing the conditioning the paper itself flags.
//
// Coefficients are solved from the Kronecker condition
//     psi_l(x_{n,i}) = delta_{li}
// at nodes placed on the physical (curved) element by the map Psi_n
// described in section 4.2 of the paper.
class AgeElementBasis
{
public:
   AgeElementBasis(int order, const AgeElementGeometry &geometry);

   int order() const { return order_; }
   int dofs() const { return dofs_; }

   // Interpolation nodes on the physical (curved) element, Psi_n-placed.
   const std::vector<CurvePoint> &nodes() const { return nodes_; }

   // Local shift / scale of the conditioned monomial basis.
   CurvePoint centroid() const { return centroid_; }
   double length_scale() const { return length_scale_; }

   // Evaluate basis function `basis` (and its gradient) at a physical
   // point (x, y). Caller is responsible for ensuring the point is in or
   // on the AGE element; outside, the polynomial is mathematically defined
   // but loses its nodal interpretation.
   double Evaluate(int basis, double x, double y) const;
   std::array<double, 2> EvaluateGradient(int basis, double x, double y) const;

   // Evaluate all basis functions at a physical point. Length = dofs().
   std::vector<double> EvaluateAll(double x, double y) const;
   std::vector<std::array<double, 2>> EvaluateGradientAll(double x, double y) const;

   // Raw coefficient access: row = basis index, col = monomial index in
   // the (u, v) shifted-scaled basis ordered as 1, u, v, u^2, uv, v^2, ...
   // (the same total-degree monomial order used elsewhere in the project).
   double Coefficient(int basis, int monomial) const;

private:
   int order_ = 0;
   int dofs_ = 0;
   CurvePoint centroid_{{0.0, 0.0}};
   double length_scale_ = 1.0;
   std::vector<CurvePoint> nodes_;
   std::vector<double> coefficients_; // dofs_ x dofs_, row-major
};

// Build per-AGE-element bases for every AGE element of `age_mesh`. The
// returned vector is parallel to `age_mesh.age_elements()`. Straight
// elements continue to use the shared NodalBasis.
std::vector<AgeElementBasis> BuildAgeElementBases(const AgeMesh &age_mesh, int order);

} // namespace callaway
