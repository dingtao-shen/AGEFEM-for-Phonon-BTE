#pragma once

#include "callaway/geometry/boundary_curve.hpp"

#include <vector>

namespace callaway
{

// Exact NURBS boundary curve (non-uniform rational B-spline). Geometry is
// supplied by the geometry sidecar — control points, weights, and knot
// vector — and evaluated with the de Boor algorithm.
//
// The parameter domain is [knots.front(), knots.back()] (the standard
// NURBS parametric range). Inversion (ParameterOf) is performed by point
// projection: a knot-span search to bracket lambda, then Newton iteration
// on (C(lambda) - x) . C'(lambda) = 0.
class NurbsCurve final : public BoundaryCurve
{
public:
   NurbsCurve(int degree,
              std::vector<CurvePoint> control_points,
              std::vector<double> weights,
              std::vector<double> knots,
              bool closed);

   CurveKind kind() const override { return CurveKind::Parametric; }
   CurveInterval domain() const override;
   bool is_closed() const override { return closed_; }

   CurvePoint Point(double lambda) const override;
   CurvePoint Tangent(double lambda) const override;
   CurvePoint Normal(double lambda) const override;
   double ParameterOf(double x, double y) const override;

   int degree() const { return degree_; }
   const std::vector<CurvePoint> &control_points() const { return control_points_; }
   const std::vector<double> &weights() const { return weights_; }
   const std::vector<double> &knots() const { return knots_; }

private:
   int degree_ = 0;
   std::vector<CurvePoint> control_points_;
   std::vector<double> weights_;
   std::vector<double> knots_;
   bool closed_ = false;
};

} // namespace callaway
