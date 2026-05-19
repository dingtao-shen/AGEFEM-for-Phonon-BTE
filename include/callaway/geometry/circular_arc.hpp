#pragma once

#include "callaway/geometry/boundary_curve.hpp"

namespace callaway
{

// Exact circular boundary: C(lambda) on a circle of given center and
// radius. The parameter lambda runs over [0, 1] (full circle).
//
// orientation = +1 traces the circle counter-clockwise with increasing
// lambda, -1 clockwise. This fixes the domain-outward normal sense:
// for a hole the material lies outside the circle, so the domain-outward
// normal points toward the center; for a disk's outer boundary it points
// away from the center.
class CircularArc final : public BoundaryCurve
{
public:
   CircularArc(CurvePoint center, double radius, int orientation);

   CurveKind kind() const override { return CurveKind::Parametric; }
   CurveInterval domain() const override;
   bool is_closed() const override { return true; }

   CurvePoint Point(double lambda) const override;
   CurvePoint Tangent(double lambda) const override;
   CurvePoint Normal(double lambda) const override;
   double ParameterOf(double x, double y) const override;

   const CurvePoint &center() const { return center_; }
   double radius() const { return radius_; }
   int orientation() const { return orientation_; }

private:
   CurvePoint center_{{0.0, 0.0}};
   double radius_ = 0.0;
   int orientation_ = 1;
};

} // namespace callaway
